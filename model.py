import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from tqdm import tqdm

from batches_helpers import batch_to_device, load_batch, load_batch_count
from chess_helpers import LEN_POSSIBLE_MOVES
from constants import (
    DROPOUT,
    INPUT_SIZE,
    LR,
    MIN_LR_RATIO,
    MODEL_FILE_PATH,
    MODELS_FOLDER_PATH,
    N_EMBED,
    N_HEADS,
    N_LOOPS,
)


def _run_block(block, x) -> torch.Tensor:
    # Autograd keeps every unroll/layer's activations. Checkpointing recomputes
    # them on backward so peak memory stays about one block, not O(depth).
    if x.requires_grad:
        return activation_checkpoint(block, x, use_reentrant=False)  # type: ignore[no-untyped-call]
    return block(x)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _scale_residual_projections(model, n_layer):
    # GPT-2 residual scaling: keeps deep / looped stacks from exploding at init.
    std = 0.02 / math.sqrt(2 * max(n_layer, 1))
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            nn.init.normal_(module.proj.weight, mean=0.0, std=std)
        elif isinstance(module, FeedForward):
            nn.init.normal_(module.net[2].weight, mean=0.0, std=std)  # type: ignore[attr-defined]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_attn_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_attn_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_attn_embd // n_head
        self.qkv = nn.Linear(n_attn_embd, 3 * n_attn_embd, bias=False)
        self.proj = nn.Linear(n_attn_embd, n_attn_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  # (B, T, n_attn_embd)
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each (B, T, n_attn_embd)
        # (B, T, n_attn_embd) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)  # (B, n_head, T, T)
        # Finite mask: -inf softmax backward NaNs on gfx1200.
        wei = wei.clamp(-50.0, 50.0)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -1.0e4)  # type: ignore[assignment]
        wei = self.attn_dropout(F.softmax(wei, dim=-1))  # (B, n_head, T, T)
        # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, n_attn_embd)
        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))  # (B, T, n_attn_embd)


# Custom LayerNorm since regular LayerNorm backward can spike to 1e18 on gfx1200 causing NaNs
class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, n_attn_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # (B, T, n_attn_embd) -> (B, T, 4 * n_attn_embd)
            nn.Linear(n_attn_embd, 4 * n_attn_embd),
            nn.ReLU(),
            # (B, T, 4 * n_attn_embd) -> (B, T, n_attn_embd)
            nn.Linear(4 * n_attn_embd, n_attn_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_attn_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_attn_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_attn_embd, dropout)
        self.ln1 = LayerNorm(n_attn_embd)
        self.ln2 = LayerNorm(n_attn_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LoopedGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_token_embd,
        n_attn_embd,
        n_head,
        block_size,
        dropout,
        loop_times,
    ):
        super().__init__()
        self.block_size = block_size
        self.loop_times = loop_times
        self.token_embedding = nn.Embedding(vocab_size, n_token_embd)
        self.position_embedding = nn.Embedding(block_size, n_token_embd)
        self.in_proj = (
            nn.Identity()
            if n_token_embd == n_attn_embd
            else nn.Linear(n_token_embd, n_attn_embd)
        )
        self.n_unroll = 2 * loop_times
        self.step_embedding = nn.Embedding(self.n_unroll, n_attn_embd)
        self.block_1 = Block(n_attn_embd, n_head, block_size, dropout)
        self.block_2 = Block(n_attn_embd, n_head, block_size, dropout)
        self.lm_head = nn.Linear(n_attn_embd, vocab_size)
        self.ln_f = LayerNorm(n_attn_embd)
        self.apply(_init_weights)
        _scale_residual_projections(self, self.n_unroll)

    def _apply_illegal_mask(self, logits, illegal_mask):
        # Finite fill: -inf softmax backward NaNs on gfx1200.
        if illegal_mask.ndim == 1:
            illegal_mask = illegal_mask.unsqueeze(0)
        if illegal_mask.ndim == 2:
            logits = logits.clone()
            logits[:, -1, :] = logits[:, -1, :].masked_fill(illegal_mask, -1.0e4)
            return logits
        return logits.masked_fill(illegal_mask, -1.0e4)

    def forward(self, idx: torch.Tensor, targets=None, illegal_mask=None):
        _, T = idx.shape  # (B, T) token ids
        pos = torch.arange(T, device=idx.device)  # (T,)
        tok = self.token_embedding(idx)  # (B, T, n_token_embd)
        x = tok + self.position_embedding(pos)  # (B, T, n_token_embd)
        x = self.in_proj(x)  # (B, T, n_attn_embd)
        step = 0
        for _ in range(self.loop_times):
            x = x + self.step_embedding.weight[step]
            x = _run_block(self.block_1, x)  # (B, T, n_attn_embd)
            step += 1
        for _ in range(self.loop_times):
            x = x + self.step_embedding.weight[step]
            x = _run_block(self.block_2, x)  # (B, T, n_attn_embd)
            step += 1
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
        if illegal_mask is not None:
            logits = self._apply_illegal_mask(logits, illegal_mask)
        loss = None
        if targets is not None:
            if targets.ndim == 1:
                # Winner-move batches: ys is (B,), supervise the last step only.
                loss = F.cross_entropy(logits[:, -1, :], targets)
            else:
                # Full-sequence LM: ys is (B, T)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        callback=None,
        illegal_mask=None,
        illegal_mask_fn=None,
    ):
        for _ in range(max_new_tokens):
            mask = illegal_mask_fn(idx) if illegal_mask_fn is not None else illegal_mask
            logits, _ = self(
                idx[:, -self.block_size :], illegal_mask=mask
            )  # (B, T, vocab_size)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            if callback is not None:
                callback(idx_next)
        return idx


def create_model(device):
    return LoopedGPT(
        vocab_size=LEN_POSSIBLE_MOVES,
        n_token_embd=N_EMBED,
        n_attn_embd=N_EMBED,
        n_head=N_HEADS,
        block_size=INPUT_SIZE,
        dropout=DROPOUT,
        loop_times=N_LOOPS,
    ).to(device=device)


def load_model(device):
    if not os.path.isfile(MODEL_FILE_PATH):
        print(f"No trained model at {MODEL_FILE_PATH}. Run train.py first.")
        raise SystemExit(1)
    model = create_model(device)
    model.load_state_dict(
        torch.load(MODEL_FILE_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def adamw_param_groups(module, weight_decay):
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(step, warmup_steps, total_steps, lr=LR, min_lr_ratio=MIN_LR_RATIO):
    min_lr = lr * min_lr_ratio
    if step < warmup_steps:
        return lr * float(step + 1) / float(warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(1.0, max(0.0, t))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * t))


def set_lr(optimizer, step, warmup_steps, total_steps, lr=LR, min_lr_ratio=MIN_LR_RATIO):
    current_lr = lr_at(step, warmup_steps, total_steps, lr, min_lr_ratio)
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    return current_lr


@torch.no_grad()
def evaluate(model, split, device):
    n = load_batch_count(split)
    if n <= 0:
        return None
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    progress = tqdm(range(n), dynamic_ncols=True, desc=f"Evaluating {split}")
    for i in progress:
        xs, ys, illegal_mask = load_batch(split, i)
        xs, ys, illegal_mask = batch_to_device(xs, ys, illegal_mask, device)
        logits, split_loss = model(xs, ys, illegal_mask=illegal_mask)
        total_loss += split_loss.item()
        preds = logits[:, -1, :].argmax(dim=-1)
        total_correct += (preds == ys).sum().item()
        total_examples += ys.size(0)
        progress.set_postfix(loss=total_loss / (i + 1), acc=total_correct / total_examples)
    if was_training:
        model.train()
    return total_loss / n, total_correct / total_examples


def save_checkpoint(model):
    os.makedirs(MODELS_FOLDER_PATH, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE_PATH)
