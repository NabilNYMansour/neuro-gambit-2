import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


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
        self.step_embedding = nn.Embedding(loop_times, n_attn_embd)
        self.block = Block(n_attn_embd, n_head, block_size, dropout)
        self.lm_head = nn.Linear(n_attn_embd, vocab_size)
        self.ln_f = LayerNorm(n_attn_embd)
        self.apply(_init_weights)
        _scale_residual_projections(self, loop_times)

    def forward(self, idx: torch.Tensor, targets=None):
        _, T = idx.shape  # (B, T) token ids
        pos = torch.arange(T, device=idx.device)  # (T,)
        tok = self.token_embedding(idx)  # (B, T, n_token_embd)
        x = tok + self.position_embedding(pos)  # (B, T, n_token_embd)
        x = self.in_proj(x)  # (B, T, n_attn_embd)
        for i in range(self.loop_times):
            x = x + self.step_embedding.weight[i]
            x = _run_block(self.block, x)  # (B, T, n_attn_embd)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
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
    def generate(self, idx, max_new_tokens, temperature=1.0, callback=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])  # (B, T, vocab_size)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            if callback is not None:
                callback(idx_next)
        return idx
