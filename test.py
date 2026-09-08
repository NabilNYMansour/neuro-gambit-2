import torch

from batches_helpers import load_batch_count
from constants import METADATA_FILE_PATH
from metadata import set_metadata
from model import evaluate, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_count = load_batch_count("test")
print(f"Test batches: {batch_count}")
if batch_count <= 0:
    print("No test batches. Run generate_batches.py first.")
    raise SystemExit(1)

model = load_model(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

metrics = evaluate(model, "test", device)
if metrics is None:
    print("No test batches. Run generate_batches.py first.")
    raise SystemExit(1)

test_loss, test_acc = metrics
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
set_metadata(METADATA_FILE_PATH, "test_loss", test_loss)
set_metadata(METADATA_FILE_PATH, "test_accuracy", test_acc)
