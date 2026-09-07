import sys

import torch

print(f"torch: {torch.__version__}")
available = torch.cuda.is_available()
print(f"cuda available: {available}")
if not available:
    print("No GPU detected — training will use CPU.")
    sys.exit()

n = torch.cuda.device_count()
print(f"device count: {n}")
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(
        f"  [{i}] {torch.cuda.get_device_name(i)}  ({props.total_memory / 1e9:.1f} GB)"
    )
