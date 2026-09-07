# Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ROCm

Double check if that is the ROCm version of your AMD GPU. Mine is RX 9060-class

```bash
pip install -r requirements-rocm.txt
python check_gpu.py
```

### CUDA

Otherwise if you have a CUDA GPU

```bash
pip install -r requirements-cuda.txt
python check_gpu.py
```
