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

### Stockfish

For Debian/Ubuntu

```bash
sudo apt install stockfish
```

For macOS

```bash
brew install stockfish
```

# Data

You can download a chess PGN from [database.lichess.org](https://database.lichess.org/) by running:

```bash
mkdir -p data/raw && cd data/raw
curl -L -O https://database.lichess.org/standard/lichess_db_standard_rated_2017-02.pgn.zst
zstd -d lichess_db_standard_rated_2017-02.pgn.zst
cd ../..
```

Set `DATA_PATH` in `generate_data.py` to the path of the downloaded PGN. I used the data from July 2017 as it is relatively small, but you can also download a larger dataset.

Also, in case you don't have `zstd`, you can install it with 
```bash
sudo apt install zstd
```
