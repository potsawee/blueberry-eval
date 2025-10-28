import sys
import os

# Import datasets config first and disable torchcodec
from datasets import config
config.TORCHCODEC_AVAILABLE = False

from datasets import load_dataset, Audio

lang = sys.argv[1] # en_de, de_en
split = sys.argv[2] # test, validation
print("lang:", lang)
print("split:", split)

base_url = f"https://huggingface.co/datasets/fixie-ai/covost2/resolve/main/{lang}/"
data_files = {split: base_url + f"{split}-*-of-*.parquet"}

print("Downloading dataset...")
data = load_dataset(
    "parquet", 
    data_files=data_files, 
    split=split,
    cache_dir="/nlp/scr/potsawee/workspace/blueberry-eval/s2tt_covost2/cache"
)

# Cast audio column to use soundfile backend instead of torchcodec
data = data.cast_column("audio", Audio(decode=True))

x = data[0]
print(f"Downloaded dataset: {lang}---{split} with {len(data)} samples")
print(f"Dataset features: {data.features}")
print(f"Sample audio shape: {x['audio']['array'].shape}")
print("Dataset cached successfully!")