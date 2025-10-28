"""
This script performs TTS inference on the seed-tts-eval dataset.

Usage:
python inference_0shot.py --data_path data/seedtts_testset/en/meta.lst --prompt_wav_dir data/seedtts_testset/en/ --output_dir tts_outputs/0shot-blueberry-v1-step238k-temp0.9-top0.8 --temperature 0.9 --top_p 0.8 --max_new_tokens 1500

# With random order:
python inference_0shot.py --data_path data/seedtts_testset/en/meta.lst --prompt_wav_dir data/seedtts_testset/en/ --output_dir tts_outputs/0shot-blueberry-v1-step238k-temp0.9-top0.8 --temperature 0.9 --top_p 0.8 --max_new_tokens 1500 --shuffle
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm

import librosa
import soundfile as sf
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import audio_to_str, resample_audio, str_to_audio

# Configure logging with immediate output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
MIMI_MODEL_ID: str = "kyutai/mimi"
MARIN_MODEL_NAME: str = "WillHeld/blueberry"
SOURCE_SAMPLE_RATE: int = 16000
TARGET_SAMPLE_RATE: int = 24000

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR inference on LibriSpeech dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the LibriSpeech JSON dataset file"
    )
    parser.add_argument(
        "--prompt_wav_dir",
        type=str,
        required=True,
        help="Directory to the prompt WAV files (e.g. prompt-wavs)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_outputs",
        help="Directory to save generated outputs (default: generated_outputs)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1500,
        help="Maximum number of new tokens to generate (default: 2000)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Generation temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Shuffle the dataset before processing (uses the same seed as --seed)"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed) 
    

def load_seedtts_data(data_path: str, prompt_wav_dir: str) -> List[dict]:
    with open(data_path, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        items = line.split("|")
        assert len(items) == 4, f"Invalid line: {line}"
        filename = items[0]
        prompt_text = items[1]
        prompt_audio = f"{prompt_wav_dir}/{items[2]}"
        target_text = items[3]
        data.append({
            "filename": filename,
            "prompt_text": prompt_text,
            "prompt_audio": prompt_audio,
            "target_text": target_text,
        })
    return data

def run_inference(
    prompt: str, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    # print(f"Prompt: '{prompt}'")
    # by default, tokenizer prepends the <|begin_of_text|> token
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=128259, # 128259 = <|audio_end|>
        )
    input_len = inputs.input_ids.shape[1]
    # skip_special_tokens=False as audio tokens are special tokens
    generated_audio_str = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=False)
    if "<|audio_end|>" in generated_audio_str:
        generated_audio_str = generated_audio_str.replace("<|audio_end|>", "")
    return generated_audio_str

def task_zero_shot_tts(
    sample: dict,
    mimi_model: MimiModel,
    marin_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:

    prompt_text = sample["prompt_text"]
    prompt_audio = sample["prompt_audio"]
    target_text = sample["target_text"]

    # ------------------------------------------------------------

    # Load and resample the prompt audio
    audio, sr = librosa.load(prompt_audio, sr=None)
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Convert to Mimi audio string
    audio_str = audio_to_str(audio_resampled, mimi_model, device)

    # Generate zero-shot TTS
    prompt = (
        f"<|text_start|>{prompt_text}<|text_end|>"
        f"<|audio_start|>{audio_str}<|audio_end|>"
        f"<|text_start|>{target_text}<|text_end|>"
        f"<|audio_start|>"
    )

    generated_audio_str = run_inference(
        prompt, marin_model, tokenizer, max_new_tokens, temperature, top_p
    )
    N = len(generated_audio_str)
    generated_audio_str = generated_audio_str[:(N//8)*8]
    return generated_audio_str

def main() -> None:
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Set random seed to {args.seed}")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wavs").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    
    # Load Mimi model
    print(f"Loading Mimi model: {MIMI_MODEL_ID}")
    mimi_model = MimiModel.from_pretrained(MIMI_MODEL_ID)
    mimi_model = mimi_model.to(device)
    
    # Load Marin audio model
    print(f"Loading Marin model: {MARIN_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MARIN_MODEL_NAME)
    marin_model = AutoModelForCausalLM.from_pretrained(
        MARIN_MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )
    marin_model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in marin_model.parameters())}")

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    print(f"Loading prompt WAV files from {args.prompt_wav_dir}")
    dataset = load_seedtts_data(args.data_path, args.prompt_wav_dir)
    print(f"Loaded {len(dataset)} samples")
    
    # Shuffle dataset if requested
    if args.shuffle:
        random.shuffle(dataset)
        print(f"Shuffled dataset with seed {args.seed}")
    
    # Check for already processed samples
    processed_count = 0
    skipped_count = 0
    for sample in dataset:
        audio_output_path = output_dir / "wavs" / f"{sample['filename']}.wav"
        metadata_output_path = output_dir / "metadata" / f"{sample['filename']}.txt"
        if audio_output_path.exists() and metadata_output_path.exists():
            skipped_count += 1
    
    print(f"Found {skipped_count} already processed samples - will skip them")
    print(f"Will process {len(dataset) - skipped_count} samples")
    
    # Process each sample
    for sample in tqdm(dataset, desc="Processing samples"):
        audio_output_path = output_dir / "wavs" / f"{sample['filename']}.wav"
        metadata_output_path = output_dir / "metadata" / f"{sample['filename']}.txt"
        
        # Skip if already processed
        if audio_output_path.exists() and metadata_output_path.exists():
            print(f"Skipping already processed sample: {sample['filename']}")
            continue
        
        # Task Zero-shot TTS
        generated_audio_str = task_zero_shot_tts(
            sample, mimi_model, marin_model, tokenizer, device,
            args.max_new_tokens, args.temperature, args.top_p
        )
        assert generated_audio_str is not None, f"Generated audio string is None for sample {sample['filename']}"
        audio_numpy = str_to_audio(generated_audio_str, mimi_model, device)
        
        # Save audio
        sf.write(str(audio_output_path), audio_numpy.T, TARGET_SAMPLE_RATE)
        
        # Write metadata text file
        with open(metadata_output_path, 'w') as f:
            f.write(f"prompt_text: {sample['prompt_text']}\n")
            f.write(f"prompt_audio: {Path(sample['prompt_audio']).resolve()}\n")
            f.write(f"target_text: {sample['target_text']}\n")
            f.write(f"target_audio: {audio_output_path.resolve()}\n")
            f.write(f"# --- generation parameters --- #\n")
            f.write(f"temperature: {args.temperature}\n")
            f.write(f"top_p: {args.top_p}\n")
            f.write(f"max_new_tokens: {args.max_new_tokens}\n")
            f.write(f"seed: {args.seed}\n")
            f.write(f"model: {MARIN_MODEL_NAME}\n")
            f.write(f"mimi_model: {MIMI_MODEL_ID}\n")
        
        processed_count += 1
        logger.info(f"[{processed_count}/{len(dataset) - skipped_count}] Saved: {audio_output_path}")
    
    print(f"\nProcessing complete!")
    print(f"Total samples: {len(dataset)}")
    print(f"Already processed (skipped): {skipped_count}")
    print(f"Newly processed: {processed_count}")

if __name__ == "__main__":
    main()

