"""
This script performs Acoustic consistency and acoustic-semantic alignment inference on the sBLIMP dataset.

Usage:
python inference_speech_0shot.py --output_dir generated_outputs/blueberry_8cb/ --num_codebook_for_log_probs 8
python inference_speech_0shot.py --output_dir generated_outputs/blueberry_4cb/ --num_codebook_for_log_probs 4
python inference_speech_0shot.py --output_dir generated_outputs/blueberry_2cb/ --num_codebook_for_log_probs 2
python inference_speech_0shot.py --output_dir generated_outputs/blueberry_1cb/ --num_codebook_for_log_probs 1

"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import librosa
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import audio_to_str, resample_audio

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
NUM_CODEBOOKS: int = 8

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR inference on LibriSpeech dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_outputs",
        help="Directory to save generated outputs (default: generated_outputs)"
    )
    parser.add_argument(
        "--num_codebook_for_log_probs",
        type=int,
        default=NUM_CODEBOOKS,
        help=f"Number of codebooks for log probabilities (default: {NUM_CODEBOOKS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

def load_sBLIMP_data() -> List[dict]:
    sBLIMP = load_dataset('speed/sBLIMP')
    return sBLIMP

def run_inference(
    prompt: str, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    num_codebook_for_log_probs: int = NUM_CODEBOOKS,
):
    # print(f"Prompt: '{prompt}'")
    # by default, tokenizer prepends the <|begin_of_text|> token
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    log_dist = torch.log_softmax(outputs.logits, dim=-1).squeeze(0)[:-1]
    log_probs = log_dist[torch.arange(log_dist.shape[0]), inputs.input_ids.squeeze(0)[1:]]

    # strip <|audio_start|> and <|audio_end|>
    log_probs = log_probs[1:-1]
    assert len(log_probs) % NUM_CODEBOOKS == 0
    log_probs = log_probs.reshape(NUM_CODEBOOKS, -1)
    log_probs = log_probs[:num_codebook_for_log_probs, :]
    return log_probs.mean()

def task_sBLIMP(
    sample: dict,
    mimi_model: MimiModel,
    marin_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_codebook_for_log_probs: int,
    device: str,
) -> str:
    # Load and resample audio
    positive_audio, pos_sr = sample["positive"]["array"], sample["positive"]["sampling_rate"]
    negative_audio, neg_sr = sample["negative"]["array"], sample["negative"]["sampling_rate"]
    pos_audio_resampled = resample_audio(positive_audio, orig_sr=pos_sr, target_sr=TARGET_SAMPLE_RATE)
    neg_audio_resampled = resample_audio(negative_audio, orig_sr=neg_sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Convert to float32 to match model dtype
    pos_audio_resampled = pos_audio_resampled.astype(np.float32)
    neg_audio_resampled = neg_audio_resampled.astype(np.float32)
    
    # Convert to Mimi audio string
    pos_audio_str = audio_to_str(pos_audio_resampled, mimi_model, device)
    neg_audio_str = audio_to_str(neg_audio_resampled, mimi_model, device)
    
    # Generate transcription
    pos_prompt = f"<|audio_start|>{pos_audio_str}<|audio_end|>"
    pos_logprob = run_inference(
        pos_prompt, marin_model, tokenizer, num_codebook_for_log_probs
    )
    neg_prompt = f"<|audio_start|>{neg_audio_str}<|audio_end|>"
    neg_logprob = run_inference(
        neg_prompt, marin_model, tokenizer, num_codebook_for_log_probs
    )
    return pos_logprob, neg_logprob

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
    print(f"Loading dataset from Huggingface")
    sBLIMP_data = load_sBLIMP_data()
    print(f"Loaded {len(sBLIMP_data)} samples")
    
    # Initialize score tracking
    all_results = []
    total_score = 0
    total_sample = 0

    num_codebook_for_log_probs = args.num_codebook_for_log_probs

    # Process consistency part first
    for sample in tqdm(sBLIMP_data["train"]):
        # Task sBLIMP
        pos_logp, neg_logp = task_sBLIMP(
            sample, mimi_model, marin_model, tokenizer, num_codebook_for_log_probs, device
        )
        if pos_logp > neg_logp:
            total_score += 1
        total_sample += 1
        
        # Compute current overall score
        current_overall_score = total_score / total_sample if total_sample > 0 else 0.0
        
        # Store result
        result = {
            "file_id": sample["id"],
            "positive_word": sample["positive_transcription"],
            "negative_word": sample["negative_transcription"],
            "pos_logp": pos_logp.item(),
            "neg_logp": neg_logp.item(),
        }
        all_results.append(result)
        
        # Log score for this sample and running average
        print(f"{sample['id']}: Score = {current_overall_score:.4f} [{total_score}/{total_sample}]")
        print(f"Running Overall score: {current_overall_score:.4f} ({current_overall_score*100:.2f}%) [{total_score}/{total_sample}]")
        sys.stdout.flush()
    
    # Create summary
    summary = {
        "total_samples": len(sBLIMP_data),
        "samples_processed": total_sample,
        "overall_score": f"{total_score}/{total_sample}={total_score/total_sample}",
    }
    
    # Save complete results
    results_file = output_dir / "sBLIMP_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": summary,
            "results": all_results
        }, f, indent=2)
    print(f"Saved complete results to {results_file}")
    
    # Save evaluation stats
    stats_file = output_dir / "evaluation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation stats to {stats_file}")
    
    # Print final summary
    print(f"\n{'=' * 80}")
    print("Evaluation Complete!")
    print(f"{'=' * 80}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Samples processed: {summary['samples_processed']}")
    print(f"Overall score: {summary['overall_score']}")
    print(f"\nResults saved to {output_dir}")
    print(f"  - Complete results: {results_file}")
    print(f"  - Stats: {stats_file}")


if __name__ == "__main__":
    main()

