"""
This script performs Acoustic consistency and acoustic-semantic alignment inference on the SALMon dataset.

Usage:
python inference_0shot.py --output_dir ./salmon_outputs/0shot-blueberry-v1-step238k

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR inference on LibriSpeech dataset")
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

def load_salmon_data() -> List[dict]:
    salmon = load_dataset('slprl/salmon', 'all')
    return salmon

def run_inference(
    prompt: str, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
):
    # print(f"Prompt: '{prompt}'")
    # by default, tokenizer prepends the <|begin_of_text|> token
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    log_dist = torch.log_softmax(outputs.logits, dim=-1).squeeze(0)[:-1]
    log_probs = log_dist[torch.arange(log_dist.shape[0]), inputs.input_ids.squeeze(0)[1:]]
    return log_probs.mean()

def task_salmon(
    sample: dict,
    mimi_model: MimiModel,
    marin_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> str:
    # Load and resample audio
    positive_audio, pos_sr = sample["positive_audio"]["array"], sample["positive_audio"]["sampling_rate"]
    negative_audio, neg_sr = sample["negative_audio"]["array"], sample["negative_audio"]["sampling_rate"]
    pos_audio_resampled = resample_audio(positive_audio, orig_sr=pos_sr, target_sr=TARGET_SAMPLE_RATE)
    neg_audio_resampled = resample_audio(negative_audio, orig_sr=neg_sr, target_sr=TARGET_SAMPLE_RATE)
    
    # Convert to Mimi audio string
    pos_audio_str = audio_to_str(pos_audio_resampled, mimi_model, device)
    neg_audio_str = audio_to_str(neg_audio_resampled, mimi_model, device)
    
    # Generate transcription
    pos_prompt = f"<|audio_start|>{pos_audio_str}<|audio_end|>"
    pos_logprob = run_inference(
        pos_prompt, marin_model, tokenizer
    )
    neg_prompt = f"<|audio_start|>{neg_audio_str}<|audio_end|>"
    neg_logprob = run_inference(
        neg_prompt, marin_model, tokenizer
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
    salmon_data = load_salmon_data()
    print(f"Loaded {len(salmon_data)} samples")
    
    # Initialize score tracking
    all_results = []
    total_score = 0
    total_sample = 0
    task_based_scores = {}
    task_based_samples = {}

    # Process consistency part first
    for sample in tqdm(salmon_data["train"]):
        # Task SALMon
        pos_logp, neg_logp = task_salmon(
            sample, mimi_model, marin_model, tokenizer, device
        )

        if sample["task"] not in task_based_scores:
            task_based_scores[sample["task"]] = 0
            task_based_samples[sample["task"]] = 0
        if pos_logp > neg_logp:
            total_score += 1
            task_based_scores[sample["task"]] += 1
        task_based_samples[sample["task"]] += 1
        total_sample += 1
        
        # Compute current overall score
        current_overall_score = total_score / total_sample if total_sample > 0 else 0.0
        
        # Store result
        result = {
            "file_id": sample["ind"],
            "task": sample["task"],
            "pos_logp": pos_logp.item(),
            "neg_logp": neg_logp.item(),
        }
        all_results.append(result)
        
        # Log score for this sample and running average
        print(f"{sample['ind']}: Score = {current_overall_score:.4f} [{total_score}/{total_sample}]")
        print(f"Running Overall score: {current_overall_score:.4f} ({current_overall_score*100:.2f}%) [{total_score}/{total_sample}]")
        sys.stdout.flush()
    
    # Create summary
    summary = {
        "total_samples": len(salmon_data),
        "samples_processed": total_sample,
        "overall_score": f"{total_score}/{total_sample}={total_score/total_sample}",
    }
    summary.update(
        {
            task: f"{score}/{task_based_samples[task]}={score/task_based_samples[task]}" for task, score in task_based_scores.items()
        }
    )
    
    # Save complete results
    results_file = output_dir / "salmon_results.json"
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
    for task, score in summary.items():
        if task not in ["total_samples", "samples_processed", "overall_score"]:
            print(f"{task}: {score}")
    print(f"\nResults saved to {output_dir}")
    print(f"  - Complete results: {results_file}")
    print(f"  - Stats: {stats_file}")


if __name__ == "__main__":
    main()

