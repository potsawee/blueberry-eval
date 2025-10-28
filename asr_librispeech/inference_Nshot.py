"""
This script performs ASR inference on the LibriSpeech dataset.

Usage:
# Same N-shot for all test samples (default)
python inference_Nshot.py --test_data_path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/test-clean.json --train_data_path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/dev-clean.json --num_shots 3 --output_dir ./asr_outputs/3shot-blueberry-v1-step238k-temp0.0001-top0.8 --temperature 0.0001 --top_p 0.8 --same_n_shot

# Different N-shot for each test sample
python inference_Nshot.py --test_data_path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/test-clean.json --train_data_path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/dev-clean.json --num_shots 3 --output_dir ./asr_outputs/3shot-different-blueberry-v1-step238k-temp0.0001-top0.8 --temperature 0.0001 --top_p 0.8

"""

import argparse
import random
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import jiwer
import librosa
import numpy as np
import torch
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
        "--test_data_path",
        type=str,
        required=True,
        help="Path to the LibriSpeech JSON test dataset file"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the LibriSpeech JSON train dataset file"
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=3,
        help="Number of shots to use for inference (default: 3)"
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
        "--same_n_shot",
        action="store_true",
        default=True,
        help="Use same N-shot examples for all test samples (default: True). If False, each test sample gets different N-shot examples."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate (default: 1200)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Generation temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling parameter (default: 0.9)"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

def load_librispeech_data(json_path: str) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def normalize_text(text: str) -> str:
    """Normalize text for WER computation."""
    # Remove punctuation (including unicode quotes)
    text = re.sub(r'[.,!?;:"\'\-\(\)\[\]\{\}\u2019\u2018\u201c\u201d]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def compute_wer_for_sample(reference: str, hypothesis: str) -> Tuple[float, Dict]:
    """
    Compute WER for a single sample.
    
    Returns:
        wer: Word Error Rate as a float
        details: Dictionary with detailed metrics
    """
    ref_normalized = normalize_text(reference)
    hyp_normalized = normalize_text(hypothesis)
    
    # Compute WER
    wer = jiwer.wer(ref_normalized, hyp_normalized)
    
    # Get detailed measures
    output = jiwer.process_words(ref_normalized, hyp_normalized)
    
    details = {
        "wer": wer,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "reference_words": len(ref_normalized.split()),
        "hypothesis_words": len(hyp_normalized.split()),
    }
    
    return wer, details

def run_inference(
    prompt: str, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, int]:
    # by default, tokenizer prepends the <|begin_of_text|> token
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=128009, # 128009 = <|eot_id|> --> this one is unused (NB: eos_token is <|end_of_text|> so don't use it as padding here)
            eos_token_id=128257, # 128257 = <|text_end|>
            
        )
    generated_text = tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True)
    return generated_text, prompt_length

def select_n_shot_examples(
    train_dataset: List[dict],
    num_shots: int,
    seed: int,
    test_sample_index: int = None,
) -> List[dict]:
    """
    Select N-shot examples from training dataset.
    
    Args:
        train_dataset: Full training dataset
        num_shots: Number of examples to select
        seed: Base random seed
        test_sample_index: Index of test sample (None for same N-shot for all)
    
    Returns:
        List of N training examples
    """
    if test_sample_index is None:
        # Same N-shot for all test samples
        rng = random.Random(seed)
        shuffled = train_dataset.copy()
        rng.shuffle(shuffled)
        return shuffled[:num_shots]
    else:
        # Different N-shot for each test sample, but reproducible
        rng = random.Random(seed + test_sample_index)
        shuffled = train_dataset.copy()
        rng.shuffle(shuffled)
        return shuffled[:num_shots]


def task_asr(
    test_sample: dict,
    train_samples: List[dict],
    mimi_model: MimiModel,
    marin_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, int]:
    """
    Returns:
        generated_text: The generated transcription
        prompt_length: Length of the prompt in tokens
    """
    # Create N-shot prompt
    prompt = ""
    for train_sample in train_samples:
        audio, sr = librosa.load(train_sample["file_path"], sr=None)
        audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        audio_str = audio_to_str(audio_resampled, mimi_model, device)
        transcript = train_sample['transcript']
        transcript = normalize_text(transcript)
        prompt += f"<|begin_of_text|><|audio_start|>{audio_str}<|audio_end|><|text_start|>{transcript}<|text_end|><|end_of_text|>"

    audio, sr = librosa.load(test_sample["file_path"], sr=None)
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    audio_str = audio_to_str(audio_resampled, mimi_model, device)
    prompt += f"<|begin_of_text|><|audio_start|>{audio_str}<|audio_end|><|text_start|>"

    # get rid of the first <|begin_of_text|> at the beginning of the prompt as the tokenizer will prepend it
    prompt = prompt[len("<|begin_of_text|>"):]

    # Generate transcription
    generated_text, prompt_length = run_inference(
        prompt, marin_model, tokenizer, max_new_tokens, temperature, top_p
    )
    return generated_text, prompt_length

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
    print(f"Loading test dataset from {args.test_data_path}")
    test_dataset = load_librispeech_data(args.test_data_path)
    print(f"Loading train dataset from {args.train_data_path}")
    train_dataset = load_librispeech_data(args.train_data_path)
    print(f"Loaded {len(test_dataset)} test samples")
    print(f"Loaded {len(train_dataset)} train samples")
    
    # Select N-shot examples based on mode
    if args.same_n_shot:
        print(f"Using same {args.num_shots}-shot examples for all test samples")
        train_samples_global = select_n_shot_examples(
            train_dataset, args.num_shots, args.seed, test_sample_index=None
        )
    else:
        print(f"Using different {args.num_shots}-shot examples for each test sample (reproducible with seed={args.seed})")
        train_samples_global = None  # Will select per test sample

    # Initialize WER tracking
    all_results = []
    total_errors = 0
    total_words = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    samples_processed = 0
    
    # Initialize prompt length tracking
    total_prompt_length = 0
    prompt_lengths = []
    
    # Process each sample
    for sample_idx, sample in enumerate(tqdm(test_dataset)):
        file_id = Path(sample["file_path"]).stem
        
        # Select N-shot examples for this test sample
        if args.same_n_shot:
            train_samples = train_samples_global
        else:
            train_samples = select_n_shot_examples(
                train_dataset, args.num_shots, args.seed, test_sample_index=sample_idx
            )
        
        # Task ASR
        transcribed_text, prompt_length = task_asr(
            sample, train_samples, mimi_model, marin_model, tokenizer, device,
            args.max_new_tokens, args.temperature, args.top_p
        )
        
        # Track prompt length
        prompt_lengths.append(prompt_length)
        total_prompt_length += prompt_length
        current_avg_prompt_length = total_prompt_length / (sample_idx + 1)
        
        # Ground truth transcript
        ground_truth_transcript = sample["transcript"]
        
        # Compute WER for this sample
        wer, details = compute_wer_for_sample(ground_truth_transcript, transcribed_text)
        
        # Update running totals for overall WER
        total_errors += details["substitutions"] + details["deletions"] + details["insertions"]
        total_words += details["reference_words"]
        total_substitutions += details["substitutions"]
        total_insertions += details["insertions"]
        total_deletions += details["deletions"]
        samples_processed += 1
        
        # Compute current overall WER and error breakdown
        current_overall_wer = total_errors / total_words if total_words > 0 else 0.0
        current_sub_pct = (total_substitutions / total_words * 100) if total_words > 0 else 0.0
        current_ins_pct = (total_insertions / total_words * 100) if total_words > 0 else 0.0
        current_del_pct = (total_deletions / total_words * 100) if total_words > 0 else 0.0
        
        # Store result
        result = {
            "file_id": file_id,
            "reference": ground_truth_transcript,
            "hypothesis": transcribed_text,
            "wer": wer,
            "details": details,
            "prompt_length": prompt_length,
        }
        all_results.append(result)
        
        # Compute error breakdown percentages
        ref_words = details["reference_words"]
        sub_pct = (details["substitutions"] / ref_words * 100) if ref_words > 0 else 0.0
        ins_pct = (details["insertions"] / ref_words * 100) if ref_words > 0 else 0.0
        del_pct = (details["deletions"] / ref_words * 100) if ref_words > 0 else 0.0
        
        # Log WER for this sample and running average
        print(f"{file_id}: WER = {wer:.4f} ({wer*100:.2f}%) [%sub={sub_pct:.2f}, %ins={ins_pct:.2f}, %del={del_pct:.2f}] | Prompt Length: {prompt_length}")
        print(f"Running Overall WER: {current_overall_wer:.4f} ({current_overall_wer*100:.2f}%) [%sub={current_sub_pct:.2f}, %ins={current_ins_pct:.2f}, %del={current_del_pct:.2f}] [{samples_processed}/{len(test_dataset)} samples]")
        print(f"Average Prompt Length: {current_avg_prompt_length:.1f} tokens")
        sys.stdout.flush()
    
    # Compute final statistics
    if total_words > 0:
        overall_wer = total_errors / total_words
        overall_sub_pct = (total_substitutions / total_words * 100)
        overall_ins_pct = (total_insertions / total_words * 100)
        overall_del_pct = (total_deletions / total_words * 100)
    else:
        overall_wer = 0.0
        overall_sub_pct = 0.0
        overall_ins_pct = 0.0
        overall_del_pct = 0.0
    
    if samples_processed > 0:
        average_wer = sum(r["wer"] for r in all_results) / samples_processed
    else:
        average_wer = 0.0
    
    # Compute prompt length statistics
    avg_prompt_length = np.mean(prompt_lengths) if prompt_lengths else 0.0
    max_prompt_length_seen = max(prompt_lengths) if prompt_lengths else 0
    min_prompt_length_seen = min(prompt_lengths) if prompt_lengths else 0
    median_prompt_length = np.median(prompt_lengths) if prompt_lengths else 0.0
    
    # Create summary
    summary = {
        "total_samples": len(test_dataset),
        "samples_processed": samples_processed,
        "overall_wer": overall_wer,
        "average_wer": average_wer,
        "total_errors": total_errors,
        "total_words": total_words,
        "total_substitutions": total_substitutions,
        "total_insertions": total_insertions,
        "total_deletions": total_deletions,
        "substitution_rate": overall_sub_pct,
        "insertion_rate": overall_ins_pct,
        "deletion_rate": overall_del_pct,
        "prompt_length_stats": {
            "average": float(avg_prompt_length),
            "median": float(median_prompt_length),
            "min": int(min_prompt_length_seen),
            "max": int(max_prompt_length_seen),
        },
        "evaluation_config": {
            "num_shots": args.num_shots,
            "same_n_shot": args.same_n_shot,
            "seed": args.seed,
        },
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
    }
    
    # Save complete results (transcripts + WER details)
    results_file = output_dir / "asr_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": summary,
            "results": all_results
        }, f, indent=2)
    print(f"Saved complete results to {results_file}")
    
    # Save simplified transcripts file
    transcripts_file = output_dir / "transcripts.json"
    transcripts_data = [
        {
            "file_id": r["file_id"],
            "reference": r["reference"],
            "hypothesis": r["hypothesis"],
        }
        for r in all_results
    ]
    with open(transcripts_file, "w") as f:
        json.dump(transcripts_data, f, indent=2)
    print(f"Saved transcripts to {transcripts_file}")
    
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
    print(f"Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"  - Substitutions: {overall_sub_pct:.2f}%")
    print(f"  - Insertions: {overall_ins_pct:.2f}%")
    print(f"  - Deletions: {overall_del_pct:.2f}%")
    print(f"Average WER: {average_wer:.4f} ({average_wer*100:.2f}%)")
    print(f"Total errors: {total_errors} (sub={total_substitutions}, ins={total_insertions}, del={total_deletions})")
    print(f"Total words: {total_words}")
    print(f"\nPrompt Length Statistics:")
    print(f"  - Average: {avg_prompt_length:.1f} tokens")
    print(f"  - Median: {median_prompt_length:.1f} tokens")
    print(f"  - Min: {min_prompt_length_seen} tokens")
    print(f"  - Max: {max_prompt_length_seen} tokens")
    print(f"\nResults saved to {output_dir}")
    print(f"  - Complete results: {results_file}")
    print(f"  - Transcripts: {transcripts_file}")
    print(f"  - Stats: {stats_file}")


if __name__ == "__main__":
    main()

