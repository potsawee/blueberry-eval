"""
This script performs speech-to-text translation inference on the COVOST2 dataset.

Usage:
python inference_Nshot.py --lang_input en --lang_output de --num_shots 2 --output_dir ./s2tt_outputs/2shot-en_de-blueberry-v1-step238k
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel
from datasets import Dataset, config, load_dataset
from sacrebleu import corpus_bleu, sentence_bleu
# Import datasets config first and disable torchcodec
config.TORCHCODEC_AVAILABLE = False

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
    parser = argparse.ArgumentParser(description="Speech-to-text translation inference on COVOST2 dataset")
    parser.add_argument(
        "--lang_input",
        type=str,
        required=True,
        help="Input language of the COVOST2 dataset"
    )
    parser.add_argument(
        "--lang_output",
        type=str,
        required=True,
        help="Output language of the COVOST2 dataset"
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
        default=0.0001,
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
    

def load_covost2_data(
    lang_input: str,
    lang_output: str,
    split: str,
) -> Dataset:
    base_url = f"https://huggingface.co/datasets/fixie-ai/covost2/resolve/main/{lang_input}_{lang_output}/"
    data_files = {split: base_url + f"{split}-*-of-*.parquet"}
    data = load_dataset(
        "parquet", 
        data_files=data_files, 
        split=split,
        cache_dir="/nlp/scr/potsawee/workspace/blueberry-eval/s2tt_covost2/cache"
    )
    return data

def compute_bleu_for_sample(reference: str, hypothesis: str) -> Tuple[float, Dict]:
    """
    Compute BLEU score for a single sample.
    
    Returns:
        bleu: BLEU score as a float (0-100)
        details: Dictionary with detailed metrics
    """
    # sacrebleu expects references as a list of strings (for multiple references)
    # and hypothesis as a single string
    bleu_result = sentence_bleu(hypothesis, [reference])
    
    details = {
        "bleu": bleu_result.score,
        "precisions": bleu_result.precisions,
        "bp": bleu_result.bp,
        "sys_len": bleu_result.sys_len,
        "ref_len": bleu_result.ref_len,
    }
    
    return bleu_result.score, details

def select_n_shot_examples(
    train_dataset: Dataset,
    num_shots: int,
    seed: int,
    test_sample_index: int = None,
) -> List[dict]:
    """
    Select N-shot examples from training dataset.
    
    Args:
        train_dataset: Full training dataset (HuggingFace Dataset)
        num_shots: Number of examples to select
        seed: Base random seed
        test_sample_index: Index of test sample (None for same N-shot for all)
    
    Returns:
        List of N training examples
    """
    if test_sample_index is None:
        # Same N-shot for all test samples
        effective_seed = seed
    else:
        # Different N-shot for each test sample, but reproducible
        effective_seed = seed + test_sample_index
    
    # Shuffle the dataset using HuggingFace's shuffle method
    shuffled_dataset = train_dataset.shuffle(seed=effective_seed)
    
    # Select first num_shots examples and convert to list of dicts
    selected = shuffled_dataset.select(range(num_shots))
    return [dict(sample) for sample in selected]


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
            pad_token_id=128009, # 128009 = <|eot_id|> --> this one is unused (NB: eos_token is <|end_of_text|> so don't use it as padding here)
            eos_token_id=128257, # 128257 = <|text_end|>
            
        )
    input_len = inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
    return generated_text

def task_s2tt(
    test_sample: dict,
    train_samples: List[dict],
    mimi_model: MimiModel,
    marin_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:

    # Create N-shot prompt
    prompt = ""
    for train_sample in train_samples:
        audio = train_sample["audio"]["array"].astype(np.float32)
        sr = train_sample["audio"]["sampling_rate"]
        audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        audio_str = audio_to_str(audio_resampled, mimi_model, device)
        translation = train_sample['translation']
        prompt += f"<|begin_of_text|><|audio_start|>{audio_str}<|audio_end|><|text_start|>Translation: {translation}<|text_end|><|end_of_text|>"

    audio = test_sample["audio"]["array"].astype(np.float32)
    sr = test_sample["audio"]["sampling_rate"]
    audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    audio_str = audio_to_str(audio_resampled, mimi_model, device)
    prompt += f"<|begin_of_text|><|audio_start|>{audio_str}<|audio_end|><|text_start|>Translation:"

    # get rid of the first <|begin_of_text|> at the beginning of the prompt as the tokenizer will prepend it
    prompt = prompt[len("<|begin_of_text|>"):]

    # Generate transcription
    generated_text = run_inference(
        prompt, marin_model, tokenizer, max_new_tokens, temperature, top_p
    )
    return generated_text

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
    print(f"Loading test dataset for {args.lang_input}->{args.lang_output}")
    test_dataset = load_covost2_data(args.lang_input, args.lang_output, "test")
    print(f"Loading train dataset for {args.lang_input}->{args.lang_output}")
    train_dataset = load_covost2_data(args.lang_input, args.lang_output, "validation")
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

    # Initialize BLEU tracking
    all_results = []
    all_references = []
    all_hypotheses = []
    total_bleu = 0.0
    samples_processed = 0
    
    # Process each sample
    for sample_idx, sample in enumerate(tqdm(test_dataset)):
        file_id = sample["id"]
        
        # Select N-shot examples for this test sample
        if args.same_n_shot:
            train_samples = train_samples_global
        else:
            train_samples = select_n_shot_examples(
                train_dataset, args.num_shots, args.seed, test_sample_index=sample_idx
            )
        
        # Task Speech-to-Text Translation
        translated_text = task_s2tt(
            sample, train_samples, mimi_model, marin_model, tokenizer, device,
            args.max_new_tokens, args.temperature, args.top_p
        )
        
        # Ground truth translation
        ground_truth_translation = sample["translation"]
        
        print("REF-translation: ", ground_truth_translation)
        print("HYP-translation: ", translated_text)

        # Compute BLEU for this sample
        bleu_score, details = compute_bleu_for_sample(ground_truth_translation, translated_text)
        
        # Update running totals
        total_bleu += bleu_score
        samples_processed += 1
        all_references.append(ground_truth_translation)
        all_hypotheses.append(translated_text)
        
        # Compute current average BLEU
        current_avg_bleu = total_bleu / samples_processed if samples_processed > 0 else 0.0
        
        # Store result
        result = {
            "file_id": file_id,
            "reference": ground_truth_translation,
            "hypothesis": translated_text,
            "bleu": bleu_score,
            "details": details,
        }
        all_results.append(result)
        
        # Log BLEU for this sample and running average
        print(f"{file_id}: BLEU = {bleu_score:.2f}")
        print(f"Running Average BLEU: {current_avg_bleu:.2f} [{samples_processed}/{len(test_dataset)} samples]")
        sys.stdout.flush()
    
    # Compute final statistics
    if samples_processed > 0:
        average_bleu = total_bleu / samples_processed
    else:
        average_bleu = 0.0
    
    # Compute corpus-level BLEU using all references and hypotheses
    corpus_bleu_result = corpus_bleu(all_hypotheses, [all_references])
    corpus_bleu_score = corpus_bleu_result.score
    
    # Create summary
    summary = {
        "total_samples": len(test_dataset),
        "samples_processed": samples_processed,
        "average_bleu": average_bleu,
        "corpus_bleu": corpus_bleu_score,
        "corpus_bleu_details": {
            "precisions": corpus_bleu_result.precisions,
            "bp": corpus_bleu_result.bp,
            "sys_len": corpus_bleu_result.sys_len,
            "ref_len": corpus_bleu_result.ref_len,
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
    
    # Save complete results (translations + BLEU details)
    results_file = output_dir / "translation_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": summary,
            "results": all_results
        }, f, indent=2)
    print(f"Saved complete results to {results_file}")
    
    # Save simplified translations file
    translations_file = output_dir / "translations.json"
    translations_data = [
        {
            "file_id": r["file_id"],
            "reference": r["reference"],
            "hypothesis": r["hypothesis"],
        }
        for r in all_results
    ]
    with open(translations_file, "w") as f:
        json.dump(translations_data, f, indent=2)
    print(f"Saved translations to {translations_file}")
    
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
    print(f"Average BLEU: {average_bleu:.2f}")
    print(f"Corpus BLEU: {corpus_bleu_score:.2f}")
    print(f"  - Precisions: {corpus_bleu_result.precisions}")
    print(f"  - Brevity Penalty: {corpus_bleu_result.bp:.4f}")
    print(f"  - System Length: {corpus_bleu_result.sys_len}")
    print(f"  - Reference Length: {corpus_bleu_result.ref_len}")
    print(f"\nResults saved to {output_dir}")
    print(f"  - Complete results: {results_file}")
    print(f"  - Translations: {translations_file}")
    print(f"  - Stats: {stats_file}")


if __name__ == "__main__":
    main()

