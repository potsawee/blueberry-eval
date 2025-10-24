import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import jiwer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(json_path: str) -> List[dict]:
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
    
    # Extract alignment information
    alignments = output.alignments[0]  # Get first alignment
    substitutions_list = []
    deletions_list = []
    insertions_list = []
    
    for chunk in alignments:
        if chunk.type == "substitute":
            substitutions_list.append({
                "ref": chunk.ref_start_idx if hasattr(chunk, 'ref_start_idx') else None,
                "hyp": chunk.hyp_start_idx if hasattr(chunk, 'hyp_start_idx') else None,
                "ref_word": output.references[0][chunk.ref_start_idx] if chunk.ref_start_idx < len(output.references[0]) else "",
                "hyp_word": output.hypotheses[0][chunk.hyp_start_idx] if chunk.hyp_start_idx < len(output.hypotheses[0]) else ""
            })
        elif chunk.type == "delete":
            deletions_list.append({
                "ref": chunk.ref_start_idx if hasattr(chunk, 'ref_start_idx') else None,
                "ref_word": output.references[0][chunk.ref_start_idx] if chunk.ref_start_idx < len(output.references[0]) else ""
            })
        elif chunk.type == "insert":
            insertions_list.append({
                "hyp": chunk.hyp_start_idx if hasattr(chunk, 'hyp_start_idx') else None,
                "hyp_word": output.hypotheses[0][chunk.hyp_start_idx] if chunk.hyp_start_idx < len(output.hypotheses[0]) else ""
            })
    
    details = {
        "wer": wer,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "reference_words": len(ref_normalized.split()),
        "hypothesis_words": len(hyp_normalized.split()),
        "substitutions_list": substitutions_list,
        "deletions_list": deletions_list,
        "insertions_list": insertions_list,
    }
    
    return wer, details


def main() -> None:
    # Paths
    data_path = "data/librispeech_test_clean_sample_100_with_zero_shot.json"
    results_dir = Path("generated_outputs")
    output_file = results_dir / "wer_results.json"
    summary_output_file = results_dir / "wer_summary.json"
    
    logger.info(f"Loading dataset from {data_path}")
    dataset = load_dataset(data_path)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Collect results
    sample_results = []
    total_errors = 0
    total_words = 0
    samples_processed = 0
    samples_missing = 0
    
    for sample in dataset:
        # Get file ID
        file_id = Path(sample["file_path"]).stem
        
        # Get ground truth transcript
        reference = sample["transcript"]
        
        # Read ASR output
        asr_file = results_dir / file_id / "asr.txt"
        
        if not asr_file.exists():
            logger.warning(f"ASR file not found for {file_id}: {asr_file}")
            samples_missing += 1
            continue
        
        with open(asr_file, "r") as f:
            hypothesis = f.read()
        
        # Compute WER
        wer, details = compute_wer_for_sample(reference, hypothesis)
        
        # Update totals for overall WER
        total_errors += details["substitutions"] + details["deletions"] + details["insertions"]
        total_words += details["reference_words"]
        samples_processed += 1
        
        # Store results
        result = {
            "file_id": file_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": wer,
            "details": details,
        }
        sample_results.append(result)
        
        logger.info(f"{file_id}: WER = {wer:.4f} ({wer*100:.2f}%)")
    
    # Compute overall WER
    if total_words > 0:
        overall_wer = total_errors / total_words
    else:
        overall_wer = 0.0
    
    # Compute average WER
    if samples_processed > 0:
        average_wer = sum(r["wer"] for r in sample_results) / samples_processed
    else:
        average_wer = 0.0
    
    # Summary
    summary = {
        "total_samples": len(dataset),
        "samples_processed": samples_processed,
        "samples_missing": samples_missing,
        "overall_wer": overall_wer,
        "average_wer": average_wer,
        "total_errors": total_errors,
        "total_words": total_words,
    }
    
    # Save results
    output_data = {
        "summary": summary,
        "sample_results": sample_results,
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Save simplified summary
    simplified_results = [
        {
            "ref": r["reference"],
            "hyp": r["hypothesis"],
            "wer": r["wer"],
            "num_substitutions": r["details"]["substitutions"],
            "num_deletions": r["details"]["deletions"],
            "num_insertions": r["details"]["insertions"],
            "substitutions": r["details"]["substitutions_list"],
            "deletions": r["details"]["deletions_list"],
            "insertions": r["details"]["insertions_list"]
        }
        for r in sample_results
    ]
    
    with open(summary_output_file, "w") as f:
        json.dump(simplified_results, f, indent=2)
    
    logger.info(f"\n{'=' * 80}")
    logger.info("WER Computation Summary:")
    logger.info(f"Total samples: {summary['total_samples']}")
    logger.info(f"Samples processed: {summary['samples_processed']}")
    logger.info(f"Samples missing: {summary['samples_missing']}")
    logger.info(f"Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    logger.info(f"Average WER: {average_wer:.4f} ({average_wer*100:.2f}%)")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Total words: {total_words}")
    logger.info(f"\nResults saved to {output_file}")
    logger.info(f"Simplified summary saved to {summary_output_file}")


if __name__ == "__main__":
    main()

