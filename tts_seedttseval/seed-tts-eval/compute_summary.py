#!/usr/bin/env python3
"""
Compute WER and SIM summary statistics from output files.

Usage:
python compute_summary.py <wer_file>
python compute_summary.py <base_path>  # will look for .wer and .sim files

Example:
python compute_summary.py ../tts_outputs/0shot-model/wavs/wav_res_ref_text.wer
python compute_summary.py ../tts_outputs/0shot-model/wavs/wav_res_ref_text
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_wer_data(wer_file: str) -> dict:
    """Load WER data with utterance IDs."""
    
    wer_data = {}
    
    with open(wer_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 7:
            continue
        
        utt_path = parts[0]
        wav_filename = Path(utt_path).name
        
        wer = float(parts[1])
        ins = float(parts[4])
        dele = float(parts[5])
        sub = float(parts[6])
        
        wer_data[wav_filename] = {
            'wer': wer,
            'ins': ins,
            'del': dele,
            'sub': sub,
        }
    
    return wer_data


def compute_wer_statistics(wer_file: str) -> dict:
    """Parse WER file and compute statistics."""
    
    wer_data = load_wer_data(wer_file)
    
    if not wer_data:
        return None
    
    wers = [v['wer'] for v in wer_data.values()]
    insertions = [v['ins'] for v in wer_data.values()]
    deletions = [v['del'] for v in wer_data.values()]
    substitutions = [v['sub'] for v in wer_data.values()]
    
    # Compute statistics
    return {
        'count': len(wers),
        'avg_wer': sum(wers) / len(wers),
        'avg_ins': sum(insertions) / len(insertions),
        'avg_del': sum(deletions) / len(deletions),
        'avg_sub': sum(substitutions) / len(substitutions),
        'min_wer': min(wers),
        'max_wer': max(wers),
        'perfect_count': sum(1 for w in wers if w == 0.0),
        'high_error_count': sum(1 for w in wers if w > 0.5),
    }


def load_sim_data(sim_file: str) -> dict:
    """Load SIM data with utterance IDs."""
    
    sim_data = {}
    
    with open(sim_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        
        # Extract generated wav path (first part before |)
        wav_path = parts[0].split('|')[0]
        wav_filename = Path(wav_path).name
        
        # Remove _0_-1 suffix if present
        if wav_filename.endswith('_0_-1'):
            wav_filename = wav_filename[:-5]
        
        sim = float(parts[1])
        sim_data[wav_filename] = sim
    
    return sim_data


def compute_sim_statistics(sim_file: str) -> dict:
    """Parse SIM file and compute statistics."""
    
    sim_data = load_sim_data(sim_file)
    
    if not sim_data:
        return None
    
    similarities = list(sim_data.values())
    
    # Compute statistics
    return {
        'count': len(similarities),
        'avg_sim': sum(similarities) / len(similarities),
        'min_sim': min(similarities),
        'max_sim': max(similarities),
        'high_sim_count': sum(1 for s in similarities if s > 0.6),
        'low_sim_count': sum(1 for s in similarities if s < 0.4),
    }


def print_wer_summary(stats: dict):
    """Display WER summary statistics."""
    print("=" * 60)
    print("WER EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {stats['count']}")
    print("-" * 60)
    print(f"Average WER:           {stats['avg_wer'] * 100:6.2f}%")
    print(f"  - Substitutions:     {stats['avg_sub'] * 100:6.2f}%")
    print(f"  - Insertions:        {stats['avg_ins'] * 100:6.2f}%")
    print(f"  - Deletions:         {stats['avg_del'] * 100:6.2f}%")
    print("=" * 60)
    
    print(f"\nDetailed Statistics:")
    print(f"  Min WER:             {stats['min_wer'] * 100:6.2f}%")
    print(f"  Max WER:             {stats['max_wer'] * 100:6.2f}%")
    
    perfect_pct = stats['perfect_count'] / stats['count'] * 100
    print(f"  Perfect predictions: {stats['perfect_count']} / {stats['count']} ({perfect_pct:.1f}%)")
    
    high_error_pct = stats['high_error_count'] / stats['count'] * 100
    print(f"  High error (>50%):   {stats['high_error_count']} / {stats['count']} ({high_error_pct:.1f}%)")
    print("=" * 60)


def print_sim_summary(stats: dict):
    """Display SIM summary statistics."""
    print("\n" + "=" * 60)
    print("SIMILARITY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {stats['count']}")
    print("-" * 60)
    print(f"Average Similarity:    {stats['avg_sim']:6.4f}")
    print("=" * 60)
    
    print(f"\nDetailed Statistics:")
    print(f"  Min Similarity:      {stats['min_sim']:6.4f}")
    print(f"  Max Similarity:      {stats['max_sim']:6.4f}")
    
    high_sim_pct = stats['high_sim_count'] / stats['count'] * 100
    print(f"  High similarity (>0.6): {stats['high_sim_count']} / {stats['count']} ({high_sim_pct:.1f}%)")
    
    low_sim_pct = stats['low_sim_count'] / stats['count'] * 100
    print(f"  Low similarity (<0.4):  {stats['low_sim_count']} / {stats['count']} ({low_sim_pct:.1f}%)")
    print("=" * 60)


def create_scatter_plot(wer_file: str, sim_file: str, output_path: str):
    """Create scatter plot of WER vs SIM."""
    
    wer_data = load_wer_data(wer_file)
    sim_data = load_sim_data(sim_file)
    
    # Match examples by utterance ID
    wer_values = []
    sim_values = []
    
    for wav_filename in wer_data.keys():
        if wav_filename in sim_data:
            wer_values.append(wer_data[wav_filename]['wer'])
            sim_values.append(sim_data[wav_filename])
    
    if not wer_values:
        print("Warning: No matching examples found between WER and SIM files.")
        return
    
    # Compute mean SIM for examples with WER = 0.0
    perfect_wer_sims = [sim for wer, sim in zip(wer_values, sim_values) if wer == 0.0]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(wer_values, sim_values, alpha=0.5, s=20, label='All samples')
    
    # Add red marker for mean SIM at WER=0.0
    if perfect_wer_sims:
        mean_sim_at_zero_wer = sum(perfect_wer_sims) / len(perfect_wer_sims)
        plt.scatter([0.0], [mean_sim_at_zero_wer], 
                   color='red', s=200, marker='*', 
                   edgecolors='darkred', linewidths=2,
                   label=f'Mean SIM at WER=0.0: {mean_sim_at_zero_wer:.4f}',
                   zorder=5)
        
        print(f"  Mean SIM at WER=0.0: {mean_sim_at_zero_wer:.4f} (n={len(perfect_wer_sims)})")
    
    plt.xlabel('Word Error Rate (WER)', fontsize=12)
    plt.ylabel('Speaker Similarity (SIM)', fontsize=12)
    plt.title('WER vs Speaker Similarity', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # Add a note with sample count
    plt.text(0.02, 0.98, f'N = {len(wer_values)} samples', 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nScatter plot saved to: {output_path}")
    print(f"  Matched samples: {len(wer_values)}")


def compute_summary(input_path: str):
    """Compute and display WER and SIM summary statistics."""
    
    # Determine file paths
    input_path_obj = Path(input_path)
    
    if input_path.endswith('.wer'):
        wer_file = input_path
        sim_file = input_path.replace('.wer', '.sim')
        base_path = input_path[:-4]
    elif input_path.endswith('.sim'):
        sim_file = input_path
        wer_file = input_path.replace('.sim', '.wer')
        base_path = input_path[:-4]
    else:
        wer_file = f"{input_path}.wer"
        sim_file = f"{input_path}.sim"
        base_path = input_path
    
    # Check which files exist
    wer_exists = Path(wer_file).exists()
    sim_exists = Path(sim_file).exists()
    
    if not wer_exists and not sim_exists:
        print(f"Error: Neither '{wer_file}' nor '{sim_file}' found.")
        sys.exit(1)
    
    # Process WER file
    if wer_exists:
        wer_stats = compute_wer_statistics(wer_file)
        if wer_stats:
            print_wer_summary(wer_stats)
        else:
            print("Warning: No valid WER data found.")
    else:
        print(f"Warning: WER file '{wer_file}' not found.")
    
    # Process SIM file
    if sim_exists:
        sim_stats = compute_sim_statistics(sim_file)
        if sim_stats:
            print_sim_summary(sim_stats)
        else:
            print("Warning: No valid SIM data found.")
    else:
        print(f"Warning: SIM file '{sim_file}' not found.")
    
    # Create scatter plot if both files exist
    if wer_exists and sim_exists:
        output_plot = f"{base_path}_wer_sim_scatter.png"
        create_scatter_plot(wer_file, sim_file, output_plot)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_summary.py <wer_file|sim_file|base_path>")
        print("\nExamples:")
        print("  python compute_summary.py ../tts_outputs/0shot-model/wavs/wav_res_ref_text.wer")
        print("  python compute_summary.py ../tts_outputs/0shot-model/wavs/wav_res_ref_text.sim")
        print("  python compute_summary.py ../tts_outputs/0shot-model/wavs/wav_res_ref_text")
        sys.exit(1)
    
    input_path = sys.argv[1]
    compute_summary(input_path)

