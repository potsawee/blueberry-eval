#!/bin/bash
set -x

meta_lst=$1
output_dir=$2
lang=$3

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

workdir=$(cd $(dirname $0); cd ../; pwd)

# Generate wav_res_ref_text file
python3 get_wav_res_ref_text.py $meta_lst $output_dir $wav_wav_text

# Prepare checkpoint (download models if needed)
python3 prepare_ckpt.py

# Create temporary directory for results
timestamp=$(date +%s)
out_dir=/tmp/thread_metas_$timestamp/results/
mkdir -p $out_dir

# Run WER calculation on single process
sub_score_file=$out_dir/result.wer.out
python3 run_wer.py $wav_wav_text $sub_score_file $lang

# Calculate average WER
cp $sub_score_file $out_dir/merge.out
python3 eval/average_wer.py $out_dir/merge.out $score_file

# Cleanup
rm $wav_wav_text
rm -rf /tmp/thread_metas_$timestamp/

echo "WER results saved to: $score_file"
cat $score_file

