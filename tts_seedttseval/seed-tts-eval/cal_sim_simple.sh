#!/bin/bash
set -x

meta_lst=$1
output_dir=$2
checkpoint_path=$3

# Convert to absolute paths
meta_lst=$(realpath "$meta_lst")
output_dir=$(realpath "$output_dir")
checkpoint_path=$(realpath "$checkpoint_path")

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.sim

# Generate wav_res_ref_text file
python3 get_wav_res_ref_text.py $meta_lst $output_dir $wav_wav_text

workdir=$(cd $(dirname $0); pwd)

cd $workdir/thirdparty/UniSpeech/downstreams/speaker_verification/

# Create temporary directory for results
timestamp=$(date +%s)
out_dir=/tmp/thread_metas_$timestamp/results/
mkdir -p $out_dir

# Run SIM calculation on single process
python3 verification_pair_list_v2.py $wav_wav_text \
    --model_name wavlm_large \
    --checkpoint $checkpoint_path \
    --scores $out_dir/result.sim.out \
    --wav1_start_sr 0 \
    --wav2_start_sr 0 \
    --wav1_end_sr -1 \
    --wav2_end_sr -1 \
    --device cuda:0

# Calculate average SIM
grep -v "avg score" $out_dir/result.sim.out > $out_dir/merge.out
python3 average.py $out_dir/merge.out $score_file

# Cleanup
rm $wav_wav_text
rm -rf /tmp/thread_metas_$timestamp/

echo "SIM results saved to: $score_file"
cat $score_file

