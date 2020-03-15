#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

# End to end script to call variants.

set -e

BAM=""
BED=""
RUN_HELP=false
MODEL=""
OUTDIR="vw_output"
REFERENCE=""
SCRIPTDIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"

if [ "$#" -eq 0 ]; then
    RUN_HELP=true
fi

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i)
            BAM="$2"
            shift
            shift
            ;;
        -b)
            BED="$2"
            shift
            shift
            ;;
        -m)
            MODEL="$2"
            shift
            shift
            ;;
        -o)
            OUTDIR="$2"
            shift
            shift
            ;;
        -r)
            REFERENCE="$2"
            shift
            shift
            ;;
        -h|--help)
            RUN_HELP=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ ${RUN_HELP} = true ]; then
    echo "End to end script for calling variants."
    echo "Options:"
    echo "-i : Input BAM file"
    echo "-b : BED file with regions for variant calling"
    echo "-m : Path to pretrained model"
    echo "-o : Output directory for files"
    echo "-r : Reference genome"
    echo "-h : Print help message"
    echo ""
    exit 0
fi

# Create output directory
mkdir -p $OUTDIR

# Generate candidate variants
printf "Generate candidate VCF...\n"
python $SCRIPTDIR/tools/candidate_generator.py \
    --input $BAM \
    --output $OUTDIR/candidates.vcf \
    --snp_min_freq 0.075 \
    --indel_min_freq 0.02 \
    --bedfile $BED \
    --keep_multialleles >& $OUTDIR/candidate_generator.log

# Generate HDF file
printf "Convert candidates to HDF...\n"
python $SCRIPTDIR/tools/convert_bam_single_reads.py \
    --input $BAM \
    --fp_vcf $OUTDIR/candidates.vcf \
    --fasta-input $REFERENCE \
    --output $OUTDIR/candidates.hdf \
    --max-reads 200 \
    --num-processes 80 \
    --locations-process-step 100000 \
    --max-insert-length 10 \
    --max-insert-length-variant 50 \
    --save-q-scores \
    --save-strand >& $OUTDIR/training_data.log

printf "Run inference...\n"
python $SCRIPTDIR/main.py \
    --lr 0.0002 \
    --grad-clip 1.0 \
    --label-smoothing 0.001 \
    --model-hidden-dropout 0.1 \
    --model-batchnorm \
    --num-data-workers 5 \
    --trust-snp-only \
    --non-snp-train-weight 2.0 \
    --fp-train-weight 0.2 \
    --model-use-q-scores \
    --model-use-strands \
    --auxillary-loss-weight 1.0 \
    --auxillary-loss-bases-weight 0.01 \
    --auxillary-loss-allele-weight 0.001 \
    --loss-debug-freq 10000 \
    --aux-keep-candidate-af \
    --model-use-reads-ref-var-mask \
    --close_match_window 2.0 \
    --focal_loss_alpha 1. \
    --focal_loss_gamma 0.2 \
    --model-conv-layers 7 \
    --model-residual-layer-start 5 \
    --model-ave-pool-layers 2 \
    --early_loss_weight 0.1 \
    --model-init-conv-channels 128 \
    --rm_var_reads_rate 0.0 \
    --rm_non_var_reads_rate 0.0 \
    --close_examples_sample_rate 0.15 \
    --delay_augmentation_epochs 1  \
    --learn_early_loss_weight \
    --model_pool_combine_dimension 0 \
    --model-final-conv-channels 128 \
    --model-bottleneck-size 32 \
    --model_final_layer_dilation 2 \
    --model_middle_layer_dilation 2 \
    --model_concat_hw_reads \
    --model-highway-single-reads \
    --log-interval 1 \
    --model-batchnorm \
    --gpus 1 \
    --test-batch-size 200 \
    --save_vcf_records \
    --save_vcf_records_file $OUTDIR/model_test.vcf \
    --test_file $OUTDIR/candidates.hdf \
    --sample_vcf $OUTDIR/candidates.vcf \
    --modelload $MODEL >& $OUTDIR/training.log

# Post-process results VCF file
printf "Sort output VCF...\n"
cat $OUTDIR/epoch1_model_test.vcf | awk '$1 ~ /^#/ {print $0;next} {print $0 | "sort -k1,1 -k2,2n"}' > $OUTDIR/model_test_sorted.vcf

printf "Threshold and combine multi-allele...\n"
python $SCRIPTDIR/tools/format_vcf.py \
    --input_file $OUTDIR/model_test_sorted.vcf \
    --output_file $OUTDIR/model_test_sorted_thres.vcf \
    --snp_threshold 0.1 \
    --indel_threshold 0.2 \
    --snp_zygo_threshold 0.75 \
    --indel_zygo_threshold 0.8 >& $OUTDIR/format_vcf.log

bcftools norm -m +any $OUTDIR/model_test_sorted_thres.vcf > $OUTDIR/model_test_sorted_thres-join.vcf 2> $OUTDIR/bcftools_norm.log
sed -i 's/0\/2/1\/2/' $OUTDIR/model_test_sorted_thres-join.vcf
sed -i 's/2\/2/1\/2/' $OUTDIR/model_test_sorted_thres-join.vcf

printf "Compress and index called VCF...\n"
bgzip -c $OUTDIR/model_test_sorted_thres-join.vcf > $OUTDIR/called_variants.vcf.gz
tabix -p vcf $OUTDIR/called_variants.vcf.gz

echo "Called variants in $OUTDIR/called_variants.vcf.gz"
