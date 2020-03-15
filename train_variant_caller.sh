#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

# Wrapper script to train DL4VC model. 

set -e

SCRIPTDIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"

RUN_HELP=false
NUM_GPUS=1
EPOCHS=5
TRAIN_BATCH_SIZE=80
TEST_BATCH_SIZE=200
TRAIN_HDF=""
TEST_HDF=""
OUT_VCF="vc_model_eval.vcf"
SAMPLE_VCF=""
OUT_MODEL="vc_model.pth"

if [ "$#" -eq 0 ]; then
    RUN_HELP=true
fi

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -g)
            NUM_GPUS="$2"
            shift
            shift
            ;;
        -e)
            EPOCHS="$2"
            shift
            shift
            ;;
        --train-batch-size)
            TRAIN_BATCH_SIZE="$2"
            shift
            shift
            ;;
        --test-batch-size)
            TEST_BATCH_SIZE="$2"
            shift
            shift
            ;;
        --train-hdf)
            TRAIN_HDF="$2"
            shift
            shift
            ;;
        --test-hdf)
            TEST_HDF="$2"
            shift
            shift
            ;;
        --out-vcf)
            OUT_VCF="$2"
            shift
            shift
            ;;
        --sample-vcf)
            SAMPLE_VCF="$2"
            shift
            shift
            ;;
        --out-model)
            OUT_MODEL="$2"
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
    echo "Wrapper script for training variant caller."
    echo "Options:"
    echo "-g : Number of GPUs to use for training (default $NUM_GPUS)"
    echo "-e : Number of epochs to train for (default $EPOCHS)"
    echo "--train-batch-size : Batch size for training (default $TRAIN_BATCH_SIZE)"
    echo "--test-batch-size : Batch size for evaluation while training (default $TEST_BATCH_SIZE)"
    echo "--train-hdf : Path to training HDF file"
    echo "--test-hdf L Path to evaluation HDF file"
    echo "--out-vcf : Output path for VCF generated during evaluation (defauilt $OUT_VCF)"
    echo "--sample-vcf : Path to sample VCF to pull VCF headers from"
    echo "--out-model : Output path and name for trained model (default $OUT_MODEL)"
    echo "-h : Print help message"
    echo ""
    exit 0
fi

printf "Run training...\n"
python $SCRIPTDIR/main.py \
    --lr 0.0002     \
    --grad-clip 1.0     \
    --epochs $EPOCHS     \
    --log-interval 1     \
    --gpus $NUM_GPUS     \
    --label-smoothing 0.001     \
    --batch-size $TRAIN_BATCH_SIZE     \
    --test-batch-size $TEST_BATCH_SIZE     \
    --model-hidden-dropout 0.1     \
    --model-batchnorm     \
    --train_file $TRAIN_HDF     \
    --test_file $TEST_HDF      \
    --num-data-workers 5     \
    --trust-snp-only     \
    --non-snp-train-weight 2.0     \
    --fp-train-weight 0.2     \
    --model-use-q-scores     \
    --model-use-strands     \
    --auxillary-loss-weight 1.0     \
    --auxillary-loss-bases-weight 0.01     \
    --auxillary-loss-allele-weight 0.001     \
    --loss-debug-freq 10000     \
    --save_vcf_records     \
    --save_vcf_records_file $OUT_VCF     \
    --aux-keep-candidate-af     \
    --model-use-reads-ref-var-mask     \
    --close_match_window 2.0     \
    --focal_loss_alpha 1.     \
    --focal_loss_gamma 0.2     \
    --model-conv-layers 7     \
    --model-residual-layer-start 5     \
    --model-ave-pool-layers 2     \
    --early_loss_weight 0.1     \
    --model-init-conv-channels 128     \
    --rm_var_reads_rate 0.0     \
    --rm_non_var_reads_rate 0.0     \
    --close_examples_sample_rate 0.15     \
    --delay_augmentation_epochs 1      \
    --save_hard_example_records     \
    --learn_early_loss_weight     \
    --model_pool_combine_dimension 0     \
    --model-final-conv-channels 128     \
    --model-bottleneck-size 32     \
    --model_final_layer_dilation 2     \
    --model_middle_layer_dilation 2     \
    --model_concat_hw_reads     \
    --model-highway-single-reads     \
    --sample_vcf $SAMPLE_VCF     \
    --modelsave $OUT_MODEL
