# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

import argparse

def create_arg_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Context module')
    parser.add_argument('--train_file', type=str,
                        help='input train file Nx5 csv format')
    parser.add_argument('--test_file', type=str, required=True,
                        help='input train file Nx5 csv format')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='print sizes, other debug info while training (extremely slow)')
    parser.add_argument('--loss-debug-freq', default=0, type=int, help='Set > 0 for loss screen print -- useful for auxillary losses especially')
    parser.add_argument('--max-train-batches', default=0, type=int, help='(for debugging) set > 0 to train for only X batches')
    parser.add_argument('--max-test-batches', default=0, type=int, help='(for debugging) set > 0 to train for only X batches')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='input batch size for training (per GPU)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing -- per GPU')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--epochs_skip_eval', type=int, default=0, help='HACK -- skip eval step for X epochs -- speed')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-decay', type=float, default=1.0,
                        help='decay LR every epoch?')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='Gradient clipping -- most likely set to 1.0 to avoid blowup for deep network.')

    # Various methods for dealing with massive problem skew.
    # label smoothing (epsilon for good enough predictions), focal loss, and closely labeled example skipping
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Set > 0 [Inception uses 0.1] to penalize over-confidence for softmax classifications. Helps with regularization.')
    parser.add_argument('--close_match_window', type=float, default=2.,
                        help='Window size (w/r/t) distance from pure label and label_smoothing -- that counts as close match -- for counting and training skip.')
    parser.add_argument('--focal_loss_gamma', type=float, default=0.,
                        help='Set gamma > 0. to turn on focal loss -- focus on hard examples. Values 0.5 - 4.0 (2.0 is high)')
    parser.add_argument('--focal_loss_alpha', type=float, default=1.,
                        help='Not really used correctly -- just used to boost focal loss examples that pass gamma down-weighting.')
    parser.add_argument('--close_examples_sample_rate', type=float, default=1.0, help='With prediction-based sampling, what percent of close examples to keep? Keep all non-close')
    parser.add_argument('--save_hard_example_records', action='store_true', default=False, help='Save up to 100k VCFs for hard examples? Per epoch output to temp file.')

    # Loss weighting and thresholding
    parser.add_argument('--use-var-type-threshold', action='store_true', default=False,
                        help='Instead of binary predicition, threshold via variant type [1 - pro(not variant)] -- include heter/homo prediction.')
    parser.add_argument('--binary-weight', type=float, default=1.0,
                        help='weight of binary class loss -- predict mutation Y/N directly')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_vcf_records', default=False, action='store_true',
                        help='Enable option to store VCF records (for variants called by our model)')
    parser.add_argument('--save_vcf_records_file', type=str, default='',
                        help='File location in which to write VCF record (give it a good name)')
    parser.add_argument('--sample_vcf',
                        help='VCF for to use for header of output VCF file')
    parser.add_argument('--gpus', type=int, default=1, help='how many gpus to use')
    parser.add_argument('--num-data-workers', type=int, default=5, help='number of data loader workers -- set to 0 if HDF problems')
    parser.add_argument('--modelsave', type=str, default="checkpoint.pth.tar", help='path save model')
    parser.add_argument('--modelload', type=str, help='path load model')
    parser.add_argument('--train_holdout_chromosomes', type=str, nargs='+', default=[], help='Holdout chromosomes, for training (do not train on these)')
    parser.add_argument('--test_holdout_chromosomes', type=str, nargs='+', default=[], help='Holdout chromosomes, for testing (only on these)')
    parser.add_argument('--shuffle_test', action='store_true', default=False, help='Now support shuffled test set -- not necessary.')
    parser.add_argument('--gatk-table', type=str, default="", help='optionally, supply GATK lookup table for comparison')
    parser.add_argument('--giab-table', type=str, default="", help='optionally, supply GIAB lookup table for debugging')
    parser.add_argument('--test-trust-region-table', type=str, default="", help='optionally, supply trust region lookup table for testing dataset')
    parser.add_argument('--train-trust-region-table', type=str, default="", help='optionally, supply trust region lookup table for training dataset')
    parser.add_argument('--non-trust-train-weight', type=float, default=0.01, help='weight of training examples in non-trust regions (if provided)')
    parser.add_argument('--fp-train-weight', type=float, default=1.0, help='assign lower weight to FP (negative) examples -- especially if these outnumber positives')
    parser.add_argument('--trust-snp-only', action='store_true', default=False, help='HACK -- on Trust Regions -- eval on SNP only? [bad datasets]')
    parser.add_argument('--non-snp-train-weight', type=float, default=1.0, help='weight non INDEL examples -- in case eval on SNPs only?')
    parser.add_argument('--auxillary-loss-weight', type=float, default=0.0, help='loss weight to predict extra information -- like variant type, confirm freq, coverage, bases being modified')
    parser.add_argument('--auxillary-loss-bases-weight', type=float, default=0.1, help='re-weight (allow to remove) weight on predicting bases -- tough for some algorithms')
    parser.add_argument('--auxillary-loss-allele-weight', type=float, default=1.0, help='re-weight (or remove) weight on allele freq -- so it doensnt dominate aux loss')
    parser.add_argument('--aux-keep-candidate-af', action='store_true', default=False, help='Dont recompute AF -- use candidate gen value -- until multiallele is fixed & validated')
    parser.add_argument('--early_loss_layers', type=int, nargs='+', default=[], help='layers (starting from 1) after which to compute early loss [for better init]')
    parser.add_argument('--early_loss_weight', type=float, default=0.1, help='relative weight of early layer loss -- training only')
    parser.add_argument('--learn_early_loss_weight', action='store_true', default=False, help='learn early loss layer weights instead')
    parser.add_argument('--layer_loss_weight', type=float, default=0.01, help='If learn loss weight (for final score), also make each layer learn the loss independently -- to avoid instabilties.')

    ### Directional data augmentation -- remove reads either explicitly supporting or not supporting the variant in consideration
    # Just variation -- then use ranking loss to ensure that more evidence == higher score
    parser.add_argument('--delay_augmentation_epochs', type=int, default=0, help='overwrite data augmentation -- delay for X epochs (typically first epoch)')
    parser.add_argument('--rm_var_reads_rate', type=float, default=0., help='turn on for random (0-1) sample of *batches* to remove variant match reads')
    parser.add_argument('--rm_non_var_reads_rate', type=float, default=0., help='turn on for random (0-1) sample of *batches* to remove variant *non* match reads (not necessarily reference')

    # Can also directionally remove reads, and make sure loss makes sense (directionaly)
    parser.add_argument('--training_use_directional_augmentation', action='store_true', default=False, help='Train data twice -- with inputs, and directional augmetation. Cut batch size by 1/2.')
    parser.add_argument('--augmented_example_weight', type=float, default=0.2, help='weight of augmented examples, relative to original data (1.0).')
    parser.add_argument('--delta_loss_weight', type=float, default=10.0, help='weight of loss function, making sure directional augment is correct -- rm read and p(var) moves correct direction')

    ### Augment data? Either through errors, masking (hiding) information, or down-sampling from full data
    parser.add_argument('--augment-single-reads', action='store_true', default=False, help='optionally, train with data augmentation (introduce read errors) -- very slow/inefficient for now')
    parser.add_argument('--augment-reference', action='store_true', default=False, help='optionally, augment (partially mask) the reference, to prevent overfitting')

    ### Augment via down-sampling reads [sam BAM, fewer data]
    # Ex: 50% of the time... remove about 30% of reads (noise added to downsample rate)
    parser.add_argument('--reads-dynamic-downsample-rate', type=float, default=0., help='[training only] Down-sample rate at read level -- noise added.')
    parser.add_argument('--reads-dynamic-downsample-prob', type=float, default=0., help='[training only] What percent of data to down-sample (keep higher percent -- at least 50 -- fully sampled)')

    ### Model parameters:
    parser.add_argument('--model-conv-layers', type=int, default=5, help='Depth of Conv1D network.')
    parser.add_argument('--model-ave-pool-layers', type=int, nargs='+', default=[2], help='Which layers to average pool across reads -- after the layer?')
    parser.add_argument('--model-residual-layer-start', type=int, default=0, help='If > 0 -- define first residual layer -- and subsequent layers.')
    parser.add_argument('--model-init-conv-channels', type=int, default=128, help='Number of initial (lowest layer) conv filters')
    parser.add_argument('--model-final-conv-channels', type=int, default=128, help='Number of final (top layer) conv filters')
    parser.add_argument('--model_final_layer_dilation', type=int, default=1, help='Set to 2 (or more?) for dilated convolutions -- bigger receptive field')
    parser.add_argument('--model_middle_layer_dilation', type=int, default=1, help='Set to 2 (or more?) for dilated convolutions -- bigger receptive field')
    parser.add_argument('--model-hidden-dropout', type=float, default=0., help='Hidden layers dropout')
    parser.add_argument('--model-batchnorm', action='store_true', default=False, help='Layer batchnorm')
    parser.add_argument('--model-use-q-scores', action='store_true', default=False, help='Use Q scores for model bases (log scale)')
    parser.add_argument('--model-use-strands', action='store_true', default=False, help='Use strand (diretion) for model bases (enum)')
    parser.add_argument('--model-highway-single-reads', action='store_true', default=False, help='bottleneck + highway from individual reads? May help with counting -- and focus on mutation at center')
    parser.add_argument('--model-bottleneck-size', type=int, default=32, help='Size of bottleneck output -- outputs per read. NOTE: big perf hit as this goes up, when highway ON')
    parser.add_argument('--model_concat_hw_reads', action='store_true', default=False, help='Concat HW outputs -- instead of average them by default (WaveNet version)')
    parser.add_argument('--model-use-naive-var-vector', action='store_true', default=False, help='distinguish multi-allele by passing naive AF and variant letters to FCN?')
    parser.add_argument('--model-use-reads-ref-var-mask', action='store_true', default=False, help='distinguish multi-allele by coding ref & var matches for each read -- adds two more channels to ConvNet')
    parser.add_argument('--model-use-AF', action='store_true', default=False, help='pass AF (allele frequency) to the model? Needed to disambiguate very long inserts')
    parser.add_argument('--model_skip_final_maxpool', action='store_true', default=False, help='only mean pool -- does it help, and save parameters?')
    parser.add_argument('--model_pool_combine_dimension', type=int, default=2048, help='dimension to save from maxpool, meanpool outputs [going into final FCN]')

    ### Transformer args:
    parser.add_argument('--use_transformer', action='store_true', default=False, help='Add transformer layers after Conv layers (before pooling)')
    parser.add_argument('--transformer_encoder_heads', type=int, default=4, help='Transformer heads (splits output dimension)')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers. No gradient checkpoint so takes memory (more than Conv layers).')
    parser.add_argument('--transformer_feedforward_dim', type=int, default=64, help='Transformer feed forward dimensions')
    parser.add_argument('--final_transformer_dims', type=int, default=64, help='Reduce dimensions to final size from Transformer to pooling. [pass 0 to keep dimension]')
    parser.add_argument('--transformer_residual', action='store_true', default=False, help='Residual Transformer network? Recommended.')
    parser.add_argument('--transformer_encoder_dropout', type=float, default=0.1, help='Transformer dropout -- 0.1 recommended')

    return parser
