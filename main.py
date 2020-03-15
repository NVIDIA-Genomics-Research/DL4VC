#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Basic minimal example for loading single reads data for network.

When complete, should provide examples for:
-- 2D CNN [Implemented]
-- some sort of RNN (for each read), combined with CNN or simple summer [TODO]
-- add position encoding [TODO]
-- add read and map quality scores [TODO]

For convenience, will hard-code some dimensions, and reduce size of the input. [currently to 100 reads]

Data generated via "convert-bam-single-reads.py" including that (numpy) data format.

@author: nyakovenko
"""

from __future__ import print_function
import os
import math
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Needs to be included *before* h5py to inherit multithreading support
import torch.multiprocessing
import time
import tqdm
from torch.utils.data import DataLoader
import h5py
import multiprocessing
from dl4vc.trainer import train, test
from dl4vc.dataset import *
from dl4vc.model import Basic2DNet
from dl4vc.utils import bin_to_string, save_checkpoint
from arguments import create_arg_parser
# For reading trust regions
from make_trust_region_filter import is_in_region


def main():
    # Parse command line arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Need 'num_workers': 0 -- to play nice with H5PY
    kwargs = {'num_workers': args.num_data_workers, 'pin_memory': True} if use_cuda else {}

    # If training file was specific, construct train loader.
    # HACK: look at file extension; load npy format
    time_zero = time.time()
    if args.train_file:
        assert args.train_file[-3:] == 'hdf', "Train dataset must be in HDF format"
        # For training, shuffle dataset, have option to augment data, and take *random* X of Y stored reads...
        train_dataset = ContextDatasetFromNumpy(args.train_file, args=args, holdout_chromosomes=args.train_holdout_chromosomes,
            augment_single_reads=args.augment_single_reads, augment_refernce=args.augment_reference,
            reads_dynamic_downsample_rate=args.reads_dynamic_downsample_rate, reads_dynamic_downsample_prob=args.reads_dynamic_downsample_prob)
        # Data sampler that removes (down-samples) easy examples in subsequent epochs
        # It also allows us to hold out chromosomes...
        if args.close_examples_sample_rate < 1.0:
            train_sampler = AdjustableDataSampler(train_dataset, args=args)
            train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, **kwargs) # shuffle = True
        else:
            print('keeping all examples -- no close example down-sampling')
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, **kwargs) # shuffle = True
        print('Time to load training dataset file %.5fs' % (time.time() - time_zero))

    # If test file was specified, construct test loader.
    time_zero = time.time()
    if args.test_file:
        assert args.test_file[-3:] == 'hdf', "Test dataset must be in HDF format"
        # For test set -- don't shuffle, don't augment
        test_dataset = ContextDatasetFromNumpy(args.test_file, args=args, holdout_chromosomes=args.test_holdout_chromosomes,
            augment_single_reads=False, augment_refernce=False)
        # NOTE: We *only* need a test data sampler -- if we use holdout chromosomes
        if len(args.test_holdout_chromosomes) > 0:
            print('creating test data sampler -- to handle holdout_chromosomes %s' % str(args.test_holdout_chromosomes))
            test_sampler = AdjustableDataSampler(test_dataset, args=args, reverse_holdout=True, shuffle=args.shuffle_test)
            test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, **kwargs)
        else:
            test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=args.shuffle_test, **kwargs)
        print('Time to load testing dataset file %.5fs' % (time.time() - time_zero))

    # Build model
    best_loss = None
    model = Basic2DNet(target_size=3, init_conv_channels=args.model_init_conv_channels, final_conv_channels=args.model_final_conv_channels,
        hidden_dropout=args.model_hidden_dropout, use_batchnorm=args.model_batchnorm,
        skip_final_maxpool=args.model_skip_final_maxpool, pool_combine_dimension=args.model_pool_combine_dimension,
        early_loss_layers=args.early_loss_layers,
        use_q_scores=args.model_use_q_scores, use_strands=args.model_use_strands,
        total_conv_layers=args.model_conv_layers, residual_layer_start=args.model_residual_layer_start,
        conv_1d_pool_layers = args.model_ave_pool_layers,
        final_layer_dilation=args.model_final_layer_dilation, middle_layer_dilation=args.model_middle_layer_dilation,
        append_bottleneck_highway_reads=args.model_highway_single_reads,
        bottleneck_channels=args.model_bottleneck_size, bottleneck_linear_outputs=args.model_bottleneck_size,
        concat_hw_reads=args.model_concat_hw_reads,
        use_naive_variant_encoding=args.model_use_naive_var_vector,
        use_reads_ref_var_mask=args.model_use_reads_ref_var_mask,
        append_allele_frequency=args.model_use_AF, args=args)
    # Display parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model includes %d trainable parameters...' % (param_count))
    optimizer=optim.Adam(model.parameters(),lr=args.lr)
    model=nn.DataParallel(model,device_ids=list(range(args.gpus))).cuda()

    # Load model state if a checkpoint file has been provided.
    # TODO: Merge proper training checkpointing from dev-chkpnt_rstrt branch.
    if args.modelload:
        print("Loading model checkpoint from {}".format(args.modelload))
        checkpoint = torch.load(args.modelload, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    # Load GATK table, if present [for validation comparison]
    # NOTE: ~1GB of data
    if args.gatk_table:
        print('loading GATK file from %s' % args.gatk_table)
        t = time.time()
        gatk_table = pickle.load(open(args.gatk_table, 'rb'))
        print('...took %.5fs to load %d lines of GATK' % (time.time() - t, len(gatk_table)))
    else:
        gatk_table = {}

    # Load (training) trust regions, so we can report results on those:
    if args.train_trust_region_table:
        print('loading (training) Trust Regions from %s' % args.train_trust_region_table)
        t = time.time()
        train_trust_starts, train_trust_ends = pickle.load(open(args.train_trust_region_table, 'rb'))
        print('...took %.5fs to load %d lines of Trust Regions' % (time.time() - t, len(train_trust_starts)))
    else:
        train_trust_starts, train_trust_ends = ({}, {})

    # Load (testing) trust regions, so we can report results on those:
    if args.test_trust_region_table:
        print('loading (testing) Trust Regions from %s' % args.test_trust_region_table)
        t = time.time()
        test_trust_starts, test_trust_ends = pickle.load(open(args.test_trust_region_table, 'rb'))
        print('...took %.5fs to load %d lines of Trust Regions' % (time.time() - t, len(test_trust_starts)))
    else:
        test_trust_starts, test_trust_ends = ({}, {})

    # If training file is specified, run training and inference.
    if args.train_file:
        print('\n\nRunning in training and inference mode...\n\n')
        for epoch in range(1, args.epochs + 1):
            s = time.time()

            # If delta loss activated (this epoch) -- create new data loader with 1/2 batch size
            if args.training_use_directional_augmentation and (args.delay_augmentation_epochs + 1) == epoch:
                half_batch = int(args.batch_size/2)
                print('Re-initializing training data loader with 1/2 batch %d' % half_batch)
                train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=half_batch, **kwargs)
            elif args.training_use_directional_augmentation and (args.delay_augmentation_epochs + 1) > epoch:
                print('Pre-train with full batch %d' % args.batch_size)

            train(args, model, device, train_loader, optimizer, epoch, train_dataset=train_dataset, debug=args.debug,
                trust_starts=train_trust_starts, trust_ends=train_trust_ends, non_trust_train_weight=args.non_trust_train_weight)

            print('\tTime elapsed for training {:.4f}\n'.format(time.time()-s), flush=True)
            s_eval = time.time()
            # Decay LR every epoch
            # TODO: Return LR by iterations -- thus support LR warmup for 1/2 epoch or so
            optimizer.param_groups[0]['lr']*=args.lr_decay

            # Inference on the test set -- with also write to VCF, upon request
            # Skip inference for faster training time
            if epoch <= args.epochs_skip_eval:
                print('Skipping eval for epoch %d' % epoch)
                print('\tTime elapsed for inference/testing {:.4f}'.format(time.time()-s_eval))
                print('\tTime elapsed overall {:.4f}\n'.format(time.time()-s), flush=True)
                torch.cuda.empty_cache()
                continue

            curloss = test(args, model, device, test_loader,
                gatk_table=gatk_table, trust_starts=test_trust_starts, trust_ends=test_trust_ends, non_trust_train_weight=args.non_trust_train_weight, epoch=epoch)
            if best_loss is None:
                is_best = True
                best_loss = curloss
            else:
                is_best = curloss < best_loss
                best_loss = min(curloss, best_loss)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                }, is_best, args.modelsave)
            print('\tTime elapsed for inference/testing {:.4f}'.format(time.time()-s_eval))
            print('\tTime elapsed overall {:.4f}\n'.format(time.time()-s), flush=True)

            # Attempt to release memory -- which somehow grows after test inference :-(
            torch.cuda.empty_cache()

            # If running out of memory -- show debug with and without test inference.
            #print('memory usage')
            #for n in range(args.gpus):
            #    print('GPU %d -- allocated, cached' % n)
            #    print(torch.cuda.memory_allocated(device=n))
            #    print(torch.cuda.memory_cached(device=n))
    # If only test file is specified, attempt to load a checkpoint and run inference.
    elif args.test_file:
        print('\n\nRunning in inference only mode...\n\n')
        assert args.modelload is not None, "--modelload argument is required when running in inference only mode"
        s_eval = time.time()
        test(args, model, device, test_loader,
            gatk_table=gatk_table, trust_starts=test_trust_starts, trust_ends=test_trust_ends, non_trust_train_weight=args.non_trust_train_weight)
        print('\tTime elapsed for inference/testing {:.4f}'.format(time.time()-s_eval))

        # Attempt to release memory -- which somehow grows after test inference :-(
        torch.cuda.empty_cache()

        # If running out of memory -- show debug with and without test inference.
        #print('memory usage')
        #for n in range(args.gpus):
        #    print('GPU %d -- allocated, cached' % n)
        #    print(torch.cuda.memory_allocated(device=n))
        #    print(torch.cuda.memory_cached(device=n))

if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
