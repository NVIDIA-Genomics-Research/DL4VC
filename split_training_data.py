#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Take huge text dataset, split into train, val, test and holdout whole chromosomes.

1. Does *not* assume that input text is presented in (chromosome) order
2. Finds per-chromosome counts and indices
3. Computes probabilities to include each line in train, val, test and holdout
4. Selects indices
5. Reads and chooses once
6. Parses results and saves to Numpy 
7. Optionally, saves results to text

@author: nyakovenko
"""

from __future__ import print_function
import argparse
import numpy as np
import tqdm

# Decode 5-channel reads
def decode(raw):
    #ref=raw.split(",")
    #ref=np.array([int(x) for x in ref])
    ref=np.fromstring(raw, dtype=np.int, sep=',')
    ref=ref.reshape([5,int(len(ref)/5)])
    return ref

# decode truth labels ["1" = true mutation]
def decode_label(label):
    if label == "FP":
        return 2
    elif label == "FN":
        return 1
    elif label == '1':
        return 0
    else:
        print('Warning -- bad label, returning -1')
        return -1
        #raise ValueError("Unknown label: {}".format(label))

# Read single dataset from 'inputs' and chunk it into train/val/test [and smaller training sets] -- save separately
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Text to numpy object')
    parser.add_argument('--input', type=str, metavar='N', help='input file ')
    parser.add_argument('--output', type=str, default='HG001_300x_full', metavar='N', help='output file ')
    parser.add_argument('--debug', action='store_true', help='print extra information')
    parser.add_argument('--max', type=int, default=10000000, help='max number of variants to read')
    parser.add_argument('--holdout', nargs='*', help='chromosomes to hold out')
    parser.add_argument('--split', action='store_true', help='split into training/val/test')
    parser.add_argument('--subsample', action='store_true', help='subsample train set')
    parser.add_argument('--savecode', action='store_true', help='save .code files')

    args = parser.parse_args()
    print(args)

    # We will have chromosomes 1-22 + Y + X + MT [but learn this from the data]
    lines_by_chrome = {}
    # Read data once -- very fast in Python
    # Record line number & chromosome
    input_filename = args.input
    chrome = ''
    count_errors = 0 
    all_labels = set(['1', 'FP', 'FN', 'TP', 'TN'])
    with open(input_filename) as fp:
        count = 0
        for line in tqdm.tqdm(fp, total=args.max):
            if count > args.max:
                break
            c = line[:line.find(':')]
            res = line[line.rfind(';')+1:-1]
            items = line.split(';')
            ref = decode(items[1])
            reads = decode(items[2])
            raw_label = items[-1].strip()
            # bounds checking -- slows things down, but necessary in case some bad inputs
            if ref.shape != (5, 201) or reads.shape != (5,201) or not(raw_label in all_labels):
                print('Error reading line -- incorrect shape')
                print(line)
                print('label |%s|' % raw_label) 
                print(c)
                print(res)
                count += 1
                count_errors += 1
                print('num errors %d' % count_errors)
                continue
            if c != chrome:
                if args.debug:
                    print(c, count)
                chrome = c
            if not (c in lines_by_chrome.keys()):
                lines_by_chrome[c] = {'1': set(), 'FP': set(), 'FN': set()}
            # more error checking
            try:
                lines_by_chrome[c][res].add(count)
            except KeyError:
                print('Error reading line -- format incomplete')
                print(line)
                print(c)
                print(res)
                print('count %s' % count)
                count += 1
                count_errors += 1
                print('num errors %d' % count_errors)
                continue
            # Make sure we keep the count
            count += 1
        print(count)
    full_count = count
    
    # Show counts by chromosome
    print([(c, [(res, len(lines_by_chrome[c][res])) for res in lines_by_chrome[c]]) for c in lines_by_chrome])

    # Full chromosome holdout
    holdout_chromes = set()
    holdout_lines = set()
    if args.holdout is not None:
        holdout_chromes = args.holdout
        for c in holdout_chromes:
            for res in lines_by_chrome[c]:
                holdout_lines.update(lines_by_chrome[c][res])
        print('Counting %d lines in holdout set for chromes %s' % (len(holdout_lines), holdout_chromes))

    # Collect lines for non-holdout
    exclude_chromes = ['chrMT', 'chrY', 'chrhs37d5', 'chrGL000192.1', 'chrGL000191.1']
    print('Excluding chromses %s' % exclude_chromes)
    non_holdout_chromes = lines_by_chrome.keys() - set(holdout_chromes) - set(exclude_chromes)
    print(non_holdout_chromes)
    non_holdout_lines = set()
    for c in non_holdout_chromes:
        for res in lines_by_chrome[c]:
            non_holdout_lines.update(lines_by_chrome[c][res])
    print('Counting %d lines in non-holdout set for chromes %s' % (len(non_holdout_lines), non_holdout_chromes))

    #shuffle
    non_holdout_lines = np.random.permutation(list(non_holdout_lines))
    nh_len = len(non_holdout_lines)
    
    #perform 80/10/10 split for train/val/test on the non-holdout dataset
    if args.split:
        non_holdout_train = set(non_holdout_lines[:int(nh_len*0.8)])
        non_holdout_val = set(non_holdout_lines[int(nh_len*0.8):int(nh_len*0.9)])
        non_holdout_test = set(non_holdout_lines[int(nh_len*0.9):])
        print('Saving train/val/test set size %d/%d/%d' % (len(non_holdout_train), len(non_holdout_val), len(non_holdout_test)))
    else:
        non_holdout_train = non_holdout_lines
        non_holdout_val = set()
        non_holdout_test = set()
    
    # Also sub-sample the training to get scaled smaller training datasets
    # 1/8th, 1/4th, 1/2 of the training set
    # TODO: We could over or under-sample categories for balance
    if args.subsample:
        nt_len = len(non_holdout_train)
        non_holdout_train_2nd = set(non_holdout_lines[:int(nt_len*0.5)])
        non_holdout_train_4th = set(non_holdout_lines[:int(nt_len*0.25)])
        non_holdout_train_8th = set(non_holdout_lines[:int(nt_len*0.125)])
        print('Saving 2nd/4th/8th training set size %d/%d/%d' % (len(non_holdout_train_2nd), len(non_holdout_train_4th), len(non_holdout_train_8th)))
    
    # Create empty numpy arrays
    dt = np.dtype([('name', np.unicode_, 16), ('ref', np.int8, (5,201)), ('reads', np.int16, (5,201)), ('label', np.int8, 1)])
    train_array = np.empty(len(non_holdout_train), dt)
    train_count = 0
    if args.holdout is not None:
        holdout_array = np.empty(len(holdout_lines), dt)
        holdout_count = 0
    if args.split:
        val_array = np.empty(len(non_holdout_val), dt)
        val_count = 0
        test_array = np.empty(len(non_holdout_test), dt)
        test_count = 0
    if args.subsample:
        train_2nd_array = np.empty(len(non_holdout_train_2nd), dt)
        train_2nd_count = 0 
        train_4th_array = np.empty(len(non_holdout_train_4th), dt)
        train_4th_count = 0
        train_8th_array = np.empty(len(non_holdout_train_8th), dt)
        train_8th_count = 0
    
    #open .code files
    write_base = args.output
    if args.savecode:
        train_writer = open(write_base + '_train.code', 'w')
        if args.holdout is not None:
            holdout_writer = open(write_base + '_holdout.code', 'w')
        if args.split:
            val_writer = open(write_base + '_val.code', 'w')
            test_writer = open(write_base + '_test.code', 'w')
        if args.subsample:
            train_2nd_writer = open(write_base + '_train_2nd.code', 'w')
            train_4th_writer = open(write_base + '_train_4th.code', 'w')
            train_8th_writer = open(write_base + '_train_8th.code', 'w')
    
    print('Writing to %s' % write_base)
    with open(input_filename) as fp:
        count = 0
        for line in tqdm.tqdm(fp, total=full_count):
            if count > args.max:
                break
            # NOTE: We could also count results, and confirm chromosomes here
            # c = line[:line.find(':')]
            # res = line[line.rfind(';')+1:-1]
            # TODO: Count alignment. 99% of the data should be aligned
            # (less than 10% disagreement except at pos #100/#101)
            # Higher mis-alignment is probably a data processing error, and should be reported
            # NOTE: We can also report, and count SNPs vs Indels, etc.
            items = line.split(';')
            c = items[0]
            ref = decode(items[1])
            reads = decode(items[2])
            res = decode_label(items[-1].strip())
            output_tuple = (c, ref, reads, res)
            if count in holdout_lines:
                if args.savecode:
                    holdout_writer.write(line)
                holdout_array[holdout_count] = output_tuple
                holdout_count += 1
            elif count in non_holdout_lines:
                if count in non_holdout_train:
                    if args.savecode:
                        train_writer.write(line)
                    train_array[train_count] = output_tuple
                    train_count += 1
                    if args.subsample:
                        if count in non_holdout_train_2nd:
                            if args.savecode:
                                train_2nd_writer.write(line)
                            train_2nd_array[train_2nd_count] = output_tuple
                            train_2nd_count += 1
                        if count in non_holdout_train_4th:
                            if args.savecode:
                                train_4th_writer.write(line)
                            train_4th_array[train_4th_count] = output_tuple
                            train_4th_count += 1
                        if count in non_holdout_train_8th:
                            if args.savecode:
                                train_8th_writer.write(line)
                            train_8th_array[train_8th_count] = output_tuple
                            train_8th_count += 1
                elif args.split:
                    if count in non_holdout_val:
                        if args.savecode:
                            val_writer.write(line)
                        val_array[val_count] = output_tuple
                        val_count += 1
                    elif count in non_holdout_test:
                        if args.savecode:
                            test_writer.write(line)
                        test_array[test_count] = output_tuple
                        test_count += 1
            else:
                #Assume we have "unsaved" chromosomes
                # TODO: Check excluded chromosome list.
                c = line[:line.find(':')]
                #assert c in exclude_chromes, 'Error parsing |%s' % line
                if not(c in exclude_chromes):
                    print('Bad count -- is it skipped?')
                    print(count)
                    print(line)
                #assert False, 'Unknown write location for line (%d) %s' % (count, line)
            count += 1
        print('wrote %d lines to several files in directory %s' % (count, write_base))

    #close .code files
    if args.savecode:
        train_writer.close()
        if args.holdout is not None:
            holdout_writer.close()
        if args.split:
            val_writer.close()
            test_writer.close()
        if args.subsample:
            train_2nd_writer.close()
            train_4th_writer.close()
            train_8th_writer.close()
    
    #save numpy arrays
    np.save(write_base + '_train.npy', train_array)
    if args.holdout is not None:
        np.save(write_base + '_holdout.npy', holdout_array)
    if args.split:
        np.save(write_base + '_val.npy', val_array)
        np.save(write_base + '_test.npy', test_array)
    if args.subsample:
        np.save(write_base + '_train_2nd.npy', train_2nd_array)
        np.save(write_base + '_train_4th.npy', train_4th_array)
        np.save(write_base + '_train_8th.npy', train_8th_array)
    
    # TODO: Asssert that line length matches expectations for all files above
    #print saved line numbers
    print('Saved ' + str(train_count) + ' lines to train')
    if args.holdout is not None:
        print('Saved ' + str(holdout_count) + ' lines to holdout')
    if args.split:
        print('Saved lines to val/test %s' % str([val_count, test_count]))
    if args.subsample:
        print('Saved lines to 2nd/4th/8th %s' % str([train_2nd_count, train_4th_count, train_8th_count]))

if __name__ == '__main__':
    main()

