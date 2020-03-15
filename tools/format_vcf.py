# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Simple but comprehensive toolset for transforming VCF output
from raw scores (Conv1D or other GPU inferencing) to fields
usable by VCFEval (and other upstream variant tools).
-- threshold by model score
-- threshold separately for SNP and Indel
-- threshold called variants by hetero/homozygous -- output 0/1 or 1/1 in the VCF
-- output a model confidence score, for ROC in VCFEval
-- TODO: If 2+ variants per location (above threhold), choose top 2 [multi-allele]
-- TODO: If single-allele and multi-allele predictions, threshold between those

This is important, because VCFEval needs 0/1 type preditions.

Probably a good idea to pre-filter the inputs -- since data will be processed
in series, and stored in memory.

Example command:
python format_vcf.py --input_file f4xnp6dc_epoch3_output_HG02_ALL-multiAF-DVAR.vcf \
  --output_file epoch3_output_multiAF_thresh_test.vcf \
  --threshold 0.5 --indel_threshold 0.1 --zygo_threshold 0.4

@author: nyakovenko
"""

import argparse
from argparse import RawTextHelpFormatter
from collections import defaultdict
import functools
import multiprocessing
from multiprocessing import Pool
import os
import tempfile
import tqdm
import logging
import subprocess
import time


# Inline -- for now
NONCE = 0.0000001
# We don't really need score bucket -- can use 50 but VCFEval will take longer
SCORE_BUCKETS = 50

# Easier to parse
def print_lines(lines):
    for l in lines:
        print(l)

def filter_format_vcf(args):
    input_fname = args.input_file
    output_fname = args.output_file
    # Unnecessary way to get length for TQDM -- can't do it any faster
    num_lines = sum(1 for line in open(input_fname, 'r'))
    print('reading %d lines from %s' % (num_lines, input_fname))
    print('writing to %s' % output_fname)

    #set thresholds
    snp_threshold = args.snp_threshold
    snp_hz_threshold = args.snp_zygo_threshold
    if args.indel_threshold > 0.:
        indel_threshold = args.indel_threshold
        indel_hz_threshold = args.indel_zygo_threshold
        if args.long_indel_threshold > 0.:
            long_indel_threshold = args.long_indel_threshold
            long_indel_hz_threshold = args. long_indel_zygo_threshold
        else:
            long_indel_threshold = indel_threshold
            long_indel_hz_threshold = indel_hz_threshold
        if args.delete_threshold > 0.:
            delete_threshold = args.delete_threshold
            delete_hz_threshold = args.delete_zygo_threshold
        else:
            delete_threshold = indel_threshold
            delete_hz_threshold = indel_hz_threshold
    else:
        indel_threshold = snp_threshold
        indel_hz_threshold = snp_hz_threshold
        long_indel_threshold = indel_threshold
        long_indel_hz_threshold = indel_hz_threshold

    debug = args.debug

    #initialize variables to store results
    curr_line = 0
    curr_pos, curr_chrom = None, None
    curr_pos_lines = list()
    curr_pos_threshold_scores = list()
    curr_pos_gts = list()

    with open(input_fname, 'r') as fin:
        with open(output_fname, 'w') as fout:
            for line in tqdm.tqdm(fin, total=num_lines):
                curr_line += 1
                if line[0] == '#':
                    # copy header
                    # TODO: Extend CVS columns?
                    fout.write(line)
                else:
                    items = line.strip('\n').split('\t')
                    if debug:
                        print(line)
                        print(items)
                    # should have 10-11 items (11th is appended "GT:1/1")
                    assert len(items) == 10 or len(items) == 11, 'Line should have 10-11 items (11th is appended "GT:1/1")\n%s' % line
                    scores = items[2].split(';')
                    scores = {a:float(b) for a,b in [s.split('=') for s in scores]}
                    if debug:
                        print(scores)
                    # Threshold on VT (variant type) score
                    # TODO: Choose based on binary score? Command line option
                    threshold_score = 1.0 - scores['NV']
                    ref_bases = items[3]
                    var_bases = items[4]
                    is_snp = (len(ref_bases) == 1 and len(var_bases) == 1)
                    is_indel = ~is_snp
                    is_long_indel = (len(ref_bases) >= 3 or len(var_bases) >= 3)
                    is_delete = (len(ref_bases) > 1 and len(var_bases) == 1) and ~is_long_indel
                    threshold = snp_threshold if is_snp else (long_indel_threshold if is_long_indel else (delete_threshold if is_delete else indel_threshold))
                    threshold_score_margin = threshold_score - threshold
                    if threshold_score_margin >= 0.:
                        # format this line
                        # default = heterozygous
                        gt_string = '0/1'
                        hz_score = scores['OV']
                        # normalize -- TODO: Option?
                        #hz_score = hz_score / (1.0 - scores['NV'] + NONCE)
                        hz_score = hz_score
                        hz_threshold = snp_hz_threshold if is_snp else (long_indel_hz_threshold if is_long_indel else (delete_hz_threshold if is_delete else indel_hz_threshold))
                        if hz_score >= hz_threshold:
                            gt_string = '1/1'
                        # Create a sort of quality score?
                        # Arbitrary, so scale to min threshold(?)
                        # NOTE: Binarize -- else too many endpoints for VCFEval
                        q_score = threshold_score_margin / (1.0 - threshold)
                        q_score = int(q_score * SCORE_BUCKETS)
                        new_items = items[0:9] + ['%s:%s' % (gt_string, q_score)]
                        new_line = '\t'.join(new_items)
                        if debug:
                            print(new_line)

                        #if this is the first variant line: store results and move on to the next line
                        if curr_pos is None:
                            curr_chrom = items[0]
                            curr_pos = items[1]
                            curr_pos_lines = [new_line]
                            curr_pos_threshold_scores = [threshold_score]
                            curr_pos_gts = [gt_string]
                        #if this position is the same as curr_pos - append results to previous ones and move on to the next line
                        elif curr_chrom == items[0] and curr_pos == items[1]:
                            if curr_pos == '182303657':
                                print('appending')
                                print(new_line)

                            curr_pos_lines.append(new_line)
                            curr_pos_threshold_scores.append(threshold_score)
                            curr_pos_gts.append(gt_string)
                        #if this is a new position - write the results for curr_pos to output, then overwrite with new results
                        else:
                            #if there is a homozygous variant - discard all others
                            if '1/1' in curr_pos_gts:
                                # TODO: Sort by OV score
                                best_index = curr_pos_gts.index('1/1')
                                if len(curr_pos_lines) > 1:
                                    print('----------------\nChoosing *single* homozygous var for possible multi-allele')
                                    print(curr_pos_lines[best_index])
                                    print('discarding %d other lines:' % (len(curr_pos_lines)-1))
                                    print_lines([curr_pos_lines[i] for i in (set(range(len(curr_pos_lines))) - set([best_index]))])
                                    # Look at *second* position. Do not over-write if second best result is still very strong
                                    #top2 = [curr_pos_threshold_scores.index(i) for i in sorted(curr_pos_threshold_scores, reverse=True)[:2]]
                                    sorted_pair = sorted(list(zip(curr_pos_threshold_scores, curr_pos_lines)), reverse=True)
                                    #print(sorted_pair)
                                    top2 = [curr_pos_lines.index(j) for (i,j) in sorted_pair][:2]
                                    assert top2[0] != top2[1]
                                    if curr_pos_threshold_scores[top2[1]] >= args.multiallele_homozygous_second_threshold:
                                        print('A*Skipping* single homozygous because second result too good %.5f' % curr_pos_threshold_scores[top2[1]])
                                    elif curr_pos_threshold_scores[top2[0]] >= args.multiallele_homozygous_second_threshold and curr_pos_gts[top2[0]] != '1/1':
                                        print('B*Skipping* single homozygous because second result too good %.5f' % curr_pos_threshold_scores[top2[0]])
                                    else:
                                        curr_pos_lines = [curr_pos_lines[best_index]]
                            #if curr_pos has >2 heterozygous variants, store 2 with highest threshold score
                            if len(curr_pos_lines)>2:
                                #top2 = [curr_pos_threshold_scores.index(i) for i in sorted(curr_pos_threshold_scores, reverse=True)[:2]]
                                sorted_pair = sorted(list(zip(curr_pos_threshold_scores, curr_pos_lines)), reverse=True)
                                #print(sorted_pair)
                                top2 = [curr_pos_lines.index(j) for (i,j) in sorted_pair][:2]
                                assert top2[0] != top2[1]
                                if len(curr_pos_lines) > 2:
                                    print('----------------\nChoosing *two* hetero var for possible multi-allele')
                                    print_lines([curr_pos_lines[i] for i in top2])
                                    # Discard second allele, if below threshold
                                    if curr_pos_threshold_scores[top2[1]] <= args.multiallele_second_threshold:
                                        print('C*Skipping* second allele because its not good enough.')
                                        print_lines([curr_pos_lines[i] for i in top2[1:]])
                                        top2 = top2[:1]
                                        #time.sleep(2)
                                    print('discarding %d other lines:' % (len(curr_pos_lines)-2))
                                    print_lines([curr_pos_lines[i] for i in (set(range(len(curr_pos_lines))) - set(top2))])
                                curr_pos_lines = [curr_pos_lines[i] for i in top2]
                            #write to file
                            for output_line in curr_pos_lines:
                                fout.write(output_line+'\n')
                            #overwrite with new position
                            curr_chrom = items[0]
                            curr_pos = items[1]
                            curr_pos_lines = [new_line]
                            curr_pos_threshold_scores = [threshold_score]
                            curr_pos_gts = [gt_string]
                        #if this is the last line in the file: write results
                        if curr_line==num_lines:
                            #if there is a homozygous variant - discard all others
                            if '1/1' in curr_pos_gts:
                                curr_pos_lines = [curr_pos_lines[curr_pos_gts.index('1/1')]]
                            #if curr_pos has >2 heterozygous variants, store 2 with highest threshold score
                            if(len(curr_pos_lines)>2):
                                top2 = [curr_pos_threshold_scores.index(i) for i in sorted(curr_pos_threshold_scores)[-2:]]
                                curr_pos_lines = [curr_pos_lines[i] for i in top2]
                            #write to file
                            for output_line in curr_pos_lines:
                                fout.write(output_line+'\n')

def main():
    # Training settings
    print("Start program")
    parser = argparse.ArgumentParser(description='Context module')
    parser.add_argument('--input_file', type=str, default="", help='input vcf file')
    parser.add_argument('--output_file', type=str, default="", help='output vcf file')
    parser.add_argument('--snp_threshold', type=float, default=0.3, help='min variant call score')
    parser.add_argument('--indel_threshold', type=float, default=0., help='(optional) threshold for indel only')
    parser.add_argument('--long_indel_threshold', type=float, default=0., help='(optional) threshold for indel only')
    parser.add_argument('--delete_threshold', type=float, default=0., help='')
    parser.add_argument('--snp_zygo_threshold', type=float, default=0.5, help='min homozygous score for 1/1 predict')
    parser.add_argument('--indel_zygo_threshold', type=float, default=0.5, help='(optional) min homozygous score for 1/1 predict for indel only')
    parser.add_argument('--long_indel_zygo_threshold', type=float, default=0.5, help='(optional) min homozygous score for 1/1 predict for indel only')
    parser.add_argument('--delete_zygo_threshold', type=float, default=0.5, help='')
    parser.add_argument('--multiallele_second_threshold', type=float, default=0.7, help='Dont add *second* allele if low probability variant (can be way above basic threshold')
    parser.add_argument('--multiallele_homozygous_second_threshold', type=float, default=0.9, help='dont over-write multi-allele if first homozygous but second var very strong alsl')
    parser.add_argument('--debug', action='store_true', default=False, help='debug while conversion?')
    # NOTE: By default, all non-homozygous predictions (that still meet variant call threshold) treated as 0/1

    args = parser.parse_args()

    print(args)

    # Perform all operations inline (while possible)
    filter_format_vcf(args)



if __name__ == '__main__':
    main()
