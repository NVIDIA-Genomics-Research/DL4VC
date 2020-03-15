#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Take GATK VCF file, and build easy to use lookup table (dictionary):
* indexed by [genome:chr:ID]
* is there or not a mutation, according to GATK
* debug for mutation type (SNP, Indel)

Table should be small enough to load into memory on a big machine.

This makes is fast to:
* compare a model to GATK
* use GATK as a proxy for reporting on error types - how we do on SNP & Indel?
* possibly save more information, and allow dynamic or multiple threshold on GATK

GATK text file for single genome is ~1GB, so need this to be smaller, ideally.

NOTE: Numpy can keep data more compressed, but lacks dict in easy lookup table.
Would need to keep dict [id -> pos] + numpy [data] for best efficiency, if needed.
That is what Gensim/W2V uses.

@author: nyakovenko
"""

from __future__ import print_function
import argparse
import numpy as np
import tqdm
import pickle
import re


# Read GATK file, skip irrelevant lines, build dictionary and save it to pickle.
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GATK VCF to lookup table')
    parser.add_argument('--input', type=str, metavar='N', help='input file ',
        default='HG001.GATK.4.10k_sample.vcf')
    parser.add_argument('--genome-id', type=str, metavar='N', help='name of genome (for multiple genome indexing) ',
        default='HG001')
    parser.add_argument('--output', type=str, default='HG001.GATK.4.10k_sample.pkl', metavar='N', help='output file ')
    parser.add_argument('--debug', action='store_true', help='print extra information')

    args = parser.parse_args()
    print(args)

    gatk_dict = {}
    input_filename = args.input
    output_filename = args.output
    MAX = 10000000
    index = {}
    genome_id = args.genome_id
    debug = args.debug
    with open(input_filename) as fp:
        count = 0
        # Collect counts on SNPs, Indel, etc -- collect depth for all
        count_mut = []
        count_non_mut = []
        count_snp = []
        count_indel = []
        count_mut_snp = []
        count_mut_indel = []
        for line in tqdm.tqdm(fp, total=MAX):
            if count > MAX:
                break
            # skip commentary lines
            if len(line) < 2 or line[:2] == '##':
                continue
            items = line.split("\t")
            # read in GATK index
            if items[0][0] == '#':
                index = dict([(b,a) for a,b in enumerate(items)])
                print(index)
                # HACK -- save the index
                gatk_dict["index"] = ("mutation", "is_snp", "is_indel","depth","ref","alt","quality")
                continue

            # If here, we should have GATK outputs per position
            # {0: '#CHROM', 1: 'POS', 2: 'ID', 3: 'REF', 4: 'ALT', 5: 'QUAL', 6: 'FILTER', 7: 'INFO', 8: 'FORMAT', 9: 'HG001\n'}
            # 1 6149286 .   G   C   5698.77 .   AC=2;AF=1.00;AN=2;DP=135;ExcessHet=3.0103;FS=0.000;MLEAC=2;MLEAF=1.00;MQ=70.00;MQ0=0;QD=35.01;SOR=0.708 GT:AD:DP:GQ:MMQ:PL  1/1:0,135:0:99:0,0:5727,406,0
            # Lots of noise. What we want is position, quality, coverage, and what base to what [mutation type]
            chrom = items[index['#CHROM']]
            pos = items[index['POS']]
            ref = items[index['REF']]
            alt = items[index['ALT']]
            q = float(items[index['QUAL']])
            info = items[index['INFO']]
            info_items = info.split(';')
            info_index = {idx:val for idx,val in [(t[:t.find('=')],t[t.find('=')+1:]) for t in info_items]}
            if 'DP' in info_index.keys():
                depth = int(info_index['DP'])
            else:
                depth = 0
            # Implement basic heuristic from @zhen
            # Quality / DP [depth] &gt; 100 is the cutoff
            mutation = (q / (depth + 0.001)) > 0.
            # SNP == letter for letter
            is_snp = len(ref) == 1 and len(alt) == 1
            # Indel == letter length changes
            is_indel = (len(ref) != len(alt))
            idx = '%s:chr%s:%s' % (genome_id, chrom, pos)
            gatk_dict[idx] = (mutation, is_snp, is_indel,depth,ref,alt,q)

            # Update counts -- with depth for each
            if debug:
                if mutation:
                    count_mut.append(depth)
                else:
                    count_non_mut.append(depth)
                if is_snp:
                    count_snp.append(depth)
                if is_indel:
                    count_indel.append(depth)
                if mutation and is_snp:
                    count_mut_snp.append(depth)
                if mutation and is_indel:
                    count_mut_indel.append(depth)

            count += 1

    print('Collected %d records from GATK' % len(gatk_dict))
    if debug:
        names = ['count_mut', 'count_non_mut', 'count_snp', 'count_indel', 'count_mut_snp', 'count_mut_indel']
        for i,record in enumerate([count_mut, count_non_mut, count_snp, count_indel, count_mut_snp, count_mut_indel]):
            name = names[i]
            count = len(record)
            depth = np.mean(record)
            std = np.std(record)
            print('%s:\t%d(%.2f%%)\tdepth: %.2f (%.2f)' % (name, count, count/len(gatk_dict)*100., depth,std))
    print('Saving to %s' % output_filename)
    with open(output_filename, 'wb') as fp:
        pickle.dump(gatk_dict, fp)





if __name__ == '__main__':
    main()
