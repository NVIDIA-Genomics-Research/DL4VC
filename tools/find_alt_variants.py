# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Hacky attempt to match DVariant. Look for pairs of regions -- where Truth & Alt match,
but with different variants.

Idea is to set these "FP" variants as true.

Basic idea: find region where  XX[affected region]XX matches
-- keep small window for consideration
-- not 100% true, but just look for small match before & after
-- need to match zygosity, within reason

Idea is to quickly exit any match that is not possible...

"""

from __future__ import print_function
from __future__ import division

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

import pysam




# Just parse, remove header
def parse_vcf_recs(fn):
    bcf_in = pysam.VariantFile(fn, 'r')
    contigs = list(bcf_in.header.contigs)
    locations = []
    print('Found XX recs in file %s' % (fn))
    for rec in bcf_in.fetch():
        #print(rec)
        name = str(rec.contig) + ":" + str(rec.pos)
        vcf_string = str(rec).strip()
        vcf_fields = vcf_string.split('\t')
        #if name in full_info_table:
        #    vcf_string += '\t%s' % full_info_table[name]
        locations.append(vcf_fields)
    return locations

# Map location -> variants
CHR_COL, LOC_COL, _, REF_BASE_COL, ALT_BASE_COL = list(range(5))
def make_map(recs):
    rec_map = {}
    for r in recs:
        l = int(r[LOC_COL])
        r, a = r[REF_BASE_COL], r[ALT_BASE_COL]
        # TODO -- collect AF as well
        if l in rec_map.keys():
            rec_map[l].append((r,a))
        else:
            rec_map[l] = []
            rec_map[l].append((r,a))
    print('created %d unique keys' % len(rec_map))
    return rec_map

# For location, look for truth <-> alt mapping
PREPAD = 4
MAXPAD = 20
MAX_DELETE = MAXPAD
def solve_region(l, fasta, alt_map, truth_map):
    print('\n------\nsolving region %s' % l)

    # Build all possible paths in alt_map and truth_map
    # All sequence locations that could be in play
    alt_locs = []
    tru_locs = []
    for i in range(l-PREPAD,l+MAXPAD,1):
        if i in alt_map.keys():
            alt_locs.append(i)
        if i in truth_map.keys():
            tru_locs.append(i)

    print('ALT:')
    print(alt_locs)
    alt_array = [(k, alt_map[k]) for k in alt_locs]
    print(alt_array)
    print('TRUTH:')
    print(tru_locs)
    tru_array = [(k, truth_map[k]) for k in tru_locs]
    print(tru_array)


    # HACK -- ignore multiple alt at position X -- for now [sort by AF]
    min_p = min(alt_array[0][0], tru_array[0][0])
    max_p = max(alt_array[-1][0], tru_array[-1][0]) + MAX_DELETE + 1
    print('range from %d, %d' % (min_p, max_p))

    #ref_seq = fasta[min_p-1:max_p]
    #alt_seq = fasta[min_p-1:max_p]
    #tru_seq = fasta[min_p-1:max_p]

    triple_seq = [[p, fasta[p-1], fasta[p-1], fasta[p-1]] for p in range(min_p, max_p)]
    print(triple_seq)

    POS_ROW, REF_ROW, ALT_ROW, TRU_ROW = list(range(4))

    # Walk through, build sequences, see if we get a match!
    for i in range(min_p, max_p, 1):
        if i in alt_map.keys():
            print(i, alt_map[i])
            r, a = alt_map[i][0]
            assert triple_seq[i - min_p][POS_ROW] == i
            assert triple_seq[i - min_p][REF_ROW] == r[0]
            apply_variant(triple_seq, i=i-min_p, row=ALT_ROW, ref=r, alt=a)

        if i in truth_map.keys():
            print(i, truth_map[i])
            r, a = truth_map[i][0]
            assert triple_seq[i - min_p][POS_ROW] == i
            assert triple_seq[i - min_p][REF_ROW] == r[0]
            apply_variant(triple_seq, i=i-min_p, row=TRU_ROW, ref=r, alt=a)

        # TODO: Check if ALT, TRU match! (but also disagree with the reference)
        ref_seq = ''.join([triple_seq[j][REF_ROW] for j in range(0,i-min_p+1)])
        alt_seq = ''.join([triple_seq[j][ALT_ROW] for j in range(0,i-min_p+1)])
        tru_seq = ''.join([triple_seq[j][TRU_ROW] for j in range(0,i-min_p+1)])

        # NOTE: Need to make sure that next base is not '' -- not finished matching yet!
        if alt_seq == tru_seq and tru_seq != ref_seq and triple_seq[i-min_p+1][ALT_ROW] != '' and triple_seq[i-min_p+1][TRU_ROW] != '':
            print('FOUND MATCH')
            print('(%d - %d) %s -> %s' % (min_p, i, ref_seq, tru_seq))
            break

    print(triple_seq)

    #assert False

# Edit table with the ref -> alt variant
def apply_variant(triple_seq, i, row, ref, alt):
    print('%s -> %s' % (ref, alt))
    print(triple_seq)
    if len(ref) == len(alt) == 1:
        print('SNP')
        assert triple_seq[i][row] == ref
        triple_seq[i][row] = alt
    elif len(ref) > len(alt) and len(alt) == 1:
        print('DEL')
        assert triple_seq[i][row] == alt
        #print([triple_seq[j][row] for j in range(i+len(ref))])
        del_string = ''.join([triple_seq[j][row] for j in range(i,i+len(ref))])
        print(del_string)
        assert del_string == ref
        for j in range(len(ref)-1):
            triple_seq[i+j+1][row] = ''
    elif len(ref) < len(alt) and len(ref) == 1:
        print('INSERT')
        assert triple_seq[i][row] == ref
        triple_seq[i][row] = alt
    # Success
    print(triple_seq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_truth', type=str, help='VCF of truth region')
    parser.add_argument('--input_alt', type=str, help='VCF of alternatives')
    parser.add_argument('--fasta_file', type=str, help='FASTA for the reference')
    args = parser.parse_args()
    print(args)

    # Naive idea is (somewhat) simple...
    # For location in Truth file -- try to build a set of alts that match exactly (over a boundary)
    # Do this by chromosome (obviously)

    # HACK -- for now assume one chromosome [small one so load seq into memory]
    chrom = '10'
    fasta = pysam.Fastafile(args.fasta_file)
    ref_seq = fasta.fetch(chrom)

    # Assume recordsa are sorted
    # TODO: Ensure they are!
    alt_recs = parse_vcf_recs(args.input_alt)
    print('got %d alt_recs' % len(alt_recs))
    #assert False
    truth_recs = parse_vcf_recs(args.input_truth)
    print('got %d truth_recs' % len(truth_recs))

    # Create an easy lookup map of
    alt_map = make_map(alt_recs)
    truth_map = make_map(truth_recs)

    # TODO: Proposal regions?
    regions = [94517577, 53963856, 59761340, 85781668, 112757286, 128414037, 128571204, 131914340, 53716911, 85121375, 120414768]
    for l in regions:
        solve_region(l=l, fasta=ref_seq, alt_map=alt_map, truth_map=truth_map)



if __name__ == '__main__':
    main()
