#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Simple converter from BED file of trust regions, to Python set of valid locations.

Inputs look like this:

1   238901  238915
1   238918  238939

We want to be able to get a quick yes/no for {ChromX;location} -- are we in the trust region?

TODO: Implement more efficiently

@author: nyakovenko
"""

from __future__ import print_function
import argparse
import numpy as np
import tqdm
import pickle
import re
from bisect import bisect_left
from bisect import bisect_right


# Test if given location is in a region?
def is_in_region(chrom, loc, start_locations, end_locations, debug=False):
    if not chrom in start_locations:
        return False
    start_list = start_locations[chrom]
    end_list = end_locations[chrom]
    if len(start_list) == 0:
        return False
    closest_start_pos = bisect_right(start_list, loc)
    closest_start_pos = max(closest_start_pos - 1,0)
    if debug:
        print((chrom, loc))
        print(closest_start_pos)
        print(start_list[closest_start_pos])
        print(end_list[closest_start_pos])
        print('next')
        print(start_list[closest_start_pos+1])
        print(end_list[closest_start_pos+1])
    if loc >= start_list[closest_start_pos] and loc <= end_list[closest_start_pos]:
        if debug:
            print('match in region')
        return True
    else:
        if debug:
            print('not in region')
        return False

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='BED regions file to lookup table')
    parser.add_argument('--input', type=str, metavar='N', help='input file ',
        default='HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel.bed')
    parser.add_argument('--output', type=str, default='HG001-trust-regions.pkl', metavar='N', help='output file ')
    parser.add_argument('--debug', action='store_true', help='print extra information')
    args = parser.parse_args()
    print(args)

    input_filename = args.input
    output_filename = args.output
    all_chroms = [str(n) for n in range(1,23)] + ['X']
    location_starts = {c:[] for c in all_chroms}
    location_ends = {c:[] for c in all_chroms}
    debug = args.debug
    MAX = 10000000
    with open(input_filename) as fp:
        count = 0
        for line in tqdm.tqdm(fp, total=MAX):
            count += 1
            if count > MAX:
                break
            chrom, start, end = line.split('\t')
            start = int(start.strip())
            end = int(end.strip())
            location_starts[chrom].append(start)
            location_ends[chrom].append(end)

            if count % 10000 == 0:
                print('-------')
                print(line)
                print('ranges: %d\ttotal_locations: %d' % (count, len(location_starts)))

    print('finished')
    print('ranges: %d\ttotal_locations: %d' % (count, len(location_starts)))
    print('Saving to %s' % output_filename)
    with open(output_filename, 'wb') as fp:
        pickle.dump((location_starts, location_ends), fp)




if __name__ == '__main__':
    main()
