# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Expand h5py dataset with appending another H5Py dataset.

NOTE: This has to be done block at a time
NOTE: *copy* the destination dataset first, just in case
"""

import argparse
from argparse import RawTextHelpFormatter
from collections import defaultdict
import functools
import os
import tempfile
import tqdm
import time
import logging
import subprocess
import h5py
import numpy as np

# - Open both datasets
# - Expand base dataset for combined size
# - Loop by step size, make copy, until done
def expand_merge_dataset(args):
    with h5py.File(args.input_file, 'a') as hf:
        print('opening file for append %s' % args.input_file)
        df = hf[args.dataset_name]
        df_len = df.size
        with h5py.File(args.merge_file, 'r') as hf_extra:
            print('opening file to be merged %s' % args.merge_file)
            df_extra = hf_extra[args.dataset_name]
            df_extra_len = df_extra.size
            df.resize((df_len+df_extra_len,))
            print('expanding size')
            print((df_len, df_extra_len, df.size))
            step = args.step
            t = time.time()
            for s in tqdm.tqdm(range(0,df_extra_len,step), total=int(df_extra_len/step + 1)):
                start = s
                end = s+step
                print((start, end))
                df[start+df_len:end+df_len] = df_extra[start:end]
                print('%.2fs' % (time.time()-t), flush=True)
    print('DONE')

def main():
    print("Start program")
    parser = argparse.ArgumentParser(description='Context module')
    parser.add_argument('--input_file', type=str, default="", help='input h5py file')
    parser.add_argument('--merge_file', type=str, default="", help='second file to attach')
    parser.add_argument('--dataset_name', type=str, default="data", help='name of file to merge (data)')
    parser.add_argument('--step', type=int, default=100000, help='size of piece to transfer')
    parser.add_argument('--debug', action='store_true', default=False, help='debug while conversion?')

    args = parser.parse_args()

    print(args)

    expand_merge_dataset(args)

if __name__ == '__main__':
    main()
