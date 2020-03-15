# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Clean up redundant split variants, like
# ['10' '133821805' 'TTA' 'TTATA']
# ['10' '134094682' 'GACACACAC' 'GACACAC']
# ['10' '134989089' 'AGATG' 'ATGGATG']
# ['10' '135203914' 'CCA' 'CCACA']
# ['10' '135273975' 'CATAT' 'CATATAT']
"""

import numpy as np
import argparse

# Canonicalize items like
def canonicalize_bases(ref, var):
    d = len(ref) - len(var)
    m = min(len(ref), len(var))
    trim = m - 1
    if trim == 0:
         return(ref, var)
    """
    if m > 1 and d > 0:
        print('canonical delete')
        print(ref, var)
    elif m > 1 and d < 0:
        print('canonical insert')
        print(ref, var)
    print(ref[-trim:], var[-trim:])
    print(ref[:-trim], var[:-trim])
    """
    assert(ref[-trim:] == var[-trim:])
    return(ref[:-trim], var[:-trim])

parser = argparse.ArgumentParser(description='Calculate thresholds for Conv1D model output')
parser.add_argument('--input', type=str, default="", help='input vcf file')
parser.add_argument('--output', type=str, default="", help='output vcf file')
args = parser.parse_args()
print(args)

#truths = np.genfromtxt(truth_file, comments='#', dtype='str', delimiter='\t', usecols=(0,1,3,4))

with open(args.input,'r') as fin:
    with open(args.output, 'w') as fout:
        for line in fin:
            if line[0] == '#':
                fout.write(line)
            else:
                items = line.split('\t')
                if len(items[3]) > 1 and len(items[4]) > 1:
                    can_ref, can_var = canonicalize_bases(items[3], items[4])
                    items[3] = can_ref
                    items[4] = can_var
                    l = '\t'.join(items)
                    #print(line, l)
                    fout.write(l)
                else:
                    fout.write(line)

