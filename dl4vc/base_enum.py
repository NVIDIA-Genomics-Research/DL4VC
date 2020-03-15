# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

# Save bases, padding and start/end tokens as int8 enum
# Also adding all these extra codes we don't see -- just in case -- cover your asss
# http://www.boekhoff.info/dna-fasta-codes/
# NOTE: All unknown & semi-unknown mapped to '?' -- since we don't really deal with them
base_enum = {'A':1,'a':1,'T':2,'t':2, 'U':2, 'u':2, 'G':3,'g':3,'C':4,'c':4,
            '':5, '-':5,'*':5, 'N':5, 'n':5, 'X':5, 'x':5, '.':5,
            's':6,'start':6,'e':7,'end':7, 'noinsert':8, 'pad':0,
            'unk':9, '?':9, 'M':9, 'm':9, 'K':9, 'k':9, 'R':9, 'r':9, 'Y':9, 'y':9,
            'S':9, 's':9, 'W':9, 'w':9, 'B':9, 'b':9, 'V':9, 'v':9, 'H':9, 'h':9, 'D':9, 'd':9}
real_bases_set = set([base_enum['a'], base_enum['t'], base_enum['g'], base_enum['c']])
enum_base = {0:'p', 1:'A', 2:'T', 3:'G', 4:'C', 5:'-', 6:'s', 7:'e', 8:'noinsert', 9:'?'}
# Encode strand direction -- aatt (lower 1) AATT (upper 2)
STRAND_PAD = 0
STRAND_LOWER = 1
STRAND_UPPER = 2
strand_enum =  {'A':STRAND_UPPER,'a':STRAND_LOWER,'T':STRAND_UPPER,'t':STRAND_LOWER,
                'G':STRAND_UPPER,'g':STRAND_LOWER,'C':STRAND_UPPER,'c':STRAND_LOWER,
                '':STRAND_PAD, '-':STRAND_PAD,'*':STRAND_PAD, 'N':STRAND_UPPER, 'n':STRAND_LOWER,
                's':STRAND_PAD,'start':STRAND_PAD,'e':STRAND_PAD,'end':STRAND_PAD,
                'noinsert':STRAND_PAD, 'pad':STRAND_PAD, 'M':STRAND_UPPER, 'm':STRAND_LOWER,
                'unk':STRAND_PAD, '?':STRAND_PAD}

real_bases_set = set(['A', 'a', 'T', 't', 'C', 'c', 'G', 'G'])
real_base_keys_set = set([base_enum[k] for k in ['A','T','C','G','-','M','noinsert']])
mutation_type_enum = {'SNP':1, 'Insert':2, 'Delete':3, 'ins': 2, 'del':3, 'unk':0, 'error':0}
