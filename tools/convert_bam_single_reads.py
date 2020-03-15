#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Read a BAM file, convert pileups to numpy object that
can be easily digested for single reads training.

* Encode sparsity and base pairs as an enum so can save in int8
* Keep dimensions fixed (and padded) for now. [downsample if needed]
* Keep pileup order, even when downsampling
* TODO: Look into format for variable size [pad dynamically at training level]

@author: nyakovenko
"""

from __future__ import print_function
import argparse
import os
import numpy as np
import tqdm
import pysam
import re
import time
import math
import pickle
import multiprocessing
import functools
import h5py

# Global debug flag
debug = False

# Save bases, padding and start/end tokens as int8 enum
# Also adding all these extra codes we don't see -- just in case -- cover your asss
# http://www.boekhoff.info/dna-fasta-codes/
# NOTE: All unknown & semi-unknown mapped to '?' -- since we don't really deal with them
base_enum = {'A':1,'a':1,'T':2,'t':2, 'U':2, 'u':2, 'G':3,'g':3,'C':4,'c':4,
            '':5, '-':5,'*':5, 'N':5, 'n':5, 'X':5, 'x':5, '.':5, ',':5,
            's':6,'start':6,'e':7,'end':7, 'noinsert':8, 'pad':0,
            'unk':9, '?':9, 'M':9, 'm':9, 'K':9, 'k':9, 'R':9, 'r':9, 'Y':9, 'y':9,
            'S':9, 's':9, 'W':9, 'w':9, 'B':9, 'b':9, 'V':9, 'v':9, 'H':9, 'h':9, 'D':9, 'd':9}
real_bases_set = set([base_enum['a'], base_enum['t'], base_enum['g'], base_enum['c']])
enum_base = {0:'p', 1:'A', 2:'T', 3:'G', 4:'C', 5:'-', 6:'s', 7:'e', 8:'noinsert', 9:'?'}
# Encode strand direction -- aatt (lower 1) AATT (upper 2)
STRAND_PAD = 200 # NOTE: unit8 so need to set to large value >> 0,1,2 but not negative
STRAND_LOWER = 1
STRAND_UPPER = 2
strand_enum =  {'A':STRAND_UPPER,'a':STRAND_LOWER,'T':STRAND_UPPER,'t':STRAND_LOWER,
                'G':STRAND_UPPER,'g':STRAND_LOWER,'C':STRAND_UPPER,'c':STRAND_LOWER,
                '':STRAND_PAD, '-':STRAND_PAD,'*':STRAND_PAD, 'N':STRAND_UPPER, 'n':STRAND_LOWER,
                's':STRAND_PAD,'start':STRAND_PAD,'e':STRAND_PAD,'end':STRAND_PAD,
                'noinsert':STRAND_PAD, 'pad':STRAND_PAD, 'M':STRAND_UPPER, 'm':STRAND_LOWER,
                'unk':STRAND_PAD, '?':STRAND_PAD}

# Shortcut, for all chromosomes we may want to encode for training [1-22 + X]
# TODO: Add other sets, with holdouts, etc
ALL_VARIANT_CHROMS_SET = [str(c) for c in range(1,22+1)] + ['X']

# (start[quality])(base)(indel)(end)
# NOTE: Could do regexp, but this is more efficient
# https://pysam.readthedocs.io/en/latest/api.html
"""
Information on match, mismatch, indel, strand, mapping
quality and start and end of a read are all encoded at the
read base column. At this column, a dot stands for a match
to the reference base on the forward strand, a comma for a
match on the reverse strand, a '>' or '<' for a reference
skip, `ACGTN' for a mismatch on the forward strand and
`acgtn' for a mismatch on the reverse strand. A pattern
`\+[0-9]+[ACGTNacgtn]+' indicates there is an insertion
between this reference position and the next reference
position. The length of the insertion is given by the
integer in the pattern, followed by the inserted
sequence. Similarly, a pattern `-[0-9]+[ACGTNacgtn]+'
represents a deletion from the reference. The deleted bases
will be presented as `*' in the following lines. Also at
the read base column, a symbol `^' marks the start of a
read. The ASCII of the character following `^' minus 33
gives the mapping quality. A symbol `$' marks the end of a
read segment
"""
def decode_base_detail(base_str):
    """
    Decodes the base string and detects base type,
    begining, ending, insertiong and deletions.

    base_str - query string as read from BAM file

    Returns a decoded tuple in the format (start, base, deletion, end, insert)

    TODO: Encode if base is A or a [capitalization encoded read order],
    since errors are non-random w/r/t base reading orders...
    NOTE: Add as 5th dimension?
    """
    #print(base_str)
    start = 0
    end = 0
    base = 0
    insert = []
    strand = STRAND_PAD
    deletion = 0
    # handle & remove start token
    if base_str[0] == '^':
        # Read quality
        if len(base_str) > 1:
            start = ord(base_str[1]) - 10
        else:
            # Weird case of '^' [no letters]
            start = 1
            return (start, base, deletion, end, strand, insert)
        if len(base_str) > 2:
            # Again, start, with missing base. Just return base == 0
            base_str = base_str[2:]
        else:
            return (start, base, deletion, end, strand, insert)
    # handle & remove end token
    if base_str[-1] == '$':
        end = 1
        base_str = base_str[:-1]
    # handle base at the current position
    base = base_enum[base_str[0]]
    strand = strand_enum[base_str[0]]

    # Debug -- why did we get "strand pad" -- unknown stand?
    #if strand == STRAND_PAD:
    #    print('strand padding!')
    #    print(base_str)

    # If additional info, handle indels
    if len(base_str)>1:
        if base_str[1] == '+':
            # Fail == for insert len 10+
            #insert_len = int(base_str[2])
            #insert = [base_enum[b] for b in base_str[3:]]
            insert_len = int(re.match(r'(\d+)(\D+)',base_str[2:]).group(1))
            insert = re.match(r'(\d+)(\D+)',base_str[2:]).group(2)
            insert = [base_enum[b] for b in insert]
            assert (insert_len == len(insert))
        elif base_str[1] == '-':
            # Deletions are weird -- we get a warning, but really just convert bases to '-'' and move on?
            # Fail == for del len 10+
            deletion = int(re.match(r'(\d+)(\D+)',base_str[2:]).group(1))
        else:
            assert False, 'unparsable indel in |%s|' % base_str
    return (start, base, deletion, end, strand, insert)

# Return numpy array of IDs in locations data -- which match "table" set
# location format -- "1\t23344444"
def find_location_from_table(locations, table):
    good_loc = []
    for i,loc in enumerate(locations):
        tag = '\t'.join([str(k) for k in loc[:2]])
        if tag in table:
            #print(loc)
            good_loc.append(i)
    return np.array(good_loc)

def get_locations_from_vcf(vcf_fn, label, full_vcf=None):
    """
    Parse variant location from VCF file.

    vcf_fn   - VCF file name
    label    - Label associated with entries in VCF file

    Returns a list of locations as a tuple
    (contig string, position, name (=contig:position), ?, ?, string from VCF line)
    """
    # HACK: If provided a "full vcf" -- pre-extract information, and index by name -- chrom/location pair
    full_info_table = {}
    if full_vcf:
        bcf_full = pysam.VariantFile(full_vcf, 'r')
        contigs_full = list(bcf_full.header.contigs)
        for rec in bcf_full.fetch():
            name = str(rec.contig) + ":" + str(rec.pos)

            # Hack -- get the GT -- 1/1 0|1 type information -- about hetero/homozygous variant
            rec_string = (str(rec).strip()).split('\t')
            rec_keys = rec_string[-2].split(':')
            rec_vals = rec_string[-1].split(':')
            if rec_keys[0] == 'GT':
                #print([rec_keys[0], rec_vals[0]])
                full_info_table[name] = '%s:%s' % (rec_keys[0], rec_vals[0])

    bcf_in = pysam.VariantFile(vcf_fn, 'r')
    contigs = list(bcf_in.header.contigs)
    locations = []
    for rec in bcf_in.fetch():
        name = str(rec.contig) + ":" + str(rec.pos)
        vcf_string = str(rec).strip()
        if name in full_info_table:
            vcf_string += '\t%s' % full_info_table[name]
        locations.append((rec.contig, rec.pos, name, label, np.NaN, np.NaN, vcf_string))

    return locations

def decode_query_sequences(query_sequences):
    """
    Decode a series of queries read from BAM file

    query_sequences   - list of queries

    Returns the queries parsed into two separate lists -
    1. a list of bases along with start, end, deletion info
    2. a corresponding list with inserts
    """
    # Decode query sequences
    # (start, base, deletion, end, insert)
    try:
        query_seq_details = [decode_base_detail(b) for b in query_sequences]  # decode_base_detail
    except e as KeyError:
        print('Key error!')
        print(query_sequences)
        assert False

    # HACK: Remove rows if base == 0 -- not sure why these are inserted in the first place
    query_seq_details = [d for d in query_seq_details if d[1] != 0]

    # Easy to store data that's not insert (thus a sequence -- handle those differently)
    query_seq_bases = np.array([bp[:-1] for bp in query_seq_details])
    query_seq_inserts = [bp[-1] for bp in query_seq_details]

    return (query_seq_bases, query_seq_inserts)

def resize_alignment_image(min_row_size, min_col_size, current_image):
    """
    Dynamically resize the alignment image. If row/column are fewer, then it doubles
    that dimension

    min_row_size  - minimum number of rows
    min_col_size  - minimum number of columns
    current_image - current image

    Returns a new image with resized row and columns
    """
    # Adjust rows
    if min_row_size > current_image.shape[0]:
        if debug:
            print('Image matrix being expanded')
            print('Extra many reads (%d) -- expanding matrix' % (min_row_size))
        new_image = np.full((max(current_image.shape[0]*2, min_row_size), current_image.shape[1]),
                base_enum['pad'], dtype=np.uint8)
        new_image[0:current_image.shape[0],0:current_image.shape[1]] = current_image
        current_image = new_image
    # Adjust columns
    if min_col_size > current_image.shape[1]:
        if debug:
            print('Extra long cols -- inserts (%d) -- expanding matrix' % (min_col_size))
        new_image = np.full((current_image.shape[0], max(current_image.shape[1]*2, min_col_size)),
                base_enum['pad'], dtype=np.uint8)
        new_image[0:current_image.shape[0], 0:current_image.shape[1]] = current_image
        current_image = new_image
    return current_image

def add_bases_to_alignment_image(query_seq_bases, query_seq_inserts, query_seq_ids,
        alignment_image, col_offset, prev_col_offset, read_row_dict,
        MAX_INSERT_LENGTH=10, query_qualities=None, read_quality_image=None, strand_image=None, args={}):
    """
    Adds base information to the alignment image.

    query_seq_bases     - A list with base query information
    query_seq_inserts   - A list with insert information
    query_seq_ids       - A list with IDs for each base query
    alignment_image     - Image into which bases need to be added
    col_offset          - Current column in alignment image where bases are to be added
    prev_col_offset     - Previous column offset
    read_row_dict       - A dict mapping sequence ID to row number in alignment image

    Returns the new local maximum column offset in the image starting from the col_offset
    for the given row
    """
    local_max_col_offset = 0
    save_read_quality = args.save_q_scores
    save_strand = args.save_strand

    # Instead, write reads one at a time -- less efficient, but impossible to avoid with
    # reads that dont 100% match position to position (even with start/end tokens)
    for i in range(query_seq_bases.shape[0]):
        row_num = read_row_dict[query_seq_ids[i]]
        alignment_image[row_num, col_offset] = query_seq_bases[i, 1]
        if save_read_quality:
            read_quality_image[row_num, col_offset] = query_qualities[i]
        if save_strand:
            strand_image[row_num, col_offset] = query_seq_bases[i, 4]
    #    B. look-behind to write start tokens
    for i in range(query_seq_bases.shape[0]):
        if query_seq_bases[i][0]:
            row_num = read_row_dict[query_seq_ids[i]]
            alignment_image[row_num, prev_col_offset] = base_enum['start']
            if save_read_quality:
                read_quality_image[row_num, prev_col_offset] = query_qualities[i]
            if save_strand:
                strand_image[row_num, prev_col_offset] = query_seq_bases[i, 4]
    #    C. look-ahead for inserts (expand columns as needed)
    for i in range(query_seq_bases.shape[0]):
        # NOTE: Just skip, MAX_INSERT_LENGTH == 0 [no inserts]
        if len(query_seq_inserts[i]) > 0 and MAX_INSERT_LENGTH > 0:
            if debug:
                print('insert %s' % str(query_seq_inserts[i]))
            # Write the insert
            insert = np.array(query_seq_inserts[i])
            if len(query_seq_inserts[i]) == 1:
                insert = np.expand_dims(insert, axis=0)
            # Trim inserts to max length [best max length 10 -- since
            # len 10 inserts happen for biological reasons]
            if insert.shape[0] > MAX_INSERT_LENGTH:
                if debug:
                    print('WARNING: long insert -- trimming to %d' % MAX_INSERT_LENGTH)
                #time.sleep(5)
                insert = insert[:MAX_INSERT_LENGTH]
            row_num = read_row_dict[query_seq_ids[i]]
            alignment_image[row_num,col_offset+1:col_offset+1+insert.shape[0]] = insert
            if save_read_quality:
                read_quality_image[row_num,col_offset+1:col_offset+1+insert.shape[0]] = np.full(insert.shape, query_qualities[i], dtype=np.uint8)
            if save_strand:
                strand_image[row_num,col_offset+1:col_offset+1+insert.shape[0]] = np.full(insert.shape, query_seq_bases[i, 4], dtype=np.uint8)
            # increase column offset if needed
            if insert.shape[0] > local_max_col_offset:
                local_max_col_offset = insert.shape[0]
                if debug:
                    print('increased local offset to %d' % local_max_col_offset)

    # If we have inserts, represent all non-inserts (not as zero/pad)
    # NOTE: Missing reads -- [2, 5, 2, 2, 2, 0, 2, 2, 2] -- will still stay as padding.
    # (but can be clear at the individual read level, what is not an insert)
    if local_max_col_offset > 0:
        for i in range(query_seq_bases.shape[0]):
            row_num = read_row_dict[query_seq_ids[i]]
            insert_block = alignment_image[row_num,col_offset+1:col_offset+local_max_col_offset+1]
            insert_block[insert_block == base_enum['pad']] = base_enum['noinsert']
            if debug:
                print(insert_block)
            # TODO: For quality scores, insert something for 'noinsert' to make sure that we match Q-scores to reads??

    # For debug, display deletes (do nothing)
    for i in range(query_seq_bases.shape[0]):
        if query_seq_bases[i][2]:
            row_num = read_row_dict[query_seq_ids[i]]
            if debug:
                print(query_seq_bases[i])
                print("delete at row %d -- %s" % (row_num, query_seq_ids[i]))
                print(alignment_image[row_num,:])

    return local_max_col_offset

def handle_ended_sequences(query_seq_bases, query_seq_ids, alignment_image, read_row_dict,
        reads_offset, col_offset, local_max_col_offset,
        query_qualities=None, read_quality_image=None, strand_image=None, args={}):
    """
    Do some housekeeping to handle those reads which had a sequence ending
    as part of the string. They need to be represented with a special token in the
    alignment image. Also remove those reads from dict.

    query_seq_bases      - A list with base query information
    query_seq_ids        - A list of IDs for the base queries
    alignment_image      - Image into which bases are to be placed
    read_row_dict        - A map for base query IDs to rows in the alignment image
    reads_offset         - A counter maintaining how many completed rows have been seen
    col_offset           - Current column offset for adding bases
    local_max_col_offset - Maximum column offset from col_offset for current row base placement

    Returns a tuple with
    (
    read_row_dict without finished sequences,
    new reads_offset accounting for finished rows,
    a list of indices for finished rows
    )
    """
    #     A. Write end token to finished rows -- lookahead past inserts
    #     B. remove finished reads from row_map
    #     C. [debug/data check on request]
    finished_rows = []
    finished_row_ids = []
    save_read_quality = args.save_q_scores
    save_strand = args.save_strand
    for i in range(query_seq_bases.shape[0]):
        if query_seq_bases[i][3]:
            finished_rows.append(i)
            finished_row_ids.append(query_seq_ids[i])
            row_num = read_row_dict[query_seq_ids[i]]
            alignment_image[row_num, col_offset+local_max_col_offset+1] = base_enum['end']
            if save_read_quality:
                read_quality_image[row_num, col_offset+local_max_col_offset+1] = query_qualities[i]
            if save_strand:
                strand_image[row_num, col_offset+local_max_col_offset+1] = query_seq_bases[i, 4]
    # Loop over finished rows separately.
    # NOTE: IDs should be unique (since we include the read as well).
    # But in case they are not, delete in a single loop
    for i in set(finished_rows):
        # Remove the key, else reads IDs may be recycled (same machine)
        # NOTE: Not really an issue, after we made the ID="machineID:read sequence"
        if debug:
            print('popping %s' % query_seq_ids[i])
        read_row_dict.pop(query_seq_ids[i], None)
        reads_offset += 1

    return (read_row_dict, reads_offset, finished_rows)

def center_image_on_column(input_img, center_index, window_size):
    """
    Bounds to center the input image on the center index, with at most window size
    columns on each side.

    This centering algorithm is intentionally a little wonky. It doesn't place the
    reference column at the center, but one column to the right of it. This was done
    for legacy reasons to be compatible with other items.
    There's also a deeper question here of how this should be
    handled in the case of indels. The VCF represents insertions
    as A -> ATTT. So the question is whether the reference should be aligned
    with the T, or with the A. Currently it being aligned with the A.
    TODO: Discuss the above and come up with a proper solution. Till then,
    live with the off by 1 column offset in centering.

    input_img      - Input image
    center_index   - Center column index
    window_size    - Number of columns on each side of center

    Return a tuple of the (min column index from original image,
                           max column index from original image
                           )
    """
    # NOTE: In basic SNP case, index should line up at pos == 100 [for window 100]
    # [backward compatible, for SNP and Delete]
    min_col_idx = max(0, center_index - (window_size))
    max_col_idx = min(center_index + window_size + 1, input_img.shape[1])
    #print('shape %s; center %d; window %d --> min, max (%d, %d)' % (str(input_img.shape),
    #    center_index, window_size, min_col_idx, max_col_idx))
    return (min_col_idx, max_col_idx)

def trim_empty_rows(image, region):
    """
    Trims empty rows (where sum of elements in row == 0 from either top,
    bottom or both sides.

    image  - Input image
    region - "top", "bottom", or "both"

    Returns the trimmed image
    """
    if (region == "top" or region == "both"):
        reads_row_sum = image.sum(axis=1)
        first_non_zero_row = 0
        for k in range(reads_row_sum.shape[0]):
            if reads_row_sum[k] > 0:
                first_non_zero_row = k
                break
        image = image[first_non_zero_row:,:]

    if (region == "bottom" or region == "both"):
        reads_row_sum = image.sum(axis=1)
        last_non_zero_row = 0
        for k in range(reads_row_sum.shape[0] - 1, -1, -1):
            if reads_row_sum[k] > 0:
                last_non_zero_row = k
                break
        image = image[:last_non_zero_row + 1,:]

    if debug:
        print('Trimmed %d empty rows after centering' % (reads_row_sum.shape[0] - image.shape[0]))
    return image

def center_image_on_row_window(input_img, window_size):
    """
    Bounds which center the input image on the window provided. If window is
    larger than the image, the original image bounds are returned.

    input_img     - Input image
    window_size   - Size of window to cut from center of image

    Returns a tuple with (index of first row from original image,
                          index of last row from original image
                          )
    """
    num_reads = input_img.shape[0]
    min_read = max(0, int((num_reads - window_size)/2))
    max_read = min(min_read + window_size, num_reads)
    return (min_read, max_read)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='BAM file to numpy object for single reads')
    parser.add_argument('--input', type=str, help='input file ',
                        default='HG001.hs37d5.300x.22.bam')
    parser.add_argument('--chrom', type=str, default='22',
                        help='(for now) mean to be used at chromosome level. Match chrom to BAM file.')
    parser.add_argument('--locations', type=str, help='numpy file with locations of interest -- mutations and not',
                        default=None)
    parser.add_argument('--tp_vcf',
                        help='VCF file for true positive variants',
                        default=None,
                        type=str)
    parser.add_argument('--tp_full_vcf',
                        help='(optional) Full VCF file for true positive variants -- including heter/homozygous etc -- from Truth BAM only',
                        default=None,
                        type=str)
    parser.add_argument('--fp_vcf',
                        help='VCF file for false positive variants',
                        default=None,
                        type=str)
    parser.add_argument('--fn_vcf',
                        help='VCF file for false negative variants',
                        default=None,
                        type=str)
    parser.add_argument('--restrict_locations_file', type=str, default='', help='(optional) pass numpy file with locations to restrict to')
    parser.add_argument('--restrict_locations', action='store_true', default=False, help='remove locations not in the file?')
    parser.add_argument('--non_restrict_match_random', action='store_true', default=False, help='include similar number of random locations, along with restricted ones?')
    parser.add_argument('--output', type=str, default='result', help='output file ')
    parser.add_argument('--locations-process-step', type=int, default=100000, help='Step size for processing reads to disk [save memory]')
    parser.add_argument('--locations-restart-pos', type=int, default=0, help='If interrupted, expand HDF dataset from this position')
    parser.add_argument('--locations-append-data', action='store_true', default=False, help='Append to existing file -- from position 0')
    parser.add_argument('--debug', action='store_true', help='print extra information')
    parser.add_argument('--min-base-quality', type=int, default=0,
                        help='Set > 0 to drop bad base reads -- on top of read level filtering. Will create holes in the data.')
    parser.add_argument('--fasta-input', type=str, default='hs37d5.fa',
                        help='FASTA file with reference bases')
    parser.add_argument('--num-processes', type=int, default=10,
                        help='How many multiprocess (threads) to process independent read locations?')
    parser.add_argument('--max-loc', type=int, default=0,
                        help='Set > 0 to limit dataset size -- for testing, memory, etc')
    parser.add_argument('--max-reads', type=int, default=1000,
                        help='Set to limit number total reads (will trim from top and bottom). Data saved in pad format.')
    parser.add_argument('--window-size', type=int, default=100,
                        help='Number of neighboring pileup columns to encode on each side of reference.')
    parser.add_argument('--max-insert-length', type=int, default=10,
                        help='For possible base insert, how long an insert to append to data? Set zero to skip all inserts. TODO: Insert only at middle pos.')
    parser.add_argument('--max-insert-length-variant', type=int, default=50,
                        help='For actual variat, allow a longer insert?')
    parser.add_argument('--save-q-scores', action='store_true', help='Save Q-score information -- base by base.')
    parser.add_argument('--save-strand', action='store_true', help='Save strand direction -- for single reads?')

    args = parser.parse_args()
    print(args)

    # Verbose mode:
    global debug
    debug = args.debug

    # Read the BAM file
    input_filename = args.input

    # FASTA file with reference bases2
    fasta_filename = args.fasta_input
    fasta = pysam.FastaFile(fasta_filename)
    print('FASTA references for chroms:')
    print(fasta.references)

    # Collect locations from NPY file, if provided.
    # TODO: Track other info, like mutation type, reference sequence, etc
    # label --> 0: TP, 1: FN, 2: TN [0 & 1 are mutations, 2 is not mutation]
    locations = []
    if args.locations:
        locations_numpy = np.load(args.locations)
        # Filter & format locations for this chrom
        # Select chromosomes to train on -- or hacks for pre-built set
        if args.chrom == 'all':
            chroms = ALL_VARIANT_CHROMS_SET
        else:
            chroms = set([args.chrom.split()])
        for d in locations_numpy:
            name = d['name'].split(':')
            if len(name) == 2 and name[0].startswith("chr") and (name[0][3:] in chroms):
                locations.append([name[0][3:], int(name[1]), d['name'], d['label'], d['ref'], d['reads']])
        print('Collected %d locations for chroms set %s' % (len(locations), chroms))
        print(locations[:min(len(locations), 2)])

    # Load from VCF
    if args.tp_vcf:
        # If available, include "full VCF" from the Truth set -- includes information like homo/hetero variant
        locations.extend(get_locations_from_vcf(args.tp_vcf, label=0, full_vcf=args.tp_full_vcf))
    if args.fn_vcf:
        locations.extend(get_locations_from_vcf(args.fn_vcf, label=1))
    if args.fp_vcf:
        locations.extend(get_locations_from_vcf(args.fp_vcf, label=2))
    print('After adding from VCF, %d total locations considered' % len(locations))

    # Upon request, filter by "difficult/interesting" locations -- greatly reduce the size of the dataset
    if args.restrict_locations:
        print('Choosing restrict_locations -- from file %s' % args.restrict_locations_file)
        loc_file = np.load(args.restrict_locations_file)
        print('Loaded %d lines of locations file %s' % (loc_file.shape[0], str(loc_file.dtype)))
        print(loc_file[:5])
        loc_table = []
        for l in loc_file:
            li = bytes.decode(l[0])
            items = li.split('\t')
            loc_str = '\t'.join(items[0:2])
            loc_table.append(loc_str)
        loc_table = set(loc_table)
        print('condensed to %d unique locations' % len(loc_table))
        relevant_locations = find_location_from_table(locations, loc_table)
        print('applies locations to %d candidates variants' % len(relevant_locations))
        if args.non_restrict_match_random:
            random_locations = np.random.permutation(range(len(locations)))[:len(relevant_locations)]
            relevant_locations = np.union1d(relevant_locations, random_locations)
            print('Extend to %d locations after extending equal random locations' % len(relevant_locations))
        relevant_locations = np.sort(relevant_locations)
        print(relevant_locations.shape)
        print(relevant_locations[:50])
        # Locations now restricted to these relevant (and added random) locations
        # NOTE: no way to sub-index Python lists
        locations = [locations[i] for i in relevant_locations]
        print(locations[:20])
        print('finished restricting locations to %d locations' % len(locations))

    num_processes = args.num_processes
    num_locations = args.max_loc
    if num_locations > 0:
        locations = locations[:num_locations]
    else:
        num_locations = len(locations)
    print('Processing %d locations with %d process' % (len(locations), num_processes))

    # Number of extra pileup columns on each side of reference column.
    window_size = args.window_size

    # Chunk our locations set into smaller pieces -- enough to fit into memory
    loc_start = args.locations_restart_pos
    output_file = args.output
    if loc_start > 0:
        restart = True
        open_file = 'a'
        assert os.path.isfile(output_file), 'Output file must exist for append mode.'
        print('Loading pre-started dataset from %s' % output_file)
    else:
        restart = False
        open_file = 'w'
        print('Initializing serialized images to %s' % output_file)
    # Alternatively, append from the start (if asked, as in missing data)
    if args.locations_append_data:
        restart = True
        open_file = 'a'
        assert os.path.isfile(output_file), 'Output file must exist for append mode.'
    print('Starting reading from location %d -- restart %s [append to file?]' % (loc_start, restart))
    loc_step = args.locations_process_step
    #hf = None
    df = None
    total_errors = 0
    chunk = 0
    num_chunks = math.ceil((num_locations - loc_start) / loc_step)
    print('Splitting locations into ~%d chunks [%d each] to save memory...' % (num_chunks, loc_step))

    # Keep pointer to opened HDF file -- in case of a crash/interruption
    with h5py.File(output_file, open_file) as hf:
        while loc_start < num_locations:
            chunk += 1
            print('%d/%d chunks -- (%d, %d)' % (chunk, num_chunks, loc_start, loc_start+loc_step))
            sample_locations = locations[loc_start:loc_start+loc_step]
            images_npy, num_errors = process_locations_chunk(locations=sample_locations, args=args, samfile=input_filename,
                fasta_filename=fasta_filename, window_size=window_size, num_processes=num_processes)
            total_errors += num_errors
            loc_start += loc_step
            print('Total errors %d through %d steps' % (total_errors, loc_start))

            # Either initialize h5py file for the first time. Or expand and append to it.
            # NOTE: Need to set maxhape == None -- else h5py will not allow expanding the file size.
            # (need initial data, so that dtype is inferred)
            if df == None and (not restart):
                df = hf.create_dataset("data", maxshape=(None,), data=images_npy, compression="gzip")
                print('created initial HDF of size %d' % (df.shape[0]))
            elif df == None and restart:
                df = hf["data"]
                d_len = df.shape[0]
                df.resize((d_len+len(images_npy),))
                df[d_len:] = images_npy
                print('expanded HDF from %d to %d -- %s' % (d_len, df.shape[0], output_file))
            else:
                d_len = df.shape[0]
                df.resize((d_len+len(images_npy),))
                df[d_len:] = images_npy
                print('expanded HDF from %d to %d -- %s' % (d_len, df.shape[0], output_file))

    print('Parsing errors in %d / %d locations -- saved to: %s' % (total_errors, num_locations, output_file))

# Multi-process a chunk of locations -- skip bad readings or BAM errors -- return numpy array ready to be saved to disk
def process_locations_chunk(locations, args, samfile, fasta_filename, window_size, num_processes):
    images = []
    start_time = time.time()
    with multiprocessing.Pool(num_processes) as pool:
        f = functools.partial(process_location, samfile=samfile, args=args, start_time=start_time, fasta=fasta_filename, window_size=window_size)
        for map_image in tqdm.tqdm(pool.imap_unordered(f, locations), total=len(locations)):
            images.append(map_image)

    print('Took %.2fs to process %d loc with %d processes' % (time.time()-start_time, len(locations), num_processes))

    print('Saving outputs')
    print(len(images))

    print('Creating output size %d' % (len(images)))
    TOTAL_SINGLE_READS = args.max_reads
    TOTAL_COLUMNS = 2 * window_size + 1
    assert args.save_q_scores, "Too many options, need to run with Q scores"
    assert args.save_strand, "Too many options, need to run with strand save"
    dt = np.dtype([('name', np.string_, 16), ('ref', np.uint8, (5, TOTAL_COLUMNS)), ('reads', np.uint16, (5, TOTAL_COLUMNS)),
                    ('single_reads', np.uint8, (TOTAL_SINGLE_READS, TOTAL_COLUMNS)), ('ref_bases', np.uint8, TOTAL_COLUMNS),
                    ('num_reads', np.int32, 1), ('label', np.uint8, 1), ('vcfrec', np.string_, 128),
                    ('q-scores', np.uint8, (TOTAL_SINGLE_READS, TOTAL_COLUMNS)),
                    ('strand', np.uint8, (TOTAL_SINGLE_READS, TOTAL_COLUMNS))])
    # NOTE: TOO many save options. Just assert we save Q-scores and Strand, going forward
    #    dt = np.dtype([('name', np.string_, 16), ('ref', np.uint8, (5, TOTAL_COLUMNS)), ('reads', np.uint16, (5, TOTAL_COLUMNS)),
    #                   ('single_reads', np.uint8, (TOTAL_SINGLE_READS, TOTAL_COLUMNS)), ('ref_bases', np.uint8, TOTAL_COLUMNS), ('num_reads', np.int32, 1), ('label', np.uint8, 1),
    #                   ('vcfrec', np.string_, 128)])

    # Count errors returned above:
    count_error = 0
    for i in range(len(images)):
        if len(images[i][0]) == 0:
            count_error += 1
    if count_error > 0:
        print('Parsing errors in %d / %d locations' % (count_error, len(images)))

    images_npy = np.empty(len(locations) - count_error, dtype=dt)
    # For Numpy saving, need to use fixed number of reads -- will keep zeroes for the rest
    # Count errors returned above:
    count_error = 0
    for i in tqdm.tqdm(range(len(images)), total=len(images)):
        if len(images[i][0]) == 0:
            count_error += 1
            continue

        # clean up single reads file
        single_read = images[i][0]
        center_index = images[i][1]
        # reference_index[idx] = (offset, reference_position, reference_base)
        reference_index = images[i][2]
        quality_read = images[i][3]
        strand_read = images[i][4]
        location = images[i][5]
        save_read_quality = args.save_q_scores

        # Create a "reference line" matching bases in the pileup
        # NOTE: Pad *between bases* with '' since these are inserts
        reference_bases = np.full(single_read.shape[1], base_enum[''], dtype=np.uint8)
        for k in reference_index.keys():
            off, ref_pos, ref_base = reference_index[k]
            reference_bases[off] = base_enum[ref_base]

        if debug:
            print('-----')
            print([reference_index[window_size - 2], reference_index[window_size - 1], reference_index[window_size], reference_index[window_size + 1]])
            print(single_read.shape)
            print('center position: %d' % center_index)

        if debug:
            print('Original images')
            print(single_read)
            print(single_read[:4,:])
            print(single_read.shape)
            print(quality_read)
            print(quality_read[:4,:])
            print(quality_read.shape)
            print(strand_read)
            print(strand_read[:4,:])
            print(strand_read.shape)

        # Trim single_read file so that it's == TOTAL_COLUMNS length, centered at center_index [could be shorter if empty reads]
        (min_col_idx, max_col_idx) = center_image_on_column(single_read, center_index, window_size)
        single_read = single_read[:,min_col_idx:max_col_idx]
        quality_read = quality_read[:,min_col_idx:max_col_idx]
        strand_read = strand_read[:,min_col_idx:max_col_idx]

        # Activate to debug if getting errrors
        if debug:
            print('Centered images')
            print(single_read)
            print(single_read[:4,:])
            print(single_read.shape)
            print(quality_read)
            print(quality_read[:4,:])
            print(quality_read.shape)
            print(strand_read)
            print(strand_read[:4,:])
            print(strand_read.shape)

        # Now that we have centered the reads -- will have some empty rows on top. Remove those rows.
        single_read = trim_empty_rows(single_read, "top")
        quality_read = trim_empty_rows(quality_read, "top")
        strand_read = trim_empty_rows(strand_read, "top")

        if debug:
            print('Trimmed empty rows')
            print(single_read)
            print(single_read[:4,:])
            print(single_read.shape)
            print(quality_read)
            print(quality_read[:4,:])
            print(quality_read.shape)
            print(strand_read)
            print(strand_read[:4,:])
            print(strand_read.shape)

        # If excessive reads (very high coverage, trim from both ends)
        (min_read, max_read) = center_image_on_row_window(single_read, TOTAL_SINGLE_READS)
        single_read = single_read[min_read:max_read,:]
        num_reads = single_read.shape[0]

        # Try if data lines up. If not, count and log the error -- errors should be rare, not frequent
        try:
            # Adjust Q-scores and strand, to same limits
            quality_read = quality_read[min_read:max_read,:]
            num_q_reads = quality_read.shape[0]
            # NOTE: If getting errors here -- tiny chance that trim_empty_rows fail since could be "noinsert" for base, or otherwise missing Q-score?
            assert num_reads == num_q_reads, "Num reads does not match num quality reads! %d != %d" % (num_reads, num_q_reads)
            assert quality_read.shape == single_read.shape, "Quality and Single Read shape does not match"
            strand_read = strand_read[min_read:max_read,:]
            num_strand_reads = strand_read.shape[0]
            assert num_reads == num_strand_reads, "Num reads does not match num strand reads! %d != %d" % (num_reads, num_strand_reads)
            assert strand_read.shape == single_read.shape, "Strand and Single Read shape does not match"

            if debug:
                print((min_col_idx, max_col_idx, num_reads, min_read, max_read))

            # Apply same indexing to reference bases
            reference_bases = reference_bases[min_col_idx:max_col_idx]
            single_read_pad = np.zeros((TOTAL_SINGLE_READS, TOTAL_COLUMNS), dtype=np.uint8)
            reference_bases_pad = np.zeros((TOTAL_COLUMNS), dtype=np.uint8)
            # IDX offset -- if we don't have enough reads (zero coverage out of scope)
            idx_offset = (window_size) - (center_index - min_col_idx)
            #print('idx offset %d' % idx_offset)
            single_read_pad[:min(TOTAL_SINGLE_READS,single_read.shape[0]),idx_offset:idx_offset+single_read.shape[1]] = single_read
            reference_bases_pad[idx_offset:idx_offset+single_read.shape[1]] = reference_bases

            # Adjust Q-scores and Strand (direction of sequencer read)
            quality_read_pad = np.zeros((TOTAL_SINGLE_READS, TOTAL_COLUMNS), dtype=np.uint8)
            quality_read_pad[:min(TOTAL_SINGLE_READS,single_read.shape[0]),idx_offset:idx_offset+single_read.shape[1]] = quality_read
            strand_read_pad = np.zeros((TOTAL_SINGLE_READS, TOTAL_COLUMNS), dtype=np.uint8)
            strand_read_pad[:min(TOTAL_SINGLE_READS,single_read.shape[0]),idx_offset:idx_offset+single_read.shape[1]] = strand_read

            # Final output
            output_tuple = (location[2], location[4], location[5], single_read_pad, reference_bases_pad,
                min(num_reads, TOTAL_SINGLE_READS), location[3], location[6], quality_read_pad, strand_read_pad)
            assert num_reads > 0, "TODO: handle no-reads case"
            images_npy[i-count_error] = output_tuple
        except AssertionError as ae:
            print('Assert (aligment) error in location %s -- skipping' % str(location))
            print(ae)
            count_error += 1

    # Re-shape output, to return only non-error cases.
    total_cases = len(locations) - count_error
    if count_error > 0:
        print('Total errors (including alignment error) %d. Reducing to %d total cases' % (count_error, total_cases))
    return (images_npy[:total_cases], count_error)

# Process a single read location
def process_location(l, samfile=None, args=None, start_time=None, fasta=None, window_size=100):
    num_loc = 0
    debug = args.debug
    MAX_INSERT_LENGTH = args.max_insert_length
    # Allow for longer inserts in the variant itself -- to handle long repeats, etc
    MAX_INSERT_VARIANT = max(args.max_insert_length_variant, MAX_INSERT_LENGTH)
    if True:
        chrom = str(l[0])
        center_position = int(l[1])
        window_size += 2 # Just in case, will get trimmed anyway
        start = center_position - window_size
        end = center_position + window_size + 1

        if debug:
            print('Processing location %s' % str(l))

        # Call Pileup with truncate=True -- to only examine pileups within the requested range.
        # NOTE: Other parameters may no be optimal settings -- trying to get all reads
        # https://pysam.readthedocs.io/en/latest/api.html
        # stepper="nofilter" [supposely more reads?]
        # stepper="samtools" [should be best match for Zhen processing?]
        # min_base_quality = 0 [what is default? -- not listed]
        # NOTE: Attaching FASTA file, but does not appear to make any difference.
        # If we set min_base_quality > 0, will have holes in the output matrix. Sometimes very big holes.
        # HACK: For multithread only, load file from scratch:
        fasta = pysam.FastaFile(fasta)
        samfile = pysam.AlignmentFile(samfile, "rb")
        pu = samfile.pileup(truncate=True, contig=chrom, start=start, stop=end, ignore_overlaps=False, fastafile=fasta,
            stepper="nofilter", ignore_orphans=True, min_base_quality = args.min_base_quality, min_mapping_quality=0)

        # Fill static N x M array
        max_reads = 1200  # NOTE: space will grow if needed
        read_window = 3 * window_size  # NOTE: space will grow if needed
        save_read_quality = args.save_q_scores
        save_strand = args.save_strand

        # Fill an index with prior window_size bases, centre base window_size bases afterward -- and enough slack for inserts
        alignment_image = np.full((max_reads, read_window), base_enum['pad'], dtype=np.uint8)
        if save_read_quality:
            if debug:
                print('initializing read quality scores!')
            read_quality_image = np.full(alignment_image.shape, 0, dtype=np.uint8)
        else:
            read_quality_image = None
        if save_strand:
            if debug:
                print('initializing strand value scores')
            strand_image = np.full(alignment_image.shape, 0, dtype=np.uint8)
        else:
            strand_image = None

        MAX = 1000
        # Index == core logic of which column in the output image, are we looking at [for current base pairs]
        idx = 0
        # Start with col_offset == 1 -- why?? Keep read starts. Doesn't really matter, but keep the look-behind logic
        prev_col_offset = 0
        col_offset = 1
        local_max_col_offset = 0

        # Keep a map of all rows by ID -- so we can align bases from a read into the same row
        read_row_dict = {}
        reads_offset = 0

        # Index to reference locations, reference bases
        # [Useful for applying final shift, output the reference]
        col_reference_map = {}

        for p in pu: # cycle per column, all reads.
            if idx > MAX:
                break

            pos = p.reference_pos
            if debug:
                print('Considering alignment for *reference* position %s' % pos)

            # Not sure why p.get_query_sequences() fails sometimes (very rare) on extra long pileups. Debug to skip bad locations
            try:
                query_sequences = p.get_query_sequences(mark_ends=True, add_indels=True, mark_matches=True)
            except AssertionError:
                print('Failed get_query_sequences')
                debug = True
                print('Center position Chrom %s -- %d' % (chrom, center_position))
                print('Considering alignment for *reference* position %s' % pos)
                print('Not sure why we died here?')
                print('--------------------------')

                # Quit early, count these later.
                return ([], -1, [])

            if debug:
                # debugging
                print(query_sequences)
                print('Mapping qualities:')
                map_qualities = p.get_mapping_qualities()
                print(map_qualities)
                print('Query qualities:')
                query_qualities = p.get_query_qualities()
                print(query_qualities)
                print(list(zip(query_sequences, map_qualities, query_qualities)))
                print(dir(p.pileups[0].alignment))
                print(p.pileups[0].alignment)
                print(p.pileups[0].alignment.query_name)
                print(p.pileups[0].alignment.next_reference_id)
                for i in range(min(len(p.pileups), 10)):
                    print(p.pileups[i].alignment.seq)
                    print(p.pileups[i].indel)
                print(query_sequences)
                print(len(query_sequences))

            # 1. Decode query sequences -- three parts -- start/end, base, indel
            try:
                (query_seq_bases, query_seq_inserts) = decode_query_sequences(query_sequences)
            except:
                # debugging
                print("chrom, center_position")
                print(chrom, center_position)
                print(query_sequences)
                print('Mapping qualities:')
                map_qualities = p.get_mapping_qualities()
                print(map_qualities)
                print('Query qualities:')
                query_qualities = p.get_query_qualities()
                print(query_qualities)
                print(list(zip(query_sequences, map_qualities, query_qualities)))
                print(dir(p.pileups[0].alignment))
                print(p.pileups[0].alignment)
                print(p.pileups[0].alignment.query_name)
                print(p.pileups[0].alignment.next_reference_id)
                for i in range(min(len(p.pileups), 10)):
                    print(p.pileups[i].alignment.seq)
                    print(p.pileups[i].indel)
                print(query_sequences)
                print(len(query_sequences))

                assert False

            # TODO: Deepder check -- verify that the reads match positions in the pileups. Just can't trust anything.
            assert len(query_seq_bases) == len(p.pileups), "mismatch between pileups & sequences"

            # Collect query qualities, upon request
            if save_read_quality:
                query_qualities = p.get_query_qualities()

            # We need to encode both ID AND sequence, since IDs are re-used
            # TODO: More efficient encoding?
            query_seq_ids = ["%s:%s" % (pu.alignment.query_name, pu.alignment.seq) for pu in p.pileups]

            # 2. Onboard new rows to row_map (check start token)
            min_row_size = len(read_row_dict) + reads_offset + len(query_seq_ids) + 1
            min_col_size = col_offset + MAX_INSERT_VARIANT + max(10, MAX_INSERT_VARIANT)
            alignment_image = resize_alignment_image(min_row_size, min_col_size, alignment_image)
            if save_read_quality:
                read_quality_image = resize_alignment_image(min_row_size, min_col_size, read_quality_image)
            if save_strand:
                strand_image = resize_alignment_image(min_row_size, min_col_size, strand_image)

            # Instead, just fill in rows as unique read IDs filter in
            for i, name in enumerate(query_seq_ids):
                if not(name in read_row_dict.keys()):
                    read_row_dict[name] = len(read_row_dict) + reads_offset

            # 3. Write bases by row_map to data matrix
            #    A. normal reads (and skips) -- also write quality scores
            is_variant = (p.reference_pos == center_position-1)
            if debug and is_variant:
                print('Variant detected, in location %s: %s' % (chrom, p.reference_pos))
                print(query_sequences)
            local_max_col_offset = add_bases_to_alignment_image(query_seq_bases, query_seq_inserts,
                    query_seq_ids, alignment_image, col_offset, prev_col_offset, read_row_dict,
                    query_qualities=query_qualities, read_quality_image=read_quality_image, strand_image=strand_image,
                    MAX_INSERT_LENGTH=(MAX_INSERT_VARIANT if is_variant else MAX_INSERT_LENGTH), args=args)

            if debug and is_variant:
                print('Local col offset %d' % local_max_col_offset)

            # 4. Perform cleanup
            (read_row_dict, reads_offset, finished_rows) = handle_ended_sequences(query_seq_bases, query_seq_ids, alignment_image,
                read_row_dict, reads_offset, col_offset, local_max_col_offset,
                query_qualities=query_qualities, read_quality_image=read_quality_image, strand_image=strand_image, args=args)

            if debug:
                print('Finished/deleting rows %s' % str(finished_rows))
                print('Finished processing idx %d, ref pos %d, col_offset %d, local_max_col_offset %d, num_reads %d' % (idx, p.reference_pos, col_offset, local_max_col_offset, len(read_row_dict)+reads_offset))

                # Assume that in a pileup, rows are in order
                print('Finished reads:')
                print([query_seq_ids[i] for i in finished_rows])

            # Save information about the current reference column
            # Unable to get reference base from the data -- look at location input instead...
            col_reference_map[idx] = (col_offset, p.reference_pos, fasta.fetch(reference=chrom, start=p.reference_pos, end=p.reference_pos+1))

            if debug and (p.reference_pos == center_position) and col_offset > len(col_reference_map):
                print(col_reference_map)
                print('center pos %s' % center_position)

            # 5. Update index and counters.
            idx += 1
            prev_col_offset = col_offset
            col_offset = col_offset + 1 + local_max_col_offset
            local_max_col_offset = 0

        # HACK -- On a line-by-line basis, we need to replace STRAND_PAD values with 1/2 from that read
        for row_num in range(strand_image.shape[0]):
            # Check if any STAND_PAD are included?
            num_pads = np.sum(strand_image[row_num, :] == STRAND_PAD)
            if num_pads == 0:
                continue
            row_nopad = strand_image[row_num, :] * ~(strand_image[row_num, :] == STRAND_PAD)
            # Find max value of STRAND over non-pad entried in a row
            strand_value = max(row_nopad)
            # HACK -- if no STAND set, set it to default [not zero]
            if strand_value == 0:
                strand_value = STRAND_UPPER
                if debug:
                    print('HACK -- unknown strand in row %d so replacing with default strand %d' % (row_num, strand_value))
            strand_image[row_num, :][strand_image[row_num, :] == STRAND_PAD] = strand_value
            if debug:
                print('Replaced %d STRAND_PAD items in row %d with strand_value %d' % (num_pads, row_num, strand_value))

        assert np.sum(strand_image == STRAND_PAD) == 0, 'ERROR: Failed to replace STRAND_PAD values for delete'

        # Revelant output so far [trim unused dimensions of the image]
        image_sample = alignment_image[:len(read_row_dict) + reads_offset, :col_offset+1]
        if save_read_quality:
            read_quality_sample = read_quality_image[:len(read_row_dict) + reads_offset, :col_offset+1]
        else:
            read_quality_sample = None
        if save_strand:
            strand_sample = strand_image[:len(read_row_dict) + reads_offset, :col_offset+1]
        else:
            strand_sample = None
        if debug:
            print(image_sample)
            print(image_sample.shape)
            if save_read_quality:
                print(read_quality_sample)
                print(read_quality_sample.shape)
            if save_strand:
                print(strand_sample)
                print(strand_sample.shape)

        # NOTE: Final saved image should be the (unshifted) training data for this location
        #if debug:
        #    print('saving to %s' % 'sample.hdf')
        #    with h5py.File('sample.hdf', 'w') as hf:
        #        hf.create_dataset("data",  data=image_sample, compression="gzip")
        #    print('saving dictionary (read IDs to matrix row) to %s' % 'row_dict.pickle')
        #    pickle.dump({read_row_dict[k]:k for k in read_row_dict.keys()}, open('row_dict.pickle', 'wb'))

        # Now for alignment...
        # A. Most important -- center the reads at the position in question
        # B. Lookup reference -- align the reference with skipped base pairs? Or not.
        center_index = -1
        for idx in col_reference_map:
            # NOTE -- center position == off by one for in p.reference_pos for some reason
            if int(col_reference_map[idx][1]) == center_position-1:
                center_index = col_reference_map[idx][0]
                if debug:
                    print('Found center index %s at position %d' % (center_position, center_index))
                break
        # Debug this error, as well. Why do we not find center index?
        if center_index == -1:
            print("Could not find center index for %s" % str(l[:3]))
            return ([], -1, [])

        # TODO: Save file, move on to new location
        if debug:
            print('%.2fs to finished processing %d locations' % (time.time() - start_time, num_loc+1))

        # Returns trimmed image, and center position -- for recovery of aligment with the mutation loc in question
        # NOTE: Make sure to return the location info -- since results in arbitary order, not in original location order!
        return image_sample, center_index, col_reference_map, read_quality_sample, strand_sample, l


if __name__ == '__main__':
    main()
