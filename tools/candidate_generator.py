#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""Variant candidate generator for DL4Mutations

Generate a VCF from a BAM file with candidate variants.

Example usage:
    candidate_generator.py --input in.bam --output out.vcf --contigs 20:1000:2000,17:0:50000,8

The above example will process contig 20 region 1000 to 2000, chromosome 17 region 0 to 50000 and chromosome 8.
Contigs are processed in separte processes.
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

import collections
import pysam
import bedutils


def create_alleles_for_substitutions(read, aligned_pairs):
    match_bases = 'ACGT'
    substitution_bases = 'acgt'

    substitution_indices = [i for i, x in enumerate(
        aligned_pairs) if x[2] is not None and x[2] in substitution_bases]

    alleles = []
    # Substitutions
    for i in substitution_indices:
        ref_pos = int(aligned_pairs[i][1])
        read_pos = int(aligned_pairs[i][0])
        ref_base = aligned_pairs[i][2]
        read_base = read.seq[read_pos]
        if read_base in match_bases:
            allele = (read.reference_name, ref_pos,
                      ref_base.upper(), read_base.upper())
            alleles.append(allele)

    return alleles


def create_alleles_for_insertions(read, aligned_pairs, max_len_indel_allele=50):
    insertion_indices = [i for i, x in enumerate(
        aligned_pairs) if x[1] is None]

    alleles = []
    # Insertions
    insertion_continue = False
    alt_seq = None
    ref_pos = None

    for j, i in enumerate(insertion_indices):
        read_pos = int(aligned_pairs[i][0])
        if insertion_continue:
            alt_seq += read.seq[read_pos].upper()
        else:
            ref_pos = int(aligned_pairs[i - 1][1])
            ref_base = aligned_pairs[i - 1][2]
            alt_seq = aligned_pairs[i - 1][2].upper() + \
                read.seq[read_pos].upper()

        if (j != (len(insertion_indices) - 1)) and ((insertion_indices[j + 1] - i) == 1):
            insertion_continue = True
        else:  # end of the insertion
            insertion_continue = False
            if len(alt_seq) <= max_len_indel_allele:
                alleles.append((read.reference_name, ref_pos,
                                ref_base.upper(), alt_seq.upper()))
            else:
                print('Dropping over-long allele %s: %s -> %s' % (ref_pos, str(ref_base), str(alt_seq)))
    return alleles


def create_alleles_for_deletions(read, aligned_pairs, max_len_indel_allele=50):
    deletion_indices = [i for i, x in enumerate(aligned_pairs) if x[0] is None]

    alleles = []
    # Deletions
    deletion_continue = False
    alt_seq = None
    ref_pos = None
    # NOTE -- will throw an exception for (None, None) pairs in aligned pairs
    # TODO: Fix it?
    for j, i in enumerate(deletion_indices):
        if deletion_continue:
            ref_base += aligned_pairs[i][2].upper()
        else:
            ref_pos = int(aligned_pairs[i - 1][1])
            ref_base = aligned_pairs[i - 1][2].upper() + \
                aligned_pairs[i][2].upper()
            alt_seq = aligned_pairs[i - 1][2]

        if (j != (len(deletion_indices) - 1)) and ((deletion_indices[j + 1] - i) == 1):
            deletion_continue = True
        else:  # end of the deletion
            deletion_continue = False
            if len(ref_base) <= max_len_indel_allele:
                alleles.append((read.reference_name, ref_pos,
                                ref_base.upper(), alt_seq.upper()))
            else:
                print('Dropping over-long allele %s: %s -> %s' % (ref_pos, str(ref_base), str(alt_seq)))
    return alleles


def detect_variants(read, max_len_indel_allele=50):
    """
    Return list of variant alleles for this read.

    If read is identical to reference returns empty list

    Allele is of form (CHR, POS, REF, ALT)
    """
    alleles = []

    aligned_pairs = read.get_aligned_pairs(with_seq=True, matches_only=False)

    # Exit early if nothing returned.
    if len(aligned_pairs) == 0:
        print('Warning -- dropping read with no aligned pairs: %s' % str(read))
        return alleles

    # Trim start
    while aligned_pairs[0][1] is None:
        aligned_pairs = aligned_pairs[1:]
        if len(aligned_pairs) == 0:
            print('(after cleaning Nones -- Warning -- dropping read with no aligned pairs: %s' % str(read))
            return alleles

    # Trim end
    while aligned_pairs[-1][1] is None:
        aligned_pairs = aligned_pairs[:-1]
        if len(aligned_pairs) == 0:
            print('(after cleaning Nones -- Warning -- dropping read with no aligned pairs: %s' % str(read))
            return alleles

    alleles.extend(create_alleles_for_substitutions(read, aligned_pairs))
    alleles.extend(create_alleles_for_insertions(read, aligned_pairs, max_len_indel_allele=max_len_indel_allele))
    try:
        alleles.extend(create_alleles_for_deletions(read, aligned_pairs, max_len_indel_allele=max_len_indel_allele))
    except TypeError as e:
        #print('skipping problem with deletion alignment error')
        #print(e)
        pass

    # Make sure alleles meet expected output format.
    # TODO: Could fix formatting here -- like uppercasing variants (since downstream comparison is string-based)
    for al_tuple in alleles:
        chrom, pos, ref, alt = al_tuple
        assert ref.isupper(), "alleles must be in uppercase form %s" % str(al_tuple)
        assert alt.isupper(), "alleles must be in uppercase form %s" % str(al_tuple)

    return alleles


def merge_vcfs(vcf_filenames, outfile):
    vcf_out_header = pysam.VariantFile(vcf_filenames[0]).header
    vcf_out = pysam.VariantFile(outfile, 'w', header=vcf_out_header)

    for v in tqdm.tqdm(vcf_filenames):
        f = pysam.VariantFile(v)
        for record in f:
            vcf_out.write(record)
    vcf_out.close()

    # Final sort of VCF file.
    logging.info("Sorting final VCF file...")
    sorted_outfile = outfile + ".tmp"
    subprocess.check_call(['cat {} | awk \'$1 ~ /^#/ {{print $0;next}} {{print $0 | "sort -k1,1 -k2,2n"}}\' > {}'.format(outfile, sorted_outfile)],
                          shell=True)
    os.rename(sorted_outfile, outfile)


def candidates_to_vcf(bamfile, candidates, out_file):
    """
    Writes a VCF file from the candidate alleles.
    """
    vcf_out = pysam.VariantFile(out_file, 'w')
    header = vcf_out.header
    candidates = list(candidates)
    candidates.sort()

    header.formats.add("GQ", "1", "Integer", "Genotype Quality")
    header.formats.add("GT", "1", "String", "Genotype")
    header.info.add("DP", 1, "Integer", "Total Depth")
    header.info.add("AF", "A", "Float", "Allele Frequency")
    header.add_sample("CALLED")

    for contig in bamfile.references:
        header.add_line('##contig=<ID={},length={}'.format(
            contig, bamfile.get_reference_length(contig)))

    for candidate in candidates:
        pos = candidate[1]
        record = vcf_out.new_record(contig=candidate[0], start=pos, alleles=(
            candidate[2], candidate[3]), qual=50)
        record.samples['CALLED']['GQ'] = 50
        record.samples['CALLED']['GT'] = (1,)
        record.info['DP'] = candidate[4]
        record.info['AF'] = candidate[5]

        record.stop = pos + len(candidate[2])
        vcf_out.write(record)

    vcf_out.close()
    pysam.tabix_index(out_file, preset='vcf', force=True, keep_original=True)


def build_allele_stats(contig, bamfile, max_len_indel_allele=50):
    """
    Generates allele frequency along with coverage of each locus.
    """
    region_start = contig[1]
    region_end = contig[2]

    if contig[1] is not None:
        logging.debug('Processing contig {}, region {}:{}'.format(
            contig[0], region_start, region_end))
    else:
        logging.debug('Processing contig {}'.format(contig[0]))

    locus_coverage = defaultdict(int)
    allele_frequency = defaultdict(int)

    reads = bamfile.fetch(contig[0], region_start, region_end)
    for i, read in enumerate(reads):
        try:
            reference_positions = read.get_reference_positions()
            for r in reference_positions:
                locus_coverage[r] += 1
            variants = detect_variants(read, max_len_indel_allele=max_len_indel_allele)
            for variant in variants:
                if (region_end is None) or ((variant[1] >= region_start) and (variant[1] <= region_end)):
                    allele_frequency[variant] += 1
        except ValueError as e:
            if 'MD tag not present' not in str(e):
                print('Warning. {} for chrom {} read {}'.format(e, contig[0], read))

    return (locus_coverage, allele_frequency)


def filter_alleles_by_frequency(locus_coverage, allele_frequency, snp_min_freq, indel_min_freq):
    """
    Filters alleles based on a minimum frequency of occurrence.
    """
    filtered_alleles = []
    for allele in allele_frequency.keys():
        allele_count = allele_frequency[allele]
        depth = locus_coverage[allele[1]]
        if depth == 0:
            print('Warning. Ignoring allele. Read depth 0 for {}'.format(allele))
            continue
        if (depth < allele_count):
            print('Warning. Allele count (%d) greater than read depth (%d) of position!' % (allele_count, depth))
        af = min(allele_count, depth) / depth
        if (len(allele[2]) == 1) and (len(allele[3]) == 1):
            min_freq = snp_min_freq
        else:
            min_freq = indel_min_freq

        if (af > min_freq):
            filtered_alleles.append(
                (allele[0], allele[1], allele[2], allele[3], depth, af))

    return filtered_alleles


def remove_multialleles(alleles):
    """
    Filter alleles so that if more than one candidate is generated for one genomic position the one with
    the highest AF is chosen.
    """
    alleles.sort()
    alleles_by_position = {}
    for allele in alleles:
        pos = str(allele[0]) + "_" + str(allele[1])
        if pos in alleles_by_position:
            if alleles_by_position[pos][5] < allele[5]:
                alleles_by_position[pos] = allele
        else:
            alleles_by_position[pos] = allele
    return alleles_by_position.values()


def vcf_for_regions(contigs, bamfile_fn, filename=None, snp_min_freq=0.01, indel_min_freq=0.01, keep_multialleles=False, max_len_indel_allele=50):
    """
    Return list of alleles for contig (contig name, start, ref, alt, depth, allele frequency) and bamfile.
    """
    bamfile = pysam.AlignmentFile(bamfile_fn, "rb")

    candidates = []
    for contig in contigs:
        (locus_coverage, allele_frequency) = build_allele_stats(contig, bamfile, max_len_indel_allele=max_len_indel_allele)

        #  Filter the allels by min_freq and add depth and af values
        filtered_alleles = filter_alleles_by_frequency(locus_coverage,
                                                       allele_frequency,
                                                       snp_min_freq,
                                                       indel_min_freq)

        if not keep_multialleles:
            filtered_alleles = remove_multialleles(filtered_alleles)

        candidates += filtered_alleles

    if filename is None:
        filename = os.path.join(tempfile.gettempdir(),
                                next(tempfile._get_candidate_names()) + ".vcf")

    candidates_to_vcf(bamfile, candidates, filename)
    return filename

# Hack -- remove pesky prefix, like "chr" in contig
def remove_prefix(s, prefix):
    return s[s.startswith(prefix) and len(prefix):]

# And vice-versa
def insert_prefix(s, prefix):
    return prefix+s

def generate_contig_regions(contig_lengths, bamfile, contig_str, bedfile, args={}):
    """
    Generate contig regions in which to look for variants. Returns the intersection
    of regions between bedfile intervals and contig_str/full contigs.

    contig_lengths   - Length of each contig from reference BAM file
    bamfile          - pysam BAM file object
    contig_str       - String defining contig regions in the format
                       contig:start:end
                       If this string is not provided, all contigs in
                       BAM file are used

    A list of tuples in the format (contig, start, end). None value for
    start/end indicates coverage from begining/till end of the contig.
    """

    # If contig_str is not defined, then grab all the contigs from the BAM
    # file. Otherwise parse the contig strings.
    regions = []
    if contig_str is None:
        contigs = bamfile.references
        regions = [(c, 0, contig_lengths[c]) for c in contigs]
    else:
        if contig_str is not None:
            contigs_str = contig_str.split(',')
            contigs_str = [c.split(':') for c in contigs_str]
            for c in contigs_str:
                contig_name = c[0]
                if len(c) == 3:
                    regions.append((contig_name, int(c[1]), int(c[2])))
                else:
                    regions.append(
                        (contig_name, 0, contig_lengths[contig_name]))

    if len(regions) == 0:
        raise RuntimeError(
            "No regions! Need to supply either via contig_str or the bamfile")

    print('Examining %d regions in the bamfile...' % (len(regions)))
    print(regions[0:7])

    # If BED file is passed in, then get the intersection of the contig regions
    # from above with intervals specified in the BED file.
    if bedfile is not None:
        bedfile_intervals = bedutils.get_intervals_from_bedfile(bedfile)
        #bed_intervals = []
        contig_intervals = collections.defaultdict(list)
        for r in regions:
            # HACK -- Remove leading "Chr" from contig if it's there...
            # NOTE: Always assume our BED file does *not* contain 'chr' prefix
            contig_intervals[remove_prefix(r[0], 'chr')].append(
                    bedutils.BedInterval(remove_prefix(r[0], 'chr'), r[1], r[2]))
        regions = []
        intersection = bedutils.intersect_intervals(contig_intervals, bedfile_intervals)
        for intersection in intersection.values():
            for i in intersection:
                # HACK -- some BAM files list contigs as "chr1" instead of "1"
                # TODO: Can automatically detect this in the BAM
                if args.keep_contig_chr:
                    regions.append((insert_prefix(i.chrom, 'chr'), i.start, i.stop))
                else:
                    regions.append((i.chrom, i.start, i.stop))

    return regions


def generate_contig_subregions(regions, subregion_size):
    """
    Split regions into subregions of at most subregion_size. Done in order
    to parallelize the candidate generation for each subregion.

    regions          - List of tuples defining (contig, start, end)
                       as per output of generate_contig_regions()
    subregion_size   - Maximum size of each subregion

    Returns a list of tuples in the format (contig, subregion_start, subregion_end)
    """
    subregions = []
    for region in regions:
        contig_name = region[0]

        if region[1] is None:
            region_start = 0
            region_end = contig_length
        else:
            region_start = region[1]
            region_end = region[2]

        subregion_start = region_start
        if subregion_start + subregion_size > region_end:
            subregion_end = region_end
        else:
            subregion_end = subregion_start + subregion_size
        subregions.append((contig_name, subregion_start, subregion_end))
        while subregion_end < region_end:
            subregion_start = subregion_end
            if subregion_start + subregion_size > region_end:
                subregion_end = region_end
            else:
                subregion_end = subregion_start + subregion_size
            subregions.append((contig_name, subregion_start, subregion_end))
    return subregions


def collate_subregions_into_groups(subregions, group_size):
    """
    Creates groups of subregions such that the total length of
    each group does not exceed the user input group_size. The
    grouping is done in a greedy manner.

    NOTE: The assumption is that each subregion is <= group_size,
    so that every subregion can be fit into a group.

    subregions - List of subregions described as (contig, start, end)
    group_size - Maximum length of a subregion to be processed in
                     a single thread.

    Returns a list of lists, where each member list is a grouping of
    subregions.
    """
    groups = [[]]

    current_group_size = 0
    for sub in subregions:
        sub_length = sub[2] - sub[1]
        logging.debug('Subregion size = {}'.format(sub_length))

        if (sub_length > group_size):
            raise RuntimeError('Found subregion {}:{}:{} greater'
                               'than group size of {}.\n'
                               'Each subregion MUST be less than group size.'
                               .format(sub[0], sub[1], sub[2], group_size))

        # If total length surpases group_size, create a new group
        if (current_group_size + sub_length > group_size):
            groups.append([])
            current_group_size = 0

        groups[-1].append(sub)
        current_group_size += sub_length
        logging.debug('Adding region {}:{}:{} to group {}'.format(
            sub[0], sub[1], sub[2], len(groups)))

    return groups


def main():
    parser = argparse.ArgumentParser(description="Generate a VCF from a BAM file with candidate variants.\n\n" +
                                     "Example usage:\n" +
                                     "    candidate_generator.py --input in.bam --output out.vcf --contigs " +
                                     "20:1000:2000,17:0:50000",
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('--input',
                        help='input BAM file')
    parser.add_argument('--output',
                        default='out.vcf',
                        help='output VCF file')
    parser.add_argument('--contigs',
                        default=None,
                        help='Comma delimited list of contigs to use in format contig:start:end')
    parser.add_argument('--keep_contig_chr',
                        action='store_true',
                        default=False,
                        help='Set true if BAM file lists contigs as "chrC" instead of "X" -- TODO: Autodetect')
    parser.add_argument('--chunk_size',
                        default=1000,
                        type=int,
                        help='Size of region for each process to calculate variants on, in kb (kilobases)')
    parser.add_argument('--threads',
                        default=None,
                        type=int,
                        help='Number of threads to use. Defaults to number of available cores')
    parser.add_argument('--snp_min_freq',
                        default=0.01,
                        type=float,
                        help='The minimum fraction of SNP alleles at a locus to be included as a candidate')
    parser.add_argument('--indel_min_freq',
                        default=0.01,
                        type=float,
                        help='The minimum fraction of indel alleles at a locus to be included as a candidate')
    parser.add_argument('--keep_multialleles',
                        action='store_true',
                        default=False,
                        help='Do not remove multiple alleles for same genomic location')
    parser.add_argument('--max_len_indel_allele',
                        default=60,
                        type=int,
                        help='In case of bad mapping, ignore long alleles.')
    parser.add_argument('--bedfile',
                        default=None,
                        help='BED file with intervals to use for candidate generation')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Print debug information')
    args = parser.parse_args()
    print(args)

    # Configure logger.
    if (args.debug):
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

    # Read input file.
    logging.debug('Reading BAM file from {} ...'.format(args.input))
    bamfile_fn = args.input
    bamfile = pysam.AlignmentFile(bamfile_fn, "rb")

    contig_lengths = dict(zip(bamfile.references, bamfile.lengths))

    # Generate regions and subregions.
    logging.info('Generating regions of interest...')
    regions = generate_contig_regions(contig_lengths,
                                      bamfile,
                                      args.contigs, args.bedfile, args)
    logging.debug(
        'Generated {} regions from BAM, contigs, and restriction on BED file.'.format(len(regions)))

    subregion_size = args.chunk_size * 1000
    subregions = generate_contig_subregions(regions, subregion_size)
    logging.info('Process {} subregions from {} regions.'.format(
        len(subregions), len(regions)))

    groups_of_subregions = collate_subregions_into_groups(
        subregions, subregion_size)
    logging.info('Grouped {} subregions into {} groups.'.format(
        len(subregions), len(groups_of_subregions)))

    if args.threads is None:
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = args.threads
    logging.info(
        'Running in multi-process mode with {} threads ...'.format(num_processes))

    vcf_files = []
    with Pool(num_processes) as p:
        f = functools.partial(vcf_for_regions,
                              bamfile_fn=bamfile_fn,
                              snp_min_freq=args.snp_min_freq,
                              indel_min_freq=args.indel_min_freq,
                              keep_multialleles=args.keep_multialleles,
                              max_len_indel_allele=args.max_len_indel_allele)
        for map_image in tqdm.tqdm(p.imap_unordered(f, groups_of_subregions), total=len(groups_of_subregions)):
            vcf_files.append(map_image)
    logging.debug('Finished processing {} groups.'.format(
        len(groups_of_subregions)))

    logging.debug('Finished generating candidates. Generating final VCF ...')
    merge_vcfs(vcf_files, args.output)
    logging.info('Generated final VCF file at {}.'.format(args.output))

    for v in vcf_files:
        os.remove(v)
        os.remove(v + '.gz')
        os.remove(v + '.gz.tbi')


if __name__ == '__main__':
    main()
