# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

"""
Calculate precision and recall for variant caller given called and ground-truth VCF files.

Example command:
    
    python called_variant_metrics.py --truth_variants=truth.vcf.gz --called_variants=called.vcf.gz

VCFs need to be compressed with bgzip and indexed with bcftools.
"""
from __future__ import print_function
from __future__ import division

import argparse
import os
import shutil
import tempfile
import uuid

from pysam import VariantFile
from pysam import bcftools

def count_variant_types(vcf, chrom, start, end):
    substitutions = 0
    insertions = 0
    deletions = 0

    for rec in vcf:
        reference_allele = rec.ref
        position = rec.pos
        if chrom is not None:
            if (rec.contig != chrom) or (rec.pos < start) or (rec.pos > end):
                continue
        alternative_alleles = rec.alts
        if (len(reference_allele) == 1) and (len(alternative_alleles[0]) == 1):
            substitutions += 1
        elif (len(reference_allele) == 1) and (len(alternative_alleles[0]) > 1):
            insertions += 1
        elif (len(reference_allele) > 1) and (len(alternative_alleles[0]) == 1):
            deletions += 1
        else:
            print('Unknown alelle: {} -> {}'.format(reference_allele, alternative_alleles))

    return (substitutions, insertions, deletions)
        

def main():
    parser = argparse.ArgumentParser(description="Analyse VCF files")

    parser.add_argument('--truth_variants',
                        help='truth set')
    parser.add_argument('--called_variants',
                        help='called set')
    parser.add_argument('--region',
                        type=str,
                        default=None,
                        help='region in form chrom:start:end. If not set defaults to whole genome')

    args = parser.parse_args()


    if args.region is not None:
        chrom, start, end = args.region.split(':')
        start = int(start)
        end = int(end)
    else:
        chrom = None
        start = None
        end = None
    
    isec_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    bcftools.isec("-p", isec_dir, args.truth_variants, args.called_variants)

    # Load the VCFs:
    false_negative_vcf_path = os.path.join(isec_dir, '0000.vcf')
    false_positive_vcf_path = os.path.join(isec_dir, '0001.vcf')
    true_positive_vcf_path = os.path.join(isec_dir, '0002.vcf')

    fn_snps, fn_insertions, fn_deletions = count_variant_types(VariantFile(false_negative_vcf_path, 'r'),chrom, start, end)
    fp_snps, fp_insertions, fp_deletions = count_variant_types(VariantFile(false_positive_vcf_path, 'r'),chrom, start, end)
    tp_snps, tp_insertions, tp_deletions = count_variant_types(VariantFile(true_positive_vcf_path, 'r'),chrom, start, end)


    fn_indels = fn_insertions + fn_deletions
    tp_indels = tp_insertions + tp_deletions
    fp_indels = fp_insertions + fp_deletions

    snp_recall = (tp_snps / (tp_snps + fn_snps))
    snp_precision = (tp_snps / (tp_snps + fp_snps))

    indel_recall = (tp_indels / (tp_indels + fn_indels))
    indel_precision = (tp_indels / (tp_indels + fp_indels))


    print('SNP Recall = {}'.format(snp_recall))
    print('SNP Precision = {}'.format(snp_precision))
    print('Indel Recall = {}'.format(indel_recall))
    print('Indel Precision = {}'.format(indel_precision))
    
    print('Cleaning up')
    shutil.rmtree(isec_dir)
    


if __name__ == '__main__':
    main()
