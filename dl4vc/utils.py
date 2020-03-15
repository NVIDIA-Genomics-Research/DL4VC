# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

import os
import pysam
import sys
import tempfile
import torch
from dl4vc.base_enum import *

# HACK: Helper function to convert binary/unicode to numpy string
# Otherwise, we get errors building batches
# TODO: Better to have HDF save in a directly usable format!
def bin_to_string(bin_text):
    return bytes.decode(bin_text)


# Parse tab-separated VCF -- return dictionary of results useful for training, debug and error understanding
var_type_enum = {'homo': 2, 'homozygous': 2, 'hetero': 1, 'heterozygous': 1, 'none': 0, 'novar': 0}
def parse_vcf(vcf_rec):
    rec = vcf_rec.strip().split('\t')
    #print(rec)
    #print('%s -> %s' % (rec[3], rec[4]))
    ref_bases = rec[3]
    var_bases = rec[4]

    # Store all results we may be interested in
    res = {}
    # Infer mutation type, and build several labels useful for softmax prediction
    if len(ref_bases) == 1 and len(var_bases) == 1 and ref_bases in real_bases_set and var_bases in real_bases_set:
        res['is_snp'] = True
        res['var_mode'] = mutation_type_enum['SNP']
        res['ref_base'] = base_enum[ref_bases]
        res['var_base'] = base_enum[var_bases]
    elif len(ref_bases) > len(var_bases):
        res['is_snp'] = False
        res['var_mode'] = mutation_type_enum['Delete']
        # NOTE: Deletes are noted as GAAA --> G -- so 'ref base' here is 'G'
        res['ref_base'] = base_enum[ref_bases[0]]
        res['var_base'] = base_enum['-']
    elif len(ref_bases) < len(var_bases):
        res['is_snp'] = False
        res['var_mode'] = mutation_type_enum['Insert']
        # NOTE: Inserts are noted as G --> GAAAA -- so 'ref base' here is 'G'
        res['ref_base'] = base_enum[ref_bases[0]]
        # HACK: Re-use 'noinsert' token here (for enum/softmax convenience)
        res['var_base'] = base_enum['noinsert']
    else:
        # In dataset, this never happens...
        print('Unknown mutation detected!!! %s' % str(rec))
        res['is_snp'] = False

    # Statistical properties passed in VCF
    candidate_stats = rec[7].split(';')
    stats = {n:v for (n,v) in [k.split('=') for k in candidate_stats]}
    #print(stats)
    res['allele_freq'] = float(stats['AF'])
    res['coverage'] = int(stats['DP'])

    # Infer if this is a variant, hetero or homozygous -- default not a variant
    res['var_type'] = var_type_enum['none']
    if len(rec) > 10:
        # Explicity look for something like GT:1/1 | GT:0|1
        gt,var = rec[10].split(':')
        if gt == 'GT' and len(var) == 3 and var[1] in set(['/','|']):
            if var[0] == '1' and var[2] == '1':
                res['var_type'] = var_type_enum['homo']
            elif var[0] == '0' and var[2] == '1':
                res['var_type'] = var_type_enum['hetero']
            elif var[0] == '1' and var[2] == '0':
                res['var_type'] = var_type_enum['hetero']

    return res

def write_test_vcfs(bin_results, vcf_records, sample_vcf, threshold=0.2, filename='out.vcf'):
    bcf_in = pysam.VariantFile(sample_vcf)  # auto-detect input format
    temp_filename = next(tempfile._get_candidate_names()) + 'tmp.vcf'
    bcf_out = pysam.VariantFile(temp_filename, 'w', header=bcf_in.header)
    bcf_out.close()

    with open(temp_filename, 'a') as f:
        for i, b in tqdm.tqdm(enumerate(bin_results), total=len(bin_results)):
            if b > threshold:
                f.write(vcf_records[i])

    # Sort results.
    # NOTE: This could be impractical for huge file
    sort_vcf_cmd = """cat {} | awk '$1 ~ /^#/ {{print $0;next}} {{print $0 | "sort -k1,1 -k2,2n"}}' > {}""".format(temp_filename, filename)
    subprocess.check_output(sort_vcf_cmd, shell=True)

# Same logic as above, but stream VCFs rather than loop over a huge file [these can be 14M records]
# hack_score --> ignore threshold, and place score in the INFO column
def stream_test_vcfs(bin_results, inputs_stream, sample_vcf, threshold=0.5, filename='out_stream.vcf', hack_score=True):
    print('Beginning to save VCF to %s' % filename)
    # Read sample record, and copy header
    bcf_in = pysam.VariantFile(sample_vcf)
    dir = os.path.dirname(filename)
    temp_filename = next(tempfile._get_candidate_names()) + 'tmp.vcf'
    temp_filename = os.path.join(dir, temp_filename)
    bcf_out = pysam.VariantFile(temp_filename, 'w', header=bcf_in.header)
    bcf_out.close()

    print('Created header, now streaming over %d records -- threshold at %.5f' % (len(bin_results), threshold))
    print('Appending to temp file: %s' % temp_filename)

    # *stream* the ouput -- this could take a while, so show progress
    recs_saved = 0
    dataloader_iterator = iter(inputs_stream)
    if hack_score:
        print('Hacking input to save score -- ignore threshold %.5f and save everything' % threshold)
        threshold = -1
    with open(temp_filename, 'a') as f:
        # Buffer by batch...
        batch = []
        batch_counter = -2
        for i, b in tqdm.tqdm(enumerate(bin_results), total=len(bin_results)):
            # Read next record from stream
            # HACK -- assume that records are unsorted, same order, and matching output results
            # TODO: Verify this? Or save entire VCF [quite a lot of space for 14M records]
            if batch_counter < 0 or batch_counter >= len(batch['vcfrec']):
                batch = next(dataloader_iterator)
                batch_counter = 0
                #print(batch['vcfrec'])
            # NOTE: Will raise StopIteration exception if we ask for too many items -- which is goood. Order was off

            # Save record if it meets the threshold
            if b > threshold:
                item = batch['vcfrec'][batch_counter]
                if hack_score:
                    items = item.split('\t')
                    assert items[2] == '.', 'DANGER -- would replace non-empty INFO -- check the hack'
                    items[2] = ('%.8f' % b)
                    item = '\t'.join(items)

                #print(item)
                f.write(item + '\n')
                recs_saved += 1

            batch_counter += 1

    print('Finished creating VCF with %d saved records' % recs_saved)

    # Do not sort? Output expected to be huge -- important is to save results.
    print('Sorting VCF. If this dies, unsorted VCF still exists in %s' % temp_filename)

# Initialize VCF with header -- also generate random filename to avoid collisions
def initialize_vcf_tempfile(sample_vcf, filename, epoch=1):
    print('Beginning to save VCF to %s' % filename)
    # Read sample record, and copy header
    bcf_in = pysam.VariantFile(sample_vcf)
    dir = os.path.dirname(filename)
    basefile = os.path.basename(filename)
    temp_filename = ('epoch%s_' % epoch) + (basefile if filename else next(tempfile._get_candidate_names()))
    temp_filename = os.path.join(dir, temp_filename)
    print('Creating header for file: %s' % temp_filename)
    bcf_out = pysam.VariantFile(temp_filename, 'w', header=bcf_in.header)
    bcf_out.close()
    print('Appending to file: %s' % temp_filename)
    return temp_filename

# Open a VCF in append mode, and add a batch of records
# Splice in the model score into INFO column
def append_vcf_records(vcf_file, bin_results, vt_results, vcf_records, hack_score=True, debug=False):
    if debug:
        print('Appending %d records to file %s' % (len(results), vcf_file))
    assert len(bin_results) == len(vcf_records), "mis-match between results and VCF to save"
    assert os.path.isfile(vcf_file), "VCF file for append does not exist -- need to initialize it"
    with open(vcf_file, 'a') as f:
        for i, b in enumerate(zip(bin_results, vt_results)):
            item = vcf_records[i].strip()
            if hack_score:
                items = item.split('\t')
                assert items[2] == '.', 'DANGER -- would replace non-empty INFO -- check the hack'
                vt = b[1]
                # HACK: Save predictions BP -- binary, NV,HV,OV -- no var, het var, homozygous var
                results_txt = ('BP=%.8f;NV=%.8f;HV=%.8f;OV=%.8f' % (b[0], vt[0], vt[1], vt[2]))
                items[2] = results_txt
                item = '\t'.join(items)
            f.write(item + '\n')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    epoch = state['epoch']
    basename = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    torch.save(state, "{}_epoch{}{}".format(basename, epoch, ext))
    if is_best:
        torch.save(state, "{}_best{}".format(basename, ext))

