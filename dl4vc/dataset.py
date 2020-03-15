# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

import time
import functools
import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
# Needs to be included *before* h5py to inherit multithreading support
import torch.multiprocessing
import multiprocessing
import h5py
from dl4vc.base_enum import *
from dl4vc.utils import bin_to_string, parse_vcf

# Give a rate, draw from normal distribution with this as the mean. Clamp result
def double_sample_rate(rate,min_rate=0.0,max_rate=0.5,debug=False,stdev=1.):
    implied_std = max(0.001, rate / 2.0)
    r = np.clip(np.random.normal(rate, implied_std*stdev, 1), min_rate, max_rate)
    if debug:
        print('Rate %.5f normal-sampled to %.5f' % (rate, r[0]))
    return r[0]

# Naive introduction of random "read errors" into the single reads
# flip bases, introduce deletes, and delete inserts
SIMPLE_NOISE_FLIP_RATE=0.002
SIMPLE_NOISE_DEL_RATE=0.002
SIMPLE_NOISE_RM_INSERT_RATE=0.0
SIMPLE_NOISE_UNKNOWN_RATE=0.02
SIMPLE_NOISE_UNKNOWN_SKIP_POS=[] # don't skip any positions in reads
SIMPLE_NOISE_DOUBLE_SAMPLE=True
def add_simple_reads_noise(reads, flip_rate=SIMPLE_NOISE_FLIP_RATE, delete_rate=SIMPLE_NOISE_DEL_RATE,
    unk_rate=SIMPLE_NOISE_UNKNOWN_RATE, unk_skip_pos=SIMPLE_NOISE_UNKNOWN_SKIP_POS,
    rm_insert_rate=SIMPLE_NOISE_RM_INSERT_RATE, double_sample=SIMPLE_NOISE_DOUBLE_SAMPLE, debug=False):
    if debug:
        print('Attempting reads noise w/ flip_rate %.5f, delete_rate %.5f, rm_insert_rate %.5f, unk_rate %.5f' % (flip_rate, delete_rate, rm_insert_rate, unk_rate))
        print(reads.shape)
    for i in range(reads.shape[1]):
        sr = reads[:,i]
        if debug:
            sr_copy = np.copy(sr)
        # Better variance, by normal-sampling the p(change) for each read?
        # NOTE: Should be a better alernative. (lots) More examples with zero errors, more examples with correlated errors
        if double_sample:
            flip_rate_x = double_sample_rate(rate=flip_rate, debug=debug)
            delete_rate_x = double_sample_rate(rate=delete_rate, debug=debug)
            rm_insert_rate_x = double_sample_rate(rate=rm_insert_rate, debug=debug)
            unk_rate_x = double_sample_rate(rate=unk_rate, debug=debug)
        else:
            flip_rate_x = flip_rate
            delete_rate_x = delete_rate
            rm_insert_rate_x = rm_insert_rate
            unk_rate_x = unk_rate
        num_changes = add_noise_single_read(sr, flip_rate=flip_rate_x, delete_rate=delete_rate_x,
            rm_insert_rate=rm_insert_rate_x, unk_rate=unk_rate_x, unk_skip_pos=unk_skip_pos)
        if debug and num_changes > 0:
            print('\n--------------\n%s\nchanged %d to\n%s' % (str(sr_copy), num_changes, str(sr)))

# ?-only noise to the reference. Does this help prevent over-fitting?
# Do not modify the reference at middle positions.
SIMPLE_REF_UNKNOWN_RATE=0.1
SIMPLE_REF_UNKNOWN_SKIP_POS=[99,100,101,102]
SIMPLE_REF_DOUBLE_SAMPLE=True
# TODO: Flip a coin on *rate* of reference error -- so we have some examples with 0% errors [2-layer model]
def add_simple_ref_noise(ref, flip_rate=0.0, delete_rate=0.0,
    unk_rate=SIMPLE_REF_UNKNOWN_RATE, unk_skip_pos=SIMPLE_REF_UNKNOWN_SKIP_POS, rm_insert_rate=0.0, double_sample=SIMPLE_REF_DOUBLE_SAMPLE, debug=False):
    if debug:
        print('Attempting ref noise w/ flip_rate %.5f, delete_rate %.5f, rm_insert_rate %.5f, ukn_rate %.5f' % (flip_rate, delete_rate, rm_insert_rate, unk_rate))
        print(ref.shape)
    if debug:
        ref_copy = np.copy(ref)
    # Better variance, by normal-sampling the p(change) for each read?
    # NOTE: Should be a better alernative. (lots) More examples with zero errors, more examples with correlated errors
    # NOTE: You want the no-error examples (maybe) since no errors on the validation evaluation...
    if double_sample:
        unk_rate = double_sample_rate(rate=unk_rate, debug=debug)
    num_changes = add_noise_single_read(ref, flip_rate=flip_rate, delete_rate=delete_rate,
        rm_insert_rate=rm_insert_rate, unk_rate=unk_rate, unk_skip_pos=unk_skip_pos)
    if debug and num_changes > 0:
            print('\n--------------\n%s\nchanged %d to\n%s' % (str(ref_copy), num_changes, str(ref)))

# Return ref and var bases int vectors -- padded/cut to max length
# Default == 10+1 -- base + 10x long insert. Technically, can handle much longer deletes...
# EDIT: Expanded read encoding to 50x bases (definition of a short read)
VAR_ENCODE_LEN = 51
def simple_variant_encoding_vectors(vcf_record, insert_limit=VAR_ENCODE_LEN, delete_limit=VAR_ENCODE_LEN, keep_pad=True):
    rec = vcf_record.strip().split('\t')
    x = rec[3]
    y = rec[4]
    if delete_limit > 0:
        x = x[:delete_limit]
    if insert_limit > 0:
        y = y[:insert_limit]
    # Backward-compatible hack to support fixed-size outputs
    ref_bases_vector = np.full((max(delete_limit,len(x)),),base_enum['pad'],dtype='uint8')
    var_bases_vector = np.full((max(insert_limit,len(y)),),base_enum['pad'],dtype='uint8')
    for i in range(len(x)):
        ref_bases_vector[i] = base_enum[x[i]]
    for i in range(len(y)):
        var_bases_vector[i] = base_enum[y[i]]
    # Upon request, clip to non-pad values:
    if not keep_pad:
        if len(np.flatnonzero(ref_bases_vector == base_enum['pad'])) > 0:
            first_pad = np.flatnonzero(ref_bases_vector == base_enum['pad'])[0]
            ref_bases_vector = ref_bases_vector[:first_pad]
        if len(np.flatnonzero(var_bases_vector == base_enum['pad'])) > 0:
            first_pad = np.flatnonzero(var_bases_vector == base_enum['pad'])[0]
            var_bases_vector = var_bases_vector[:first_pad]
    return (ref_bases_vector, var_bases_vector)

# Return mask [of bases] for reference and variant, for specific allele suggested
def get_read_mask_vectors(vcf_record, reference, debug=False):
    length = len(reference)
    assert length==201, print('get_read_mask_vectors assume length 201 reads -- adjust if wrong here')
    rec = vcf_record.strip().split('\t')
    if debug:
        print(rec)
        print('%s -> %s' % (rec[3], rec[4]))
    x = rec[3]
    y = rec[4]
    # Bases sequence for reference, variant
    ref_bases_vector, var_bases_vector = simple_variant_encoding_vectors(vcf_record, delete_limit=0, keep_pad=False)
    ref_mask = np.full((length, ),base_enum['pad'],dtype='uint8')
    var_mask = np.full((length, ),base_enum['pad'],dtype='uint8')
    # Depending on variant type -- simply compute the offset for inserting bases vector into mask
    if len(x) == 1 and len(y) == 1 and x in real_bases_set and y in real_bases_set:
        #return mutation_type_enum['SNP']
        # SNP is simple -- single base at middle position
        # Handle case -- insert at pos 100 -- rewind in reference... but only past '5'
        offset = 100
        while reference[offset] == base_enum['-']:
            offset -= 1
        ref_offset = offset
        var_offset = offset
    elif len(x) > len(y):
        #return mutation_type_enum['Delete']
        # Delete has the form AT -> A -- so both technically start at middle position

        # Handle case -- insert at pos 100 -- rewind in reference... but only past '5'
        offset = 100
        while reference[offset] == base_enum['-']:
            offset -= 1
        ref_offset = offset
        var_offset = offset
        # NOTE: Not that simple -- as there could also be an insert... so *ref* could be [A .... T]
        assert len(y) == 1, "For deletes, expect exactly one base in variant. [%s -> %s]" % (x, y)
    elif len(y) > len(x):
        #return mutation_type_enum['Insert']
        # Inserts are annoying, and tricky.
        # [Perhaps incorrectly -- certainly inconveniently] -- we insert *ending* at middle position
        # To make things worse, inserts are of variable length -- so if there are reads with longer insert...
        # TODO: Fix the data -- move toward insert *starting* at middle pos and we'll be fine
        # NOTE: To do this correctly -- rely on the reference.
        # Pattern will be [base, -, -, -, ...,] <- base will give us location for the insert [in reference]
        # Nice thing is that insert will be continuous --
        # this is sub-optimal for the case of long partial matches... but technically correct here

        assert len(x) == 1, "For inserts, expect exactly one base in reference. [%s -> %s]" % (x, y)
        # NOTE: After alignment fix, inserts should also start from middle position!
        offset = 100
        while reference[offset] == base_enum['-']:
            offset -= 1
        ref_offset = offset
        var_offset = offset
        # Look for reference base in the reference -- then we know our offset!
        #for offset in range(100,100-VAR_ENCODE_LEN-3,-1):
        #    if reference[offset] == ref_bases_vector[0]:
        #        ref_offset = offset
        #        var_offset = offset
        #        break
        #    # only skip past '5' padding in the reference...
        #    assert reference[offset] == base_enum['-'], "Rewinding in the reference -- should only skip '5' padding. Instead %s at pos %d" % (reference[offset], offset)
        assert reference[ref_offset] == ref_bases_vector[0], "Did not find ref base in reference! [%s -> %s]\n%s\n%s" % (x, y, str(ref_bases_vector), str(reference))

    # Ok now we have bases and offsets. All good... except in case of long *reference* -- could need to pad it (correcting for inserts)
    if debug:
        print(ref_bases_vector)
        print(reference[ref_offset:ref_offset+len(ref_bases_vector)])
    # TODO: fix this error -- until then -- turn it into a warning...
    #if reference[ref_offset] != ref_bases_vector[0]:
    #    print("WARNING Did not find (first) ref base in reference! [%s -> %s] %d: %s\n%s\n%s" % (x, y, ref_offset, reference[ref_offset], str(ref_bases_vector), str(reference)))
    #    print("TODO: get location, debug")
    assert reference[ref_offset] == ref_bases_vector[0], "Did not find (first) ref base in reference! [%s -> %s] %d: %s\n%s\n%s" % (x, y, ref_offset, reference[ref_offset], str(ref_bases_vector), str(reference))

    # Add padding for deletes to reference vector
    if len(ref_bases_vector) > 1:
        assert len(y) == 1, "Can not handle long ref & variant at the same time! [%s -> %s]" % (x, y)

        # First, expand the variant -- to match the reference -- explicitly as delete
        var_pad_delete = np.array([base_enum['-']] * (len(ref_bases_vector)-len(var_bases_vector)))
        var_bases_vector = np.concatenate((var_bases_vector, var_pad_delete))

        # Just notice cases where ref needs gap inserts
        if not np.array_equal(ref_bases_vector, reference[ref_offset:ref_offset+len(ref_bases_vector)]):
            if debug:
                print(ref_bases_vector)
                print(reference)

            # Insert gaps into the delete...
            # Basically pad with 5's -- the only option -- iterate thru ref while matching next character or '5' padding
            new_ref_bases_vector = []
            ref_bases_idx = 0
            new_var_bases_vector = []
            for ref_idx in range(ref_offset, len(reference)):
                # Absorb base, add padding element, or we have a mis-match!
                if ref_bases_idx >= len(ref_bases_vector):
                    break
                elif reference[ref_idx] == ref_bases_vector[ref_bases_idx]:
                    new_ref_bases_vector.append(reference[ref_idx])
                    new_var_bases_vector.append(var_bases_vector[ref_bases_idx])
                    ref_bases_idx += 1
                elif reference[ref_idx] == base_enum['-']:
                    # Keep padding -- from spurious insert in another allele
                    new_ref_bases_vector.append(base_enum['-'])
                    new_var_bases_vector.append(base_enum['noinsert'])
                else:
                    assert reference[ref_idx] == ref_bases_vector[ref_bases_idx], "Mis-match inserting pad delete into reference. [%s -> %s]" % (x, y)
            assert ref_bases_idx >= len(ref_bases_vector), 'Finished padding, did not reach end of pad insert %s' % str(new_ref_bases_vector)
            new_ref_bases_vector = np.array(new_ref_bases_vector)
            new_var_bases_vector = np.array(new_var_bases_vector)
            if debug:
                print('expanded bases vector')
                print(ref_bases_vector)
                print(new_ref_bases_vector)
            ref_bases_vector = new_ref_bases_vector
            var_bases_vector = new_var_bases_vector
            assert np.array_equal(ref_bases_vector, reference[ref_offset:ref_offset+len(ref_bases_vector)]), "Error in padding ref vector for deletes"

            # Weird hack -- now that we padded the (longer) ref -- replace 'noinsert' with skip-pad.
            # WHY?? We want to match any reference *as long as they don't delete*
            ref_bases_vector[ref_bases_vector==base_enum['-']] = base_enum['pad']
            # Similarly, we can ignore the padding inside the variant as well (as long as we keep the delete value)
            var_bases_vector[var_bases_vector==base_enum['noinsert']] = base_enum['pad']
            if debug:
                print('Delete padding is great success')
                #time.sleep(2)

    # Add "no insert" to reference, in the case of Insert -- pretty simple
    if len(ref_bases_vector) == 1 and len(var_bases_vector) > 1:
        ref_pad_noinsert = np.array([base_enum['noinsert']] * (len(var_bases_vector)-len(ref_bases_vector)))
        ref_bases_vector = np.concatenate((ref_bases_vector, ref_pad_noinsert))

    assert len(ref_bases_vector) == len(var_bases_vector), 'Need to adjust ref, var vectors for same length! %s -- %s' % (str(ref_bases_vector), str(var_bases_vector))

    # Update mask with offsets...
    ref_len = len(ref_bases_vector)
    ref_mask[ref_offset:ref_offset+ref_len] = ref_bases_vector
    var_len = len(var_bases_vector)
    var_mask[var_offset:var_offset+var_len] = var_bases_vector
    return (ref_mask, var_mask)

# Return middle -- or random -- set of reads for single reads item
# NOTE: That we may only have *total* num_reads < max_reads. Make sure we don't sample zeros for no reason!
# NOTE: Also return the index in original array -- useful if we need to add Q-scores for this data...
TOTAL_SINGLE_READS=1000
def sample_single_reads(reads, max_reads, num_reads=TOTAL_SINGLE_READS, random=True, dynamic_downsample_rate=0., debug=False):
    # If asked for dynamic downsample -- first sample from that distribution [choose a rate normally centered at rate, and clip at 0.]
    if dynamic_downsample_rate > 0.:
        downsample_rate = double_sample_rate(rate=dynamic_downsample_rate,min_rate=0.0,max_rate=0.8,stdev=1.)
        if debug:
            print('%.5f rate (sampled %.5f): %d reads -> %d reads' % (dynamic_downsample_rate, downsample_rate, num_reads, int((1.0 - downsample_rate) * num_reads)))
        sampled_num_reads = int((1.0 - downsample_rate) * num_reads)
    else:
        sampled_num_reads = num_reads
    # NOTE: If we downsample -- and even if we don't -- need to keep track of which reads are not used [will effect allele freq, etc]

    max_reads = min(max_reads, reads.shape[1])
    # HACK -- if we want to downsample (50 real reads, we want 40) -- need to fill in the data with padded zeros.
    # Make sure last row is 0's -- and pass that index
    sample_zero_pad_reads = num_reads - sampled_num_reads
    if max_reads >= num_reads and sample_zero_pad_reads == 0:
        return reads[:,:max_reads], np.array([i for i in range(max_reads)])
    total_nonzero_reads = min(reads.shape[1], num_reads)
    if random:
        # Choose random X of Y reads -- in sorted order
        permutation = np.sort(np.random.choice(total_nonzero_reads, min(max_reads, sampled_num_reads), replace=False))
        # Expand permutation with last row -- which we ensure is zero -- to compensate for down-sampled reads
        if len(permutation) < max_reads:
            permutation = np.concatenate((permutation, np.full(max_reads-len(permutation), reads.shape[1]-1)))
            reads[:,-1] = 0
        return reads[:, permutation], permutation
    else:
        # Choose middle X reads
        single_reads_middle = int(total_nonzero_reads/2)
        single_reads_start = max(0, int(single_reads_middle - max_reads/2))
        permutation = np.array([i for i in range(single_reads_start,single_reads_start+max_reads)])
        return reads[:,permutation], permutation

# Iterate through the read, flip bases as needed (independently) -- return number or change made
# NOTE: Substitute base to "?" as well -- artificial masking, especially on reference. In ref, don't mask center, by default
# TODO: Can this be parallelize with lambda function?
def add_noise_single_read(sr, flip_rate=0.01, delete_rate=0.01, rm_insert_rate=0.01,
    unk_rate=0.01, unk_skip_pos=[]):
    changes = 0
    start = False
    end = False
    for i in range(sr.shape[0]):
        # skip all bases until we start the read
        if not start:
            if sr[i] == base_enum['pad']:
                continue
            elif sr[i] == base_enum['start']:
                start = True
                continue
        # skip all bases if we are at the end, as well
        if end:
            continue
        elif sr[i] == base_enum['pad'] and np.sum(sr[i:]) == 0:
            end = True
            continue
        # If here, we are in the middle of a read -- possible modifications
        # NOTE: This will flip bases in inserts, as well
        if sr[i] in real_bases_set:
            # Flip to random base
            # TODO: Use conditional matrix for A->T, T->G, etc
            if flip_rate > 0 and np.random.random() < flip_rate:
                changes  += 1
                sr[i] = np.random.choice(list(real_bases_set - set([sr[i]])))
            if delete_rate > 0 and np.random.random() < delete_rate:
                changes  += 1
                sr[i] = base_enum['-']
        elif delete_rate > 0 and sr[i] == base_enum['-'] and np.random.random() < delete_rate:
            # Naive "undelete" -- assign a random base
            # NOTE: No check for if this is part of a larger indel, rather than simple unread/deleted base
            changes  += 1
            sr[i] = np.random.choice(list(real_bases_set))
        # Sub bases as "?" at a [possibly much higher] rate. In place for any legal token [real base or -]
        if unk_rate > 0 and (i not in unk_skip_pos) and (sr[i] in real_bases_set or sr[i] == base_enum['-']) and np.random.random() < unk_rate:
            # Substitute explicit "unknown" tag in the sequence
            changes += 1
            sr[i] = base_enum['?']
        # TODO: Add logic for removing/modifying inserts
        # Indels are trickier -- we need info from the reference, to know if this is an insert

        # TODO: If base deleted, randomly assign the reference (un-delete)
    return changes

# Count agreements and disagreements at position 100 [ignore non-bases]
# return (total_count, agree, disagree)
def count_variants_from_single_reads(single_reads, reference, var_type):
    # Handle mutation count types differently!
    # SNPs -- Simply compare bases at 100 position
    # NOTE: Deletes are at the *next* position (pos 101)
    # NOTE: Inserts are 8 = no insert [ref base is 5]
    if var_type == mutation_type_enum['SNP']:
        ref_base = reference[100]
        reads_bases = single_reads[100]
    elif var_type == mutation_type_enum['Delete']:
        ref_base = reference[101]
        reads_bases = single_reads[101]
    elif var_type == mutation_type_enum['Insert']:
        ref_base = base_enum['noinsert']
        reads_bases = single_reads[101]
    unique_bases, counts = np.unique(reads_bases, return_counts=True)
    bases_map = {b:c for (b,c) in zip(unique_bases, counts)}
    # Count agreements -- and any real base that disagrees [ignore padding, start and end tokens]
    agreements = bases_map.get(ref_base,0)
    disagreements = 0
    for key in (real_base_keys_set - set([ref_base])):
        disagreements += bases_map.get(key,0)
    return (agreements+disagreements, agreements, disagreements)

# Simple reader, to break up VCFs by SNP, Insert, Delete
def vcf_type(vcf_rec):
    rec = vcf_rec.strip().split('\t')
    #print(rec)
    #print('%s -> %s' % (rec[3], rec[4]))
    x = rec[3]
    y = rec[4]
    if len(x) == 1 and len(y) == 1 and x in real_bases_set and y in real_bases_set:
        return mutation_type_enum['SNP']
    elif len(x) > len(y):
        return mutation_type_enum['Delete']
    elif len(y) > len(x):
        return mutation_type_enum['Insert']
    else:
        print('Unknown mutation detected!!! %s' % str(rec))
        return mutation_type_enum['error']

# Helper furnction, so we can iterate over large HDF quickly
# In this case to count locations that match the holdout chromosome
def process_location(start_loc, block_size=100, path='', holdout_set=[]):
    found_cases = []
    #print((start_loc, block_size, holdout_set))
    with h5py.File(path,'r') as hdfile:
        #print('Iterating thru %d locations to update_holdout_chromosomes on %s' % (len(hdfile['data']), holdout_set))
        data = hdfile['data'][start_loc:start_loc+block_size]
        for i in range(len(data)):
            #chrom = str(np.random.randint(22))
            chrom = bin_to_string(data[i]['vcfrec'])
            chrom = chrom[:chrom.index('\t')]
            if chrom in holdout_set:
                #print(chrom)
                found_cases.append(i+start_loc)
    return found_cases

# How many single reads to consider? Sample from the middle of the pileup. Or, randomly?
MAX_READS = 100
#MAX_READS = 50
#MAX_READS = 30
STORE_MAX_READS = 200
HDF_MULTI_BLOCK_SIZE = 20000 # records to load at once with multi-process
class ContextDatasetFromNumpy(Dataset):
    """
    Wrapper loading from npy file.
    """
    # During the loading process, reduce number of reads, for a smaller model.
    # TODO: Control this from the command line
    def __init__(self, file_name, args, max_reads = MAX_READS, store_max_reads = min(TOTAL_SINGLE_READS, STORE_MAX_READS),
        reads_random_sample=True, augment_single_reads=True, augment_refernce=True, holdout_chromosomes = [],
        reads_dynamic_downsample_rate=0., reads_dynamic_downsample_prob=0., debug=False):
        self.debug=debug
        # Reads per input data
        self.max_reads = max_reads
        # Reads stored on disk -- of which we take random (or middle) X above
        self.store_max_reads = store_max_reads
        # Do we sample single reads randomly -- or from the middle? [random = less coverage in the middle]
        self.reads_random_sample = reads_random_sample
        self.augment_single_reads = augment_single_reads
        if self.augment_single_reads:
            print('Augmenting single reads')
            print('Attempting reads noise w/ flip_rate %.5f, delete_rate %.5f, rm_insert_rate %.5f, unk_rate %.5f' % (SIMPLE_NOISE_FLIP_RATE, SIMPLE_NOISE_DEL_RATE, SIMPLE_NOISE_RM_INSERT_RATE, SIMPLE_NOISE_UNKNOWN_RATE))
        self.augment_refernce = augment_refernce
        if self.augment_refernce:
            print('Augmenting reference sequence')
            print('Attempting ref noise w/ flip_rate %.5f, delete_rate %.5f, rm_insert_rate %.5f, ukn_rate %.5f' % (0.0, 0.0, 0.0, SIMPLE_REF_UNKNOWN_RATE))
        self.use_q_scores = args.model_use_q_scores
        self.use_strands = args.model_use_strands
        # Might be better to keep AF from candidate set -- until multiallele is fixed?
        self.keep_candidate_af = args.aux_keep_candidate_af
        # Optional parameters to train with further downsampling
        # Ex: 50% of the time... remove about 30% of reads (noise added to downsample rate)
        self.reads_dynamic_downsample_rate = reads_dynamic_downsample_rate
        self.reads_dynamic_downsample_prob = reads_dynamic_downsample_prob
        # H5PY only
        self.h5_path = file_name
        self.file = None
        self._h5_gen = None
        self.data = None

        # Initialize close/easy examples data
        self.close_examples = np.full((self.__len__()), False, dtype=bool)
        print('Initialized empty close_examples of size %s' % str(self.close_examples.shape))
        # Also track blacklist [bad examples] -- probably inefficient, but simple logic
        self.blacklist = np.full((self.__len__()), False, dtype=bool)
        print('Initialized empty blacklist of size %s' % str(self.blacklist.shape))

        # Also -- upon request -- calculate chromosome holdout list -- (to skip, save for eval, thresholding)
        self.holdout_chromosomes = set([str(c) for c in holdout_chromosomes])
        self.chromosome_holdout = np.full((self.__len__()), False, dtype=bool)
        if len(self.holdout_chromosomes) > 0:
            print('Computing chromosome holdout on %s' % self.holdout_chromosomes)
            self.update_holdout_chromosomes(self.holdout_chromosomes)
            print('\titems in holdout set %d' % (np.sum(self.chromosome_holdout == True)))

        print('Finished pre-loading data -- will dynamically load from HDF')

    # Iterate thru the dataset, match locations to chromosomes we care about
    def update_holdout_chromosomes(self, holdout_set):
        images = []
        time_zero = time.time()
        block_size = HDF_MULTI_BLOCK_SIZE
        loc_chunks = list(range(0,self.__len__(),block_size))
        num_processes = multiprocessing.cpu_count()
        print('Process with %d processors' % num_processes)
        with multiprocessing.Pool(num_processes) as pool:
            f = functools.partial(process_location, block_size=block_size, path=self.h5_path, holdout_set=holdout_set)
            for map_image in tqdm.tqdm(pool.imap_unordered(f, loc_chunks), total=len(loc_chunks)):
                images.extend(map_image)

        print('Saving num output locations:')
        print(len(images))
        # Not necessary -- just for debug consistency -- remove to save time if big
        images.sort()
        print(images[:10])
        print('Took %.2fs to process %d loc with %d processes' % (time.time()-time_zero, self.__len__(), num_processes))
        for idx in tqdm.tqdm(images, total=len(images)):
            self.chromosome_holdout[idx] = True

    def update_close_example(self, idx, value):
        self.close_examples[idx] = value

    def count_close_examples(self):
        return np.sum(self.close_examples == True)

    def update_blacklist(self, idx, value):
        self.blacklist[idx] = value

    def __len__(self):
        with h5py.File(self.h5_path,'r') as file:
            return len(file['data'])

    # Attempt getitem that supporst multi-worker
    def __getitem__(self, idx):
        if self._h5_gen is None:
            self._h5_gen = self._get_generator()
            next(self._h5_gen)
        return self._h5_gen.send(idx)

    def _get_generator(self):
        with h5py.File(self.h5_path, 'r') as hdfile:
            print('loading H5Py file %s' % self.h5_path)
            record = hdfile['data']
            #assert record.dtype == DATA_TYPE, 'Incorrect datatype loading from npy %s' % str(self.data.dtype)
            idx = yield
            while True:
                if self.debug:
                    print('------')
                    print(record[idx]['name'])
                    print(idx)
                    print(record[idx]['num_reads'])
                single_reads = record[idx]['single_reads'].astype(dtype='uint8')
                if self.debug:
                    print(single_reads.shape)
                    print(np.sum(single_reads, axis=1))
                # A. Transpose from num_reads X read_len to read_len x max_store_reads
                single_reads_middle = int(max(record[idx]['num_reads'], self.store_max_reads) / 2)
                single_reads_start = max(0, int(single_reads_middle - self.store_max_reads/2))
                if self.debug:
                    print((single_reads_start, single_reads_middle, single_reads_start+self.store_max_reads))
                single_reads = np.transpose(single_reads[single_reads_start:single_reads_start+self.store_max_reads,:])

                if self.debug:
                    print(single_reads.shape)
                    print(np.sum(single_reads, axis=0))
                    print('sampling')
                # B. Down_sample to max_reads
                # Dynamically down-sample for regularization -- choose P of downsampling (0.5?) and downsample rate (0.3?) to which noise is added
                sampled_reads, single_reads_permutation = sample_single_reads(single_reads, max_reads=self.max_reads,
                    num_reads=record[idx]['num_reads'], random=self.reads_random_sample,
                    dynamic_downsample_rate=(self.reads_dynamic_downsample_rate if np.random.random() < self.reads_dynamic_downsample_prob else 0.), debug=self.debug)
                not_single_reads_permutation = np.setxor1d(np.indices((single_reads.shape[1],)), single_reads_permutation)
                if self.debug:
                    print(sampled_reads.shape)
                    print(np.sum(sampled_reads, axis=0))
                    print(sampled_reads[:,0])
                    print(sampled_reads[:,10])
                    print(sampled_reads[:,21])
                    print('Single reads permuations:')
                    print(single_reads_permutation)
                    print(not_single_reads_permutation)
                # C. Augment, upon request
                if self.augment_single_reads:
                    add_simple_reads_noise(sampled_reads)
                reference = record[idx]['ref_bases'].astype(dtype='uint8')
                if self.augment_refernce:
                    add_simple_ref_noise(reference)
                # Q-scores, if requested
                if self.use_q_scores:
                    # TODO: Gracefully handle no "q-scores" in data?
                    q_scores = record[idx]["q-scores"].astype('uint8')
                    q_scores = np.transpose(q_scores[single_reads_start:single_reads_start+self.store_max_reads,:])
                    #q_scores *= Q_SCORE_SCALE_FACTOR
                    # Apply same sampling as single reads above...
                    sample_q_scores = q_scores[:,single_reads_permutation]
                    # TODO -- verify that single reads & q-scores have same shape, and otherwise match!
                    if self.debug:
                        print(sample_q_scores.shape)
                        print(np.sum(sample_q_scores, axis=0))
                        print(sample_q_scores[:,0])
                        print(sample_q_scores[:,10])
                        print(sample_q_scores[:,21])
                else:
                    # If not using Q-Scores -- return zeros
                    # TODO: Avoid passing all this data?
                    sample_q_scores = np.zeros(sampled_reads.shape, dtype='uint8')
                # Strand, if requested (similar to Q scores)
                if self.use_strands:
                    strands = record[idx]["strand"].astype('uint8')
                    strands = np.transpose(strands[single_reads_start:single_reads_start+self.store_max_reads,:])
                    sample_strands = strands[:,single_reads_permutation]
                    # TODO -- verify that strand has same shape and order as reads & q-scores
                    if self.debug:
                        print(sample_strands.shape)
                        print(np.sum(sample_strands, axis=0))
                        print(sample_strands[:,0])
                        print(sample_strands[:,10])
                        print(sample_strands[:,21])
                else:
                    # If not using strand, return zeros.
                    # TODO: Avoid passing empty data [but we should always use strand if available]
                    sample_strands = np.zeros(sampled_reads.shape, dtype='uint8')
                # Finally -- other information we may use in training, or useful for breaking down errors
                vcf_record = bin_to_string(record[idx]['vcfrec'])
                #print(vcf_record)
                vcf_info = parse_vcf(vcf_record)
                if self.debug:
                    print('VCFInfo enums:')
                    print(vcf_info)
                # i. is SNP?
                is_snp = vcf_info['is_snp']
                mutation_var_mode = vcf_info['var_mode']
                # ii. variant type -- homozygous, heterozygous, not a variant
                variant_type = vcf_info['var_type']
                # iii. allele frequency -- % of reads comprising the "variation"
                allele_freq = vcf_info['allele_freq']
                # iv. coverage -- simply count # reads covering the centre location
                vcf_coverage = vcf_info['coverage']
                # v. variant base -- enum for each type of SNP, otherwise "delete" or "insert" for indels
                mutation_var_base = vcf_info['var_base']
                # vi. reference base -- what is the reference base for our modification
                mutation_ref_base = vcf_info['ref_base']

                # Get variant, agreement, coverage numbers -- directly from the reads
                # NOTE: AF is not correct in the case of multi-allele
                # TODO: Actually match exact allele suggested -- with a mask, offsets, etc [may not be cheap]
                cover_count, agree_count, variant_count = count_variants_from_single_reads(single_reads=single_reads[:,single_reads_permutation], reference=reference, var_type=mutation_var_mode)
                if self.debug:
                    print(vcf_record.strip().split('\t'))
                    print('bases and reference at reads chosen')
                    print(single_reads[99, single_reads_permutation])
                    print(single_reads[100, single_reads_permutation])
                    print(single_reads[101, single_reads_permutation])
                    print('----')
                    print(reference[99:102])
                    print('-------')
                    print('Count, agree, disagree')
                    print([cover_count, agree_count, variant_count])

                # Over-write coverage and allele count (assume all variants are *our* variant)
                # TODO: Add option to use this count or not
                if cover_count > 0:
                    vcf_coverage = cover_count
                    # Options to keep stated AF -- reason for this is:
                    # * Calculations above not correct [multi-allele]
                    # * To distinguish alleles (by original AF)
                    if not self.keep_candidate_af:
                        allele_freq = variant_count / cover_count

                # Get vector encoding of the ref and alt bases "A --> TC" -- for multiallele to pass to model
                ref_bases_vector, var_bases_vector = simple_variant_encoding_vectors(vcf_record)
                if self.debug:
                    print('Bases encoding -- AF ref/var up to 10 bases:')
                    print(vcf_record.strip().split('\t'))
                    print(ref_bases_vector)
                    print(var_bases_vector)
                    print('----')

                # Get vector masks for reference and variant -- single base for SNPs and longer for Indels.
                # Why? Add ref/variant "highlighting" directly at the reads level -- feed in the ConvNet
                # Note -- more efficient to pass vector here (in Numpy) and do matching/expansion in the GPU

                # NOTE: Getting errors here on new datasets. Do not let this break training
                blacklist = False
                try:
                    ref_read_mask_vector, var_read_mask_vector = get_read_mask_vectors(vcf_record, reference=reference, debug=self.debug)
                except AssertionError as e:
                    print('WARNING -- read/ref mask generation fail!')
                    print('will continue, but try to debug this case...')
                    print('%d: %s' % (idx, bin_to_string(record[idx]['vcfrec'])))
                    print(e)
                    print(vcf_record)
                    print(reference)
                    try:
                        ref_read_mask_vector, var_read_mask_vector = get_read_mask_vectors(vcf_record, reference=reference, debug=True)
                    except AssertionError as e2:
                        pass
                    length = len(reference)
                    ref_read_mask_vector = np.full((length, ),base_enum['pad'],dtype='uint8')
                    var_read_mask_vector = np.full((length, ),base_enum['pad'],dtype='uint8')
                    # Add bad location to the blacklist...
                    print('adding element %s to the blacklist' % str(idx))
                    blacklist = True
                    print('-----------------')

                if self.debug:
                    print('ref and variant *read* masks -- check if read matches reference or variant...')
                    print(ref_read_mask_vector)
                    print(var_read_mask_vector)
                    # Count how many reads actually match the ref and variant...

                idx = yield {"name": bin_to_string(record[idx]['name']), "label": record[idx]['label'], "ref": reference,
                            "reads": sampled_reads, "vcfrec": bin_to_string(record[idx]['vcfrec']),
                            "q-scores": sample_q_scores, "strands": sample_strands,
                            "num_reads": record[idx]['num_reads'],
                            'is_snp': is_snp, 'var_type':variant_type, 'allele_freq':allele_freq, 'coverage':vcf_coverage,
                            'var_base_enum': mutation_var_base, 'var_ref_enum': mutation_ref_base,
                            'var_base_vector': var_bases_vector, 'var_ref_vector': ref_bases_vector,
                            'var_mask': var_read_mask_vector, 'ref_mask': ref_read_mask_vector, 'idx':idx,
                            'blacklist':blacklist}

# Dataset sampler -- allowing up to downsample (skip over) easy example the model already gets right
class AdjustableDataSampler(Sampler):
    def __init__(self, dataset, args={}, reverse_holdout=False, shuffle=True):
        self.total_len = len(dataset)
        self.close_keep_per = args.close_examples_sample_rate
        # Reference to the close examples table -- which we update during training
        self.close_examples = dataset.close_examples
        self.blacklist = dataset.blacklist
        self.chromosome_holdout = dataset.chromosome_holdout
        self.reverse_holdout = reverse_holdout
        self.shuffle = shuffle
        print('loaded dataset %d with %d close examples' % (self.total_len, np.sum(self.close_examples == True)))
        self.epochs = 0
        self.epoch_len = self.total_len

    def __iter__(self):
        self.epochs += 1
        print('calling AdjustableDataSampler.__iter__ (prepare new random sample) for epoch %d' % self.epochs)
        print('%d close examples' % np.sum(self.close_examples == True))

        # How to down-sample close cases?
        # A. choose all "not close" examples -- save those
        #not_close_examples = np.nonzero(self.close_examples == False)[0]
        # Choose all "not close" examples... not in the blacklist
        print('blacklist examples to skip: %d' % (np.sum(self.blacklist == True)))
        if self.reverse_holdout:
            print('holdout chromosome examples to use! %d' % (np.sum(self.chromosome_holdout == True)))
            # NOTE: In the reverse case... we *only* want examples in the holdout.
            # NOTE: This is for eval -- do *not* set close==True for any examples here...
            not_close_examples = np.nonzero(~self.close_examples * ~self.blacklist * self.chromosome_holdout)[0]
        else:
            print('holdout chromosome examples to skip: %d' % (np.sum(self.chromosome_holdout == True)))
            # NOTE: We *assume* that blacklist and holdout do *NOT* end up in the set of "not close" examples...
            not_close_examples = np.nonzero(~self.close_examples * ~self.blacklist * ~self.chromosome_holdout)[0]
        print('non-close examples to save: %s' % str(not_close_examples.shape))
        # B. sample multi-nomial distribution -- with all close examples same weight
        if not self.reverse_holdout:
            num_close = np.sum(self.close_examples == True)
            num_close_to_sample = int(self.close_keep_per * num_close)
            print('keeping %d/%d (%.2f%%) of close examples' % (num_close_to_sample, num_close, (100.*num_close_to_sample/max(num_close,1))))

            # NOTE -- can't use torch.multinomial since it's unreasonably slow! [Tried to get non-colliding sample in one shot, not yet fixed]
            # https://github.com/pytorch/pytorch/issues/11931

            # For now, use numpy instead
            close_indices = np.argsort(self.close_examples)[-num_close:]
            assert np.sum(self.close_examples[close_indices] == True) == num_close, "screwed up sampling of close cases"
            close_indices_sample = np.random.permutation(close_indices)[:num_close_to_sample]

            # C. merge, and return shuffled iteration
            sample_list = np.concatenate((not_close_examples, close_indices_sample))
        else:
            print('Do NOT add close examples (there should be none) with self.reverse_holdout')
            sample_list = not_close_examples
        print('merged list to %d examples (shuffled)' % len(sample_list))
        print('-----------------------------')
        self.epoch_len = len(sample_list)

        # TODO -- use Torch RNG state?
        # TODO: Use the blacklist -- to exclude (efficiently) from permutation...
        if self.shuffle:
            return(iter(np.random.permutation(sample_list)))
        else:
            print('Returning iterator with no shuffle -- if needed in test set?')
            return(iter(sample_list))

    def __len__(self):
        return self.epoch_len

