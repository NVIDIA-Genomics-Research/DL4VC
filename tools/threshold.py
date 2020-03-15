# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

import numpy as np
import argparse
from sklearn.metrics import precision_recall_curve

def get_optimal_threshold(labels, scores):
    print('optimal thresholds for %d labels %d scores' % (len(labels), len(scores)))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
	# Loop over thresholds, and choose best F1:
    best_i, best_f1 = -1, -1
    find_03,find_05,find_07 = False, False, False
    for i,t in enumerate(thresholds):
        f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        if f1 > best_f1:
            best_i = i
            best_f1 = f1
        if t >= 0.3 and not find_03:
            print('\tthreshold %f: F1 %f (prec: %f; recall: %f)' % (t, f1, precision[i], recall[i]))
            find_03 = True
        if t >= 0.5 and not find_05:
            print('\tthreshold %f: F1 %f (prec: %f; recall: %f)' % (t, f1, precision[i], recall[i]))
            find_05 = True
        if t >= 0.7 and not find_07:
            print('\tthreshold %f: F1 %f (prec: %f; recall: %f)' % (t, f1, precision[i], recall[i]))
            find_07 = True

    best_precision = precision[best_i]
    best_recall = recall[best_i]
    best_threshold = thresholds[best_i]
    print('Best threshold %f: F1 %f (prec: %f; recall: %f)' % (best_threshold, best_f1, best_precision, best_recall))
    return(best_threshold)

# Canonicalize items like
# ['10' '133821805' 'TTA' 'TTATA']
# ['10' '134094682' 'GACACACAC' 'GACACAC']
# ['10' '134989089' 'AGATG' 'ATGGATG']
# ['10' '135203914' 'CCA' 'CCACA']
# ['10' '135273975' 'CATAT' 'CATATAT']
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
parser.add_argument('--input_file', type=str, default="", help='input vcf file')
parser.add_argument('--truth_file', type=str, default="", help='truth set')
args = parser.parse_args()
print(args)

input_file = args.input_file
truth_file = args.truth_file

##Input data
print('reading input')
inputs = open(input_file, 'r').readlines()
inputs = [x.split('\t') for x in inputs if x[0]!='#']

print('extracting information from input')
#get variant - position, ref and alt - to compare with truth set
input_variants = np.array(['\t'.join(x[:2] + x[3:5]) for x in inputs])
#SNP - if length of ref and var are both 1
is_snp = np.logical_and([len(x[3])==1 for x in inputs], [len(x[4])==1 for x in inputs])
is_insert = np.logical_and([len(x[3])==1 for x in inputs], [len(x[4])>1 for x in inputs])
is_delete = np.logical_and([len(x[3])>1 for x in inputs], [len(x[4])==1 for x in inputs])
is_long_indel = np.logical_or([len(x[3])>=3 for x in inputs], [len(x[4])>=3 for x in inputs])
#is_long_del = np.logical_or([len(x[3])>=3 for x in inputs], [len(x[4])>=3 for x in inputs])
#threshold = 1-NV score
thresh = np.array([1-float(x[2].split(';')[1].split('=')[1]) for x in inputs])
#OV score
ov = np.array([float(x[2].split(';')[3].split('=')[1]) for x in inputs])
#label for homozygosity
gt = np.array([x[-1].strip('\n') for x in inputs])
#gt[gt=='1:50']=None
gt = np.isin(gt,['GT:1|1', 'GT:1/1'])

print('splitting snps and indel inputs')
#for snps
input_snps = input_variants[is_snp]
thresh_snps = thresh[is_snp]
ov_snps = ov[is_snp]
gt_snps = gt[is_snp]
#for indels
input_indels = input_variants[~is_snp]
thresh_indels = thresh[~is_snp]
ov_indels = ov[~is_snp]
gt_indels = gt[~is_snp]
# long indels
input_long_indels = input_variants[is_long_indel]
thresh_long_indels = thresh[is_long_indel]
ov_long_indels = ov[is_long_indel]
gt_long_indels = gt[is_long_indel]
# long deletes
input_long_dels = input_variants[is_long_indel & ~is_insert]
thresh_long_dels = thresh[is_long_indel & ~is_insert]
ov_long_dels = ov[is_long_indel & ~is_insert]
gt_long_dels = gt[is_long_indel & ~is_insert]
# Inserts only
input_inserts = input_variants[is_insert & ~is_long_indel]
thresh_inserts = thresh[is_insert & ~is_long_indel]
ov_inserts = ov[is_insert & ~is_long_indel]
gt_inserts = gt[is_insert & ~is_long_indel]
# Deletes only
input_deletes = input_variants[is_delete & ~is_long_indel]
thresh_deletes = thresh[is_delete & ~is_long_indel]
ov_deletes = ov[is_delete & ~is_long_indel]
gt_deletes = gt[is_delete & ~is_long_indel]

#How many true variants are not included in candidate set?
print('reading truth set')
truths = np.genfromtxt(truth_file, comments='#', dtype='str', delimiter='\t', usecols=(0,1,3,4))

# Clearn up truths -- need to get canonical for Indel "ATT > ATTTT" [created because of splitting]
print(truths[:10])
for i,x in enumerate(truths):
    if len(x[2]) > 1 and len(x[3]) > 1:
        can_ref, can_var = canonicalize_bases(x[2], x[3])
        truths[i][2] = can_ref
        truths[i][3] = can_var

print('extracting information from truth set')
is_truth_snp = np.logical_and([len(x[2])==1 for x in truths], [len(x[3])==1 for x in truths])
is_truth_long_indels = np.logical_or([len(x[2])>=3 for x in truths], [len(x[3])>=3 for x in truths])
is_truth_deletes = np.logical_and([len(x[2])>1 for x in truths], [len(x[3])==1 for x in truths])
is_truth_inserts = np.logical_and([len(x[2])==1 for x in truths], [len(x[3])>1 for x in truths])
truth_variants = np.array(['\t'.join(x) for x in truths])

print('splitting snps and indel truth')
truth_snps = truth_variants[is_truth_snp]
truth_indels = truth_variants[~is_truth_snp]
truth_long_indels = truth_variants[is_truth_long_indels]
truth_long_dels = truth_variants[is_truth_long_indels & ~is_truth_inserts]
truth_inserts = truth_variants[is_truth_inserts & ~is_truth_long_indels]
truth_deletes = truth_variants[is_truth_deletes & ~is_truth_long_indels]

#Get number of SNPs not in input (candidate) dataset
is_truth_snp_called = np.in1d(truth_snps, input_snps)
#FN = number of true SNPs not called
base_fn_snps = sum(~is_truth_snp_called)
##NOTE - these base FN numbers, and the thresholds that use them, only make sense if the truth file is limited to trust regions. Ignore otherwise.

#max possible recall for SNPs = FP/number of true SNPs(FP+FN)
max_recall_snps = sum(is_truth_snp_called)/len(truth_snps)
print('base FN number for SNPs: ' + str(base_fn_snps) + '. Max recall = ' + str(max_recall_snps))

#Indels
is_truth_indel_called = np.in1d(truth_indels, input_indels)
base_fn_indels = sum(~is_truth_indel_called)
max_recall_indels = sum(is_truth_indel_called)/len(truth_indels)
print('base FN number for indels: ' + str(base_fn_indels) + '. Max recall = ' + str(max_recall_indels))

#Calculate thresholds for SNPs

#is input variant found in truth set?
is_snp_true = np.in1d(input_snps, truth_snps)

#variant call threshold
print('-------------------')
print('variant call threshold for SNPs')
opt_thresh_snp_1 = get_optimal_threshold(is_snp_true, thresh_snps)
print('variant call threshold for SNPs with all FNs included - assigning them a score of -1')
opt_thresh_snp_2 = get_optimal_threshold(np.concatenate([is_snp_true, np.repeat(1, base_fn_snps)]), np.concatenate([thresh_snps, np.repeat(-1, base_fn_snps)]))

#homozygosity threshold
print('-------------------')
print('homozygosity threshold for SNPs')
opt_ov_snp_1 = get_optimal_threshold(gt_snps, ov_snps)
print('homozygosity threshold for SNPs using only variants called with variant-call threshold')
opt_ov_snp_2 = get_optimal_threshold(gt_snps[thresh_snps>=opt_thresh_snp_1], ov_snps[thresh_snps>=opt_thresh_snp_1])

#Calculate thresholds for indels

#is input variant found in truth set?
is_indel_true = np.in1d(input_indels, truth_indels)
is_long_indel_true = np.in1d(input_long_indels, truth_long_indels)
is_long_del_true = np.in1d(input_long_dels, truth_long_dels)
is_delete_true = np.in1d(input_deletes, truth_deletes)
is_insert_true = np.in1d(input_inserts, truth_inserts)

#variant call threshold
print('-------------------')
print('variant call threshold for indels')
opt_thresh_indel_1 = get_optimal_threshold(is_indel_true, thresh_indels)
print('variant call threshold for indels with base FNs included')
opt_thresh_indel_2 = get_optimal_threshold(np.concatenate([is_indel_true, np.repeat(1, base_fn_indels)]), np.concatenate([thresh_indels, np.repeat(-1, base_fn_indels)]))

#homozygosity threshold
print('-------------------')
print('homozygosity threshold for indels')
opt_ov_indel_1 = get_optimal_threshold(gt_indels, ov_indels)
print('homozygosity threshold for indels using only variants called with variant-call threshold')
opt_ov_indel_2 = get_optimal_threshold(gt_indels[thresh_indels>=opt_thresh_indel_1], ov_indels[thresh_indels>=opt_thresh_indel_1])


#variant call threshold
print('-------------------')
print('variant call threshold for *long* DELS')
opt_thresh_indel_1 = get_optimal_threshold(is_long_del_true, thresh_long_dels)

#homozygosity threshold
print('-------------------')
print('homozygosity threshold for *long* DELS')
opt_ov_indel_1 = get_optimal_threshold(gt_long_dels, ov_long_dels)

#variant call threshold
print('-------------------')
print('variant call threshold for *long* indels')
opt_thresh_indel_1 = get_optimal_threshold(is_long_indel_true, thresh_long_indels)

#homozygosity threshold
print('-------------------')
print('homozygosity threshold for *long* indels')
opt_ov_indel_1 = get_optimal_threshold(gt_long_indels, ov_long_indels)

#variant call threshold
print('-------------------')
print('variant call threshold for *deletes*')
opt_thresh_indel_1 = get_optimal_threshold(is_delete_true, thresh_deletes)

#homozygosity threshold
print('-------------------')
print('homozygosity threshold for *deletes*')
opt_ov_indel_1 = get_optimal_threshold(gt_deletes, ov_deletes)
#print('homozygosity threshold for indels using only variants called with variant-call threshold')
#opt_ov_indel_2 = get_optimal_threshold(gt_indels[thresh_indels>=opt_thresh_indel_1], ov_indels[thresh_indels>=opt_thresh_indel_1])

#variant call threshold
print('-------------------')
print('variant call threshold for *inserts*')
opt_thresh_indel_1 = get_optimal_threshold(is_insert_true, thresh_inserts)

#homozygosity threshold
print('-------------------')
print('homozygosity threshold for *inserts*')
opt_ov_indel_1 = get_optimal_threshold(gt_inserts, ov_inserts)
