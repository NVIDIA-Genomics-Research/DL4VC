# Step By Step Guideline
The step by step instructions in this section can be used to train your own DL4VC based variant caller
from scratch, or run inference using our pre-trained model.

* [Infer on custom dataset](#inference)
* [Train your own model](#training)
* [VCFEval evaluation](#vcfeval)

## Inference
A wrapper script has been created to make it easy to run inference on a new BAM file. The example below
uses the pre-processed [HG002 evaluation dataset](Data.md#evaluation) from the precisionFDA Truth challenge and
a model pre-trained on the HG001 training dataset from precisionFDA.

The pre-trained model can be downloaded from https://dl4vc.s3.us-east-2.amazonaws.com/checkpoint.pth.tar

Example command for calling variants -

```
  ./call_variants.sh -i HG002-NA24385-50x.sort.bam \
      -b HG002_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_CHROM1-22_v3.2.2_highconf.bed \
      -m checkpoint.pth.tar \
      -o vc_inference \
      -r hs37d5.fa
```

To get details about options, run -

```
  ./call_variants.sh -h
```

A pre-generated inferred HG002 VCF is available here:

https://dl4vc.s3.us-east-2.amazonaws.com/bcf_hg002_results-sorted-thres-join.vcf.gz

https://dl4vc.s3.us-east-2.amazonaws.com/bcf_hg002_results-sorted-thres-join.vcf.gz.tbi

## Training
### Step 1: Generate candidates

The fist step is to generate candidate locations for variants. This step performs a basic pileup based
analysis of the aligned reads at every loci in the specified region of genome to determine if it is
a potential variant location. This step can be considered as a high recall, low precision variant calling
pipeline.

The command lines below use the pre-processed [HG001 training dataset](Data.md#training) from precisionFDA 
challenge to highlight usage. However, the same steps would apply for any custom BAM file as well.

```
time python tools/candidate_generator.py \
    --input HG001-NA12878-50x.sort.bam \
    --output HG001-NA12878-50x-candidates.vcf \
    --snp_min_freq 0.075 \
    --indel_min_freq 0.02 \
    --keep_multialleles >& candidate_generator.log
```

Pre-generated candidate file for HG001 is available here:

https://dl4vc.s3.us-east-2.amazonaws.com/HG001-NA12878-50x-candidates.vcf.gz

https://dl4vc.s3.us-east-2.amazonaws.com/HG001-NA12878-50x-candidates.vcf.gz.csi

## Step 2: Generate training dataset
Once candidate variant locations are found, we need to prepare ground truth labels. The variant candidate VCF
file is intersected with a known turth set for the BAM file. The `bcftools` intersect command produces separate
files for true positives, false positives and false negative. These files are then used to generate examples for
training.

```
bcftools isec \
    -p HG001-NA12878-50x \
    NA12878_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_ALLCHROM_v3.2.2_highconf.bcf \
    HG001-NA12878-50x-candidates.vcf.gz
```

Once the relevant VCF files are generated the training examples are encoded into numpy arrays
through an offline process. In a nutshell, the `convert_bam_single_reads` tool paralelly processes
chunks of the genome and generates a pileup of aligned reads around each proposed variant location and
dumps the tensor into an HDF filesystem.

```
time python tools/convert_bam_single_reads.py \
    --input HG001-NA12878-50x.sort.bam \
    --tp_vcf HG001-NA12878-50x/0003.vcf \
    --tp_full_vcf HG001-NA12878-50x/0002.vcf \
    --fp_vcf HG001-NA12878-50x/0001.vcf \
    --fasta-input hs37d5.fa \
    --output HG001-NA12878-50x-bcf.hdf \
    --max-reads 200 \
    --num-processes 80 \
    --locations-process-step 100000 \
    --max-insert-length 10 \
    --max-insert-length-variant 50 \
    --save-q-scores \
    --save-strand >& training_data.log
```

- Pre-generated HDF files for HG001 training is available here:

https://dl4vc.s3.us-east-2.amazonaws.com/HG001-NA12878-50x-bcf.hdf

## Step 3: Run training
After the encoding of examples is done, it's off to training! By default the active learning component
of the training pipeline is turned on, so easy examples are automatically downsampled after each new epoch.

A wrapper script has been written to simplify the training setup process. An example command line is presented
below.

```
  ./train_variant_caller.sh \
      -g 1 \
      -e 5 \
      --train-batch-size 80 \
      --test-batch-size 200 \
      --train-hdf HG001-NA12878-50x-bcf.hdf \
      --test-hdf HG001-NA12878-50x-bcf.hdf \
      --out-vcf model_eval.vcf \
      --sample-vcf HG001-NA12878-50x/0003.vcf \
      --out-model checkpoint.pth >& training.log
```

To get details about options, run -

```
  ./train_variant_caller.sh -h
```

NOTE - The core training scripty (`main.py`) has a lot of configurability though command line options. Our best settings
for those options are hard coded in the wrapper script. Please feel free to poke around and adjust them based on your needs
(especially options around number of GPUs to train on, etc).

A pre-trained model on HG001 dataset is available here:

https://dl4vc.s3.us-east-2.amazonaws.com/checkpoint.pth.tar

## VCFEval 
We have found the following `vcfeval` command line useful in measuring accuracy.

```
time rtg vcfeval \
    --baseline=HG002_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_CHROM1-22_v3.2.2_highconf.bcf \
    --bed-regions=HG002_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_CHROM1-22_v3.2.2_highconf.bed \
    --calls=model_test_sorted_thres-join.vcf.gz \
    --output=vcfeval_results \
    --no-gzip \
    -t hs37d5.fa \
    --evaluation-regions=HG002_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_CHROM1-22_v3.2.2_highconf.bed \
    --ref-overlap
```
