# Data Setup
To help get started quickly, we have pre-processed some of the public domain data (more specifically,
the data and truth sets shared during the [Precision FDA Truth challenge](https://precision.fda.gov/challenges/truth)) 
and made them available for instant use. They can be downloaded from our public S3 bucket links mentioned below.

## Common
- hs37d5 reference

https://dl4vc.s3.us-east-2.amazonaws.com/hs37d5.fa

https://dl4vc.s3.us-east-2.amazonaws.com/hs37d5.fa.fai

## Training
- HG001 50x BAM (generated from precisionFDA HG001 FASTQ file)

https://dl4vc.s3.us-east-2.amazonaws.com/HG001-NA12878-50x.sort.bam

https://dl4vc.s3.us-east-2.amazonaws.com/HG001-NA12878-50x.sort.bam.bai

- Truth set split by multi-allele and normalized

https://dl4vc.s3.us-east-2.amazonaws.com/NA12878_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_ALLCHROM_v3.2.2_highconf.bcf

https://dl4vc.s3.us-east-2.amazonaws.com/NA12878_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_ALLCHROM_v3.2.2_highconf.bcf.csi

- High confidence region

https://dl4vc.s3.us-east-2.amazonaws.com/NA12878_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_ALLCHROM_v3.2.2_highconf.bed

## Evaluation
- HG002 50x BAM (generated from pFGA HG002 FASTQ file)

https://dl4vc.s3.us-east-2.amazonaws.com/HG002-NA24385-50x.sort.bam

https://dl4vc.s3.us-east-2.amazonaws.com/HG002-NA24385-50x.sort.bam.bai

- Truth set split by multi-allele and normalized

https://dl4vc.s3.us-east-2.amazonaws.com/HG002_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_CHROM1-22_v3.2.2_highconf.bcf

https://dl4vc.s3.us-east-2.amazonaws.com/NA12878_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_ALLCHROM_v3.2.2_highconf.bcf.csi

- High confidence region

https://dl4vc.s3.us-east-2.amazonaws.com/HG002_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_CHROM1-22_v3.2.2_highconf.bed

- High recall variant candidates in ihgh confidence region

https://dl4vc.s3.us-east-2.amazonaws.com/HG002-NA24385-50x-candidates.vcf.gz

https://dl4vc.s3.us-east-2.amazonaws.com/HG002-NA24385-50x-candidates.vcf.gz.csi

