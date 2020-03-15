# Deep Learning For Variant Calling (DL4VC)

DL4VC is an advanced deep learning based variant caller for short-read based germline variant calling.
It proposes a deep averaging network (DAN) designed specifically for variant calling. The model takes as
input a tensor encoding aligned reads in a proposed variant region, a variant proposal,
and outputs a softmax over three cateogies: `{no variant, heterozygous variant, homozygous variant}`. 
This model takes into account the independence of each short input read sequence by transforming individual 
reads through a series of 1D convolutional layers, limiting the communication between individual reads 
to averaging and concatenating operations, before passing them into a fully connected network.

Our purpose-built model achieves state of the art results on the precisionFDA germline variant calling dataset
(compared post competition).

To facilitate future work, we release our code, trained models and pre-processed public domain datasets through this repo.

## Accuracy Highlights
PrecisionFDA Truth Challenge results vs DL4VC

| Variant Caller | Type | F1 | Recall | Precision |
| -------------- | ----- | -- | ---- | ------ |
| rpoplin-dv42 | Overall <br> Indels <br> SNPS | 0.998597 <br> 0.989802 <br> 0.999587 | 0.998275 <br> 0.987883 <br> 0.999447 | 0.998919 <br> 0.991728 <br> **0.999728** |
| dgrover-gatk | Overall <br> Indels <br> SNPS | 0.998905 <br> **0.994008** <br> 0.999456 | 0.999005 <br> 0.993455 <br> 0.999631 | 0.998804 <br> **0.994561** <br> 0.999282 |
| astatham-gatk | Overall <br> Indels <br> SNPS | 0.995679 <br> 0.993422 <br> 0.995934 | 0.992122 <br> 0.992401 <br> 0.992091 | 0.999261 <br> 0.994446 <br> **0.999807** |
| bgallagher-sentieon | Overall <br> Indels <br> SNPS | 0.998626 <br> 0.992676 <br> 0.999296 | 0.998910 <br> 0.992140 <br> **0.999673** | 0.998342 <br> 0.993213 <br> 0.998919 |
| DL4VC | Overall <br> Indels <br> SNPS | **0.998924** <br> 0.992949 <br> **0.999596** | **0.999076** <br> **0.994708** <br> 0.999566 | 0.998772 <br> 0.991196 <br> 0.999625 |

## Feature Highlights

* PyTorch-based model training and inference
* 1D convolutional model with learned embeddings of bases
* Variant proposal encodings
* Down-sampling of easy examples to speed up training by 5x

| Section | Description |
| -----   | ----------- |
| [Installation](#installation) | System and code setup instructions |
| [Data](#data) | Pre-processed datasets from precisionFDA to reproduce DL4VC results |
| [Step by step guidelines](#step-by-step-guideline) | Instructions to train and run inference with DL4VC |

## Installation
The installation has been tested on bare metal as well as conda virtual environments. We recommend conda environments 
because they simplify the installation of non-python dependencies.

Core dependencies - 
1. BCF Tools
2. Tabix
3. Python 3.5+ environment
4. vcfeval (optional, only needed for comparing with other VCFs)

Setup
1. git clone https://github.com/clara-genomics/DL4VC.git
2. cd DL4VC
3. pip install -r requirements.txt

## Data
Please follow the dataset instructions in the [Dataset Readme](docs/Data.md) to download pre-processed
data and model checkpoints from our experiment. The results mentioned in the [Accuracy Highlights](#accuracy-highlights)
section can be reproduced using the same datasets.

## Step By Step Guideline
We have created a detailed step by step guideline to run both training and inference using
the DL4VC pipeline in our [Step by Step Guide](docs/Step-by-step.md).
