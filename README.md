# Implicit Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation
This is the code for our paper: Implicit Class-Conditioned Domain Alignment for Unsupervised Domain Adaptation.

The code was adapted from https://github.com/thuml/MDD.
## Installation
Install implicit alignment and its dependencies as a Python module:
```bash
./install.sh
```
The detailed dependencies are specified in [setup.py](./setup.py):

- python3
- torch>=1.4.0
- easydict
- pyyaml
- tensorboardX
- tqdm
- torchvision
- scikit-learn
- numpy

It is highly recommended to use PyTorch >=1.4.0 because we find that the adaptation performance on PyTorch 1.4 is better earlier versions.
It is normal to have empirical results slightly better than what we reported in the paper.

## Data
Download the *Office-31* and *Office-Home* datasets using the following links:
- Office-31: https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view
- Office-Home: https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view

Organize the datasets into the following file structure where `domain_adaptation` is the parent folder of the datasets.
```
domain_adaptation
├── office31
│   ├── amazon
│   ├── dslr
│   └── webcam
└── OfficeHomeDataset_10072016
    ├── Art
    ├── Clipart
    ├── ImageInfo.csv
    ├── imagelist.txt
    ├── Product
    └── Real_World
```
During training, the model loads the data in `domain_adaptation` directory where the label information is specified in the form of `(img, label)` pairs.

As an example, the file names and labels of the imbalanced Office-Home (RS-UT) are defined in [./domain_adaptation/data/office-home-imbalanced](ai/domain_adaptation/data/office-home-imbalanced) (kindly provided by Shuhan Tan in the [COAL](https://arxiv.org/abs/1910.10320) paper, and we thank them for providing the data split).


Note that in [setup.py](./setup.py), we define an entry point `implicit_alignment`, which will be used to run our code.

## Training
### Training with `Makefile`
Makefile is the most convenient way to train a model with pre-specified arguments.
To execute a makefile command:
```bash
cd ai/domain_adaptation/makefiles
make office31.A2W.implicit
```
The global `Makefile` is located at [ai/domain_adaptation/makefiles](./ai/domain_adaptation/makefiles/Makefile), and all instructions and sub-makefiles are included in the same directory.
Please make sure to **change** the `datapath := /your/data/path` to your own data directory in [ai/domain_adaptation/makefiles/Makefile](./ai/domain_adaptation/makefiles/Makefile).

`Makefile` makes it easy to document different command lines options and organize different model training procedures in different sub-makefiles.


### Training with command line arguments

To train the model with command line arguments, one needs to specify the `--dataset_dir` to the `domain_adaptation` data directory that we created.
All `argparse` commands are defined in [./domain_adaptation/main.py](ai/domain_adaptation/main.py).
During training, the model stores its training log in the `tensorboard` directory.

Please find the scripts for different models and datasets in the following makefiles [ai/domain_adaptation/makefiles](./ai/domain_adaptation/makefiles/Makefile):

- [Office-31 Makefile](./ai/domain_adaptation/makefiles/Makefile_office31)
- [Office-Home Makefile](./ai/domain_adaptation/makefiles/Makefile_office_home_standard)
- [Office-Home-Imbalanced (RS-UT) Makefile](./ai/domain_adaptation/makefiles/Makefile_officehome_imbalanced)

For example, to run Rw->Pr on the imbalanced Office-Home (RS-UT) with implicit alignment:
```bash
implicit_alignment \
    --datasets_dir [yourdatadirectory] \
    --optimizer_config ../config/sgd_0.001.yml \
    --dataset Office-Home \
    --class_num 65 \
    --src_address ../data/office-home-imbalanced/Real_World_RS.txt \
    --tgt_address ../data/office-home-imbalanced/Product_UT.txt \
    --name MDD.baseline \
    --train_steps 50000 \
    --seed 10 \
    --eval_interval 50 \
    --machine $(UNAME_N) \
    --tensorboard_dir $(tensorboardpath) \
    --batch_size 50 \
    --mask_classifier --mask_divergence \
    --train_loss total_loss --group_name office_home \
    --bottleneck_dim 2048  \
    --disable_prompt
```



## Main source files
- [./MDD/model/MDD.py](ai/domain_adaptation/models/MDD.py): MDD as domain divergence measure, together with explicit alignment
- [./MDD/preprocess/sampler.py](ai/domain_adaptation/datasets/sampler.py): implicit class-aligned sampling
- [./MDD/trainer/train.py](ai/domain_adaptation/main.py): main file

## Issues
Please create an [issue](https://github.com/xiangdal/implicit_alignment/issues) if you have any problem, bug, or feature request with this code. Thank you!
