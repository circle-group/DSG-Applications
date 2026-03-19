# AGC U-ResNet

## Installation

Install requirements:

```bash
python -m pip install -r requirements.txt
```


## Data

Download the human segmentation dataset of "Convolutional Neural Networks on Surfaces via Seamless Toric Covers" by Maron et. al. 2017 from here (link by the original authors): https://www.dropbox.com/sh/cnyccu3vtuhq1ii/AADgGIN6rKbvWzv0Sh-Kr417a?dl=0

Unzip it in to the data subdirectory, the meshes and segmentation should then be stored in `data/sig17_seg_benchmark`.

Run the preprocessing script to normalize and subsample the meshes:
```bash
python data_setup.py
```


## Running the model

There are two config files available (located in the configs subdirectory). The model can be trained using the `train_sig17.py` script, it takes as parameters the config file, an output directory (to save model checkpoints and logs) a device.

```bash
python train_sig17.py --config configs/sig17_hks.yaml --save output/sig17_hks1 --device cuda:0
```