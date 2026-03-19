# MeshFlow

## Installation

Install requirements:

```bash
python -m pip install -r requirements.txt
```

Sample points from the eigenfunctions before training or evaluating a model,
```bash
python data_setup.py
```

## Running

```bash
python train.py \
  --config configs/meshflow.yml \
  --mesh_file data/bunny.obj \
  --eigfn_file data/bunny_eigfn009.npy \
  --device cuda
```

The logs and checkpoints are stored in `output/`.

Given a checkpoint, it is possible to evaluate it, and visualize the results easily:
```bash
python eval.py \
  --checkpoint_path checkpoint.pt \
  --mesh_path data/bunny.obj \
  --eigenfn_path data/eignfn_bunny10.npy \
  --device cuda \
  --visualize
```