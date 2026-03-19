# Geodesic Centroidal Voronoi Tessellation

## Installation

Install the dependencies:
```bash
python -m pip install -r requirements.txt
```

## Running the code

```bash
python comparison.py <mesh-name> <distribution> --device <device>
```

Where `<mesh-name>` is the name of the mesh file in the `data/` folder (without the `.obj` extension), `<distribution>` is either `uniform` or `cluster`.

Example:
```bash
python comparison.py spot uniform --device cuda
```

It is also possible to do a single run and visualise the Voronoi tesselation, with the train script:
```bash
python train.py <mesh-path> <optim> --device <device>
```

Where `<mesh-path>` is the path to the mesh, and `<optim>` is the optimizer (either `gd` or `lbfgs`)?

Example:
```bash
python train.py data/spot.obj lbfgs --device cpu
```

## Visualising

To visualize the results once the  script is done, run:
```bash
python viz_comparison.py <mesh-name> <distribution>
```

Example:
```bash
python viz_comparison.py spot uniform
```
