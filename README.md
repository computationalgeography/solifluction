# Solifluction

**Numerical simulation of solifluction processes**

Solifluction is a numerical simulation package based on the finite difference method (FDM) and the [LUE](https://lue.computationalgeography.org) library.
It models solifluction as a multiphase process involving muddy soil movement and ice sheet expansion or dissipation.
The model solves the mass conservation, momentum conservation, and heat transfer equations to simulate the coupled dynamics of soil and ice.

## Features
- Finite difference method (FDM) solver
- Multiphase modeling: muddy soil and ice sheets
- Mass, momentum, and heat conservation equations
- Built using [LUE](https://lue.computationalgeography.org)

## Dependencies

Before running the project, you will need to install the following dependencies:

- **Python** (version >= 3.10)
- **LUE** (version 0.3.9)
- **NumPy** (version >= 2.2.3)

<!-- ## Installation -->
<!-- TODO: Add installation instructions here -->
<!-- Example: requirements, build steps, how to install -->

<!-- ## Usage -->
<!-- TODO: Add usage examples here if needed -->
<!-- Example: how to run a basic simulation -->

<!-- ## Links
- [Homepage](https://lue.computationalgeography.org)
- [Documentation](https://lue.computationalgeography.org/doc)
- [Publications](https://lue.computationalgeography.org/publication)
- [R&D Team](https://www.computationalgeography.org)

## Community
- [![Chat with users on Matrix](https://img.shields.io/badge/chat-on%20Matrix-%230098D4)](https://matrix.to/#/#lue:matrix.org)
- [![Chat with developers on Matrix](https://img.shields.io/badge/chat-on%20Matrix-%230098D4)](https://matrix.to/#/#lue-dev:matrix.org)

## Citation
If you use this software, please cite it:
[![Latest release](https://zenodo.org/badge/DOI/10.5281/zenodo.5535685.svg)](https://doi.org/10.5281/zenodo.5535685) -->

## Create Environment and Install Dependencies

The recommended way to set up the environment is with **Conda**.
A ready-to-use environment file is provided in `environment/configuration/conda_environment.yml`.

```bash
cd solifluction

# Create the environment
conda env create -f environment/configuration/conda_environment.yml

# Activate the environment
conda activate soli3d

```

## Run

Activate the environment:

```bash
conda activate soli3d
```

On some systems, you may need to provide a path to libtcmalloc to run LUE:

```bash
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
```

Set the path to the package directory:

```bash
export PYTHONPATH=/full/path/to/source/package:$PYTHONPATH
```

Run the simulation with a parameter file and specify the number of threads:

```bash
python ../source/script/run_simulation.py --hpx:threads=<nr_threads> <path/to/param.txt>
```

## Test

PYTHONPATH=path/to/source/package python -m unittest source.test.model_test -v

## License
This project is licensed under the [MIT License](LICENSE).

---
