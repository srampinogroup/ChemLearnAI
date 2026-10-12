# ChemLearnAI

## Project

This project is linked to the article "Tracing the Physical Roots of
Chemical Concepts through Artificial Intelligence" by Erwan Privat,
Giovanni Parolin, Ajay M. S. Rawat, Antonino Polimeno, and Sergio
Rampino.

## Setup

The recommended way to run the code and reproduce the results is to
create a virtual environment using Anaconda, miniconda or mamba using
Python 3.13.1:
```bash
conda create -n chemlearnai python=3.13.1
conda activate chemlearnai
```

Then, clone the repository
```bash
git clone https://github.com/srampinogroup/ChemLearnAI
cd ChemLearnAI
```
and install requirements using `pip`:
```bash
pip install -r requirements.txt
```
To use GPU acceleration, also install cuML with
```bash
pip install -r requirements-gpu.txt
```

Calculations can be run with:
```bash
cd src
./atom_type_prediction.py
./bond_type_prediction.py
```

## Operational datasets

In the paper, we refer to the "reduced" and "full" datasets. This
corresponds to the files in the `data` folder, that is HDF5 files
with respective suffixes `1-6.hdf` and `1-7.hdf`. The so called `1-7`
datasets use QM9 molecules with ID going from 1 to 4237, and thus
also include 236 molecules having 8 C, N, O or F atoms and 9
molecules having 9 of these.
