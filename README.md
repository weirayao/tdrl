# Causal Disentanglement for Time Series
Experiment results are showcased in Jupyter Notebooks in `/tests` folder. Each notebook contains the scripts for analysis and visualization for one specific experiment.

Run the scripts in `scripts/run.sh` to generate results for experiment.

Further details are documented within the code.

### Requirements
To install it, create a conda environment with `Python>=3.7` and follow the instructions below. Note, that the current implementation of LEAP requires a GPU.
```
conda create -n lily python=3.7
cd lily
pip install -e .
```

### Datasets

- Synthetic data: `python LiLY/tools/gen_dataset.py `
- Modified Cartpole: `python LiLY/tools/extract_new.py`
- CMU Mocap databse: http://mocap.cs.cmu.edu/
