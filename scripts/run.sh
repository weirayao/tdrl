#/bin/bash
# Note: change the dataset path and log path in config files in configs folder
# Stationary dynamics
python train_stationary.py -e stationary_2lag --seed 770
# Changing dynamics
python train_change.py -e change_5 --seed 770
# Modular changes
# This one may need some tuning
python train_modular.py -e modular_5 --seed 770

python train_cartpole -e cartpole --seed 770