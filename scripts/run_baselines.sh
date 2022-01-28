#/bin/bash
# Fixed
python train_baselines.py \
    --exp fixed_ts_bvae \
    --seed 770

python train_baselines.py \
    --exp fixed_ts_svae \
    --seed 770

python train_fixed_pcl.py \
    --exp fixed_ts_pcl \
    --seed 770

# Change

python train_baselines.py \
    --exp change_ts_bvae \
    --seed 770

python train_baselines.py \
    --exp change_ts_svae \
    --seed 770

python train_baselines.py \
    --exp change_ts_ivae \
    --seed 770

python train_baselines.py \
    --exp change_ts_tcl \
    --seed 770

python train_change_pcl.py \
    --exp change_ts_pcl \
    --seed 770

python train_leap_ns.py \
    --exp change_ts_leap \
    --seed 770

# Modular

python train_baselines.py \
    --exp modular_ts_bvae \
    --seed 770

python train_baselines.py \
    --exp modular_ts_svae \
    --seed 770

python train_baselines.py \
    --exp modular_ts_ivae \
    --seed 770

python train_baselines.py \
    --exp modular_ts_tcl \
    --seed 770

python train_change_pcl.py \
    --exp modular_ts_pcl \
    --seed 770

python train_leap_ns.py \
    --exp modular_ts_leap \
    --seed 770