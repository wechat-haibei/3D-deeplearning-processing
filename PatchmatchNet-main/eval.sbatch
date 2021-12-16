#!/usr/bin/env bash
#SBATCH -J patch
#SBATCH --comment=WRF
#SBATCH -n 8
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -p kshctest
#SBATCH -o %j
#SBATCH -e %j
module rm compiler/rocm/2.9
module load compiler/rocm/3.3
#module load apps/python/3.8.10  
#`module load apps/anaconda3/4.9.2
module load mathlib/opencv/3.1.0/gcc
#module load apps/PyTorch/1.5.0a0/hpcx-2.4.1-gcc-7.3.1-rocm3.3
conda create -n patchmatch python=3.7
source activate patchmatch
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
pip uninstall datasets
#virtualenv -p python3 --system-site-packages venv
#source venv/bin/activate
#module load apps/python/3.8.10  
#conda install --yes --file requirements.txt

# test on DTU's evaluation set
DTU_TESTING="/home/dtu/"
# CUSTOM_TESTING
# python eval.py --dataset=dtu_yao_eval --batch_size=1 --n_views 5 \
# --patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
# --patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
# --patchmatch_interval_scale 0.005 0.0125 0.025 \
# --testpath=$DTU_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres 0.8 \
# --outdir=./outputs --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@

# -------------------------------------------------------------------------------------
# test on eth3d benchmark
ETH3d_TESTING="/home/eth3d_high_res_test/"
# python eval_eth.py --dataset=eth3d --split train --batch_size=1 --n_views 7 \
# --patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
# --patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
# --patchmatch_interval_scale 0.005 0.0125 0.025 \
# --testpath=$ETH3d_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.6 \
# --outdir ./outputs_eth --loadckpt $CKPT_FILE $@

# -------------------------------------------------------------------------------------
# test on tanks & temples
TANK_TESTING="/home/TankandTemples/"
# python eval_tank.py --dataset=tanks --split intermediate --batch_size=1 --n_views 7 \
# --patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
# --patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
# --patchmatch_interval_scale 0.005 0.0125 0.025 \
# --testpath=$TANK_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 \
# --outdir ./outputs_tanks --loadckpt $CKPT_FILE $@

# -------------------------------------------------------------------------------------
# test on your custom dataset
CUSTOM_TESTING="./data/scan2"
CKPT_FILE="./checkpoints/model_000007.ckpt"
python3 eval_custom.py --dataset=custom --batch_size=1 --n_views 5 \
--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
--patchmatch_interval_scale 0.005 0.0125 0.025 \
--testpath=$CUSTOM_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres 0.8 \
--outdir ./outputs_custom --loadckpt $CKPT_FILE $@
source deactivate
