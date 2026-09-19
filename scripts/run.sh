#!/bin/bash

mkdir -p logs && setsid nohup bash -c '
  export PYTHONPATH=. WANDB_MODE=offline

  CKPT_DIR=model_iclr bash scripts/train.sh --all && \
  CKPT_DIR=model_iclr_capacity bash scripts/train.sh --model vae-3d-spatio-spectral --dataset IIRS --loss physics --seed 69 --set vae_3d_base_ch=30 && \
  CKPT_DIR=model_iclr_capacity bash scripts/train.sh --model vae-1d-pixelwise --dataset IIRS --loss physics --seed 69 --set "vae_1d_hidden_dims=[1788,894,447]" && \
  CKPT_DIR=model_iclr_capacity bash scripts/train.sh --model vae-3d-spatio-spectral --dataset AVIRIS --loss physics --seed 69 --set vae_3d_base_ch=32 && \
  CKPT_DIR=model_iclr_capacity bash scripts/train.sh --model vae-1d-pixelwise --dataset AVIRIS --loss physics --seed 69 --set "vae_1d_hidden_dims=[1764,882,441]" && \
  CKPT_DIR=model_iclr OUT_DIR=results_iclr bash scripts/inference.sh && \
  mkdir -p results_iclr/capacity && \
  python inference/inference.py --model vae-3d-spatio-spectral --dataset IIRS --loss physics --seed 69 --select sam --ckpt-dir model_iclr_capacity --set vae_3d_base_ch=30 --out-json results_iclr/capacity/IIRS__vae-3d_capacity.json && \
  python inference/inference.py --model vae-1d-pixelwise --dataset IIRS --loss physics --seed 69 --select sam --ckpt-dir model_iclr_capacity --set "vae_1d_hidden_dims=[1788,894,447]" --out-json results_iclr/capacity/IIRS__vae-1d_capacity.json && \
  python inference/inference.py --model vae-3d-spatio-spectral --dataset AVIRIS --loss physics --seed 69 --select sam --ckpt-dir model_iclr_capacity --set vae_3d_base_ch=32 --out-json results_iclr/capacity/AVIRIS__vae-3d_capacity.json && \
  python inference/inference.py --model vae-1d-pixelwise --dataset AVIRIS --loss physics --seed 69 --select sam --ckpt-dir model_iclr_capacity --set "vae_1d_hidden_dims=[1764,882,441]" --out-json results_iclr/capacity/AVIRIS__vae-1d_capacity.json
' > logs/iclr_run_$(date +%F_%H%M).log 2>&1 &

echo "PID $!  log: logs/iclr_run_*.log"