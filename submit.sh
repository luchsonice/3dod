#!/bin/sh
#BSUB -J omni3d_example_run
#BSUB -o hpc_logs/%J.out
#BSUB -e hpc_logs/%J.err
#BSUB -n 4
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 12:00
#BSUB -R 'rusage[mem=5GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B

source /work3/s194235/python310/bin/activate

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file configs/Base_Omni3D.yaml \
  OUTPUT_DIR output/Baseline_trial
