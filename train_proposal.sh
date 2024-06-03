#!/bin/sh
#BSUB -J omni3d_example_run
#BSUB -o hpc_logs/%J.out
#BSUB -e hpc_logs/%J.err
#BSUB -n 4
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 6:00
#BSUB -R 'rusage[mem=16GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B

source /work3/s194369/3dod_hpc_env/bin/activate
export PYTHONPATH=/work3/s194369/3dod

# Run evaluation
python tools/train_score.py \
    --config-file configs/ScoreNet.yaml \
    OUTPUT_DIR output/ScoreNet
