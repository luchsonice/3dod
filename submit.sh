#!/bin/sh
#BSUB -J omni3d_example_run
#BSUB -o hpc_logs/%J.out
#BSUB -e hpc_logs/%J.err
#BSUB -n 4
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=shared' ###:mode=exclusive_process
#BSUB -W 4:00
#BSUB -R 'rusage[mem=5GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B

source /work3/s194369/3dod_hpc_env/bin/activate
export PYTHONPATH=/work3/s194369/3dod

# Run evaluation
CUDA_VISIBLE_DEVICES=0 python tools/eval_boxes.py --eval-only \
    --config-file configs/BoxNet.yaml \
    PLOT.EVAL AP \
    PLOT.MODE2D PRED \
    PLOT.PROPOSAL_FUNC propose \
    MODEL.WEIGHTS output/Baseline_sgd/model_final.pth \
    OUTPUT_DIR output/propose

# Run baseline
#CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#  --config-file configs/Base_Omni3D.yaml \
#  OUTPUT_DIR output/Baseline_trial
