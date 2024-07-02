#!/bin/sh
#BSUB -J omni3d_example_run
#BSUB -o hpc_logs/%J.out
#BSUB -e hpc_logs/%J.err
#BSUB -n 4
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 0:20
#BSUB -R 'rusage[mem=8GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B

source /work3/s194369/3dod_hpc_env/bin/activate
export PYTHONPATH=/work3/s194369/3dod

# Run evaluation
python tools/eval_boxes.py --eval-only \
    --config-file configs/BoxNet.yaml \
    PLOT.EVAL MABO \
    PLOT.MODE2D GT \
    PLOT.PROPOSAL_FUNC random \
    MODEL.WEIGHTS output/Baseline_sgd/model_final.pth \
    OUTPUT_DIR output/propose_random \
    PLOT.SCORING_FUNC True

# Run baseline
#CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#  --config-file configs/Base_Omni3D.yaml \
#  OUTPUT_DIR output/Baseline_trial
