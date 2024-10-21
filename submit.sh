#!/bin/sh
#BSUB -J weak_cube_w_high
#BSUB -o hpc_logs/%J.out
#BSUB -e hpc_logs/%J.err
#BSUB -n 4
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 20:00
#BSUB -R 'rusage[mem=8GB]'
#BSUB -R 'span[hosts=1]'

source /work3/s194369/3dod/3dod_paper/bin/activate
export PYTHONPATH=/work3/s194369/3dod

# Run
python tools/train_net.py \
    --resume \
    --config-file configs/Omni_combined.yaml \
    OUTPUT_DIR output/weak-cube-2_20_2_001_2 \
    log True \
    loss_functions "['iou', 'z_pseudo_gt_center', 'pose_alignment', 'pose_ground']" \
    MODEL.WEIGHTS output/omni3d-2d-only/model_final.pth \
    MODEL.ROI_CUBE_HEAD.LOSS_W_IOU 2.0 \
    MODEL.ROI_CUBE_HEAD.LOSS_W_NORMAL_VEC 20.0 \
    MODEL.ROI_CUBE_HEAD.LOSS_W_Z 2.0 \
    MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS 0.01 \
    MODEL.ROI_CUBE_HEAD.LOSS_W_POSE 2.0 \