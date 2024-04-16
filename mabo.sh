#!/bin/sh
#BSUB -J mabo_run
#BSUB -o hpc_logs/%J.out
#BSUB -e hpc_logs/%J.err
#BSUB -n 4
#BSUB -W 01:30
#BSUB -R 'rusage[mem=5GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B

source /work3/s194369/python310/bin/activate

python tools/eval_boxes.py --eval-only --config-file configs/BoxNet.yaml PLOT.EVAL MABO
