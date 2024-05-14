#!/bin/sh
#BSUB -J mabo_run
#BSUB -o hpc_logs/%J.out
#BSUB -e hpc_logs/%J.err
#BSUB -n 4
#BSUB -q gpuv100
#BSUB -gpu 'num=1:mode=exclusive_process
#BSUB -W 08:30
#BSUB -R 'rusage[mem=5GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B


source /work3/s194369/python310/bin/activate
export PYTHONPATH=/work3/s194369/3dod
# single
# python tools/eval_boxes.py --config-file configs/BoxNet.yaml OUTPUT_DIR output/xy PLOT.EVAL MABO PLOT.MODE2D GT PLOT.PROPOSAL_FUNC xy

# This will run all the methods to run MABO for all methods
# random z xy dim rotation aspect full
for method in random z xy dim rotation aspect full; do
    python tools/eval_boxes.py --eval_only --config-file configs/BoxNet.yaml OUTPUT_DIR output/$method PLOT.EVAL MABO PLOT.MODE2D GT PLOT.PROPOSAL_FUNC $method
done

# This will run all the methods to generate pseudo gt
for method in random z xy dim rotation aspect full; do
    python tools/eval_boxes.py --config-file configs/BoxNet.yaml OUTPUT_DIR output/gt/$method "TRAIN.pseudo_gt", "learn" PLOT.PROPOSAL_FUNC $method
done
