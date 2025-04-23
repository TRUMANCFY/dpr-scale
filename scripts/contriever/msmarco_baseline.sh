#!/bin/bash
export ROOT_DIR=/pfss/mlde/workspaces/mlde_wsp_PI_Heinz/cai/dpr-scale
export DATA_DIR=$ROOT_DIR/data
export OUTPUT_DIR=$ROOT_DIR/logs
export MODEL_DIR=$ROOT_DIR/models

export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
export HYDRA_FULL_ERROR=1
export WANDB_MODE=dryrun

python $ROOT_DIR/dpr_scale/main.py --config-name msmarco_baseline_contriever.yaml \
    logger.name=$OUTPUT_DIR/contriever.msmarco.ep20.bs16.neg7.max_length512.lr1e-5 \
    checkpoint_callback.dirpath="$MODEL_DIR"/contriever.msmarco.ep10.bs64.neg7.max_length512.lr1e-5/checkpoints
