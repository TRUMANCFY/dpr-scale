#!/bin/bash
export ROOT_DIR=/pfss/mlde/workspaces/mlde_wsp_PI_Heinz/cai/dpr-scale
export DATA_DIR=$ROOT_DIR/data
export OUTPUT_DIR=$ROOT_DIR/logs
export MODEL_DIR=$ROOT_DIR/models

export CUDA_VISIBLE_DEVICES=4,5,6,7

export PYTHONPATH="$ROOT_DIR":$PYTHONPATH
export HYDRA_FULL_ERROR=1

python $ROOT_DIR/dpr_scale/main.py --config-name msmarco_baseline_bert_contriever_40.yaml \
    logger.name=bert.contriever.40 \
    checkpoint_callback.dirpath="$MODEL_DIR"/bert.contriever.40/checkpoints
