#!/bin/bash
export ROOT_DIR=/pfss/mlde/workspaces/mlde_wsp_PI_Heinz/cai/dpr-scale
export DATA_DIR=$ROOT_DIR/data
export OUTPUT_DIR=$ROOT_DIR/logs
export MODEL_DIR=$ROOT_DIR/models

export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONPATH="$ROOT_DIR":$PYTHONPATH
export HYDRA_FULL_ERROR=1
# export WANDB_API_KEY="6524c9fb101a70ff6712c8d66740fcfd289a2486"
# export WANDB_PROJECT="mixgr"
# export WANDB_ENTITY="my-team" 
# export WANDB_MODE=dryrun

python $ROOT_DIR/dpr_scale/main.py --config-name msmarco_baseline_contriever_prop_trainable_logsumexp.yaml \
    logger.name=contriever.prop.trainable.logsumexp \
    checkpoint_callback.dirpath="$MODEL_DIR"/contriever.prop.trainable.logsumexp/checkpoints
