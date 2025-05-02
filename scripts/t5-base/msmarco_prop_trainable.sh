#!/bin/bash
export ROOT_DIR=/pfss/mlde/workspaces/mlde_wsp_PI_Heinz/cai/dpr-scale
export DATA_DIR=$ROOT_DIR/data
export OUTPUT_DIR=$ROOT_DIR/logs
export MODEL_DIR=$ROOT_DIR/models

export CUDA_VISIBLE_DEVICES=4,5,6,7

export PYTHONPATH="$ROOT_DIR":$PYTHONPATH
export HYDRA_FULL_ERROR=1
# export WANDB_API_KEY="6524c9fb101a70ff6712c8d66740fcfd289a2486"
# export WANDB_PROJECT="mixgr"
# export WANDB_ENTITY="my-team" 
# export WANDB_MODE=dryrun

python $ROOT_DIR/dpr_scale/main.py --config-name msmarco_baseline_t5_base_prop_trainable.yaml \
    logger.name=gtr.base.msmarco.prop.trainable.10 \
    checkpoint_callback.dirpath="$MODEL_DIR"/gtr.base.msmarco.prop.trainable.10/checkpoints
