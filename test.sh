#!/bin/bash

export ROOT_DIR=/storage/ukp/work/cai_e/instruction_pir/instruct-dense-retrieval

# PYTHONPATH=.:$PYTHONPATH \
# python dpr_scale/generate_query_embeddings.py -m --config-name nq.yaml \
# trainer.gpus=1 \
# datamodule.test_path=$ROOT_DIR/data/official.dev.query.jsonl \
# +task.ctx_embeddings_dir=$ROOT_DIR/outputs \
# +task.checkpoint_path=$ROOT_DIR/../models/dpr.msmarco.ep10.bs64.neg7.max_length512.lr1e-5/checkpoints/checkpoint_best.ckpt



# PYTHONPATH=.:$PYTHONPATH \
# python dpr_scale/generate_embeddings.py -m --config-name nq.yaml \
# datamodule=generate \
# +task.ctx_embeddings_dir=$ROOT_DIR/outputs \
# +task.checkpoint_path=$ROOT_DIR/../models/dpr.msmarco.ep10.bs64.neg7.max_length512.lr1e-5/checkpoints/checkpoint_best.ckpt


export ROOT_DIR=/storage/ukp/work/cai_e/instruction_pir
OUTPUT_DIR=$ROOT_DIR/outputs
MODEL_DIR=$ROOT_DIR/models

export DATA_DIR OUTPUT_DIR MODEL_DIR

export PYTHONPATH=.:$PYTHONPATH
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=dryrun

python $ROOT_DIR/dpr_scales/dpr_scale/main.py --config-name msmarco_baseline_t5.yaml \
    logger.name=$OUTPUT_DIR/gtr_test \
    checkpoint_callback.dirpath="$MODEL_DIR"/gtr_test/checkpoints