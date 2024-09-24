#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional
import torch.nn as nn
from dpr_scale.utils.utils import PathManager
import torch
import torch.nn.functional as F
import shutil

# @manual=//python/wheel/transformers3:transformers3
from transformers import AutoModel, AutoConfig, T5EncoderModel, T5Config
from sentence_transformers import SentenceTransformer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class HFEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "roberta-base",
        dropout: float = 0.1,        
        projection_dim: Optional[int] = None,
        query_model_hf: Optional[str] = None,
        ctx_model_hf: Optional[str] = None,
        hf_model_mode: str = 'query',
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        self.model_path = model_path
        self.hf_model_mode = hf_model_mode
        print('Initializing model with path:', model_path)

        if model_path.startswith('bert') or 'contriever' in model_path:
            local_model_path = PathManager.get_local_path(model_path)
            cfg = AutoConfig.from_pretrained(local_model_path)
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
            self.transformer = AutoModel.from_pretrained(local_model_path, config=cfg)

        elif model_path.startswith('t5'):
            # dpr
            local_model_path = PathManager.get_local_path(model_path)
            print('local_model_path:', local_model_path)
            cfg = T5Config.from_pretrained(local_model_path)
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
            try:
                self.transformer = T5EncoderModel.from_pretrained(local_model_path, config=cfg)
            except:
                print('Failed to load model from local path. Trying to remove hub cache...')
                shutil.rmtree('/storage/ukp/work/cai_e/.cache/huggingface/hub')
                # RuntimeError: unable to open file </storage/ukp/work/cai_e/.cache/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1/model.safetensors> in read-only mode: No such file or directory (2)
                self.transformer = T5EncoderModel.from_pretrained(local_model_path, config=cfg)

        # load from hf -  we put it in the init function, therefore, it won't affect loading the checkpoint
        checkpoint = None
        if hf_model_mode == 'query' and query_model_hf is not None:
            checkpoint = self.load_from_hf(query_model_hf)
        
        if hf_model_mode == 'ctx' and ctx_model_hf is not None:
            checkpoint = self.load_from_hf(ctx_model_hf)
        
        if checkpoint is not None:
            self.transformer.load_state_dict(checkpoint, strict=False)

        self.project = nn.Identity()  # to make torchscript happy
        if projection_dim == -1:
            projection_dim = cfg.hidden_size
        if projection_dim:
            linear = nn.Linear(cfg.hidden_size, projection_dim)
            linear.weight.data.normal_(mean=0.0, std=0.02)
            self.project = nn.Sequential(
                linear, nn.LayerNorm(projection_dim)
            )

    def forward(self, tokens):
        # make it transformer 4.x compatible
        outputs = self.transformer(**tokens)  # B x T x C
        # not t5
        if self.model_path.startswith('bert'):
            last_layer = outputs[0]
            sentence_rep = self.project(last_layer[:, 0, :])
        elif self.model_path.startswith('t5') or 'contriever' in self.model_path:
            # t5 and contriever
            mean_pooled = mean_pooling(outputs, tokens['attention_mask'])
            sentence_rep = self.project(mean_pooled)
            # sentence_rep = F.normalize(sentence_rep, p=2, dim=1)
        return sentence_rep.clone()
    
    def load_from_hf(self, hf_model_path):
        print('Loading model from hf:', hf_model_path)

        if self.model_path.startswith('bert'):
            if self.hf_model_mode == 'query':
                # dpr - query
                model = DPRQuestionEncoder.from_pretrained(hf_model_path)
                return model.question_encoder.bert_model.state_dict()
            elif self.hf_model_mode == 'ctx':
                # dpr - ctx
                model = DPRContextEncoder.from_pretrained(hf_model_path)
                return model.ctx_encoder.bert_model.state_dict()
        
        elif self.model_path.startswith('t5'):
            # gtr
            try:
                model = SentenceTransformer(hf_model_path)
            except:
                print('Failed to load model from hf. Trying to remove hub cache...')
                shutil.rmtree('/storage/ukp/work/cai_e/.cache/huggingface/hub')
                model = SentenceTransformer(hf_model_path)
        return model[0].auto_model.state_dict()