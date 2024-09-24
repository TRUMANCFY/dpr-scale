#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hydra
import torch
import torch.nn as nn
from dpr_scale.utils.utils import PathManager, ScriptEncoder
from pytorch_lightning import LightningModule
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location
from copy import deepcopy


# Implementation of https://arxiv.org/abs/2004.04906.
# Logic and some code from the original https://github.com/facebookresearch/DPR/
class DenseRetrieverTask(LightningModule):
    def __init__(
        self,
        transform,
        model,
        datamodule,
        optim,
        k=1,  # k for accuracy@k metric
        shared_model: bool = True,  # shared encoders
        in_batch_eval: bool = True,  # use only in-batch contexts for val
        in_batch_negatives: bool = True,  # train using in-batch negatives
        warmup_steps: int = 0,
        fp16_grads: bool = False,
        pretrained_checkpoint_path: str = "",
        softmax_temperature: float = 1.0,
    ):
        super().__init__()
        # save all the task hyperparams
        # so we can instantiate it much easily later.
        self.save_hyperparameters()
        self.transform_conf = (
            transform.text_transform
            if hasattr(transform, "text_transform")
            else transform
        )
        # this is a dictionary
        self.model_conf = model
        self.shared_model = shared_model
        self.optim_conf = optim
        self.k = k
        self.loss = nn.CrossEntropyLoss()
        self.in_batch_eval = in_batch_eval
        self.in_batch_negatives = in_batch_negatives
        self.warmup_steps = warmup_steps
        self.fp16_grads = fp16_grads
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.softmax_temperature = softmax_temperature
        self.setup_done = False
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def setup(self, stage: str):
        # skip building model during test.
        # Otherwise, the state dict will be re-initialized
        if stage == "test" and self.setup_done:
            return
        # resetting call_configure_sharded_model_hook attribute so that we could configure model
        self.call_configure_sharded_model_hook = False

        query_conf = deepcopy(self.model_conf)
        query_conf['hf_model_mode'] = 'query'
        self.query_encoder = hydra.utils.instantiate(
            query_conf,
        )

        if self.shared_model:
            self.context_encoder = self.query_encoder
        else:
            ctx_conf = deepcopy(self.model_conf)
            ctx_conf['hf_model_mode'] = 'ctx'
            self.context_encoder = hydra.utils.instantiate(
                ctx_conf,
            )

        if self.pretrained_checkpoint_path:
            checkpoint_dict = torch.load(
                PathManager.open(self.pretrained_checkpoint_path, "rb"),
                map_location=lambda s, l: default_restore_location(s, "cpu"),
            )
            self.load_state_dict(checkpoint_dict["state_dict"])
            print(f"Loaded state dict from {self.pretrained_checkpoint_path}")

        self.setup_done = True

    def on_load_checkpoint(self, checkpoint) -> None:
        """
        This hook will be called before loading state_dict from a checkpoint.
        setup("fit") will built the model before loading state_dict
        """
        self.setup("fit")

    def on_pretrain_routine_start(self):
        if self.fp16_grads:
            self.trainer.strategy._model.register_comm_hook(None, fp16_compress_hook)

    def _encode_sequence(self, token_ids, encoder_model):
        encoded_seq = encoder_model(token_ids)  # bs x d
        return encoded_seq

    def sim_score(self, query_repr, context_repr, mask=None):
        scores = torch.matmul(
            query_repr, torch.transpose(context_repr, 0, 1)
        )  # bs x ctx_cnt
        if mask is not None:
            # bs x ctx_cnt
            scores[mask] = float("-inf")
        return scores

    def encode_queries(self, query_ids):
        query_repr = self._encode_sequence(query_ids, self.query_encoder)  # bs x d
        return query_repr

    def encode_contexts(self, contexts_ids):
        contexts_repr = self._encode_sequence(
            contexts_ids, self.context_encoder
        )  # ctx_cnt x d
        return contexts_repr

    def forward(self, query_ids, contexts_ids):
        # encode query and contexts
        query_repr = self.encode_queries(query_ids)  # bs x d
        contexts_repr = self.encode_contexts(contexts_ids)  # ctx_cnt x d
        return query_repr, contexts_repr

    def configure_optimizers(self):
        self.optimizer = hydra.utils.instantiate(self.optim_conf, self.parameters())
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            training_steps = self.trainer.max_steps
        else:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            training_steps = steps_per_epoch * self.trainer.max_epochs
        print(
            f"Configured LR scheduler for total {training_steps} training steps, "
            f"with {self.warmup_steps} warmup steps."
        )

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                float(training_steps - current_step)
                / float(max(1, training_steps - self.warmup_steps)),
            )

        scheduler = LambdaLR(self.optimizer, lr_lambda)
        scheduler = {
            "scheduler": LambdaLR(self.optimizer, lr_lambda),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """
        This receives queries, each with multiple contexts.
        """
        query_ids = batch["query_ids"]  # bs x tokens
        contexts_ids = batch["contexts_ids"]  # ctx_cnt x ctx_len
        pos_ctx_indices = batch["pos_ctx_indices"]  # bs
        mask = batch["ctx_mask"]  # ctx_cnt
        query_repr, context_repr = self(query_ids, contexts_ids)  # bs

        if batch_idx == 0 and self.trainer.is_global_zero and self.trainer.current_epoch == 0:
            print(f"[{self.trainer.local_rank}] query_ids: {query_ids.input_ids.shape}")
            print(f"[{self.trainer.local_rank}] contexts_ids: {contexts_ids.input_ids.shape}")
            print(f"[{self.trainer.local_rank}] pos_ctx_indices: {pos_ctx_indices.shape}")
            print(f"[{self.trainer.local_rank}] mask: {mask.shape}")
            print(f"[{self.trainer.local_rank}] query_repr: {query_repr.shape}")
            print(f"[{self.trainer.local_rank}] context_repr: {context_repr.shape}")
            print(f"[{self.trainer.local_rank}] pos_ctx_indices: {pos_ctx_indices}")

        if self.in_batch_negatives:
            from pytorch_lightning.strategies import DDPStrategy
            # gather all tensors for training w/ in_batch_negatives
            if isinstance(self.trainer.strategy, (DDPStrategy)):
                query_to_send = query_repr.detach()
                context_to_send = context_repr.detach()
                # assumes all nodes have same number of contexts
                (
                    all_query_repr,
                    all_context_repr,
                    all_labels,
                    all_mask,
                ) = self.all_gather(
                    (query_to_send, context_to_send, pos_ctx_indices, mask)
                )
                offset = 0
                all_query_list = []
                all_context_list = []

                for i in range(all_labels.size(0)):
                    if i != self.global_rank:
                        all_query_list.append(all_query_repr[i])
                        all_context_list.append(all_context_repr[i])
                    else:
                        # to calculate grads for this node only
                        all_query_list.append(query_repr)
                        all_context_list.append(context_repr)
                    all_labels[i] += offset
                    offset += all_context_repr[i].size(0)

                context_repr = torch.cat(all_context_list, dim=0)  # total_ctx x dim
                query_repr = torch.cat(all_query_list, dim=0)  # total_query x dim
                pos_ctx_indices = torch.flatten(all_labels)  # total_query
                mask = torch.flatten(all_mask)  # total_ctx
            else:
                raise NotImplementedError(
                    "Have not implemented in_batch_negatives for this strategy."
                )
            # create a query-ctx mask where all ctxs except dummies will be unmasked for each query.
            query_ctx_mask = mask.repeat(query_repr.shape[0], 1)  # bs x ctx_cnt
        else:
            # create a query-ctx mask where only non-dummy ctxs directly attached to the query will be unmasked.
            num_ctx_per_batch = int(mask.shape[0] / query_repr.shape[0])  # ctx_cnt / bs
            query_ctx_mask = torch.ones(
                query_repr.shape[0], mask.shape[0], dtype=torch.bool
            )  # bs x ctx_cnt
            for i, pos_ctx_id in enumerate(pos_ctx_indices):
                query_ctx_mask[i, pos_ctx_id : pos_ctx_id + num_ctx_per_batch] = mask[
                    pos_ctx_id : pos_ctx_id + num_ctx_per_batch
                ]

        scores = self.sim_score(query_repr, context_repr, query_ctx_mask)
        # temperature scaling
        scores /= self.softmax_temperature
        loss = self.loss(scores, pos_ctx_indices)
        
        if torch.isnan(loss):
            print('contexts_ids: ', contexts_ids)
            print("context_repr: ", context_repr)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _eval_step(self, batch, batch_idx):
        query_ids = batch["query_ids"]  # bs x tokens
        contexts_ids = batch["contexts_ids"]  # bs x ctx_cnt x ctx_len
        pos_ctx_indices = batch["pos_ctx_indices"]  # bs x ctx_cnt
        mask = batch["ctx_mask"]  # ctx_cnt
        query_repr, contexts_repr = self(query_ids, contexts_ids)
        query_ctx_mask = mask.repeat(query_repr.shape[0], 1)
        pred_context_scores = self.sim_score(query_repr, contexts_repr, query_ctx_mask)
        loss = self.loss(pred_context_scores, pos_ctx_indices)

        return (
            self.compute_rank_metrics(pred_context_scores, pos_ctx_indices),
            query_repr,
            contexts_repr,
            pos_ctx_indices,
            mask,
            loss,
        )

    def compute_rank_metrics(self, pred_scores, target_labels):
        # Compute total un_normalized avg_ranks, mrr
        values, indices = torch.sort(pred_scores, dim=1, descending=True)
        rank = 0
        mrr = 0.0
        score = 0
        for i, idx in enumerate(target_labels):
            gold_idx = torch.nonzero(indices[i] == idx, as_tuple=False)
            rank += gold_idx.item() + 1
            score += gold_idx.item() < self.k
            mrr += 1 / (gold_idx.item() + 1)
        return rank, mrr, score

    def _eval_epoch_end(self, outputs, log_prefix="valid"):
        total_avg_rank, total_ctx_count, total_count = 0, 0, 0
        total_mrr = 0
        total_loss = 0
        total_score = 0
        if self.in_batch_eval:
            for metrics, query_repr, contexts_repr, _, mask, loss in outputs:
                rank, mrr, score = metrics
                total_avg_rank += rank
                total_mrr += mrr
                total_score += score
                total_ctx_count += contexts_repr.size(0) - torch.sum(mask)
                total_count += query_repr.size(0)
                total_loss += loss
            total_ctx_count = total_ctx_count / len(outputs)
            total_loss = total_loss / len(outputs)
        else:
            # collate the representation and gold +ve labels
            all_query_repr = []
            all_context_repr = []
            all_labels = []
            all_mask = []
            offset = 0
            for _, query_repr, context_repr, target_labels, mask, _ in outputs:
                all_query_repr.append(query_repr)
                all_context_repr.append(context_repr)
                all_mask.append(mask)
                all_labels.extend([offset + x for x in target_labels])
                offset += context_repr.size(0)
            # gather all contexts
            all_context_repr = torch.cat(all_context_repr, dim=0)
            all_mask = torch.cat(all_mask, dim=0)
            if self.trainer.world_size > 1:
                all_context_repr, all_mask = self.all_gather(
                    (all_context_repr, all_mask)
                )
                all_labels = [
                    x + all_context_repr.size(1) * self.global_rank for x in all_labels
                ]
                all_context_repr = torch.cat(tuple(all_context_repr), dim=0)
                all_mask = torch.cat(tuple(all_mask), dim=0)
            all_query_repr = torch.cat(all_query_repr, dim=0)
            all_query_ctx_mask = all_mask.repeat(all_query_repr.shape[0], 1)
            scores = self.sim_score(
                all_query_repr, all_context_repr, all_query_ctx_mask
            )
            total_count = all_query_repr.size(0)
            total_ctx_count = scores.size(1) - torch.sum(all_mask)
            total_avg_rank, total_mrr, total_score = self.compute_rank_metrics(
                scores, all_labels
            )
            total_loss = self.loss(
                scores,
                torch.tensor(all_labels).to(scores.device, dtype=torch.long),
            )
        metrics = {
            log_prefix + "_avg_rank": total_avg_rank / total_count,
            log_prefix + "_mrr": total_mrr / total_count,
            log_prefix + f"_accuracy@{self.k}": total_score / total_count,
            log_prefix + "_ctx_count": total_ctx_count,
            log_prefix + "_loss": total_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        res = self._eval_step(batch, batch_idx)
        self.validation_step_outputs.append(res)
        return res

    def on_validation_epoch_end(self):
        # self._eval_epoch_end(valid_outputs) if valid_outputs else None
        if self.validation_step_outputs:
            self._eval_epoch_end(self.validation_step_outputs, "valid")
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        res = self._eval_step(batch, batch_idx)
        self.test_step_outputs.append(res)
        return res

    def on_test_epoch_end(self):
        # self._eval_epoch_end(test_outputs, "test") if test_outputs else None
        if self.test_step_outputs:
            self._eval_epoch_end(self.test_step_outputs, "test")
            self.test_step_outputs.clear()

    @torch.no_grad()
    def to_torchscript(
        self,
        file_path=None,
        method="script",
        example_inputs=None,
        **kwargs,
    ):

        mode = self.training
        if method == "script":
            transform = hydra.utils.instantiate(self.transform_conf)
            ctx_encoder = ScriptEncoder(transform, self.context_encoder)
            ctx_encoder = torch.jit.script(ctx_encoder.eval(), **kwargs)
            result = {"ctx_encoder": ctx_encoder}
            # Quantize. TODO when PL has better handling link this with the save_quantized
            # flag in ModelCheckpoint
            ctx_encoder_qt = ScriptEncoder(
                transform, self.context_encoder, quantize=True
            )
            ctx_encoder_qt = torch.jit.script(ctx_encoder_qt.eval(), **kwargs)
            result["ctx_encoder_qt"] = ctx_encoder_qt
            if not self.shared_model:
                q_encoder = ScriptEncoder(transform, self.query_encoder)
                q_encoder = torch.jit.script(q_encoder.eval(), **kwargs)
                result["q_encoder"] = q_encoder
                # Quantize. TODO when PL has better handling link this with the save_quantized
                # flag in ModelCheckpoint
                q_encoder_qt = ScriptEncoder(
                    transform, self.query_encoder, quantize=True
                )
                q_encoder_qt = torch.jit.script(q_encoder_qt.eval(), **kwargs)
                result["q_encoder_qt"] = q_encoder_qt
        else:
            raise ValueError(
                "The 'method' parameter only supports 'script',"
                f" but value given was: {method}"
            )

        self.train(mode)

        if file_path is not None:
            torch.jit.save(ctx_encoder, file_path)

        return result
