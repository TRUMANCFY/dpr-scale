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
import torch.nn.functional as F
from typing import Literal, Optional

def check_tensor(name, t, max_idx_print=10):
    """Pretty-print a quick health report for any tensor."""
    nan_mask = torch.isnan(t)
    inf_mask = torch.isinf(t)
    n_nan = nan_mask.sum().item()
    n_inf = inf_mask.sum().item()
    print(f"\n─── {name} ───")
    print("shape:", tuple(t.shape), "dtype:", t.dtype)
    print(f"NaNs: {n_nan:,}   Infs: {n_inf:,}")

    if n_nan:
        idx = nan_mask.nonzero(as_tuple=False)
        print("  first NaN at index:", tuple(idx[0].tolist()))
        if idx.size(0) > 1:
            print("  more NaNs at (up to) first", max_idx_print, "indices:",
                  [tuple(i.tolist()) for i in idx[:max_idx_print]])
    if n_inf:
        idx = inf_mask.nonzero(as_tuple=False)
        print("  first Inf at index:", tuple(idx[0].tolist()))

    finite = t[~nan_mask & ~inf_mask]
    if finite.numel():
        print("  min:", finite.min().item(),
              "max:", finite.max().item(),
              "mean:", finite.mean().item())

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
        prop_trainable : bool = True,
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
        self.prop_trainable = prop_trainable

        
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
    
    # ✅ 👇👇 Add this method to fix the error
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

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


class DensePropRetrieverTask(DenseRetrieverTask):
    """
    Extends DenseRetrieverTask to include a query-prop loss and KL divergence
    between query-context and query-prop distributions, while preserving
    in-batch negative handling (including prop representations broadcast).
    """
    # ------------------------------------------------------------------
    # ctor
    # ------------------------------------------------------------------
    def __init__(
        self,
        *args,
        prop_pooling: Literal["max", "logsumexp", "topk"] = "max",
        prop_topk: int = 2,
        prop_tau: float = 0.05,
        kl_target: Literal["prop", "ctx", "skip"] = "prop",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if prop_pooling not in {"max", "logsumexp", "topk"}:
            raise ValueError("prop_pooling must be 'max', 'logsumexp' or 'topk'")
        self.prop_pooling: str = prop_pooling
        self.prop_topk: int = prop_topk
        self.prop_tau: float = prop_tau  # temperature inside pooling
        self.kl_target: str = kl_target

    # ------------------------------------------------------------------
    # helper ------------------------------------------------------------
    # ------------------------------------------------------------------
    def _aggregate_prop_scores(
        self,
        scores_prop3: torch.Tensor,  # (bs, ctx_cnt, max_p)
        prop_mask3: torch.Tensor,    # (bs, ctx_cnt, max_p)  True where *masked*
    ) -> torch.Tensor:
        """Collapse proposition‑level logits to one score per context.

        The behaviour depends on ``self.prop_pooling``.
        """
        # mask invalid propositions first
        scores_prop3 = scores_prop3.masked_fill(prop_mask3, float("-inf"))

        if self.prop_pooling == "max":
            # hard MIL – original implementation
            return scores_prop3.max(dim=2).values

        if self.prop_pooling == "logsumexp":
            # differentiable soft‑max (smooth‑max)
            return (scores_prop3 / self.prop_tau).logsumexp(dim=2)

        if self.prop_pooling == "topk":
            k_full = scores_prop3.size(2)
            k = min(self.prop_topk, k_full)
            # ``torch.topk`` will return –inf if fewer than k valid props;
            # we handle that below.
            topk_vals, _ = torch.topk(scores_prop3, k, dim=2)
            valid_mask = topk_vals.ne(float('-inf'))
            # sum only valid values; avoid NaNs when all are -inf
            sum_vals = topk_vals.masked_fill(~valid_mask, 0.0).sum(dim=2)
            denom = valid_mask.sum(dim=2).clamp(min=1)
            mean_vals = sum_vals / denom
            # If *all* props were masked the denom is 1 but the sum is 0;
            # we restore the sentinel so downstream max/softmax behave.
            mean_vals = mean_vals.masked_fill(denom == 0, float('-inf'))
            return mean_vals

        # if self.prop_pooling == "topk":
        #     # mean over top‑k props; k clipped to available propositions
        #     k = min(self.prop_topk, scores_prop3.size(2))
        #     topk, _ = torch.topk(scores_prop3, k, dim=2)
        #     return topk.mean(dim=2)

        # should be unreachable – keep mypy happy
        raise RuntimeError(f"Unknown prop_pooling: {self.prop_pooling}")

    def training_step(self, batch, batch_idx):
        # unpack batch
        q_ids           = batch["query_ids"]         # bs x tok
        ctx_ids         = batch["contexts_ids"]      # ctx_cnt x tok
        prop_ids        = batch["prop_ctx_ids"]      # ctx_cnt x MAX_PROPS x tok
        pos_ctx_indices = batch["pos_ctx_indices"]   # bs
        mask            = batch["ctx_mask"]          # ctx_cnt
        prop_mask       = batch["prop_mask"]         # ctx_cnt x MAX_PROPS

        # encode queries and contexts
        query_repr, context_repr = self(q_ids, ctx_ids)  # bs x d, ctx_cnt x d

        # prepare prop representations
        bs, dim      = query_repr.size()
        c_cnt, max_p = prop_ids["input_ids"].size(0), prop_ids["input_ids"].size(1)
        flat_pids    = {_k: _v.view(c_cnt * max_p, -1) for _k, _v in prop_ids.items()}
        flat_pref    = self._encode_sequence(flat_pids, self.context_encoder)
        p_repr       = flat_pref.view(c_cnt, max_p, dim)  # ctx_cnt x MAX_PROPS x d

        # --- in-batch negatives handling with p_repr broadcast ---
        if self.in_batch_negatives:
            from pytorch_lightning.strategies import DDPStrategy
            if isinstance(self.trainer.strategy, DDPStrategy):
                # detach for gathering
                q_send, c_send = query_repr.detach(), context_repr.detach()
                p_send, pm_send = p_repr.detach(), prop_mask.detach()
                # gather across nodes
                all_q, all_c, all_p, all_pm, all_labels, all_mask = self.all_gather(
                    (q_send, c_send, p_send, pm_send, pos_ctx_indices, mask)
                )
                offset = 0
                qs, cs, ps, pms = [], [], [], []
                for rank in range(all_labels.size(0)):
                    qs.append(all_q[rank] if rank != self.global_rank else query_repr)
                    cs.append(all_c[rank] if rank != self.global_rank else context_repr)
                    ps.append(all_p[rank] if rank != self.global_rank else p_repr)
                    pms.append(all_pm[rank] if rank != self.global_rank else prop_mask)
                    all_labels[rank] += offset
                    offset += all_c[rank].size(0)
                # concat gathered reps and masks
                query_repr      = torch.cat(qs, dim=0)
                context_repr    = torch.cat(cs, dim=0)
                p_repr          = torch.cat(ps, dim=0)
                prop_mask       = torch.cat(pms, dim=0)
                pos_ctx_indices = torch.flatten(all_labels)
                mask            = torch.flatten(all_mask)
            else:
                raise NotImplementedError("in_batch_negatives not supported for this strategy.")
            # context mask
            query_ctx_mask = mask.repeat(query_repr.shape[0], 1)
        else:
            num_ctx_per_q = int(mask.shape[0] / query_repr.shape[0])
            query_ctx_mask = torch.ones(
                query_repr.size(0), mask.size(0), dtype=torch.bool, device=mask.device
            )
            for i, pos_id in enumerate(pos_ctx_indices):
                query_ctx_mask[i, pos_id: pos_id + num_ctx_per_q] = \
                    mask[pos_id: pos_id + num_ctx_per_q]

        # --- context-based loss ---
        scores_ctx = self.sim_score(query_repr, context_repr, query_ctx_mask)
        scores_ctx = scores_ctx / self.softmax_temperature
        loss_ctx   = self.loss(scores_ctx, pos_ctx_indices)

        bs, dim  = query_repr.size()
        c_cnt    = context_repr.size(0)
        max_p    = p_repr.size(1)

        # --- proposition scores with selectable pooling -----------------
        m3 = prop_mask.view(c_cnt, max_p).unsqueeze(0).expand(bs, -1, -1)
        scores_prop3 = torch.einsum("bd,cmd->bcm", query_repr, p_repr)

        scores_prop = self._aggregate_prop_scores(scores_prop3, m3)
        scores_prop = scores_prop / self.softmax_temperature
        scores_prop = scores_prop.masked_fill(mask, float("-inf"))
        loss_prop   = self.loss(scores_prop, pos_ctx_indices)
        
        mask_2d = mask.view(bs, c_cnt // bs)  # reshape flattened mask
        assert (~mask_2d).any(dim=1).all(), "Each query must have at least one unmasked context!"
        
        scores_ctx = scores_ctx.masked_fill(mask.unsqueeze(0), torch.finfo(scores_ctx.dtype).min)
        ctx_prob = F.softmax(scores_ctx, dim=1)
        scores_prop = scores_prop.masked_fill(mask.unsqueeze(0), torch.finfo(scores_prop.dtype).min)
        prop_prob = F.softmax(scores_prop, dim=-1)

        if self.kl_target == 'prop':
            target_prop_prob = prop_prob.detach()
            kl_loss = F.kl_div(
                ctx_prob.clamp(min=1e-8).log(),
                target_prop_prob,
                reduction='batchmean',
            )
        else:
            target_ctx_prob = ctx_prob.detach()
            kl_loss = F.kl_div(
                prop_prob.clamp(min=1e-8).log(),
                target_ctx_prob,
                reduction='batchmean',
            )
    
    
        # ctx_prob: [B, C]
        nan_idx_ctx = torch.isnan(ctx_prob).nonzero(as_tuple=False)
        if nan_idx_ctx.numel() > 0:
            # --- call it for everything that feeds into softmax / KL ---
            check_tensor("query_repr",  query_repr)
            check_tensor("context_repr", context_repr)
            check_tensor("p_repr",       p_repr)
            
            check_tensor("scores_ctx",   scores_ctx)
            check_tensor("scores_prop3", scores_prop3)
            check_tensor("scores_prop",  scores_prop)
            
            check_tensor("ctx_prob",     ctx_prob)
            check_tensor("prop_prob",    prop_prob)

        alpha      = getattr(self, 'prop_kl_weight', 1.0)
        
        if self.kl_target == 'skip':
            total_loss = loss_ctx + loss_prop
        else:
            if self.prop_trainable:
                total_loss = loss_ctx + loss_prop + alpha * kl_loss
            else:
                total_loss = loss_ctx + alpha * kl_loss

        # logging
        self.log("train/loss_ctx",  loss_ctx,    prog_bar=True)
        self.log("train/loss_prop", loss_prop,   prog_bar=True)
        self.log("train/kl_loss",   kl_loss,     prog_bar=True)
        self.log("train/loss",      total_loss,  prog_bar=True)

        return total_loss
    # --------------------------------------------------------------------
    # Add to DensePropRetrieverTask
    # --------------------------------------------------------------------
    def _eval_step(self, batch, batch_idx):
        """
        Computes *both* context-level and proposition-level scores
        (max-over-props) and returns the pieces we need for epoch-end
        aggregation.
        """
        # ---------- unpack ----------
        q_ids           = batch["query_ids"]          # bs x tok
        ctx_ids         = batch["contexts_ids"]       # ctx_cnt x tok
        prop_ids        = batch["prop_ctx_ids"]       # ctx_cnt x MAX_P x tok
        pos_ctx_indices = batch["pos_ctx_indices"]    # bs
        mask            = batch["ctx_mask"]           # ctx_cnt
        prop_mask       = batch["prop_mask"]          # ctx_cnt x MAX_P

        # ---------- encode ----------
        q_repr, c_repr = self(q_ids, ctx_ids)                           # bs x d , ctx_cnt x d
        c_cnt, max_p   = prop_ids["input_ids"].shape[:2]
        flat_pids      = {k: v.view(c_cnt * max_p, -1) for k, v in prop_ids.items()}
        flat_pref      = self._encode_sequence(flat_pids, self.context_encoder)
        p_repr         = flat_pref.view(c_cnt, max_p, -1)               # ctx_cnt x MAX_P x d

        # ---------- masks ----------
        bs = q_repr.size(0)
        q_ctx_mask  = mask.repeat(bs, 1)                                # bs x ctx_cnt
        q_prop_mask = prop_mask.view(c_cnt, max_p).unsqueeze(0).expand(bs, -1, -1)

        # ---------- context scores ----------
        scores_ctx  = self.sim_score(q_repr, c_repr, q_ctx_mask)        # bs x ctx_cnt
        scores_ctx  = scores_ctx / self.softmax_temperature

        # ---------- proposition scores (max over props) ----------
        #   scores_prop3: bs x ctx_cnt x MAX_P
        scores_prop3 = torch.einsum("bd,cmd->bcm", q_repr, p_repr)
        scores_prop  = self._aggregate_prop_scores(scores_prop3, q_prop_mask)
        scores_prop  = scores_prop / self.softmax_temperature
        scores_prop  = scores_prop.masked_fill(mask, float("-inf"))

        # ---------- losses (needed so Lightning doesn’t complain) ----------
        loss_ctx  = self.loss(scores_ctx,  pos_ctx_indices)
        loss_prop = self.loss(scores_prop, pos_ctx_indices)             # not used for opt step

        # ---------- metrics ----------
        ctx_metrics  = self.compute_rank_metrics(scores_ctx,  pos_ctx_indices)
        prop_metrics = self.compute_rank_metrics(scores_prop, pos_ctx_indices)

        return (
            ctx_metrics, prop_metrics,
            q_repr, c_repr, p_repr,
            pos_ctx_indices, mask, prop_mask,
            loss_ctx, loss_prop
        )


    def _eval_epoch_end(self, outputs, log_prefix="valid"):
        """
        Aggregates context-level *and* proposition-level metrics.
        Mirrors the parent logic but keeps two separate metric groups.
        """
        # --- helpers ---
        def _aggregate(metric_idx):
            tot_avg_rank = tot_mrr = tot_score = tot_loss = 0.
            tot_q = tot_ctx = 0
            for out in outputs:
                (rank, mrr, score) = out[metric_idx]
                tot_avg_rank += rank
                tot_mrr      += mrr
                tot_score    += score
                tot_q        += out[2].size(0)           # q_repr
                tot_ctx      += out[3].size(0) - torch.sum(out[6])  # ctx_repr - mask
                tot_loss     += out[8 + metric_idx]      # loss_ctx or loss_prop
            n_batches = len(outputs)
            return {
                "avg_rank":     tot_avg_rank / tot_q,
                "mrr":          tot_mrr      / tot_q,
                f"accuracy@{self.k}": tot_score / tot_q,
                "ctx_count":    tot_ctx / n_batches,
                "loss":         tot_loss / n_batches,
            }

        ctx_stats  = _aggregate(0)
        prop_stats = _aggregate(1)

        # ---------- log ----------
        metric_dict = {
            f"{log_prefix}_ctx_"  + k: v for k, v in ctx_stats.items()
        }
        metric_dict.update({
            f"{log_prefix}_prop_" + k: v for k, v in prop_stats.items()
        })
        self.log_dict(metric_dict, on_epoch=True, sync_dist=True)


    # Make sure validation_step / test_step use the new _eval_step
    def validation_step(self, batch, batch_idx):
        res = self._eval_step(batch, batch_idx)
        self.validation_step_outputs.append(res)
        return res

    def test_step(self, batch, batch_idx):
        res = self._eval_step(batch, batch_idx)
        self.test_step_outputs.append(res)
        return res
