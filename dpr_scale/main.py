# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from dpr_scale.conf.config import MainConfig

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
import wandb.util

import os

"""
Sample commands:
Default: $ buck run //deeplearning/projects/dpr-scale:main

For debugging Hydra:
$ HYDRA_FULL_ERROR=1 buck run //deeplearning/projects/dpr-scale:main -- --info
"""


@hydra.main(config_path="conf", config_name="config")
def main(cfg: MainConfig):
    # import pdb; pdb.set_trace()
    print(OmegaConf.to_yaml(cfg))
    # Temp patch for datamodule refactoring
    cfg.task.datamodule = None
    task = hydra.utils.instantiate(cfg.task, _recursive_=False)

    assert cfg.task.model.model_path == cfg.task.transform.model_path
    transform = hydra.utils.instantiate(cfg.task.transform)
    datamodule = hydra.utils.instantiate(cfg.datamodule, transform=transform)
    
    print(f"*** Checkpoint path is {cfg.checkpoint_callback.dirpath}")
    os.makedirs(cfg.checkpoint_callback.dirpath, exist_ok=True)
    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint_callback)
    latest_checkpoint = None
    if os.path.exists(cfg.checkpoint_callback.dirpath + "/last.ckpt"):
        latest_checkpoint = cfg.checkpoint_callback.dirpath + "/last.ckpt"
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if os.path.exists(cfg.checkpoint_callback.dirpath + "/wandb_id.txt"):
        with open(cfg.checkpoint_callback.dirpath + "/wandb_id.txt", "r") as f:
            wandb_id = f.read()
        wandb_logger = WandbLogger(project=cfg.logger.project, name=cfg.logger.name, id=wandb_id, resume="must")
    else:
        wandb_id = wandb.util.generate_id()
        with open(cfg.checkpoint_callback.dirpath + "/wandb_id.txt", "w") as f:
            f.write(wandb_id)
        wandb_logger = WandbLogger(project=cfg.logger.project, name=cfg.logger.name, id=wandb_id)

    # delete gpus from trainer config and add [accelerator="auto"] to use all available GPUs
    cfg_trainer = dict(**cfg.trainer)
    cfg_trainer.pop("gpus")
    cfg_trainer["accelerator"] = "auto"
    from pprint import pprint
    pprint(cfg_trainer)
    trainer = Trainer(**cfg_trainer, callbacks=[checkpoint_callback, lr_monitor], logger=wandb_logger)

    if cfg.test_only:
        ckpt_path = cfg.task.pretrained_checkpoint_path
        trainer.test(
            model=task,
            ckpt_path=ckpt_path,
            verbose=True,
            datamodule=datamodule,
        )
    else:
        # trainer.validate(model=task, verbose=True, datamodule=datamodule)
        if latest_checkpoint is not None:
            trainer.fit(task, datamodule=datamodule, ckpt_path=latest_checkpoint)
        else:
            trainer.fit(task, datamodule=datamodule)
        print(f"*** Best model path is {checkpoint_callback.best_model_path}")
        trainer.test(
            model=None,
            ckpt_path="best",
            verbose=True,
            datamodule=datamodule,
        )


if __name__ == "__main__":
    main()
