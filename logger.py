import os

import wandb
from torch import nn
from omegaconf import OmegaConf


class WandBLogger:
    def __init__(
            self,
            config: dict,
            model: nn.Module,
    ):
        self.config = config
        self.run = wandb.init(
            project='coursework',
            name=self.config['run_name'],
            config=self.config,
            resume='allow',
        )
        wandb.watch(model, log='all', log_freq=100, log_graph=True)
        self.save_config()

    def __call__(self, metric_name: str, value: float):
        self.log(metric_name, value)

    def log(self, metric_name: str, value: float):
        self.run.log({metric_name: value})

    def log_image(self):
        pass

    def save_config(self):
        config_name = 'train_config.yaml'
        path_to_save_config = os.path.join(self.run.dir, config_name)
        OmegaConf.save(self.config, path_to_save_config)
        wandb.save(config_name, policy='now')

    def finish(self):
        self.run.finish()
