import argparse
import os
import shutil
from functools import partial
from typing import List, Tuple, Optional, Sequence

from omegaconf import OmegaConf
import torch
import torch.nn.parallel
from torch.optim.lr_scheduler import LinearLR
from dotenv import load_dotenv

from models.unetr import UNETR
from trainer import run_training
from utils.data_utils import get_loader
from optimizers.lr_schedule import LinearWarmupCosineAnnealingLR

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction


def main(config: dict):
    load_dotenv()
    config = parse_congig(config)
    run(config=config, **config)


def parse_congig(config: dict):
    config['data_dir'] = os.path.expanduser(config['data_dir'])
    return config


def run(
        log_dir: str,
        batch_size: int,
        inf_size: int,
        in_channels: int,
        out_channels: int,
        feature_size: int,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        pos_embed: str,
        norm_name: Tuple or str,
        dropout_rate: float,
        pretrained: bool,
        path_to_pretrain: Optional[str],
        path_to_checkpoint: Optional[str],
        smooth_nr: float,
        smooth_dr: float,
        sw_batch_size: int,
        infer_overlap: float,
        optim_name: str,
        optim_lr: float,
        optim_weight_decay: float,
        momentum: Optional[float],
        lrschedule_name: Optional[str],
        warmup_epochs: Optional[int],
        max_epochs: Optional[int],
        val_every: int,
        save_every: int,
        data_dir: str,
        spacing: Sequence[float],
        modality: int or Sequence,
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        RandFlipd_prob: float,
        RandRotate90d_prob: float,
        RandScaleIntensityd_prob: float,
        RandShiftIntensityd_prob: float,
        n_workers: int,
        cache_num: int,
        device: str,
        config: dict,
        **kwargs,
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'device: {device.type}')

    loader = get_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        spacing=spacing,
        modality=modality,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
        roi_size=inf_size,
        RandFlipd_prob=RandFlipd_prob,
        RandRotate90d_prob=RandRotate90d_prob,
        RandScaleIntensityd_prob=RandScaleIntensityd_prob,
        RandShiftIntensityd_prob=RandShiftIntensityd_prob,
        n_workers=n_workers,
        cache_num=cache_num,
        device=device,
    )
    print(f"Batch size is: {batch_size}")

    model = UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=(inf_size, inf_size, inf_size),
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        pos_embed=pos_embed,
        norm_name=norm_name,
        conv_block=True,
        res_block=True,
        dropout_rate=dropout_rate,
    )

    if pretrained:
        model_dict = torch.load(path_to_pretrain)
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=smooth_nr, smooth_dr=smooth_dr
    )
    post_label = AsDiscrete(to_onehot=True, n_classes=out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.NONE, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=(inf_size, inf_size, inf_size),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    start_epoch = 0

    checkpoint = None
    if path_to_checkpoint is not None:
        checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        print("=> loaded checkpoint '{}' (epoch {})".format(path_to_checkpoint, start_epoch))

    model.to(device)

    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr, weight_decay=optim_weight_decay)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optim_lr, weight_decay=optim_weight_decay)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=optim_lr, momentum=momentum, nesterov=True,
            weight_decay=optim_weight_decay
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(optim_name))

    if lrschedule_name == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs
        )
    elif lrschedule_name == "warmup_linear":
        scheduler = LinearLR(optimizer, start_factor=1e-12, end_factor=1.0, total_iters=warmup_epochs)
    elif lrschedule_name == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        scheduler = None

    run_training(
        log_dir=log_dir,
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        val_every=val_every,
        save_every=save_every,
        device=device,
        config=config,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_unetr_config.yaml")
    args = parser.parse_args()
    config_name = args.config

    print(f'Using config {config_name}')

    config_folder = "configs"
    config_path = os.path.join(config_folder, config_name)

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["config_path"] = config_path
    main(cfg)
