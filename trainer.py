import os
from typing import Callable, Optional

from torch import nn
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from monai.data import decollate_batch

from utils import utils


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: Optimizer,
        epoch: int,
        loss_func: Callable,
        device: str or torch.device,
) -> float or torch.Tensor:
    model.train()
    epoch_loss = 0.
    batch_size = loader.batch_size

    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)

        utils.zero_grad(model)

        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        epoch_loss += loss_value / batch_size
        print(
            "Epoch {} {}/{}".format(epoch, idx, len(loader)),
            "loss: {:.4f}".format(loss_value),
            )

    utils.zero_grad(model)

    return epoch_loss


def val_epoch(
        model: nn.Module,
        loader: DataLoader,
        epoch: int,
        acc_func: Callable,
        device: str or torch.device,
        model_inferer=None,
        post_label=None,
        post_pred=None
):
    if model_inferer is None:
        model_inferer = lambda x: model(x)

    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)

            if not logits.is_cuda:
                target = target.cpu()

            val_labels_list = decollate_batch(target)
            if post_label is not None:
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]

            val_outputs_list = decollate_batch(logits)
            if post_pred is not None:
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)

            acc_list = acc.detach().cpu().numpy()
            avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            print(
                "Val {} {}/{}".format(epoch, idx, len(loader)),
                "acc",
                avg_acc,
            )
    return avg_acc


def save_checkpoint(
        model: nn.Module,
        epoch: int,
        log_dir: str,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    snapshot_path = os.path.join(log_dir, 'snapshots')
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    filename = os.path.join(snapshot_path, f'chk_{epoch}.pt')
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_func: Callable,
        acc_func: Callable,
        max_epochs: int,
        log_dir: str,
        device: str or torch.device,
        val_every: int,
        save_every: int,
        model_inferer=None,
        scheduler=None,
        start_epoch: int = 0,
        post_label=None,
        post_pred=None,
):
    writer = SummaryWriter(log_dir=log_dir)

    val_acc_max = 0.0

    for epoch in range(start_epoch, max_epochs):
        print("Epoch:", epoch)
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
            device=device,
        )

        print(
            "training {}".format(epoch),
            "loss: {:.4f}".format(train_loss),
        )

        writer.add_scalar("loss/train", train_loss, epoch)

        if (epoch + 1) % val_every == 0:
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                device=device,
                model_inferer=model_inferer,
                post_label=post_label,
                post_pred=post_pred,
            )
            print(
                "validation {}".format(epoch),
                "acc/val",
                val_avg_acc,
            )
            writer.add_scalar("acc/val", val_avg_acc, epoch)

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, epoch, log_dir, optimizer=optimizer, scheduler=scheduler
            )

        if scheduler is not None:
            scheduler.step()

    return val_acc_max
