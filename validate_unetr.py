import os

import torch
import numpy as np
from omegaconf import OmegaConf
from monai.inferers import sliding_window_inference

from models.unetr import UNETR
from utils.data_utils import get_loader
from utils.utils import dice
from visualize import create_image_visual


def main(config: dict):
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device('cpu')
    loader = get_loader(
        data_dir=os.path.expanduser(config['data_dir']),
        batch_size=1,
        roi_size=config['inf_size'],
        spacing=config['spacing'],
        modality=config['modality'],
        a_min=config['a_min'],
        a_max=config['a_max'],
        b_min=config['b_min'],
        b_max=config['b_max'],
        RandFlipd_prob=config['RandFlipd_prob'],
        RandRotate90d_prob=config['RandRotate90d_prob'],
        RandScaleIntensityd_prob=config['RandScaleIntensityd_prob'],
        RandShiftIntensityd_prob=config['RandShiftIntensityd_prob'],
        n_workers=0,
        cache_num=0,
    )[1]
    model = UNETR(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        img_size=(config['inf_size'], config['inf_size'], config['inf_size']),
        feature_size=config['feature_size'],
        hidden_size=config['hidden_size'],
        mlp_dim=config['mlp_dim'],
        num_heads=config['num_heads'],
        pos_embed=config['pos_embed'],
        norm_name=config['norm_name'],
        conv_block=True,
        res_block=True,
        dropout_rate=config['dropout_rate'],
    )
    model_dict = torch.load(config['path_to_checkpoint'], map_location=device)['state_dict']
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    labels = loader.dataset.get_properties('labels')['labels']

    with torch.inference_mode():
        dice_scores = []
        for i, batch in enumerate(loader):
            val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
            val_outputs = sliding_window_inference(
                val_inputs,
                roi_size=(config['inf_size'], config['inf_size'], config['inf_size']),
                sw_batch_size=config['sw_batch_size'],
                predictor=model,
                overlap=config['infer_overlap']
            )
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy().squeeze()
            dice_score_sample = [
                dice(val_outputs == int(label), val_labels == int(label)) for label in labels
            ]
            dice_scores.append(dice_score_sample)
            print(val_inputs.shape, val_outputs.shape, val_labels.shape)
            create_image_visual(val_inputs.cpu().numpy(), val_labels, val_outputs, f'{i}_{round(np.mean(dice_score_sample), 3)}')

            print("Class Dice: {}".format(dice_score_sample))
        print("Overall Mean Dice: {}".format(np.mean(dice_scores, axis=0)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--path_to_checkpoint', type=str, default=None)
    args = parser.parse_args()
    path_to_config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(path_to_config, resolve=True)

    if args.path_to_checkpoint is not None:
        config['path_to_checkpoint'] = args.path_to_checkpoint

    main(config)
