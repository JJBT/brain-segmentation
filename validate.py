import os

import torch
import numpy as np
from omegaconf import OmegaConf
from monai.inferers import sliding_window_inference

from models.unetr import UNETR
from utils.data_utils import get_loader
from utils.model_utils import get_model
from utils.utils import dice, ImageSaver
from visualize import create_image_visual


def main(config: dict, title: str = ''):
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
        gauss_noise_prob=config['gauss_noise_prob'],
        gauss_noise_std=config['gauss_noise_std'],
        gauss_smooth_prob=config['gauss_smooth_prob'],
        gauss_smooth_std=config['gauss_smooth_std'],
        device=device,
        n_workers=0,
        cache_num=0,
    )[1]
    model = get_model(
        model_name=config['model_name'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        inf_size=config['inf_size'],
        feature_size=config['feature_size'],
        hidden_size=config['hidden_size'],
        mlp_dim=config['mlp_dim'],
        num_heads=config['num_heads'],
        pos_embed=config['pos_embed'],
        norm_name=config['norm_name'],
        conv_block=True,
        res_block=True,
        dropout_rate=config['dropout_rate'],
        device=device
    )
    model_dict = torch.load(config['path_to_checkpoint'], map_location=device)['state_dict']
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    labels = loader.dataset.get_properties('labels')['labels']
    image_saver = ImageSaver('images', save_name=title)

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

            dice_scores_by_class = np.round(dice_score_sample, 3).tolist()
            dice_sample = np.mean(dice_score_sample)
            sample_name = os.path.basename(batch["filename_or_obj"][0])
            title = f'Dice: {dice_sample:.3f} | {dice_scores_by_class} | {sample_name}'
            image_visualization = create_image_visual(val_inputs.cpu().numpy(), val_labels, val_outputs, title)
            image_saver.save_image(image_visualization, sample_name)

            print("Class Dice: {}".format(dice_score_sample))

        overall_dice_by_class = np.mean(dice_scores, axis=0).round(3)
        overall_dice = np.mean(dice_scores).round(3)
        print("Overall Mean Dice: {}".format(overall_dice_by_class))



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
