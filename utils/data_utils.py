import os
from typing import Sequence

from monai import data, transforms, apps
from torch.utils.data import Subset


def get_msd_vanilla_transforms(
        roi_size: int,
        spacing: Sequence[float],
        modality: int or Sequence = 0,
        **kwargs,
):
    modality = [modality] if isinstance(modality, int) else modality    # modality = 0
    train_transforms_list = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AsChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[modality]),
        transforms.AddChanneld(keys=["label"]),
        transforms.Spacingd(
            keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
        ),
        transforms.ScaleIntensityd(keys=["image"]),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.RandSpatialCropd(keys=["image", "label"],
                                    roi_size=(roi_size, roi_size, roi_size), random_size=False),
        transforms.Resized(keys=["image", "label"], spatial_size=(roi_size, roi_size, roi_size)),
        # transforms.RandCropByPosNegLabeld(
        #     keys=["image", "label"],
        #     label_key="label",
        #     spatial_size=(roi_size, roi_size, roi_size),
        #     pos=1,
        #     neg=1,
        #     num_samples=4,
        #     image_key="image",
        #     image_threshold=0,
        # ),
        transforms.ToTensord(keys=["image", "label"]),
    ]

    val_transforms_list = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AsChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[modality]),
        transforms.AddChanneld(keys=["label"]),
        transforms.Spacingd(
            keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
        ),
        transforms.ScaleIntensityd(keys=["image"]),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.ToTensord(keys=["image", "label"]),
    ]

    train_transform = transforms.Compose(train_transforms_list)
    val_transform = transforms.Compose(val_transforms_list)
    return train_transform, val_transform


def get_loader(
        data_dir: str,
        batch_size: int,
        spacing: Sequence[float],
        modality: int or Sequence,
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        roi_size: int,
        RandFlipd_prob: float,
        RandRotate90d_prob: float,
        RandScaleIntensityd_prob: float,
        RandShiftIntensityd_prob: float,
        n_workers: int,
        cache_num: int,
        **kwargs,
):
    train_transform, val_transform = get_msd_vanilla_transforms(
        roi_size=roi_size,
        spacing=spacing,
        modality=modality,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
    )

    train_ds = apps.DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        section="training",
        transform=train_transform,
        download=False,
        cache_num=cache_num,
    )
    # train_ds = Subset(train_ds, indices=range(0, len(train_ds), 100))

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )

    val_ds = apps.DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        section="validation",
        transform=val_transform,
        download=False,
        cache_num=cache_num,
    )
    # val_ds = Subset(val_ds, indices=range(0, len(val_ds), 20))
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=n_workers > 0,
        drop_last=True,
    )
    loader = [train_loader, val_loader]

    return loader
