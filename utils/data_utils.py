import os
from typing import Sequence

from monai import data, transforms, apps
from monai.data import load_decathlon_datalist
from torch.utils.data import Subset


def get_msd_vanilla_transforms(
        roi_size: int,
        spacing: Sequence[float],
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        modality: int or Sequence = 0,
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
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(roi_size, roi_size, roi_size),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
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


def get_btcv_transforms(
        spacing: Sequence[float],
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        roi_size: Sequence[int],
        RandFlipd_prob: float,
        RandRotate90d_prob: float,
        RandScaleIntensityd_prob: float,
        RandShiftIntensityd_prob: float,
):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),

            transforms.Spacingd(
                keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transform, val_transform


def get_loader(
        data_dir: str,
        json_name: str,
        batch_size: int,
        test_mode: bool,
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

):

    datalist_json = os.path.join(data_dir, json_name)

    train_transform, val_transform = get_msd_vanilla_transforms(
        roi_size=roi_size,
        spacing=spacing,
        modality=modality,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
    )

    if test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)

        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        # datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)

        # train_ds = data.CacheDataset(
        #     data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=n_workers
        # )

        train_ds = apps.DecathlonDataset(
            root_dir=data_dir,
            task="Task01_BrainTumour",
            section="training",
            transform=train_transform,
            download=False,
            cache_num=cache_num,
        )
        # train_ds = Subset(train_ds, indices=range(0, len(train_ds), 40))

        train_loader = data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )
        # val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        # val_ds = data.Dataset(data=val_files, transform=val_transform)

        val_ds = apps.DecathlonDataset(
            root_dir=data_dir,
            task="Task01_BrainTumour",
            section="validation",
            transform=val_transform,
            download=False,
            cache_num=cache_num,
        )
        # val_ds = Subset(val_ds, indices=range(0, len(val_ds), 40))
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
