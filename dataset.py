from pathlib import Path
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


def get_train_transform(
    input_size, color_jitter, auto_augment, re_prob, re_mode, re_count, crop_boundary
):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    transform = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation="bicubic",
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        mean=mean,
        std=std,
    )

    transform.transforms.insert(
        0, transforms.Lambda(lambda img: img.crop(crop_boundary))
    )
    return transform


def get_val_test_transform(input_size, crop_boundary):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(transforms.Lambda(lambda img: img.crop(crop_boundary)))
    t.append(
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    return transforms.Compose(t)


def get_train_dataset(
    data_path,
    input_size,
    color_jitter,
    auto_augment,
    re_prob,
    re_mode,
    re_count,
    crop_boundary,
):

    transform = get_train_transform(
        input_size,
        color_jitter,
        auto_augment,
        re_prob,
        re_mode,
        re_count,
        crop_boundary,
    )
    folder_path = Path(data_path) / "train"
    dataset = datasets.ImageFolder(folder_path, transform=transform)

    return dataset


def get_val_test_dataset(val_or_test, data_path, input_size, crop_boundary):
    transforms = get_val_test_transform(input_size, crop_boundary)
    folder_path = Path(data_path) / val_or_test
    dataset = datasets.ImageFolder(folder_path, transform=transforms)

    return dataset
