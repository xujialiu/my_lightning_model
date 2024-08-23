from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import InterpolationMode
from timm.data import create_transform


class PairedImageDataset(Dataset):
    def __init__(self, data, data1_folder, data2_folder, transform=None):
        self.data = data
        self.data1_folder = data1_folder
        self.data2_folder = data2_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_name = self.data.iloc[idx, 0]
        img2_name = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]

        img1_path = Path(self.data1_folder, img1_name)
        img2_path = Path(self.data2_folder, img2_name)

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label)


class PairedImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        excel_file: str,
        data1_folder: str,
        data2_folder: str,
        batch_size: int = 32,
        num_workers: int = 4,
        input_size: int = 224,
        crop_boundary: list[int] = [0, 0, 256, 256],
        color_jitter: float | tuple[float, ...] = 0.4,
        auto_augment: str | None = None,
        re_prob: float = 0,
        re_mode: str = "const",
        re_count: int = 1,
    ):
        super().__init__()
        self.excel_file = excel_file
        self.data1_folder = data1_folder
        self.data2_folder = data2_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = get_train_transform(
            input_size=input_size,
            crop_boundary=crop_boundary,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
        )

        self.val_test_predict_transform = get_val_test_predict_transform(
            input_size=input_size, crop_boundary=crop_boundary
        )

    def setup(self, stage: Optional[str] = None):

        data = pd.read_excel(self.excel_file)
        train_data, temp_data = train_test_split(
            data, test_size=0.3, stratify=data["label"], random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42
        )

        if stage == "fit" or stage is None:
            self.train_dataset = PairedImageDataset(
                train_data, self.data1_folder, self.data2_folder, self.transform
            )
            self.val_dataset = PairedImageDataset(
                val_data, self.data1_folder, self.data2_folder, self.transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = PairedImageDataset(
                test_data, self.data1_folder, self.data2_folder, self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class SingleImageDataset(Dataset):
    def __init__(self, data, data_folder, transform=None):
        self.data = data
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, :].loc[:, "macula_filename"]
        label = self.data.iloc[idx, :].loc[:, "label"]

        img_path = Path(self.data_folder, img_name)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)


class SingleImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        excel_file: str,
        data_folder: str,
        batch_size: int = 32,
        num_workers: int = 4,
        input_size: int = 224,
        crop_boundary: list[int] = [0, 0, 256, 256],
        color_jitter: float | tuple[float, ...] = 0.4,
        auto_augment: str | None = None,
        re_prob: float = 0,
        re_mode: str = "const",
        re_count: int = 1,
    ):
        super().__init__()
        self.excel_file = excel_file
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = get_train_transform(
            input_size=input_size,
            crop_boundary=crop_boundary,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
        )

        self.val_test_predict_transform = get_val_test_predict_transform(
            input_size=input_size, crop_boundary=crop_boundary
        )

    def setup(self, stage: Optional[str] = None):
        data = pd.read_excel(self.excel_file)
        train_data, temp_data = train_test_split(
            data, test_size=0.3, stratify=data["label"], random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42
        )

        if stage == "fit" or stage is None:
            self.train_dataset = SingleImageDataset(
                train_data, self.data_folder, self.train_transform
            )
            self.val_dataset = SingleImageDataset(
                val_data, self.data_folder, self.val_test_predict_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = SingleImageDataset(
                test_data, self.data_folder, self.val_test_predict_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


def get_val_test_predict_transform(input_size: int, crop_boundary: list[int]):
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


def get_train_transform(
    input_size: int,
    crop_boundary: list[int],
    color_jitter: float | tuple[float, ...] = 0.4,
    auto_augment: str | None = None,
    re_prob: float = 0,
    re_mode: str = "const",
    re_count: int = 1,
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
