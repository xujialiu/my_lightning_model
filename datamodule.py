from pathlib import Path
from typing import Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.utils
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    SequentialSampler,
)
import lightning.pytorch as pl
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import InterpolationMode
from timm.data import create_transform
from torchvision.transforms import Compose


class PairedImageDataset(Dataset):
    def __init__(
        self,
        df,
        macula_folder,
        disc_folder,
        macula_img_path_col="macula_filename",
        disc_img_path_col="disc_filename",
        label_col="label",
        transform=None,
    ):
        self.df = df
        self.macula_folder = macula_folder
        self.disc_folder = disc_folder
        self.macula_img_path_col = macula_img_path_col
        self.disc_img_path_col = disc_img_path_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name_macula = self.df.iloc[idx].loc[self.macula_img_path_col]
        img_name_disc = self.df.iloc[idx].loc[self.disc_img_path_col]
        label = self.df.iloc[idx].loc[self.label_col]

        img_path_macula = Path(self.macula_folder) / img_name_macula
        img_path_disc = Path(self.disc_folder) / img_name_disc

        img_macula = Image.open(img_path_macula).convert("RGB")
        img_disc = Image.open(img_path_disc).convert("RGB")

        if self.transform:
            img_macula = self.transform(img_macula)
            img_disc = self.transform(img_disc)

        return img_macula, img_disc, torch.tensor(label)


class PairedImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        excel_file: str,
        macula_folder: str,
        disc_folder: str,
        macula_img_path_col: str,
        disc_img_path_col: str,
        batch_size: int = 32,
        num_workers: int = 4,
        input_size: int = 224,
        crop_boundary: list[int] = [583, 124, 3194, 2324],
        color_jitter: float | tuple[float, ...] = 0.4,
        auto_augment: str | None = None,
        re_prob: float = 0,
        re_mode: str = "const",
        re_count: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.excel_file = excel_file
        self.macula_folder = macula_folder
        self.disc_folder = disc_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.macula_img_path_col = macula_img_path_col
        self.disc_img_path_col = disc_img_path_col

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

    def prepare_data(self):
        # 数据读取和分割操作移到这里
        data = pd.read_excel(self.excel_file)
        train_data, temp_data = train_test_split(
            data, test_size=0.3, stratify=data["label"], random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42
        )

        # 存储分割后的数据
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # Pre-compute sampling weights
        class_counts = self.train_data["label"].value_counts().sort_index()
        total_samples = len(self.train_data)
        class_weights = torch.FloatTensor(
            total_samples / (len(class_counts) * class_counts)
        )
        self.sample_weights = [
            class_weights[label] for label in self.train_data["label"]
        ]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = PairedImageDataset(
                self.train_data,
                self.macula_folder,
                self.disc_folder,
                self.macula_img_path_col,
                self.disc_img_path_col,
                self.train_transform,
            )
            self.val_dataset = PairedImageDataset(
                self.val_data,
                self.macula_folder,
                self.disc_folder,
                self.macula_img_path_col,
                self.disc_img_path_col,
                self.val_test_predict_transform,
            )

            self.sampler = WeightedRandomSampler(
                self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True,
            )

        if stage == "test" or stage is None:
            self.test_dataset = PairedImageDataset(
                self.test_data,
                self.macula_folder,
                self.disc_folder,
                self.macula_img_path_col,
                self.disc_img_path_col,
                self.val_test_predict_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class SingleImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        img_path_col: str,
        data_folder: str,
        transform: Optional[Compose] = None,
    ):
        self.df = df
        self.data_folder = data_folder
        self.transform = transform
        self.label_col = label_col
        self.img_path_col = img_path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, :].loc["macula_filename"]
        label = self.df.iloc[idx, :].loc[self.label_col]

        img_path = Path(self.data_folder) / img_name

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)


class SingleImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        excel_file: str,
        img_path_col: str,
        label_col: str,
        data_folder: str,
        batch_size: int = 32,
        num_workers: int = 4,
        input_size: int = 224,
        crop_boundary: list[int] = [583, 124, 3194, 2324],
        color_jitter: float | tuple[float, ...] = 0.4,
        auto_augment: str | None = None,
        re_prob: float = 0,
        re_mode: str = "const",
        re_count: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.excel_file = excel_file
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_path_col = img_path_col
        self.label_col = label_col

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

    def prepare_data(self):
        data = pd.read_excel(self.excel_file)

        train_data, temp_data = train_test_split(
            data, test_size=0.3, stratify=data["label"], random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42
        )

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # Pre-compute sampling weights
        class_counts = self.train_data["label"].value_counts().sort_index()
        total_samples = len(self.train_data)
        class_weights = torch.FloatTensor(
            total_samples / (len(class_counts) * class_counts)
        )
        self.sample_weights = [
            class_weights[label] for label in self.train_data["label"]
        ]

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
                df =train_data,
                label_col = self.label_col,
                img_path_col = self.img_path_col,
                data_folder =self.data_folder,
                transform =self.train_transform,
            )
            self.val_dataset = SingleImageDataset(
                df = val_data,
                label_col = self.label_col,
                img_path_col = self.img_path_col,
                data_folder =self.data_folder,
                transform =self.train_transform,
            )

            self.sampler = WeightedRandomSampler(
                self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SingleImageDataset(
                df = test_data,
                label_col = self.label_col,
                img_path_col = self.img_path_col,
                data_folder =self.data_folder,
                transform =self.train_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
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
