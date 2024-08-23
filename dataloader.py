from collections import Counter

from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler
from dataset import get_train_dataset, get_val_test_dataset


def get_train_sampler(dataset):
    # 计算每个类别的样本数量
    class_counts = Counter(dataset.targets)

    # 计算权重
    weights = [1.0 / class_counts[label] for label in dataset.targets]

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_val_test_sampler(dataset):
    return SequentialSampler(dataset)


def get_train_loader(
    data_path,
    train_batch_size,
    num_workers,
    input_size,
    color_jitter,
    auto_augment,
    re_prob,
    re_mode,
    re_count,
    crop_boundary,
):
    train_dataset = get_train_dataset(
        data_path,
        input_size,
        color_jitter,
        auto_augment,
        re_prob,
        re_mode,
        re_count,
        crop_boundary,
    )
    train_sampler = get_train_sampler(train_dataset)

    return DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_val_test_loader(
    val_or_test,
    data_path,
    num_workers,
    val_or_test_batch_size,
    input_size,
    crop_boundary,
):

    val_or_test_dataset = get_val_test_dataset(
        val_or_test, data_path, input_size, crop_boundary
    )
    val_or_test_sampler = get_val_test_sampler(val_or_test_dataset)

    return DataLoader(
        val_or_test_dataset,
        sampler=val_or_test_sampler,
        batch_size=val_or_test_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
