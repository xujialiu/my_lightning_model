import lightning.pytorch as pl
from model_retfound import create_retfound_model
import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch.nn.functional as F


class RETFoundLightning(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        learning_rate: float = 0.001,
        num_classes: int = 5,
        drop_path_rate: float = 0.1,
        global_pool: str = "avg",
        mixup: float = 0,
        cutmix: float = 0,
        cutmix_minmax=None,
        mixup_prob: float = 1.0,
        mixup_switch_prob: float = 0.5,
        mixup_mode: str = "batch",
        smoothing: float = 0.1,
    ):
        super().__init__()

        pl.seed_everything(1)

        self.learning_rate = learning_rate
        self.model = create_retfound_model(
            img_size, num_classes, drop_path_rate, global_pool
        )
        self.num_classes = num_classes

        mixup_active = (mixup > 0) or (cutmix > 0.0) or (cutmix_minmax is not None)
        # mixup
        if mixup_active:
            print("Mixup is activated!")

        mixup_fn = Mixup(
            mixup_alpha=mixup,
            cutmix_alpha=cutmix,
            cutmix_minmax=cutmix_minmax,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
            mode=mixup_mode,
            label_smoothing=smoothing,
            num_classes=num_classes,
        )

        # 定义损失函数
        if mixup_fn:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif smoothing > 0.0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # 初始化指标
        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.test_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro", top_k=1
        )
        self.test_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro", top_k=1
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro", top_k=1
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        y = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        self.train_accuracy(logits, y)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        y = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)

        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_accuracy(logits, y)
        self.log(
            "val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)

        loss = self.criterion(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.test_accuracy(logits, y)
        self.test_precision(logits, y)
        self.test_recall(logits, y)
        self.test_f1(logits, y)

        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
