import lightning.pytorch as pl
from matplotlib import pyplot as plt
from torch.nn.init import trunc_normal_
from torch.optim.adamw import AdamW
import retfound_lr_sched
from retfound_pos_embed import interpolate_pos_embed
from model_retfound import create_retfound_model
import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from torchmetrics import MetricCollection
import torch.nn.functional as F
from retfoud_lr_decay import param_groups_lrd
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix
import seaborn as sns


class RETFoundLightning(pl.LightningModule):
    def __init__(
        self,
        use_original_retfound_ckpt: str = None,
        img_size: int = 224,
        batch_size: int = 32,
        learning_rate: float = None,
        base_learning_rate: float = 5e-3,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
        weight_decay: float = 0.05,
        layer_decay: float = 0.65,
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

        pl.seed_everything(42)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_learning_rate = base_learning_rate
        self.use_original_retfound_ckpt = use_original_retfound_ckpt
        self.num_classes = num_classes

        self.model = create_retfound_model(
            img_size, num_classes, drop_path_rate, global_pool
        )

        if use_original_retfound_ckpt:
            checkpoint = torch.load(use_original_retfound_ckpt, map_location="cpu")
            checkpoint_model = checkpoint["model"]

            interpolate_pos_embed(self.model, checkpoint_model)

            self.model.load_state_dict(checkpoint_model, strict=False)
            trunc_normal_(self.model.head.weight, std=2e-5)

        mixup_active = (mixup > 0) or (cutmix > 0.0) or (cutmix_minmax is not None)

        # mixup指标
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

        self.train_metrics = self._get_train_metrics()
        self.val_metrics = self._get_eval_metrics(prefix="val_")
        self.test_metrics = self._get_eval_metrics(prefix="test_")
        self.val_confusion_matrix = self._get_confusion_matrix()

    def _get_train_metrics(self):
        if self.num_classes == 2:
            return MetricCollection({"train_acc": BinaryAccuracy()})
        else:
            return MetricCollection(
                {"train_acc": MulticlassAccuracy(num_classes=self.num_classes, top_k=1)}
            )

    def _get_eval_metrics(self, prefix=""):
        if self.num_classes == 2:
            return MetricCollection(
                {
                    "acc": BinaryAccuracy(),
                    "auc_roc": BinaryAUROC(),
                    "auc_pr": BinaryAveragePrecision(),
                    "f1": BinaryF1Score(),
                    "mcc": BinaryMatthewsCorrCoef(),
                    "precision": BinaryPrecision(),
                    "recall": BinaryRecall(),
                    "specificity": BinarySpecificity(),
                },
                prefix=prefix,
            )
        else:
            return MetricCollection(
                {
                    "acc_macro": MulticlassAccuracy(
                        num_classes=self.num_classes, average="macro", top_k=1
                    ),
                    "acc_weighted": MulticlassAccuracy(
                        num_classes=self.num_classes, average="weighted", top_k=1
                    ),
                    "sensitivity": MulticlassRecall(
                        num_classes=self.num_classes, average="macro"
                    ),
                    "specificity": MulticlassSpecificity(
                        num_classes=self.num_classes, average="macro"
                    ),
                    "precision": MulticlassPrecision(
                        num_classes=self.num_classes, average="macro"
                    ),
                    "auc_roc": MulticlassAUROC(
                        num_classes=self.num_classes, average="macro"
                    ),
                    "auc_pr": MulticlassAveragePrecision(
                        num_classes=self.num_classes, average="macro"
                    ),
                    "f1": MulticlassF1Score(
                        num_classes=self.num_classes, average="macro"
                    ),
                    "mcc": MulticlassMatthewsCorrCoef(num_classes=self.num_classes).to(
                        torch.float32
                    ),
                },
                prefix=prefix,
            )

    def _get_confusion_matrix(self):
        if self.num_classes == 2:
            return BinaryConfusionMatrix()
        else:
            return MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        self.opt = self.optimizers()

    def training_step(self, batch, batch_idx):
        # 每一个batch都要更新学习率
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            retfound_lr_sched.adjust_learning_rate(
                self.optimizers(),
                (
                    batch_idx / self.trainer.num_training_batches
                    + self.trainer.current_epoch
                ),
                warmup_epochs=self.warmup_epochs,
                min_lr=self.min_lr,
                lr=self.learning_rate,
                epochs=self.trainer.max_epochs,
            )

        x, y = batch
        y_hat_logits = self(x)
        y_logits = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        y_hat = torch.argmax(y_hat_logits, dim=1)
        y_hat_probs = F.softmax(y_hat_logits, dim=1)

        # log train loss
        loss = self.criterion(y_hat_logits, y_logits)
        self.log("train_loss_epoch", loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log("train_loss_step", loss, on_epoch=False, on_step=True, prog_bar=True)

        # log train metrics
        train_metrics = self.train_metrics(y_hat_probs, y)
        self.log_dict(train_metrics, on_step=True, prog_bar=True)

        # log train lr
        lr, lr_scale = (
            self.optimizers().param_groups[-1]["lr"],
            self.optimizers().param_groups[-1]["lr_scale"],
        )
        effective_lr = lr * lr_scale
        self.log("lr", effective_lr, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self(x)
        y_logits = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        y_hat = torch.argmax(y_hat_logits, dim=1)
        y_hat_probs = F.softmax(y_hat_logits, dim=1)

        self.val_confusion_matrix(y_hat, y)

        loss = self.criterion(y_hat_logits, y_logits)
        self.log("val_loss", loss, on_epoch=True)

        # log val metrics
        val_metrics = self.val_metrics(y_hat_probs, y)
        if "val_mcc" in val_metrics:
            val_metrics["val_mcc"] = val_metrics["val_mcc"].float()
        self.log_dict(val_metrics, on_epoch=True)

    def on_validation_epoch_end(self):
        confusion_matrix = self.val_confusion_matrix.compute()
        self.save_confusion_metrics(confusion_matrix)
        self.val_confusion_matrix.reset()

    def save_confusion_metrics(self, confusion_matrix):
        # 归一化混淆矩阵
        norm_confusion_matrix = confusion_matrix / confusion_matrix.sum(
            dim=1, keepdim=True
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            norm_confusion_matrix.cpu().numpy(), annot=True, fmt=".2f", cmap="Blues"
        )

        plt.title(f"Normalized Confusion Matrix - Epoch {self.current_epoch}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.savefig(
            f"{self.trainer.logger.log_dir}/normalized_confusion_matrix_epoch_{self.current_epoch}.png"
        )
        plt.close()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self(x)
        y_logits = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        y_hat = torch.argmax(y_hat_logits, dim=1)
        y_hat_probs = F.softmax(y_hat_logits, dim=1)

        loss = self.criterion(y_hat_logits, y_logits)

        self.log("test_loss", loss, on_epoch=True)

        # log test metrics
        test_metrics = self.test_metrics(y_hat_probs, y)
        self.log_dict(test_metrics, on_epoch=True)

    def configure_optimizers(self):

        if self.learning_rate is None:
            eff_batch_size = (
                self.batch_size
                * self.trainer.accumulate_grad_batches
                * self.trainer.world_size
            )
            self.learning_rate = self.base_learning_rate * eff_batch_size / 256.0
            print(f"base lr: {self.base_learning_rate}")
            print(f"actual lr: {self.learning_rate}")

        param_groups = param_groups_lrd(
            self.model,
            self.weight_decay,
            no_weight_decay_list=["pos_embed", "cls_token", "dist_token"],
            layer_decay=self.layer_decay,
        )

        # 3. 初始化AdamW优化器
        optimizer = AdamW(param_groups, lr=self.learning_rate)

        return [optimizer], []
