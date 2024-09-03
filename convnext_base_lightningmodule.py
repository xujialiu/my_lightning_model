from typing import Literal
import lightning.pytorch as pl
from matplotlib import pyplot as plt
from torch.nn.init import trunc_normal_
from torch.optim.adamw import AdamW
import convnext_lr_sched as lr_sched
from retfound_pos_embed import interpolate_pos_embed
from model_retfound import create_retfound_model
import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from convnext_optim_factory import create_optimizer, LayerDecayValueAssigner
from timm.data.mixup import Mixup
from timm import create_model
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


class ConvNextLightning(pl.LightningModule):
    def __init__(
        self,
        pretrained_from_timm: bool = True,
        img_size: Literal[384, 512] = 512,
        batch_size: int = 32,
        learning_rate: float = None,
        base_learning_rate: float = 6.25e-4,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
        weight_decay: float = 0.05,
        layer_decay: float = 0.65,
        layer_decay_type: Literal["single", "group"] = "single",
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
        is_save_confusion_matrix: bool = True,
    ):
        super().__init__()

        pl.seed_everything(42)

        self.val_outputs = {}
        self.test_outputs = {}

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_learning_rate = base_learning_rate
        self.num_classes = num_classes
        self.is_save_confusion_matrix = is_save_confusion_matrix
        self.layer_decay_type = layer_decay_type

        self.model = create_model(
            f"convnextv2_huge.fcmae_ft_in22k_in1k_{img_size}",
            pretrained=pretrained_from_timm,
            num_classes=num_classes,
        )

        self.model.depths = [3, 3, 27, 3]

        self.model.drop_rate = drop_path_rate

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

    def _get_layer_decay_assigner(self):
        if self.layer_decay < 1.0 or self.layer_decay > 1.0:
            assert self.layer_decay_type in ["single", "group"]
            if self.layer_decay_type == "group":  # applies for Base and Large models
                num_layers = 12
            else:
                num_layers = sum(self.model.depths)
            self.assigner = LayerDecayValueAssigner(
                list(
                    self.layer_decay ** (num_layers + 1 - i)
                    for i in range(num_layers + 2)
                ),
                depths=self.model.depths,
                layer_decay_type=self.layer_decay_type,
            )
        else:
            self.assigner = None

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
                    "sensitivity": BinaryRecall(),
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
                    "mcc": MulticlassMatthewsCorrCoef(num_classes=self.num_classes),
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

    def _lr_scheduler(self, batch, batch_idx):
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            lr_sched.adjust_learning_rate(
                optimizer=self.optimizers(),
                epoch=self.current_epoch,
                epochs=self.trainer.max_epochs,
                warmup_epochs=self.warmup_epochs,
                min_lr=self.min_lr,
                lr=self.learning_rate,
            )

    def training_step(self, batch, batch_idx):
        # 每一个batch都要更新学习率
        self._lr_scheduler(batch, batch_idx)

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
        if self.num_classes == 2:
            y_hat_probs = y_hat_probs[:, 1]
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

        loss = self.criterion(y_hat_logits, y_logits)
        self.log("val_loss", loss, on_epoch=True)

        # log val metrics
        if self.num_classes == 2:
            y_hat_probs = y_hat_probs[:, 1]

        self.val_outputs.setdefault("y_hat_probs", []).append(y_hat_probs)
        self.val_outputs.setdefault("y", []).append(y)
        return loss

    def on_validation_epoch_end(self):

        # log val metrics
        y_hat_probs = torch.cat(self.val_outputs["y_hat_probs"], dim=0)
        y = torch.cat(self.val_outputs["y"], dim=0)
        val_metrics = self.val_metrics(y_hat_probs, y)
        self.log_dict(val_metrics, on_epoch=True)

        # save and plot val confusion matrix
        if self.is_save_confusion_matrix:
            confusion_matrix = self.val_confusion_matrix(y_hat_probs, y)
            self._save_confusion_metrics_fig(confusion_matrix)
            self.val_confusion_matrix.reset()

        self.val_outputs.clear()

    def _save_confusion_metrics_fig(self, confusion_matrix):
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

    def on_test_epoch_start(self):
        self.test_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self(x)
        y_logits = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        y_hat = torch.argmax(y_hat_logits, dim=1)
        y_hat_probs = F.softmax(y_hat_logits, dim=1)

        # log test metrics
        if self.num_classes == 2:
            y_hat_probs = y_hat_probs[:, 1]

        self.test_outputs.setdefault("y_hat_probs", []).append(y_hat_probs)
        self.test_outputs.setdefault("y", []).append(y)

        loss = self.criterion(y_hat_logits, y_logits)
        self.log("test_loss", loss, on_epoch=True)

    def on_test_epoch_end(self):
        y_hat_probs = torch.cat(self.test_outputs["y_hat_probs"], dim=0)
        y = torch.cat(self.test_outputs["y"], dim=0)

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

        self._get_layer_decay_assigner()

        # 3. 初始化AdamW优化器
        # optimizer = AdamW(param_groups, lr=self.learning_rate)
        optimizer = create_optimizer(
            optimizer_name="adamw",
            model=self.model,
            weight_decay=self.weight_decay,
            lr=self.learning_rate,
            opt_eps=None,
            opt_betas=None,
            momentum=None,
            get_num_layer=(
                self.assigner.get_layer_id if self.assigner is not None else None
            ),
            get_layer_scale=(
                self.assigner.get_scale if self.assigner is not None else None
            ),
            filter_bias_and_bn=True,
            skip_list=None,
        )

        return [optimizer], []
