import lightning.pytorch as pl
from torch.nn.init import trunc_normal_
from torch.optim.adamw import AdamW
import lr_sched
from model_retfound import create_retfound_model
import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch.nn.functional as F
from retfoud_lr_decay import param_groups_lrd


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

        # 初始化指标
        # train_step指标
        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )

        # test_step指标
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

        # val_step指标
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.val_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro", top_k=1
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro", top_k=1
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro", top_k=1
        )

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        self.opt = self.optimizers()

    def training_step(self, batch, batch_idx):
        # 每一个batch都要更新学习率
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            lr_sched.adjust_learning_rate(
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

        lr, lr_scale = (
            self.optimizers().param_groups[-1]["lr"],
            self.optimizers().param_groups[-1]["lr_scale"],
        )
        effective_lr = lr * lr_scale

        print(f"effective lr: {effective_lr}")

        x, y = batch
        y_hat_logits = self(x)
        y_logits = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        y_hat = torch.argmax(y_hat_logits, dim=1)

        loss = self.criterion(y_hat_logits, y_logits)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.train_accuracy(y_hat, y)
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "learning_rate", effective_lr, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self(x)
        y_logits = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        y_hat = torch.argmax(y_hat_logits, dim=1)

        loss = self.criterion(y_hat_logits, y_logits)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.val_accuracy(y_hat, y)
        self.log(
            "val_acc", self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self(x)
        y_logits = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        y_hat = torch.argmax(y_hat_logits, dim=1)
        loss = self.criterion(y_hat_logits, y_logits)

        self.log("test_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        self.test_accuracy(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)

        self.log(
            "test_acc", self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_precision",
            self.test_precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_recall", self.test_recall, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("test_f1", self.test_f1, on_step=True, on_epoch=True, prog_bar=True)

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
