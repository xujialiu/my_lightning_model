from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns


class ValMixin:
    def on_validate_epoch_start(self):
        self.val_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat_logits = self(x1, x2)
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


class TestMixin:
    def on_test_epoch_start(self):
        self.test_outputs.clear()

    def test_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat_logits = self(x1, x2)
        y_logits = F.one_hot(y.to(torch.int64), num_classes=self.num_classes)
        y_hat = torch.argmax(y_hat_logits, dim=1)
        y_hat_probs = F.softmax(y_hat_logits, dim=1)

        loss = self.criterion(y_hat_logits, y_logits)

        self.log("test_loss", loss, on_epoch=True)

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
