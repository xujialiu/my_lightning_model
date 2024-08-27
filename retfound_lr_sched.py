import math


def adjust_learning_rate(optimizer, current_epoch, warmup_epochs, lr, min_lr, epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if current_epoch < warmup_epochs:
        lr = lr * current_epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (current_epoch - warmup_epochs)
                / (epochs - warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
