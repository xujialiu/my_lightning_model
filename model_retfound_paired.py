from functools import partial
from timm.models.vision_transformer import VisionTransformer
from torch import nn
import torch


class DualInputRETFoundModel(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_embed.requires_grad_(False)  # freeze positional embedding
        self.global_pool = kwargs["global_pool"]

        # Modify the fc_norm to handle the concatenated features
        self.fc_norm = nn.LayerNorm(self.embed_dim * 2, eps=1e-6)

        # Modify the head to handle the concatenated features
        self.head = nn.Linear(self.embed_dim * 2, kwargs["num_classes"])

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Process each input image
        x1 = self.forward_features(x1)
        x2 = self.forward_features(x2)

        x = torch.cat((x1, x2), dim=-1)
        print(x.shape)
        x = self.forward_head(x)

        if not self.global_pool:
            x = x[:, 0]
        return x


def create_dual_input_retfound_model(
    img_size: int = 224,
    num_classes: int = 5,
    drop_path_rate: float = 0.1,
    global_pool: str = "avg",
    **kwargs
):
    model = DualInputRETFoundModel(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=img_size,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
        **kwargs
    )

    return model
