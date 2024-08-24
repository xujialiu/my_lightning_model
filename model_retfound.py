from functools import partial

from timm.models.vision_transformer import VisionTransformer
from torch import nn
import torch


class RETFoundModel(VisionTransformer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_embed.requires_grad_(False)  # freeze positional embedding
        self.global_pool = kwargs["global_pool"]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        if not self.global_pool:
            x = x[:, 0]
            print(x.shape)
            
        print(x.shape)
        return x


def create_retfound_model(img_size, num_classes, drop_path_rate, global_pool, **kwargs):

    model = RETFoundModel(
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