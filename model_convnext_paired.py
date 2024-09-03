from typing import Optional
import torch
import torch.nn as nn
from timm import create_model
from timm.layers import NormMlpClassifierHead


class DualInputConvNeXt(nn.Module):
    def __init__(
        self,
        img_size=512,
        pretrained=True,
        num_classes: int = 2,
        global_pool: str = "avg",
        drop_path_rate: float = 0.1,
        head_hidden_size: Optional[int] = None,
    ):
        super(DualInputConvNeXt, self).__init__()

        # 创建一个ConvNeXt模型实例
        self.convnext = create_model(
            f"convnextv2_huge.fcmae_ft_in22k_in1k_{img_size}",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

        # 替换最后的head块
        self.convnext.head = NormMlpClassifierHead(
            in_features=self.convnext.num_features * 2,
            num_classes=num_classes,
            hidden_size=head_hidden_size,
            pool_type=global_pool,
            act_layer="gelu",
        )

    def forward(self, x1, x2):
        # 使用同一个ConvNeXt模型处理两张图片
        features1 = self.convnext.forward_features(x1)
        features2 = self.convnext.forward_features(x2)

        # 连接特征
        combined_features = torch.cat((features1, features2), dim=1)

        # 通过分类器
        output = self.convnext.forward_head(combined_features)

        return output
