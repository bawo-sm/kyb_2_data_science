import torch
from torch import nn, Tensor
from torchvision.models import convnext_tiny, convnext_small
from typing import Literal


class CustomConvNeXt(nn.Module):

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor):
        self.validate_input(x)
        x = self.backbone(x)
        return self.head(x)

    @staticmethod
    def validate_input(x: Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be torch.Tensor, got {type(x)}")

        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {x.shape}")

        B, C, H, W = x.shape

        if C != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {C}")

        if H <= 0 or W <= 0:
            raise ValueError(f"Invalid image size: H={H}, W={W}")

        if x.dtype != torch.float32:
            raise TypeError(f"Expected float32, got {x.dtype}")

        if torch.isnan(x).any():
            raise ValueError("Input contains NaNs")

        if torch.isinf(x).any():
            raise ValueError("Input contains infs")

        if x.min() < -5 or x.max() > 5:
            raise ValueError(
                f"Unexpected value range: min={x.min()}, max={x.max()} "
                "(did you forget normalization?)"
            )

        if H < 224 or W < 224:
            raise ValueError(
                f"ConvNeXt expects >=224 resolution, got {H}x{W}"
            )


def classification_head(input_features: int, output_features: int):
    return nn.Sequential(
        nn.Linear(input_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, output_features)
    )


def regression_head(input_features: int):
    return nn.Sequential(
        nn.Linear(input_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )


def custom_convnext_factory(
        version: Literal["tiny", "small"],
        prediction: Literal["classification", "regression"],
        n_classes: int | None = None
) -> CustomConvNeXt:

    match version:
        case "tiny":
            backbone = convnext_tiny(pretrained=True)
            input_features = 768
        case "small":
            backbone = convnext_small(pretrained=True)
            input_features = 768
        case _:
            raise ValueError(f"Unsupported ConvNeXt version: {version}")

    backbone.classifier = nn.Identity()

    match prediction:
        case "classification":
            if n_classes is None:
                raise ValueError("n_classes must be provided for classification")
            head = classification_head(input_features, n_classes)

        case "regression":
            head = regression_head(input_features)

        case _:
            raise ValueError(f"Unsupported prediction type: {prediction}")

    return CustomConvNeXt(backbone, head)
