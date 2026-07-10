import torch
from torch import nn, Tensor
from torchvision.models import resnet18, resnet34, resnet50
from typing import Literal


class CustomResnet(nn.Module):

    def __init__(self, resnet: nn.Module, head: nn.Module):
        super().__init__()
        self.resnet = resnet
        self.head = head

    def forward(self, x: Tensor):
        self.validate_input(x)
        x = self.resnet(x)
        result = self.head(x)
        return result

    @staticmethod
    def validate_input(x: Tensor):
        assert isinstance(x, Tensor), f"x must be torch.Tensor, got {type(x)}"
        assert x.ndim == 4, f"Expected 4D tensor [B, C, H, W], got {x.shape}"
        
        B, C, H, W = x.shape
        assert C == 3, f"Expected 3 channels (RGB), got {C}"
        assert H > 0 and W > 0, f"Invalid image size: H={H}, W={W}"
        assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
        assert not torch.isnan(x).any(), "Input contains NaNs"
        assert not torch.isinf(x).any(), "Input contains infs"


def classification_head(input_features: int, output_features: int):
    return nn.Sequential(
        nn.Linear(input_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, output_features)
    )


def regression_head(input_features: int):
    return nn.Sequential(
        nn.Linear(input_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1)
    )


def custom_resnet_factory(
        resnet_version: Literal[18, 34, 50],
        prediction: Literal["classification", "regression"],
        n_classes: int | None = None
) -> CustomResnet:
    match resnet_version:
        case 18:
            resnet = resnet18(pretrained=True)
            input_features = 512
        case 34:
            resnet = resnet34(pretrained=True)
            input_features = 512
        case 50:
            resnet = resnet50(pretrained=True)
            input_features = 2048
        case _:
            raise ValueError

    resnet.fc = nn.Identity()

    match prediction:
        case "classification":
            if n_classes is None:
                raise ValueError("n_classes must be provided for classification")
            head = classification_head(input_features, n_classes)
        case "regression":
            head = regression_head(input_features)
        case _:
            raise ValueError

    return CustomResnet(resnet, head)
