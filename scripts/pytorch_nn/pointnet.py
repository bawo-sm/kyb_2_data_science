import torch
import torch.nn as nn
import torch.nn.functional as F


FIRST_MLP_LAYER = 64
SECOND_MLP_LAYER = 128
THIRD_MLP_LAYER = 1024


class PointNetUtils:

    @staticmethod
    def validate_input(x: torch.Tensor):
        assert isinstance(x, torch.Tensor), f"x must be torch.Tensor, got {type(x)}"
        assert x.ndim == 3, f"Expected 3D tensor [B, N, 3], got {x.shape}"
        assert x.shape[2] == 3, f"Last dim must be 3 (XYZ), got {x.shape}"
        assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
        assert not torch.isnan(x).any(), "Input contains NaNs"

    @staticmethod
    def get_basic_mlp() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(3, FIRST_MLP_LAYER, 1),
            nn.BatchNorm1d(FIRST_MLP_LAYER),
            nn.ReLU(),

            nn.Conv1d(FIRST_MLP_LAYER, SECOND_MLP_LAYER, 1),
            nn.BatchNorm1d(SECOND_MLP_LAYER),
            nn.ReLU(),

            nn.Conv1d(SECOND_MLP_LAYER, THIRD_MLP_LAYER, 1),
            nn.BatchNorm1d(THIRD_MLP_LAYER),
            nn.ReLU()
        )


class PointNetBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = PointNetUtils.get_basic_mlp()

    def forward(self, x: torch.Tensor):
        PointNetUtils.validate_input(x)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]
        return x


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, FIRST_MLP_LAYER, 1),
            nn.ReLU(),
            nn.Conv1d(FIRST_MLP_LAYER, SECOND_MLP_LAYER, 1),
            nn.ReLU(),
            nn.Conv1d(SECOND_MLP_LAYER, THIRD_MLP_LAYER, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(THIRD_MLP_LAYER, 512),
            nn.ReLU(),
            nn.Linear(512, k * k)
        )

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.conv(x)
        x = torch.max(x, dim=2)[0]   # [B, THIRD_MLP_LAYER]
        x = self.fc(x)

        identity = torch.eye(self.k, device=x.device).view(1, -1)
        x = x + identity

        x = x.view(B, self.k, self.k)
        return x


class PointNetWithTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet = TNet(k=3)
        self.mlp = PointNetUtils.get_basic_mlp()

    def forward(self, x: torch.Tensor):
        PointNetUtils.validate_input(x)

        x = x.transpose(1, 2)
        T = self.tnet(x)
        x = torch.bmm(T, x)
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]
        return x


class PointNetClassifier(nn.Module):

    def __init__(self, pointnet: PointNetBasic | PointNetWithTNet, num_classes: int):
        super().__init__()
        self.pointnet = pointnet
        self.head = nn.Linear(THIRD_MLP_LAYER, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.pointnet(x)
        return self.head(h)


class PointNetRegressor(nn.Module):

    def __init__(self, pointnet: PointNetBasic | PointNetWithTNet):
        super().__init__()
        self.pointnet = pointnet
        self.head = nn.Linear(THIRD_MLP_LAYER, 1)

    def forward(self, x: torch.Tensor):
        h = self.pointnet(x)
        return self.head(h)


class PointNetEmbedder(nn.Module):

    def __init__(self, pointnet: PointNetBasic | PointNetWithTNet, emb_size: int):
        super().__init__()
        self.pointnet = pointnet
        self.head = nn.Linear(THIRD_MLP_LAYER, emb_size)

    def forward(self, x: torch.Tensor):
        h = self.pointnet(x)
        return self.head(h)
    