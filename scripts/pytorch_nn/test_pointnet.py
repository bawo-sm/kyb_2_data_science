import logging, pprint
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from scripts.pytorch_nn.supervised_trainer import PytorchSupervisedTrainer
from scripts.pytorch_nn.pointnet import PointNetRegressor, PointNetBasic


class SyntheticLidarData:
    MAX_X = 10**2
    MAX_Y = 10**2
    MAX_Z = 10**2
    POINTS_PER_OBJECT = 10**4
    CLASS_0_SUBSET = 100
    CLASS_1_SUBSET = 100

    def generate_synthetic_data(self, n: int) -> dict[int, np.ndarray]:
        x = np.array([self._generate_cloud() for i in range(n)] + [self._generate_pole() for i in range(n)])
        y = np.array([0 for i in range(n)] + [0 for i in range(n)])
        return {"x": x, "y": y}

    def _generate_cloud(self) -> np.ndarray:
        random_x = [x + np.random.rand() for x in np.random.randint(0, self.MAX_X+1, self.POINTS_PER_OBJECT)]
        random_y = [x + np.random.rand() for x in np.random.randint(0, self.MAX_Y+1, self.POINTS_PER_OBJECT)]
        random_z = [x + np.random.rand() for x in np.random.randint(0, self.MAX_Z+1, self.POINTS_PER_OBJECT)]
        points = [np.array([x, y, z]) for x, y, z in zip(random_x, random_y, random_z)]
        return points

    def _generate_pole(self) -> np.ndarray:
        random_x = [x + np.random.rand() for x in np.random.randint(0, 11, self.POINTS_PER_OBJECT)]
        random_y = [x + np.random.rand() for x in np.random.randint(0, self.MAX_Y+1, self.POINTS_PER_OBJECT)]
        random_z = [x + np.random.rand() for x in np.random.randint(0, 11, self.POINTS_PER_OBJECT)]
        points = [np.array([x, y, z]) for x, y, z in zip(random_x, random_y, random_z)]
        return points


class CustomDataset(Dataset):

    def __init__(self, n: int):
        data = SyntheticLidarData().generate_synthetic_data(n)
        self.x = data["x"]
        self.y = data["y"]
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index].astype(np.float32), self.y[index]



LEARNING_RATE = 1e-5
BATCH_SIZE = 25
EPOCHS = 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = PointNetRegressor(pointnet=PointNetBasic())
    loss_fn = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    train_dataloader = DataLoader(CustomDataset(100), batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(CustomDataset(10), batch_size=BATCH_SIZE, shuffle=True)

    history = PytorchSupervisedTrainer().train(
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer
    )
    pprint.pp(history)
    # train_history, test_history = PytorchSupervisedTrainer().train_with_early_stopping(
    #     n_epochs=100,
    #     batch_size=BATCH_SIZE,
    #     model=model,
    #     loss_fn=loss_fn,
    #     train_dataloader=train_dataloader,
    #     test_dataloader=test_dataloader,
    #     optimizer=optimizer,
    #     epochs_before_early_stopping=20,
    #     early_stopping_epochs_agg=5,
    #     min_loss_decrease=1
    # )