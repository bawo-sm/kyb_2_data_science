import logging
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from scripts.pytorch_nn.trainer import PytorchTrainer


class TestNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class RandomDataset(Dataset):

    def __init__(self):
        self.x = np.random.randint(0, 100, (100, 2), dtype=int)
        self.y = np.random.randint(0, 100, 100, dtype=int)
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index].astype(np.float32), self.y[index]



LEARNING_RATE = 1e-3
BATCH_SIZE = 1
EPOCHS = 5


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = TestNeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    train_dataloader = DataLoader(RandomDataset(), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(RandomDataset(), batch_size=64, shuffle=True)

    # train_history, test_history = PytorchTrainer().train(
    #     n_epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     model=model,
    #     loss_fn=loss_fn,
    #     train_dataloader=train_dataloader,
    #     test_dataloader=test_dataloader,
    #     optimizer=optimizer
    # )

    train_history, test_history = PytorchTrainer().train_with_early_stopping(
        n_epochs=100,
        batch_size=BATCH_SIZE,
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        epochs_before_early_stopping=20,
        early_stopping_epochs_agg=5,
        min_loss_decrease=1
    )