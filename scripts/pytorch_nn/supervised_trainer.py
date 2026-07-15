import torch, logging
from torch import nn, optim
from torch.utils.data import DataLoader


logger = logging.getLogger(name="PytorchTrainer")


class PytorchSupervisedTrainer:
    
    _model: nn.Module = None
    _loss_fn: nn.Module = None

    def train(
            self,
            n_epochs: int,
            batch_size: int,
            model: nn.Module,
            loss_fn: nn.Module,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            optimizer: optim.Optimizer
    ) -> dict[int, dict[str, list[float] | float]]:
        assert n_epochs > 0

        self._model = model
        self._loss_fn = loss_fn

        history = {}

        for i in range(n_epochs):
            logger.info(f"Epoch {i+1} / {n_epochs}")
            train_loss = self._step_train(train_dataloader, optimizer)
            test_loss = self._step_test(test_dataloader)
            history[i] = {"train": train_loss, "test": [test_loss]}
        
        return model, history

    def train_with_early_stopping(
            self,
            n_epochs: int,
            batch_size: int,
            model: nn.Module,
            loss_fn: nn.Module,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            optimizer: optim.Optimizer,
            epochs_before_early_stopping: int,
            early_stopping_epochs_agg: int,
            min_loss_decrease: float
    ) -> dict[int, dict[str, list[float] | float]]:
        assert n_epochs > 0
        assert n_epochs > epochs_before_early_stopping
        assert epochs_before_early_stopping > early_stopping_epochs_agg

        self._model = model
        self._loss_fn = loss_fn

        test_loss_history = []
        history = {}

        for i in range(epochs_before_early_stopping):
            logger.info(f"Epoch {i+1} / {n_epochs}")
            train_loss = self._step_train(train_dataloader, optimizer)
            test_loss = self._step_test(test_dataloader)

            test_loss_history.append(test_loss)
            history[i] = {"train": train_loss, "test": [test_loss]}

        for j in range(n_epochs - epochs_before_early_stopping):
            logger.info(f"Epoch {epochs_before_early_stopping+j+1} / {n_epochs}")
            prev_test_loss = sum(test_loss_history[-early_stopping_epochs_agg:]) / early_stopping_epochs_agg

            train_loss = self._step_train(train_dataloader, optimizer)
            test_loss = self._step_test(test_dataloader)

            test_loss_history.append(test_loss)
            history[epochs_before_early_stopping+j] = {"train": train_loss, "test": [test_loss]}

            loss_decrease = prev_test_loss - test_loss
            if loss_decrease <= 0:
                logger.warning("Early stopping! Loss decrease <= 0")
                break
            elif loss_decrease < min_loss_decrease:
                logger.warning(f"Early stopping! Loss decrease [{loss_decrease}] lower than user's assumption [{min_loss_decrease}]")
                break

        return history
    
    def _step_train(
            self,
            dataloader: DataLoader, 
            optimizer: optim.Optimizer
    ) -> list[float]:
        loss_history = []
        self._model.train()
        for i, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self._model(X)
            loss = self._loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_history.append(loss.item())
            logger.info(f"Train loss: {round(loss.item(), 5)}")
        return loss_history

    def _step_test(self, dataloader: DataLoader) -> float:
        self._model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self._model(X)
                test_loss += self._loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        logger.info(f"*Test loss: {round(test_loss, 5)} \n")

        return test_loss
    