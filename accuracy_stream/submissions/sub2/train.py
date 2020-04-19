# TODO: Add logging
# TODO: Add WRMSSE as metric (for all 42,840 series)
# TODO: Add WRMSSE as loss (for 30,490 series)
# TODO: Add callbacks

import torch
from tqdm import tqdm
import numpy as np
from importlib import import_module

from data_loader.data_generator import DataLoader
from losses_and_metrics import loss_functions
from config import Config

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)


class Trainer:
    def __init__(self, config):
        self.config = config

        # Model
        print(f'x ---- Model: {self.config.architecture} ---- x')
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config)
        print(self.model)

        # Loss and Optimizer
        self.criterion = getattr(loss_functions, config.loss_fn)()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        print(f'\nx ---- Loading data ---- x')
        data_loader = DataLoader(self.config)
        self.train_loader = data_loader.create_train_loader()
        self.val_loader = data_loader.create_val_loader()

    def _get_val_loss(self):
        self.model.eval()
        losses = []
        for i, [x, y, loss_input] in enumerate(tqdm(self.val_loader)):
            x = [inp.to(self.config.device) for inp in x]
            y = y.to(self.config.device)
            loss_input = [inp.to(self.config.device) for inp in loss_input]

            preds = self.model(*x)
            loss = self.criterion(preds, y, loss_input[0])
            loss_iter = loss.data.cpu().numpy()
            losses.append(loss_iter)

        return np.mean(losses)

    def train(self):
        print(f'x ---- Training ---- x\n')
        min_val_loss = 100
        for epoch in range(self.config.num_epochs):
            self.model.train()
            progbar = tqdm(self.train_loader)
            losses = []
            for i, [x, y, loss_input] in enumerate(progbar):
                x = [inp.to(self.config.device) for inp in x]
                y = y.to(self.config.device)
                loss_input = [inp.to(self.config.device) for inp in loss_input]

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()
                preds = self.model(*x)
                loss = self.criterion(preds, y, loss_input[0])
                loss_iter = loss.data.cpu().numpy()
                progbar.set_description("loss = %0.3f " % np.round(loss_iter, 3))
                losses.append(loss_iter)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch + 1}/{self.config.num_epochs}], Loss: {np.mean(losses):.4f}')
            val_loss = self._get_val_loss()
            print(f'Validation Loss: {val_loss:.4f}')

            # save best model
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), './weights/model.pth.tar')


if __name__ == "__main__":
    config = Config
    trainer = Trainer(config)
    trainer.train()
