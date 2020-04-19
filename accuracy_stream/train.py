# TODO: Add WRMSSE as metric (for all 42,840 series)
# TODO: Add WRMSSE as loss (for 30,490 series)
# TODO: Add callbacks

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from importlib import import_module
import shutil
import glob
import os

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
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        # Model
        print(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'))
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config)
        print(self.model, end='\n\n')

        # Loss and Optimizer
        self.criterion = getattr(loss_functions, config.loss_fn)()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        print(f' Loading data '.center(self.terminal_width, '*'))
        data_loader = DataLoader(self.config)
        self.train_loader = data_loader.create_train_loader()
        self.val_loader = data_loader.create_val_loader()

        # logging
        # remove previous logs, if any
        logs = glob.glob('./logs/.*') + glob.glob('./logs/*')
        for f in logs:
            os.remove(f)
        self.writer = SummaryWriter('logs')

    def _get_val_loss(self):
        self.model.eval()
        progbar = tqdm(self.val_loader)
        progbar.set_description("             ")
        losses = []
        for i, [x, y, loss_input] in enumerate(progbar):
            x = [inp.to(self.config.device) for inp in x]
            y = y.to(self.config.device)
            loss_input = [inp.to(self.config.device) for inp in loss_input]

            preds = self.model(*x)
            loss = self.criterion(preds, y, loss_input[0])
            loss_iter = loss.data.cpu().numpy()
            losses.append(loss_iter)

        return np.mean(losses)

    def train(self):
        print(f' Training '.center(self.terminal_width, '*'), end='\n\n')
        min_val_loss = 100
        for epoch in range(self.config.num_epochs):
            print(f' Epoch [{epoch + 1}/{self.config.num_epochs}] '.center(self.terminal_width, 'x'))
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
                losses.append(loss_iter)
                progbar.set_description("loss = %0.3f " % np.round(np.mean(losses), 3))
                loss.backward()
                self.optimizer.step()

            val_loss = self._get_val_loss()
            print(f'Training Loss: {np.mean(losses):.4f}, Validation Loss: {val_loss:.4f}')

            # save best model
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), './weights/model.pth.tar')

            # write logs
            self.writer.add_scalar(f'{self.config.loss_fn}/train', np.mean(losses), (epoch + 1) * i)
            self.writer.add_scalar(f'{self.config.loss_fn}/val', val_loss, (epoch + 1) * i)

        self.writer.close()


if __name__ == "__main__":
    config = Config
    trainer = Trainer(config)
    trainer.train()
