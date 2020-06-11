from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from importlib import import_module
import shutil
import glob
import os

from data_loader.data_generator_representation import DataLoader
from losses_and_metrics import loss_functions
from utils.data_utils import *
from utils.training_utils import ModelCheckpoint, EarlyStopping
from config import *

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        # Load model trained on raw data to extract weight for embeddings
        model_type = import_module('models.' + config.architecture)
        create_model = getattr(model_type, 'create_model')
        model_raw = create_model(config)
        model_checkpoint = ModelCheckpoint(weight_dir='./weights/raw/')
        model_raw, _, _, _ = model_checkpoint.load(model_raw, load_best=True)
        self.model_embedder = model_raw.embedder

        # Freeze weights for pre-trained Embedder module
        for param in self.model_embedder.parameters():
            param.requires_grad = False
        self.model_embedder.eval()

        # Initialize representation model
        print(f' Model: {self.config.rs_architecture} '.center(self.terminal_width, '*'))
        model_type = import_module('models.' + self.config.rs_architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config)
        print(self.model, end='\n\n')

        # Loss, Optimizer and LRScheduler
        self.criterion = getattr(loss_functions, config.rs_loss_fn)(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.rs_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                                    patience=3, verbose=True)
        self.early_stopping = EarlyStopping(patience=8)

        print(f' Loading data '.center(self.terminal_width, '*'))
        data_loader = DataLoader(self.config)
        self.ids = data_loader.ids
        self.train_loader, self.val_loader = data_loader.create_train_val_loaders()
        self.n_windows = data_loader.n_windows

        self.start_epoch, self.min_val_error = 1, None
        # Load checkpoint if training is to be resumed
        self.model_checkpoint = ModelCheckpoint(weight_dir='./weights/representation/')
        if config.resume_training:
            self.model, self.optimizer, self.scheduler, [self.start_epoch, self.min_val_error, num_bad_epochs] = \
                self.model_checkpoint.load(self.model, self.optimizer, self.scheduler)
            self.early_stopping.best = self.min_val_error
            self.early_stopping.num_bad_epochs = num_bad_epochs
            print(f'Resuming model training from epoch {self.start_epoch}')
        else:
            # remove previous logs, if any
            logs = glob.glob('./logs/representation/.*') + glob.glob('./logs/representation/*')
            for f in logs:
                os.remove(f)

        # logging
        self.writer = SummaryWriter('logs/representation/')

    def _get_val_loss_and_err(self):
        self.model.eval()
        progbar = tqdm(self.val_loader)
        progbar.set_description("             ")
        losses = []
        for i, x in enumerate(progbar):
            x = [inp.to(self.config.device) for inp in x]

            # Pass x through pre-trained Embedder to obtain inputs for autoencoder
            x_autoenc = self.model_embedder(*x).permute(1, 0, 2)

            preds = self.model(x_autoenc)
            loss = self.criterion(preds, x_autoenc)
            losses.append(loss.data.cpu().numpy())

        return np.mean(losses)

    def train(self):
        print(f' Training '.center(self.terminal_width, '*'), end='\n\n')

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            print(f' Epoch [{epoch}/{self.config.rs_num_epochs}] '.center(self.terminal_width, 'x'))
            self.model.train()
            progbar = tqdm(self.train_loader)
            losses = []
            for i, x in enumerate(progbar):
                x = [inp.to(self.config.device) for inp in x]

                # Pass x through pre-trained Embedder to obtain inputs for autoencoder
                x_autoenc = self.model_embedder(*x).permute(1, 0, 2)

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()
                preds = self.model(x_autoenc)

                loss = self.criterion(preds, x_autoenc)
                losses.append(loss.data.cpu().numpy())
                progbar.set_description("loss = %0.3f " % np.round(np.mean(losses), 3))

                loss.backward()
                self.optimizer.step()

            # Get training and validation loss
            train_loss = np.mean(losses)

            val_loss = self._get_val_loss_and_err()

            print(f'Training Loss: {train_loss:.4f}\n'
                  f'Validation Loss: {val_loss:.4f}')

            # Change learning rate according to scheduler
            self.scheduler.step(val_loss)

            # save checkpoint and best model
            if self.min_val_error is None:
                self.min_val_error = val_loss
                is_best = True
                print(f'Best model obtained at the end of epoch {epoch}')
            else:
                if val_loss < self.min_val_error:
                    self.min_val_error = val_loss
                    is_best = True
                    print(f'Best model obtained at the end of epoch {epoch}')
                else:
                    is_best = False
            self.model_checkpoint.save(is_best, self.min_val_error, self.early_stopping.num_bad_epochs,
                                       epoch, self.model, self.optimizer, self.scheduler)

            # write logs
            self.writer.add_scalar(f'{self.config.loss_fn}/train', train_loss, epoch * i)
            self.writer.add_scalar(f'{self.config.loss_fn}/val', val_loss, epoch * i)

            # Early Stopping
            if self.early_stopping.step(val_loss):
                print(f' Training Stopped'.center(self.terminal_width, '*'))
                print(f'Early stopping triggered after epoch {epoch}')
                break

        self.writer.close()


if __name__ == "__main__":
    config = Config
    trainer = Trainer(config)
    trainer.train()
