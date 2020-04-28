import torch
import shutil
import os


class ModelCheckpoint:
    def __init__(self, weight_dir='./weights'):
        self.weight_dir = weight_dir
        self.filename = os.path.join(self.weight_dir, 'model_latest_checkpoint.pth.tar')
        self.best_filename = os.path.join(self.weight_dir, 'model_best.pth.tar')

    def save(self, is_best, min_val_error, epoch, model, optimizer, scheduler=None):
        if scheduler is not None:
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'min_val_error': min_val_error,
                'scheduler': scheduler.state_dict()
            }
        else:
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'min_val_error': min_val_error,
                'scheduler': scheduler
            }
        torch.save(save_dict, self.filename)
        if is_best:
            shutil.copyfile(self.filename, self.best_filename)

    def load(self, model, optimizer=None, scheduler=None, load_best=False):
        load_filename = self.best_filename if load_best else self.filename
        if os.path.isfile(load_filename):
            checkpoint = torch.load(load_filename)
            model.load_state_dict(checkpoint['model'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            min_val_error = checkpoint['min_val_error']
        else:
            raise FileNotFoundError(f'No checkpoint found at {load_filename}')

        return model, optimizer, scheduler, start_epoch, min_val_error
