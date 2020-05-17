import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from importlib import import_module
import os
import shutil

from data_loader.data_generator import DataLoader
from utils.training_utils import ModelCheckpoint
from utils.data_utils import *
from config import Config


class SubmissionGenerator:
    def __init__(self, config):
        self.config = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns

        # Model
        print(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'), end='\n\n')
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config)
        print(self.model)
        model_checkpoint = ModelCheckpoint()
        self.model, _, _, _ = model_checkpoint.load(self.model, load_best=True)

        print(f' Loading data '.center(self.terminal_width, '*'))
        data_loader = DataLoader(self.config)
        self.ids = data_loader.ids
        self.test_loader = data_loader.create_test_loader()

        self.sub_dir = self._prepare_dir()

    def _prepare_dir(self):
        print(f' Create submission directory '.center(self.terminal_width, '*'))
        subs = [int(i[3:]) for i in os.listdir('./submissions')]
        sub_idx = max(subs) + 1 if len(subs) > 0 else 1

        os.makedirs(f'./submissions/sub{sub_idx}')

        # copy model code to submission directory
        shutil.copytree('losses_and_metrics/', f'./submissions/sub{sub_idx}/losses_and_metrics/')
        shutil.copytree('./models/', f'./submissions/sub{sub_idx}/models/')
        shutil.copytree('./weights/', f'./submissions/sub{sub_idx}/weights/')
        shutil.copytree('./data_loader/', f'./submissions/sub{sub_idx}/data_loader/')
        shutil.copytree('./utils/', f'./submissions/sub{sub_idx}/utils/')
        shutil.copytree('./logs/', f'./submissions/sub{sub_idx}/logs/')

        os.makedirs(f'./submissions/sub{sub_idx}/data')
        shutil.copyfile('./data/prepare_data.py', f'./submissions/sub{sub_idx}/data/prepare_data.py')
        shutil.copyfile('config.py', f'./submissions/sub{sub_idx}/config.py')
        shutil.copyfile('generate_submission.py', f'./submissions/sub{sub_idx}/generate_submission.py')
        shutil.copyfile('train.py', f'./submissions/sub{sub_idx}/train.py')

        return f'./submissions/sub{sub_idx}/'

    def generate_submission_file(self):
        print(f' Predict '.center(self.terminal_width, '*'))
        self.model.eval()
        preds = []
        for i, [x, norm_factor] in enumerate(tqdm(self.test_loader)):
            x = [inp.to(self.config.device) for inp in x]
            norm_factor = norm_factor.to(self.config.device)
            preds.append((self.model(*x) * norm_factor[:, None, None]).data.cpu().numpy())

        predictions = np.concatenate(preds, 0)
        sample_submission = pd.read_csv('../data/sample_submission_uncertainty.csv')
        sample_submission.iloc[:9*predictions.shape[0], 1:] = predictions.transpose(2, 0, 1).reshape(-1, 28)
        sample_submission.to_csv(f'{self.sub_dir}/submission.csv.gz', compression='gzip', index=False,
                                 float_format='%.3g')


if __name__ == "__main__":
    config = Config
    predictor = SubmissionGenerator(config)
    predictor.generate_submission_file()
