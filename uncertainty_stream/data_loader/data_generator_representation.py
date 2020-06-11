import sys

sys.path.extend(['..'])

import torch
import torch.utils.data
import torch.utils.data as data_utils
import pickle as pkl

from utils.data_utils import *


# Dataset (Input Pipeline)
class CustomDataset(data_utils.Dataset):
    """
    Custom dataset

    Let:
    training period timesteps = [0, N]
    prediction period timesteps = [N+1, N+P]

    Arguments:
    X_prev_day_sales : previous day sales for training period ([0, N])
    X_enc_only_feats : aggregated series' previous day sales for training period ([0, N])
    X_enc_dec_feats : sell price and categorical features for training and prediction period ([0, N+P])
    X_calendar : calendar features for training and prediction period ([0, N+P])
    X_last_day_sales : the actual sales for the day before the start of the prediction period (for timestep N)
                       (this will serve as the first timestep's input for the decoder)

    Returns:
    List of torch arrays:
    """

    def __init__(self, X_prev_day_sales, X_enc_only_feats, X_enc_dec_feats, X_calendar, window_id):

        self.X_prev_day_sales = X_prev_day_sales
        self.X_enc_only_feats = X_enc_only_feats
        self.X_enc_dec_feats = X_enc_dec_feats
        self.X_calendar = X_calendar
        self.window_id = window_id

    def __len__(self):
        return self.X_prev_day_sales.shape[1]

    def __getitem__(self, idx):
        if self.window_id is not None:
            X_calendar = self.X_calendar[self.window_id[idx]]
        else:
            X_calendar = self.X_calendar

        enc_timesteps = self.X_prev_day_sales.shape[0]
        num_embedding = 5
        num_cal_embedding = 2

        # input data for embedder
        x_enc_dec_feats = self.X_enc_dec_feats[:enc_timesteps, idx, :-num_embedding].reshape(enc_timesteps, -1)
        x_prev_day_sales = self.X_prev_day_sales[:, idx].reshape(-1, 1)
        x_calendar = X_calendar[:enc_timesteps, :-num_cal_embedding]
        x_calendar_emb = X_calendar[:enc_timesteps, -num_cal_embedding:].reshape(enc_timesteps, -1)

        x = np.concatenate([x_enc_dec_feats, x_calendar, x_prev_day_sales], axis=1)
        x_emb = self.X_enc_dec_feats[:enc_timesteps, idx, -num_embedding:].reshape(enc_timesteps, -1)

        return [torch.from_numpy(x).float(), torch.from_numpy(x_emb).long(), torch.from_numpy(x_calendar_emb).long()]


class DataLoader:
    def __init__(self, config):
        self.config = config

        # load data
        with open(f'{self.config.data_file}', 'rb') as f:
            data_dict = pkl.load(f)

        self.ids = data_dict['sales_data_ids']
        self.enc_dec_feat_names = data_dict['enc_dec_feat_names']
        self.sell_price_i = self.enc_dec_feat_names.index('sell_price')

        self.X_prev_day_sales, self.agg_ids, _ = get_aggregated_series(data_dict['X_prev_day_sales'].T, self.ids)
        self.X_prev_day_sales = self.X_prev_day_sales.T

        self.X_enc_dec_feats = data_dict['X_enc_dec_feats']
        self.sell_price_l12 = self.X_enc_dec_feats[:, :, self.sell_price_i]
        sell_price_all, _, _ = get_aggregated_series(self.X_enc_dec_feats[:, :, self.sell_price_i].T, self.ids, 'mean')
        encodings_all, _ = get_aggregated_encodings(self.X_enc_dec_feats[:, :, 1:].transpose(1, 0, 2), self.ids)
        self.X_enc_dec_feats = np.concatenate([sell_price_all[:, :, np.newaxis], encodings_all], axis=2)\
            .transpose(1, 0, 2)

        self.Y_l12 = data_dict['Y']
        self.Y, _, _ = get_aggregated_series(data_dict['Y'], self.ids)

        self.X_enc_only_feats = data_dict['X_enc_only_feats']
        self.X_calendar = data_dict['X_calendar']
        self.n_windows = 1

    def create_train_val_loaders(self, data_start_t=None, horizon_start_t=None, horizon_end_t=None, val_split=0.2):
        if (data_start_t is None) | (horizon_start_t is None) | (horizon_end_t is None):
            data_start_t = self.config.training_ts['data_start_t']
            horizon_start_t = self.config.training_ts['horizon_start_t']
            horizon_end_t = self.config.training_ts['horizon_end_t']

        # Run a sliding window of length "window_length" and train for the next month of each window
        if self.config.sliding_window:
            window_length = self.config.window_length
            X_prev_day_sales, X_enc_only_feats, X_enc_dec_feats, X_calendar, Y, norm_factor = [], [], [], [], [], []

            for idx, i in enumerate(range(data_start_t + window_length, horizon_end_t, 28)):
                w_data_start_t, w_horizon_start_t = data_start_t + (idx * 28), i
                w_horizon_end_t = w_horizon_start_t + 28

                # Normalize sale features by dividing by mean of each series (as per the selected input window)
                w_X_prev_day_sales = self.X_prev_day_sales[w_data_start_t:w_horizon_start_t].copy().astype(float)
                w_norm_factor = np.mean(w_X_prev_day_sales, 0)
                w_norm_factor[w_norm_factor == 0] = 1.
                w_X_prev_day_sales = w_X_prev_day_sales / w_norm_factor

                w_X_enc_dec_feats = self.X_enc_dec_feats[w_data_start_t:w_horizon_end_t]
                w_X_sell_p = self.X_enc_dec_feats[w_data_start_t:w_horizon_start_t, :, self.sell_price_i].copy().astype(float)
                w_norm_factor_sell_p = np.median(w_X_sell_p, 0)
                w_norm_factor_sell_p[w_norm_factor_sell_p == 0] = 1.
                w_X_enc_dec_feats[:, :, self.sell_price_i] = w_X_enc_dec_feats[:, :, self.sell_price_i] / w_norm_factor_sell_p

                X_prev_day_sales.append(w_X_prev_day_sales)
                X_enc_only_feats.append(self.X_enc_only_feats[w_data_start_t:w_horizon_start_t])
                X_enc_dec_feats.append(w_X_enc_dec_feats)
                X_calendar.append(self.X_calendar[w_data_start_t:w_horizon_end_t])
                norm_factor.append(w_norm_factor)

            self.n_windows = idx + 1
            X_prev_day_sales = np.concatenate(X_prev_day_sales, 1)
            X_enc_only_feats = np.concatenate(X_enc_only_feats, 1)
            X_enc_dec_feats = np.concatenate(X_enc_dec_feats, 1)
            X_calendar = np.stack(X_calendar, 0)
            norm_factor = np.concatenate(norm_factor, 0)
            window_id = np.arange(idx + 1).repeat(self.X_enc_dec_feats.shape[1])

        else:
            # Normalize sale features by dividing by mean of each series (as per the selected input window)
            X_prev_day_sales = self.X_prev_day_sales[data_start_t:horizon_start_t].copy().astype(float)
            norm_factor = np.mean(X_prev_day_sales, 0)
            norm_factor[norm_factor == 0] = 1.
            X_prev_day_sales = X_prev_day_sales / norm_factor

            X_enc_dec_feats = self.X_enc_dec_feats[data_start_t:horizon_end_t]
            X_sell_p = self.X_enc_dec_feats[data_start_t:horizon_start_t, :, self.sell_price_i].copy().astype(float)
            norm_factor_sell_p = np.median(X_sell_p, 0)
            norm_factor_sell_p[norm_factor_sell_p == 0] = 1.
            X_enc_dec_feats[:, :, self.sell_price_i] = X_enc_dec_feats[:, :, self.sell_price_i] / norm_factor_sell_p

            X_enc_only_feats = self.X_enc_only_feats[data_start_t:horizon_start_t]
            X_calendar = self.X_calendar[data_start_t:horizon_end_t]
            window_id = None

        dataset = CustomDataset(X_prev_day_sales, X_enc_only_feats, X_enc_dec_feats, X_calendar, window_id=window_id)

        # Split validation data and create train and validation data loaders
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.config.rs_batch_size,
                                                   shuffle=True, num_workers=3, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.config.rs_batch_size,
                                                 shuffle=True, num_workers=3, pin_memory=True)

        return train_loader, val_loader
