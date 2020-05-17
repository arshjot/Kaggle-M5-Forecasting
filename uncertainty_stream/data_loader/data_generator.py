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
    Y : actual sales, denoting targets for prediction period ([N+1, N+P])

    Returns:
    List of torch arrays:
    x_enc: concatenated encoder features (except embedding)
    x_enc_emb: concatenated encoder embedding features
    x_dec: concatenated decoder features (except embedding)
    x_dec_emb: concatenated decoder embedding features
    x_last_day_sales: the actual sales for the day before the start of the prediction period
    y: targets (only in training phase)
    """

    def __init__(self, X_prev_day_sales, X_enc_only_feats, X_enc_dec_feats, X_calendar, X_last_day_sales,
                 norm_factor, Y=None, rmsse_denominator=None, wrmsse_weights=None, window_id=None):

        self.X_prev_day_sales = X_prev_day_sales
        self.X_enc_only_feats = X_enc_only_feats
        self.X_enc_dec_feats = X_enc_dec_feats
        self.X_calendar = X_calendar
        self.X_last_day_sales = X_last_day_sales
        self.norm_factor = norm_factor
        self.window_id = window_id

        if Y is not None:
            self.Y = torch.from_numpy(Y).float()
            self.rmsse_denominator = torch.from_numpy(rmsse_denominator).float()
            self.wrmsse_weights = torch.from_numpy(wrmsse_weights).float()
        else:
            self.Y = None

    def __len__(self):
        return self.X_prev_day_sales.shape[1]

    def __getitem__(self, idx):
        if self.window_id is not None:
            X_calendar = self.X_calendar[self.window_id[idx]]
            scale = self.rmsse_denominator[idx - (self.window_id[idx] * 42840)]
            weight = self.wrmsse_weights[idx - (self.window_id[idx] * 42840)]
            ids_idx = idx - (self.window_id[idx] * 42840)
            window_id = self.window_id[idx]
        else:
            X_calendar = self.X_calendar
            if self.Y is not None:
                scale = self.rmsse_denominator[idx]
                weight = self.wrmsse_weights[idx]
                ids_idx = idx
                window_id = 0

        enc_timesteps = self.X_prev_day_sales.shape[0]
        dec_timesteps = self.X_enc_dec_feats.shape[0] - enc_timesteps
        num_embedding = 5
        num_cal_embedding = 2

        # input data for encoder
        x_enc_dec_feats_enc = self.X_enc_dec_feats[:enc_timesteps, idx, :-num_embedding].reshape(enc_timesteps, -1)
        # x_enc_only_feats = self.X_enc_only_feats[:, idx, :].reshape(enc_timesteps, -1)
        x_prev_day_sales_enc = self.X_prev_day_sales[:, idx].reshape(-1, 1)
        x_calendar_enc = X_calendar[:enc_timesteps, :-num_cal_embedding]
        x_calendar_enc_emb = X_calendar[:enc_timesteps, -num_cal_embedding:].reshape(enc_timesteps, -1)
        # x_enc = np.concatenate([x_enc_dec_feats_enc, x_calendar_enc,
        #                         x_prev_day_sales_enc, x_enc_only_feats], axis=1)
        x_enc = np.concatenate([x_enc_dec_feats_enc, x_calendar_enc, x_prev_day_sales_enc], axis=1)
        x_enc_emb = self.X_enc_dec_feats[:enc_timesteps, idx, -num_embedding:].reshape(enc_timesteps, -1)

        # input data for decoder
        x_enc_dec_feats_dec = self.X_enc_dec_feats[enc_timesteps:, idx, :-num_embedding].reshape(dec_timesteps, -1)
        x_calendar_dec = X_calendar[enc_timesteps:, :-num_cal_embedding]
        x_calendar_dec_emb = X_calendar[enc_timesteps:, -num_cal_embedding:].reshape(dec_timesteps, -1)
        x_last_day_sales = self.X_last_day_sales[idx].reshape(-1)
        x_dec = np.concatenate([x_enc_dec_feats_dec, x_calendar_dec], axis=1)
        x_dec_emb = self.X_enc_dec_feats[enc_timesteps:, idx, -num_embedding:].reshape(dec_timesteps, -1)

        if self.Y is None:
            return [[torch.from_numpy(x_enc).float(), torch.from_numpy(x_enc_emb).long(),
                     torch.from_numpy(x_calendar_enc_emb).long(),
                     torch.from_numpy(x_dec).float(), torch.from_numpy(x_dec_emb).long(),
                     torch.from_numpy(x_calendar_dec_emb).long(),
                     torch.from_numpy(x_last_day_sales).float()], self.norm_factor[idx]]

        return [[torch.from_numpy(x_enc).float(), torch.from_numpy(x_enc_emb).long(),
                 torch.from_numpy(x_calendar_enc_emb).long(),
                 torch.from_numpy(x_dec).float(), torch.from_numpy(x_dec_emb).long(),
                 torch.from_numpy(x_calendar_dec_emb).long(),
                 torch.from_numpy(x_last_day_sales).float()],
                self.Y[idx, :], torch.from_numpy(np.array(self.norm_factor[idx])).float(),
                ids_idx,
                [scale, weight],
                window_id]


class DataLoader:
    def __init__(self, config):
        self.config = config

        # load data
        with open(f'{self.config.data_file}', 'rb') as f:
            data_dict = pkl.load(f)

        self.ids = data_dict['sales_data_ids']
        self.enc_dec_feat_names = data_dict['enc_dec_feat_names']
        sell_price_i = self.enc_dec_feat_names.index('sell_price')

        self.X_prev_day_sales, _, _ = get_aggregated_series(data_dict['X_prev_day_sales'].T, self.ids)
        self.X_prev_day_sales = self.X_prev_day_sales.T

        self.X_enc_dec_feats = data_dict['X_enc_dec_feats']
        self.sell_price_l12 = self.X_enc_dec_feats[:, :, sell_price_i]
        sell_price_all, _, _ = get_aggregated_series(self.X_enc_dec_feats[:, :, sell_price_i].T, self.ids)
        encodings_all, _ = get_aggregated_encodings(self.X_enc_dec_feats[:, :, 1:].transpose(1, 0, 2), self.ids)
        self.X_enc_dec_feats = np.concatenate([sell_price_all[:, :, np.newaxis], encodings_all], axis=2)\
            .transpose(1, 0, 2)

        self.Y_l12 = data_dict['Y']
        self.Y, _, _ = get_aggregated_series(data_dict['Y'], self.ids)

        self.X_enc_only_feats = data_dict['X_enc_only_feats']
        self.X_calendar = data_dict['X_calendar']
        self.n_windows = 1

    def create_train_loader(self, data_start_t=None, horizon_start_t=None, horizon_end_t=None):
        if (data_start_t is None) | (horizon_start_t is None) | (horizon_end_t is None):
            data_start_t = self.config.training_ts['data_start_t']
            horizon_start_t = self.config.training_ts['horizon_start_t']
            horizon_end_t = self.config.training_ts['horizon_end_t']

        # Run a sliding window of length "window_length" and train for the next month of each window
        if self.config.sliding_window:
            window_length = self.config.window_length
            X_prev_day_sales, X_enc_only_feats, X_enc_dec_feats, X_calendar, Y, norm_factor = [], [], [], [], [], []
            last_day_sales, weights, scales = [], [], []

            for idx, i in enumerate(range(data_start_t + window_length, horizon_end_t, 28)):
                w_data_start_t, w_horizon_start_t = data_start_t + (idx * 28), i
                w_horizon_end_t = w_horizon_start_t + 28

                # calculate denominator for rmsse loss
                absolute_movement = np.absolute(self.Y.T[:w_horizon_start_t] -
                                                self.X_prev_day_sales[:w_horizon_start_t]).astype(np.int64)
                actively_sold_in_range = (self.X_prev_day_sales[:w_horizon_start_t] != 0).argmax(axis=0)
                rmsse_den = []
                for idx_active_sell, first_active_sell_idx in enumerate(actively_sold_in_range):
                    den = absolute_movement[first_active_sell_idx:, idx_active_sell].mean()
                    den = den if den != 0 else 1
                    rmsse_den.append(den)
                scales.append(np.array(rmsse_den))

                # Get weights for WRMSSE and SPL loss
                w_weights, _ = get_weights_all_levels(self.Y_l12[:, w_horizon_start_t - 28:w_horizon_start_t],
                                                      self.sell_price_l12[w_horizon_start_t - 28:w_horizon_start_t, :].T,
                                                      self.ids)
                weights.append(w_weights)

                # Normalize sale features by dividing by median of each series (as per the selected input window)
                w_X_prev_day_sales = self.X_prev_day_sales[w_data_start_t:w_horizon_start_t].copy().astype(float)
                w_norm_factor = np.median(w_X_prev_day_sales, 0)
                w_norm_factor[w_norm_factor == 0] = 1.
                w_X_prev_day_sales = w_X_prev_day_sales / w_norm_factor

                X_prev_day_sales.append(w_X_prev_day_sales)
                X_enc_only_feats.append(self.X_enc_only_feats[w_data_start_t:w_horizon_start_t])
                X_enc_dec_feats.append(self.X_enc_dec_feats[w_data_start_t:w_horizon_end_t])
                X_calendar.append(self.X_calendar[w_data_start_t:w_horizon_end_t])
                last_day_sales.append(w_X_prev_day_sales[-1])
                norm_factor.append(w_norm_factor)
                Y.append(self.Y[:, w_horizon_start_t:w_horizon_end_t])

            self.n_windows = idx + 1
            X_prev_day_sales = np.concatenate(X_prev_day_sales, 1)
            X_enc_only_feats = np.concatenate(X_enc_only_feats, 1)
            X_enc_dec_feats = np.concatenate(X_enc_dec_feats, 1)
            X_calendar = np.stack(X_calendar, 0)
            last_day_sales = np.concatenate(last_day_sales, 0)
            Y = np.concatenate(Y, 0)
            scales = np.concatenate(scales, 0)
            weights = np.concatenate(weights, 0)
            norm_factor = np.concatenate(norm_factor, 0)
            window_id = np.arange(idx + 1).repeat(self.X_enc_dec_feats.shape[1])

        else:
            # calculate denominator for rmsse loss
            absolute_movement = np.absolute(self.Y.T[:horizon_start_t] -
                                            self.X_prev_day_sales[:horizon_start_t]).astype(np.int64)
            actively_sold_in_range = (self.X_prev_day_sales[:horizon_start_t] != 0).argmax(axis=0)
            rmsse_den = []
            for idx, first_active_sell_idx in enumerate(actively_sold_in_range):
                den = absolute_movement[first_active_sell_idx:, idx].mean()
                den = den if den != 0 else 1
                rmsse_den.append(den)

            # Get weights for WRMSSE and SPL loss
            weights, _ = get_weights_all_levels(self.Y_l12[:, horizon_start_t - 28:horizon_start_t],
                                                self.sell_price_l12[horizon_start_t - 28:horizon_start_t, :].T,
                                                self.ids)

            # Normalize sale features by dividing by median of each series (as per the selected input window)
            X_prev_day_sales = self.X_prev_day_sales[data_start_t:horizon_start_t].copy().astype(float)
            norm_factor = np.median(X_prev_day_sales, 0)
            norm_factor[norm_factor == 0] = 1.
            X_prev_day_sales = X_prev_day_sales / norm_factor

            X_enc_only_feats = self.X_enc_only_feats[data_start_t:horizon_start_t]
            X_enc_dec_feats = self.X_enc_dec_feats[data_start_t:horizon_end_t]
            X_calendar = self.X_calendar[data_start_t:horizon_end_t]
            last_day_sales = X_prev_day_sales[-1]
            Y = self.Y[:, horizon_start_t:horizon_end_t]
            scales = np.array(rmsse_den)
            window_id = None

        dataset = CustomDataset(X_prev_day_sales,
                                X_enc_only_feats,
                                X_enc_dec_feats,
                                X_calendar, last_day_sales,
                                norm_factor,
                                Y=Y,
                                rmsse_denominator=scales, wrmsse_weights=weights, window_id=window_id)

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=3, pin_memory=True)

    def create_val_loader(self, data_start_t=None, horizon_start_t=None, horizon_end_t=None):
        if (data_start_t is None) | (horizon_start_t is None) | (horizon_end_t is None):
            data_start_t = self.config.validation_ts['data_start_t']
            horizon_start_t = self.config.validation_ts['horizon_start_t']
            horizon_end_t = self.config.validation_ts['horizon_end_t']

        # calculate denominator for rmsse loss
        absolute_movement = np.absolute(self.Y.T[:horizon_start_t] -
                                        self.X_prev_day_sales[:horizon_start_t]).astype(np.int64)
        actively_sold_in_range = (self.X_prev_day_sales[:horizon_start_t] != 0).argmax(axis=0)
        rmsse_den = []
        for idx, first_active_sell_idx in enumerate(actively_sold_in_range):
            den = absolute_movement[first_active_sell_idx:, idx].mean()
            den = den if den != 0 else 1
            rmsse_den.append(den)

        # Get weights for WRMSSE and SPL loss
        weights, _ = get_weights_all_levels(self.Y_l12[:, horizon_start_t-28:horizon_start_t],
                                            self.sell_price_l12[horizon_start_t-28:horizon_start_t, :].T,
                                            self.ids)

        # Normalize sale features by dividing by median of each series (as per the selected input window)
        X_prev_day_sales = self.X_prev_day_sales[data_start_t:horizon_start_t].copy().astype(float)
        norm_factor = np.median(X_prev_day_sales, 0)
        norm_factor[norm_factor == 0] = 1.
        X_prev_day_sales = X_prev_day_sales / norm_factor

        dataset = CustomDataset(X_prev_day_sales,
                                self.X_enc_only_feats[data_start_t:horizon_start_t],
                                self.X_enc_dec_feats[data_start_t:horizon_end_t],
                                self.X_calendar[data_start_t:horizon_end_t], X_prev_day_sales[-1],
                                norm_factor,
                                Y=self.Y[:, horizon_start_t:horizon_end_t],
                                rmsse_denominator=np.array(rmsse_den), wrmsse_weights=weights)

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, num_workers=3,
                                           pin_memory=True)

    def create_test_loader(self, data_start_t=None, horizon_start_t=None, horizon_end_t=None):
        if (data_start_t is None) | (horizon_start_t is None) | (horizon_end_t is None):
            data_start_t = self.config.test_ts['data_start_t']
            horizon_start_t = self.config.test_ts['horizon_start_t']
            horizon_end_t = self.config.test_ts['horizon_end_t']

        # Normalize sale features by dividing by median of each series (as per the selected input window)
        X_prev_day_sales = self.X_prev_day_sales[data_start_t:horizon_start_t].copy().astype(float)
        norm_factor = np.median(X_prev_day_sales, 0)
        norm_factor[norm_factor == 0] = 1.
        X_prev_day_sales = X_prev_day_sales / norm_factor

        dataset = CustomDataset(X_prev_day_sales,
                                self.X_enc_only_feats[data_start_t:horizon_start_t],
                                self.X_enc_dec_feats[data_start_t:horizon_end_t],
                                self.X_calendar[data_start_t:horizon_end_t], X_prev_day_sales[-1],
                                norm_factor)

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, num_workers=3,
                                           pin_memory=True)