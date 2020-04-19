import torch
import torch.utils.data
import torch.utils.data as data_utils
import numpy as np
import pickle as pkl


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
                 Y=None, rmsse_denominator=None, transform=None):
        self.X_prev_day_sales = X_prev_day_sales
        self.X_enc_only_feats = X_enc_only_feats
        self.X_enc_dec_feats = X_enc_dec_feats
        self.X_calendar = X_calendar
        self.X_last_day_sales = X_last_day_sales

        if Y is not None:
            self.Y = torch.from_numpy(Y).float()
            self.rmsse_denominator = torch.from_numpy(rmsse_denominator).float()
        else:
            self.Y = None

        self.transform = transform

    def __len__(self):
        return self.X_prev_day_sales.shape[1]

    def __getitem__(self, idx):
        enc_timesteps = self.X_prev_day_sales.shape[0]
        dec_timesteps = self.X_enc_dec_feats.shape[0] - enc_timesteps
        num_embedding = 3

        # input data for encoder
        x_enc_dec_feats_enc = self.X_enc_dec_feats[:enc_timesteps, idx, :-num_embedding].reshape(enc_timesteps, -1)
        # x_enc_only_feats = self.X_enc_only_feats[:, idx, :].reshape(enc_timesteps, -1)
        x_prev_day_sales_enc = self.X_prev_day_sales[:, idx].reshape(-1, 1)
        x_calendar_enc = self.X_calendar[:enc_timesteps, :]
        # x_enc = np.concatenate([x_enc_dec_feats_enc, x_calendar_enc,
        #                         x_prev_day_sales_enc, x_enc_only_feats], axis=1)
        x_enc = np.concatenate([x_enc_dec_feats_enc, x_calendar_enc, x_prev_day_sales_enc], axis=1)
        x_enc_emb = self.X_enc_dec_feats[:enc_timesteps, idx, -num_embedding:].reshape(enc_timesteps, -1)

        # input data for decoder
        x_enc_dec_feats_dec = self.X_enc_dec_feats[enc_timesteps:, idx, :-num_embedding].reshape(dec_timesteps, -1)
        x_calendar_dec = self.X_calendar[enc_timesteps:, :]
        x_last_day_sales = self.X_last_day_sales[idx].reshape(-1)
        x_dec = np.concatenate([x_enc_dec_feats_dec, x_calendar_dec], axis=1)
        x_dec_emb = self.X_enc_dec_feats[enc_timesteps:, idx, -num_embedding:].reshape(dec_timesteps, -1)

        if self.Y is None:
            return [[torch.from_numpy(x_enc).float(), torch.from_numpy(x_enc_emb).long(),
                    torch.from_numpy(x_dec).float(), torch.from_numpy(x_dec_emb).long(),
                    torch.from_numpy(x_last_day_sales).float()]]

        return [[torch.from_numpy(x_enc).float(), torch.from_numpy(x_enc_emb).long(),
                torch.from_numpy(x_dec).float(), torch.from_numpy(x_dec_emb).long(),
                torch.from_numpy(x_last_day_sales).float()],
                self.Y[idx, :],
                [self.rmsse_denominator[idx]]]


class DataLoader:
    def __init__(self, config):
        self.config = config

        # load data
        with open(f'{self.config.data_file}', 'rb') as f:
            data_dict = pkl.load(f)

        self.X_prev_day_sales = data_dict['X_prev_day_sales']
        self.X_enc_only_feats = data_dict['X_enc_only_feats']
        self.X_enc_dec_feats = data_dict['X_enc_dec_feats']
        self.X_calendar = data_dict['X_calendar']
        self.Y = data_dict['Y']

    def create_train_loader(self, data_start_t=1969 - 1 - 1968, horizon_start_t=1969 - 1 - (28*4),
                            horizon_end_t=1969 - 1 - (28*3)):

        # calculate denominator for rmsse loss
        squared_movement = ((self.Y.T[data_start_t:horizon_start_t] -
                             self.X_prev_day_sales[data_start_t:horizon_start_t]).astype(np.int64) ** 2)
        actively_sold_in_range = (self.X_prev_day_sales[data_start_t:horizon_start_t] != -1).argmax(axis=0)
        rmsse_den = []
        for idx, first_active_sell_idx in enumerate(actively_sold_in_range):
            rmsse_den.append(squared_movement[first_active_sell_idx:, idx].mean())

        dataset = CustomDataset(self.X_prev_day_sales[data_start_t:horizon_start_t],
                                self.X_enc_only_feats[data_start_t:horizon_start_t],
                                self.X_enc_dec_feats[data_start_t:horizon_end_t],
                                self.X_calendar[data_start_t:horizon_end_t], self.X_prev_day_sales[horizon_start_t],
                                Y=self.Y[:, horizon_start_t:horizon_end_t],
                                rmsse_denominator=np.array(rmsse_den))

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=3, pin_memory=True)

    def create_val_loader(self, data_start_t=1969 - 1 - 1968, horizon_start_t=1969 - 1 - (28*3),
                          horizon_end_t=1969 - 1 - (28 * 2)):

        # calculate denominator for rmsse loss
        squared_movement = ((self.Y.T[data_start_t:horizon_start_t] -
                             self.X_prev_day_sales[data_start_t:horizon_start_t]).astype(np.int64) ** 2)
        actively_sold_in_range = (self.X_prev_day_sales[data_start_t:horizon_start_t] != -1).argmax(axis=0)
        rmsse_den = []
        for idx, first_active_sell_idx in enumerate(actively_sold_in_range):
            rmsse_den.append(squared_movement[first_active_sell_idx:, idx].mean())

        dataset = CustomDataset(self.X_prev_day_sales[data_start_t:horizon_start_t],
                                self.X_enc_only_feats[data_start_t:horizon_start_t],
                                self.X_enc_dec_feats[data_start_t:horizon_end_t],
                                self.X_calendar[data_start_t:horizon_end_t], self.X_prev_day_sales[horizon_start_t],
                                Y=self.Y[:, horizon_start_t:horizon_end_t],
                                rmsse_denominator=np.array(rmsse_den))

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, num_workers=3,
                                           pin_memory=True)

    def create_test_loader(self, data_start_t=1969 - 1 - 1968, horizon_start_t=1969 - 1 - (28 * 2),
                           horizon_end_t=1969 - 1 - (28 * 1)):

        dataset = CustomDataset(self.X_prev_day_sales[data_start_t:horizon_start_t],
                                self.X_enc_only_feats[data_start_t:horizon_start_t],
                                self.X_enc_dec_feats[data_start_t:horizon_end_t],
                                self.X_calendar[data_start_t:horizon_end_t], self.X_prev_day_sales[horizon_start_t])

        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, num_workers=3,
                                           pin_memory=True)
