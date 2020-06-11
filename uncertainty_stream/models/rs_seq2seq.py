import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random
from models.model_utils.drnn import DRNN
random.seed(0)


# Build an autoencoder model
# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, config):
        super(Autoencoder, self).__init__()
        self.config = config
        self.input_size = input_size

        rnn_in_size = [self.input_size] + [h * (config.rs_bidirectional + 1) for h in config.rs_rnn_num_hidden]
        rnn_out_size = [h for h in config.rs_rnn_num_hidden] + [self.input_size]
        self.rnns = nn.ModuleList([nn.LSTM(rnn_in_size[i], rnn_out_size[i], 1, bidirectional=config.rs_bidirectional)
                                   for i in range(len(config.rs_rnn_num_hidden) + 1)])
        self.rnn_dropouts = nn.ModuleList(
            [nn.Dropout(config.rs_rnn_dropout[i]) for i in range(len(config.rs_rnn_num_hidden))])

        self.adaptor = nn.Linear(rnn_in_size[0] * (config.rs_bidirectional + 1), rnn_in_size[0])

    def forward(self, x_rnn):
        x_rnn = x_rnn.permute(1, 0, 2)

        for i, rnn in enumerate(self.rnns):
            if i != 0:
                x_rnn = self.rnn_dropouts[i - 1](x_rnn)
            x_rnn, h = rnn(x_rnn)

        output = self.adaptor(x_rnn.permute(1, 0, 2))

        return output


def create_model(config):
    # for item_id, dept_id, cat_id, store_id, state_id respectively
    embedding_sizes = [(3049 + 1, 50), (7 + 1, 4), (3 + 1, 2), (10 + 1, 5), (3 + 1, 2)]
    cal_embedding_sizes = (31, 16)
    num_features = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2
    model = Autoencoder(num_features, config)
    model.to(config.device)

    return model
