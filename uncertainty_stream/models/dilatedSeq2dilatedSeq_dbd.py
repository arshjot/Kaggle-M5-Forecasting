import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random
from models.model_utils.drnn import DRNN
random.seed(0)


# Build a seq2seq model
class Embedder(nn.Module):
    def __init__(self, input_size, embedding_sizes, cal_embedding_sizes, config):
        super(Embedder, self).__init__()
        self.config = config
        self.input_size = input_size

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.cal_embedding = nn.Embedding(cal_embedding_sizes[0], cal_embedding_sizes[1])

    def forward(self, x, x_emb, x_cal_emb):
        x, x_emb, x_cal_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2), x_cal_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        # share embedding layer for both the calendar events
        output_emb_cal = [self.cal_embedding(x_cal_emb[:, :, 0]), self.cal_embedding(x_cal_emb[:, :, 1])]
        output_emb_cal = torch.cat(output_emb_cal, 2)

        return torch.cat([x, output_emb, output_emb_cal], 2)


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, config):
        super(Encoder, self).__init__()
        self.config = config
        self.input_size = input_size

        self.drnns = nn.ModuleList([DRNN(self.input_size, config.rnn_num_hidden, config.rnn_num_layers,
                                         dropout=config.enc_rnn_dropout, cell_type='LSTM')
                                    for i in range(config.bidirectional + 1)])

    def forward(self, x_rnn):
        batch_size = x_rnn.shape[1]

        rnn_input = [x_rnn, torch.flip(x_rnn, [0])] if self.config.bidirectional else [x_rnn]
        drnn_outputs, drnn_hiddens = [], []
        for i, drnn in enumerate(self.drnns):
            last_output, [h0, h1] = drnn(rnn_input[i])
            drnn_outputs.append(last_output)
            layer_wise_hidden = [[i.view(-1, batch_size, self.config.rnn_num_hidden),
                                  j.view(-1, batch_size, self.config.rnn_num_hidden)] for i, j in zip(h0, h1)]
            drnn_hiddens.append(layer_wise_hidden)

        return drnn_outputs, drnn_hiddens


# Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.config = config

        self.drnns = nn.ModuleList([DRNN(self.input_size, config.rnn_num_hidden, config.rnn_num_layers,
                                         dropout=config.enc_rnn_dropout, cell_type='LSTM')
                                    for i in range(config.bidirectional + 1)])
        self.pred = nn.Linear(config.rnn_num_hidden * (config.bidirectional + 1), output_size)

    def forward(self, x_rnn, hidden):
        rnn_input = [x_rnn, torch.flip(x_rnn, [0])] if self.config.bidirectional else [x_rnn]
        drnn_outputs = []
        for i, drnn in enumerate(self.drnns):
            last_output, _ = drnn(rnn_input[i], hidden[i])
            drnn_outputs.append(last_output)

        output = torch.cat(drnn_outputs, 2) if self.config.bidirectional else drnn_outputs[0]

        output = F.relu(self.pred(output.permute(1, 0, 2)))

        return output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, embedder, config):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.embedder = embedder
        self.config = config

    def forward(self, x_enc, x_enc_emb, x_cal_enc_emb, x_dec, x_dec_emb, x_cal_dec_emb, x_prev_day_sales_dec):
        # Prepare inputs and send to encoder
        rnn_input = self.embedder(x_enc, x_enc_emb, x_cal_enc_emb)
        encoder_output, hidden = self.encoder(rnn_input)

        # Prepare inputs and send to decoder
        rnn_input = self.embedder(x_dec, x_dec_emb, x_cal_dec_emb)
        # as it is a day-by-day model (not a recursive one), no need for a loop
        predictions = self.decoder(rnn_input, hidden)

        return predictions


def create_model(config):
    # for item_id, dept_id, cat_id, store_id, state_id respectively
    embedding_sizes = [(3049 + 1, 50), (7 + 1, 4), (3 + 1, 2), (10 + 1, 5), (3 + 1, 2)]
    cal_embedding_sizes = (31, 16)
    num_features_enc = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2
    num_features_dec = 11 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2
    emb = Embedder(num_features_enc, embedding_sizes, cal_embedding_sizes, config)
    enc = Encoder(num_features_enc, config)
    dec = Decoder(num_features_dec, 9, config)
    model = Seq2Seq(enc, dec, emb, config)
    model.to(config.device)

    return model
