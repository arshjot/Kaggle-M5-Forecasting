import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.model_utils.drnn import DRNN
import random


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
                                         dropout=config.enc_rnn_dropout, cell_type='LSTM', layer_outputs=True)
                                    for i in range(config.bidirectional + 1)])

    def forward(self, x_rnn):
        batch_size = x_rnn.shape[1]

        rnn_input = [x_rnn, torch.flip(x_rnn, [0])] if self.config.bidirectional else [x_rnn]
        drnn_outputs, drnn_h0, drnn_h1 = [], [], []
        for i, drnn in enumerate(self.drnns):
            last_output, [h0, h1], all_outputs = drnn(rnn_input[i])
            drnn_outputs.append(all_outputs)
            drnn_h0.append([h.view(-1, batch_size, self.config.rnn_num_hidden) for h in h0])
            drnn_h1.append([h.view(-1, batch_size, self.config.rnn_num_hidden) for h in h1])
        drnn_hiddens = [drnn_h0, drnn_h1]

        if self.config.bidirectional:
            for j, drnn_h in enumerate(drnn_hiddens):
                drnn_hiddens[j] = [torch.cat([drnn_h[0][i], drnn_h[1][i]], 0)
                                   for i in range(self.config.rnn_num_layers)]
            drnn_outputs = [torch.cat([drnn_outputs[0][i], drnn_outputs[1][i]], 2)
                            for i in range(self.config.rnn_num_layers)]
        else:
            drnn_hiddens = [drnn_hiddens[0][0], drnn_hiddens[1][0]]
            drnn_outputs = drnn_outputs[0]

        return drnn_outputs[-1], drnn_hiddens


# Attention Decoder
class AttnDecoder(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(AttnDecoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.max_length = config.window_length if config.sliding_window \
            else config.training_ts['horizon_start_t'] - config.training_ts['data_start_t']

        self.attn = nn.Linear(self.input_size + (config.rnn_num_hidden * 2), self.max_length)
        self.attn_combine = nn.Linear(self.input_size + (config.rnn_num_hidden * (config.bidirectional + 1)),
                                      self.input_size)
        self.rnn = nn.LSTM(self.input_size, config.rnn_num_hidden,
                           config.rnn_num_layers, dropout=config.dec_rnn_dropout, bidirectional=config.bidirectional)
        self.pred = nn.Linear(config.rnn_num_hidden * (config.bidirectional + 1), output_size)

    def forward(self, x_rnn, hidden, encoder_outputs):
        attn_weights = F.softmax(
            self.attn(torch.cat((x_rnn[0], hidden[0][0], hidden[1][0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2))

        output = torch.cat((x_rnn[0], attn_applied[:, 0, :]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output, hidden = self.rnn(output, hidden)
        output = F.relu(self.pred(output[0]))

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, embedder, config):
        super().__init__()

        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.max_length = config.window_length if config.sliding_window \
            else config.training_ts['horizon_start_t'] - config.training_ts['data_start_t']

        self.fc_h0 = nn.Linear(sum([2**i for i in range(config.rnn_num_layers)] * (config.bidirectional + 1)),
                               config.rnn_num_layers * (config.bidirectional + 1))
        self.fc_h1 = nn.Linear(sum([2**i for i in range(config.rnn_num_layers)] * (config.bidirectional + 1)),
                               config.rnn_num_layers * (config.bidirectional + 1))

    def forward(self, x_enc, x_enc_emb, x_cal_enc_emb, x_dec, x_dec_emb, x_cal_dec_emb, x_prev_day_sales_dec):
        batch_size, pred_len = x_dec.shape[0:2]

        # Ignore some initial timesteps of encoder data, according to the max_length allowed
        x_enc, x_enc_emb = x_enc[:, -self.max_length:], x_enc_emb[:, -self.max_length:]
        x_cal_enc_emb = x_cal_enc_emb[:, -self.max_length:]

        # create a tensor to store the outputs
        predictions = torch.zeros(batch_size, pred_len, 9).to(self.config.device)

        # Prepare inputs and send to encoder
        rnn_input = self.embedder(x_enc, x_enc_emb, x_cal_enc_emb)
        encoder_output, hidden = self.encoder(rnn_input)
        h0 = self.fc_h0(torch.cat(hidden[0], 0).permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        h1 = self.fc_h1(torch.cat(hidden[1], 0).permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        hidden = [h0, h1]

        # for each prediction timestep, use the output of the previous step,
        # concatenated with other features as the input

        # enable teacher forcing only if model is in training phase
        use_teacher_forcing = True if (random.random() < self.config.teacher_forcing_ratio) & self.training else False

        for timestep in range(0, pred_len):

            if timestep == 0:
                # for the first timestep of decoder, use previous steps' sales
                dec_input = torch.cat([x_dec[:, 0, :], x_prev_day_sales_dec[:, 0]], dim=1).unsqueeze(1)
            else:
                if use_teacher_forcing:
                    dec_input = torch.cat([x_dec[:, timestep, :], x_prev_day_sales_dec[:, timestep]], dim=1).unsqueeze(1)
                else:
                    # for next timestep, current timestep's output will serve as the input along with other features
                    dec_input = torch.cat([x_dec[:, timestep, :], decoder_output[:, 4].unsqueeze(1)], dim=1).unsqueeze(1)

            # Prepare inputs and send to decoder
            # the hidden state of the encoder will be the initialize the decoder's hidden state
            rnn_input = self.embedder(dec_input, x_dec_emb[:, timestep, :].unsqueeze(1),
                                      x_cal_dec_emb[:, timestep, :].unsqueeze(1))
            decoder_output, hidden, _ = self.decoder(rnn_input, hidden, encoder_output)

            # add predictions to predictions tensor
            predictions[:, timestep] = decoder_output

        return predictions


def create_model(config):
    # for item_id, dept_id, cat_id, store_id, state_id respectively
    embedding_sizes = [(3049 + 1, 50), (7 + 1, 4), (3 + 1, 2), (10 + 1, 5), (3 + 1, 2)]
    cal_embedding_sizes = (31, 16)
    num_features_enc = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2
    num_features_dec = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2
    emb = Embedder(num_features_enc, embedding_sizes, cal_embedding_sizes, config)
    enc = Encoder(num_features_enc, config)
    dec = AttnDecoder(num_features_dec, 9, config)
    model = Seq2Seq(enc, dec, emb, config)
    model.to(config.device)

    return model
