import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random
from models.model_utils.drnn import DRNN
from importlib import import_module
from utils.training_utils import ModelCheckpoint
random.seed(0)


# Build a seq2seq model
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, config):
        super(Encoder, self).__init__()
        self.config = config
        self.input_size = input_size

        self.drnns = nn.ModuleList([DRNN(self.input_size, config.ors_rnn_num_hidden, config.ors_rnn_num_layers,
                                         dropout=config.ors_enc_rnn_dropout, cell_type='LSTM')
                                    for i in range(config.ors_bidirectional + 1)])

    def forward(self, x_rnn):
        batch_size = x_rnn.shape[1]

        rnn_input = [x_rnn, torch.flip(x_rnn, [0])] if self.config.ors_bidirectional else [x_rnn]
        drnn_outputs, drnn_h0, drnn_h1 = [], [], []
        for i, drnn in enumerate(self.drnns):
            last_output, [h0, h1] = drnn(rnn_input[i])
            drnn_outputs.append(last_output)
            drnn_h0.append([h.view(-1, batch_size, self.config.ors_rnn_num_hidden) for h in h0])
            drnn_h1.append([h.view(-1, batch_size, self.config.ors_rnn_num_hidden) for h in h1])
        drnn_hiddens = [drnn_h0, drnn_h1]

        if self.config.ors_bidirectional:
            for j, drnn_h in enumerate(drnn_hiddens):
                drnn_hiddens[j] = [torch.cat([drnn_h[0][i], drnn_h[1][i]], 0)
                                   for i in range(self.config.ors_rnn_num_layers)]
        else:
            drnn_hiddens = [drnn_hiddens[0][0], drnn_hiddens[1][0]]

        return drnn_outputs, drnn_hiddens


# Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(Decoder, self).__init__()
        self.input_size = input_size

        self.rnn = nn.LSTM(self.input_size, config.ors_rnn_num_hidden, config.ors_rnn_num_layers,
                           bidirectional=config.ors_bidirectional, dropout=config.ors_dec_rnn_dropout)
        self.dropout = nn.Dropout(0.0)
        self.pred = nn.Linear(config.ors_rnn_num_hidden * (config.ors_bidirectional + 1), output_size)

    def forward(self, x_rnn, hidden):
        output, hidden = self.rnn(x_rnn, hidden)
        output = F.relu(self.pred(self.dropout(output[0])))

        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, embedder, representator, config):
        super().__init__()

        self.embedder = embedder
        self.representator = representator
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.fc_h0 = nn.Linear(sum([2**i for i in range(config.ors_rnn_num_layers)] * (config.ors_bidirectional + 1)),
                               config.ors_rnn_num_layers * (config.ors_bidirectional + 1))
        self.fc_h1 = nn.Linear(sum([2**i for i in range(config.ors_rnn_num_layers)] * (config.ors_bidirectional + 1)),
                               config.ors_rnn_num_layers * (config.ors_bidirectional + 1))

    def forward(self, x_enc, x_enc_emb, x_cal_enc_emb, x_dec, x_dec_emb, x_cal_dec_emb, x_prev_day_sales_dec):
        batch_size, pred_len = x_dec.shape[0:2]

        # create a tensor to store the outputs
        predictions = torch.zeros(batch_size, pred_len, 9).to(self.config.device)

        # Prepare inputs and send to encoder
        rnn_input = self.embedder(x_enc, x_enc_emb, x_cal_enc_emb)
        ors_inputs = [rnn_input]
        for i, rnn in enumerate(self.representator):
            rnn_input, h = rnn(rnn_input)
            ors_inputs.append(rnn_input)

        ors_inputs = torch.cat(ors_inputs, 2)
        encoder_output, hidden = self.encoder(ors_inputs)
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
            ors_inputs = [rnn_input]
            for i, rnn in enumerate(self.representator):
                rnn_input, h = rnn(rnn_input)
                ors_inputs.append(rnn_input)

            ors_inputs = torch.cat(ors_inputs, 2)
            decoder_output, hidden = self.decoder(ors_inputs, hidden)

            # add predictions to predictions tensor
            predictions[:, timestep] = decoder_output

        return predictions


def create_model(config):
    num_features_enc = sum(config.rs_rnn_num_hidden[:2]) * (config.rs_bidirectional + 1) + 107
    num_features_dec = sum(config.rs_rnn_num_hidden[:2]) * (config.rs_bidirectional + 1) + 107

    # Load model trained on raw data to extract weight for embeddings
    model_type = import_module('models.' + config.architecture)
    create_model_raw = getattr(model_type, 'create_model')
    model_raw = create_model_raw(config)
    model_checkpoint = ModelCheckpoint(weight_dir='./weights/raw/')
    model_raw, _, _, _ = model_checkpoint.load(model_raw, load_best=True)
    model_embedder = model_raw.embedder

    # Freeze the weights for pre-trained Embedder module
    for param in model_embedder.parameters():
        param.requires_grad = False
    model_embedder.eval()

    # Load representation model and extract data to be used for training
    model_type = import_module('models.' + config.rs_architecture)
    create_model_rs = getattr(model_type, 'create_model')
    model_rs = create_model_rs(config)
    model_checkpoint = ModelCheckpoint(weight_dir='./weights/representation/')
    model_rs, _, _, _ = model_checkpoint.load(model_rs, load_best=True)
    model_rs = model_rs.rnns[:config.rs_representation_layer + 1]

    # Freeze the weights for pre - trained Representation model
    for param in model_rs.parameters():
        param.requires_grad = False
    model_rs.eval()

    enc = Encoder(num_features_enc, config)
    dec = Decoder(num_features_dec, 9, config)
    model = Seq2Seq(enc, dec, model_embedder, model_rs, config)
    model.to(config.device)

    return model
