import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# Build a seq2seq model
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, config):
        super(Encoder, self).__init__()
        self.input_size = input_size

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.rnn = nn.LSTM(self.input_size, config.rnn_num_hidden,
                           config.rnn_num_layers, dropout=config.enc_rnn_dropout, bidirectional=config.bidirectional)

    def forward(self, x, x_emb):
        x, x_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)

        x_rnn = torch.cat([x, output_emb], 2)

        output, hidden = self.rnn(x_rnn)
        return output, hidden


# Attention Decoder
class AttnDecoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, output_size, config):
        super(AttnDecoder, self).__init__()
        self.input_size = input_size
        self.max_length = config.training_ts['horizon_start_t'] - config.training_ts['data_start_t']

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.attn = nn.Linear(self.input_size + (config.rnn_num_hidden * 2), self.max_length)
        self.attn_combine = nn.Linear(self.input_size + (config.rnn_num_hidden * (config.bidirectional + 1)),
                                      self.input_size)
        self.rnn = nn.LSTM(self.input_size, config.rnn_num_hidden,
                           config.rnn_num_layers, dropout=config.dec_rnn_dropout, bidirectional=config.bidirectional)
        self.pred = nn.Linear(config.rnn_num_hidden * (config.bidirectional + 1), output_size)

    def forward(self, x, x_emb, hidden, encoder_outputs):
        x, x_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)
        x_rnn = torch.cat([x, output_emb], 2)

        attn_weights = F.softmax(
            self.attn(torch.cat((x_rnn[0], hidden[0][0], hidden[1][0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute(1, 0, 2))

        output = torch.cat((x_rnn[0], attn_applied[:, 0, :]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        # output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.pred(output[0])

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.max_length = config.training_ts['horizon_start_t'] - config.training_ts['data_start_t']

    def forward(self, x_enc, x_enc_emb, x_dec, x_dec_emb, x_last_day_sales):
        batch_size, pred_len = x_dec.shape[0:2]

        # Ignore some initial timesteps of encoder data, according to the max_length allowed
        x_enc, x_enc_emb = x_enc[:, :self.max_length], x_enc_emb[:, :self.max_length]

        # create a tensor to store the outputs
        predictions = torch.zeros(batch_size, pred_len).to(self.config.device)

        encoder_output, hidden = self.encoder(x_enc, x_enc_emb)

        # for each prediction timestep, use the output of the previous step,
        # concatenated with other features as the input

        for timestep in range(0, pred_len):

            if timestep == 0:
                # for the first timestep of decoder, use previous steps' sales
                dec_input = torch.cat([x_dec[:, 0, :], x_last_day_sales], dim=1).unsqueeze(1)
            else:
                # for next timestep, current timestep's output will serve as the input along with other features
                dec_input = torch.cat([x_dec[:, timestep, :], decoder_output], dim=1).unsqueeze(1)

            # the hidden state of the encoder will be the initialize the decoder's hidden state
            decoder_output, hidden, _ = self.decoder(dec_input, x_dec_emb[:, timestep, :].unsqueeze(1), hidden,
                                                     encoder_output)

            # add predictions to predictions tensor
            predictions[:, timestep] = decoder_output.view(-1)

        return predictions


def create_model(config):
    embedding_sizes = [(3049, 50), (7, 4), (10, 5)]  # for item_id, dept_id, store_id respectively
    num_features_enc = 46 + sum([j for i, j in embedding_sizes])
    num_features_dec = 46 + sum([j for i, j in embedding_sizes])
    enc = Encoder(num_features_enc, embedding_sizes, config)
    dec = AttnDecoder(num_features_dec, embedding_sizes, 1, config)
    model = Seq2Seq(enc, dec, config)
    model.to(config.device)

    return model