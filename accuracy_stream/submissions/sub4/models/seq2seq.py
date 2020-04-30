import torch
import torch.nn as nn
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


# Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_sizes, output_size, config):
        super(Decoder, self).__init__()
        self.input_size = input_size

        self.embeddings = nn.ModuleList([nn.Embedding(classes, hidden_size)
                                         for classes, hidden_size in embedding_sizes])
        self.rnn = nn.LSTM(self.input_size, config.rnn_num_hidden,
                           config.rnn_num_layers, dropout=config.dec_rnn_dropout, bidirectional=config.bidirectional)
        self.pred = nn.Linear(config.rnn_num_hidden * (config.bidirectional + 1), output_size)

    def forward(self, x, x_emb, hidden):
        x, x_emb = x.permute(1, 0, 2), x_emb.permute(1, 0, 2)  # make time-major
        output_emb = [emb(x_emb[:, :, i]) for i, emb in enumerate(self.embeddings)]
        output_emb = torch.cat(output_emb, 2)
        x_rnn = torch.cat([x, output_emb], 2)

        output, hidden = self.rnn(x_rnn, hidden)
        #         shape = output.size()
        #         output = self.pred(output.view(-1, output.size(2)))
        #         output = output.view(shape[0], shape[1]).permute(1, 0)
        output = self.pred(output[0])
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def forward(self, x_enc, x_enc_emb, x_dec, x_dec_emb, x_last_day_sales):
        batch_size, pred_len = x_dec.shape[0:2]

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
            decoder_output, hidden = self.decoder(dec_input, x_dec_emb[:, timestep, :].unsqueeze(1), hidden)

            # add predictions to predictions tensor
            predictions[:, timestep] = decoder_output.view(-1)

        return predictions


def create_model(config):
    embedding_sizes = [(3049, 50), (7, 4), (10, 5)]  # for item_id, dept_id, store_id respectively
    num_features_enc = 46 + sum([j for i, j in embedding_sizes])
    num_features_dec = 46 + sum([j for i, j in embedding_sizes])
    enc = Encoder(num_features_enc, embedding_sizes, config)
    dec = Decoder(num_features_dec, embedding_sizes, 1, config)
    model = Seq2Seq(enc, dec, config)
    model.to(config.device)

    return model
