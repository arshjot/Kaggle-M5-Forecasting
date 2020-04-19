import torch


class Config:

    architecture = 'seq2seq'
    data_file = 'data/data.pickle'
    loss_fn = 'RMSSELoss'

    # hidden dimension and no. of layers will be the same for both encoder and decoder
    rnn_num_hidden = 256
    rnn_num_layers = 2
    enc_rnn_dropout = 0.0
    dec_rnn_dropout = 0.0

    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001

    device = torch.device('cuda')