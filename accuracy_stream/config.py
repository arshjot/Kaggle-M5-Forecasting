import torch


class Config:

    loss_fn = 'RMSSELoss'
    architecture = 'seq2seq'

    # hidden dimension and no. of layers will be the same for both encoder and decoder
    rnn_num_hidden = 256
    rnn_num_layers = 2
    bidirectional = True
    enc_rnn_dropout = 0.0
    dec_rnn_dropout = 0.0

    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001

    data_file = 'data/data.pickle'
    device = torch.device('cuda')
