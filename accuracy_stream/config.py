import torch


class Config:

    loss_fn = 'RMSSELoss'
    metric = 'WRMSSEMetric'
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

    # training, validation and test periods
    training_ts = {'data_start_t': 1969 - 1 - 1968, 'horizon_start_t': 1969 - 1 - (28 * 4),
                   'horizon_end_t': 1969 - 1 - (28 * 3)}
    validation_ts = {'data_start_t': 1969 - 1 - 1968, 'horizon_start_t': 1969 - 1 - (28 * 3),
                     'horizon_end_t': 1969 - 1 - (28 * 2)}
    test_ts = {'data_start_t': 1969 - 1 - 1968, 'horizon_start_t': 1969 - 1 - (28 * 2),
               'horizon_end_t': 1969 - 1 - (28 * 1)}

    data_file = 'data/data.pickle'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
