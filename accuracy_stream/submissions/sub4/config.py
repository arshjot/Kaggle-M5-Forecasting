import torch


class Config:

    resume_training = False

    loss_fn = 'WRMSSELevel12Loss'
    metric = 'WRMSSEMetric'
    secondary_metric = 'RMSSELoss'
    architecture = 'seq2seq_w_attention'

    # hidden dimension and no. of layers will be the same for both encoder and decoder
    rnn_num_hidden = 256
    rnn_num_layers = 2
    bidirectional = True
    enc_rnn_dropout = 0.1
    dec_rnn_dropout = 0.1

    num_epochs = 200
    batch_size = 128
    learning_rate = 0.001

    # training, validation and test periods
    training_ts = {'data_start_t': 1969 - 1 - 1968, 'horizon_start_t': 1969 - 1 - (28 * 4),
                   'horizon_end_t': 1969 - 1 - (28 * 3)}
    validation_ts = {'data_start_t': 1969 - 1 - 1968, 'horizon_start_t': 1969 - 1 - (28 * 3),
                     'horizon_end_t': 1969 - 1 - (28 * 2)}
    test_ts = {'data_start_t': 1969 - 1 - 1968, 'horizon_start_t': 1969 - 1 - (28 * 2),
               'horizon_end_t': 1969 - 1 - (28 * 1)}

    data_file = '../../data/data.pickle'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
