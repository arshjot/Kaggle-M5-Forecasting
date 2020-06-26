import torch


class Config:

    resume_training = False

    loss_fn = 'SPLLoss'
    metric = 'SPLMetric'
    secondary_metric = 'WRMSSEMetric'
    architecture = 'dilated_seq2seq'

    # Running a sliding window training will help increase the training data
    sliding_window = True  # Note: sliding window has not been tested with WRMSSELoss
    window_length = 28 * 13

    lag_and_roll_feats = True  # Note: Currently only works with dilated_seq2seq architecture
    lags = list(range(27, 42))  # Do not put lags < 27 - currently not compatible with seq2seq decoder code
    rolling = [7, 14, 30, 60]

    # *** RNN *** #
    # hidden dimension and no. of layers will be the same for both encoder and decoder
    rnn_num_hidden = 128
    rnn_num_layers = 2
    bidirectional = True
    enc_rnn_dropout = 0.2
    dec_rnn_dropout = 0.0
    teacher_forcing_ratio = 0.0

    # *** Transformer *** #
    enc_nhead = 4
    enc_nlayers = 2
    enc_dropout = 0.1
    dec_nhead = 4
    dec_nlayers = 2
    dec_dropout = 0.1

    num_epochs = 200
    batch_size = 160
    learning_rate = 0.0003

    # training, validation and test periods
    training_ts = {'data_start_t': 1969 - 1 - (28 * 29), 'horizon_start_t': 1969 - 1 - (28 * 4),
                   'horizon_end_t': 1969 - 1 - (28 * 3)}
    validation_ts = {'data_start_t': 1969 - 1 - (28 * 16), 'horizon_start_t': 1969 - 1 - (28 * 3),
                     'horizon_end_t': 1969 - 1 - (28 * 2)}
    test_ts = {'data_start_t': 1969 - 1 - (28 * 15), 'horizon_start_t': 1969 - 1 - (28 * 2),
               'horizon_end_t': 1969 - 1 - (28 * 1)}

    data_file = '../accuracy_stream/data/data.pickle'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
