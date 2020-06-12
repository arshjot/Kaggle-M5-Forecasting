import torch.nn as nn
import random
from importlib import import_module
from utils.training_utils import ModelCheckpoint
random.seed(0)


# Build an autoencoder model
# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, config):
        super(Autoencoder, self).__init__()
        self.config = config
        self.input_size = input_size

        rnn_in_size = [self.input_size] + [h * (config.rs_bidirectional + 1) for h in config.rs_rnn_num_hidden]
        rnn_out_size = [h for h in config.rs_rnn_num_hidden] + [self.input_size]
        self.rnns = nn.ModuleList([nn.LSTM(rnn_in_size[i], rnn_out_size[i], 1, bidirectional=config.rs_bidirectional)
                                   for i in range(len(config.rs_rnn_num_hidden) + 1)])
        self.rnn_dropouts = nn.ModuleList(
            [nn.Dropout(config.rs_rnn_dropout[i]) for i in range(len(config.rs_rnn_num_hidden))])

        self.adaptor = nn.Linear(rnn_in_size[0] * (config.rs_bidirectional + 1), rnn_in_size[0])

    def forward(self, x_rnn):
        for i, rnn in enumerate(self.rnns):
            if i != 0:
                x_rnn = self.rnn_dropouts[i - 1](x_rnn)
            x_rnn, h = rnn(x_rnn)

        output = self.adaptor(x_rnn.permute(1, 0, 2))

        return output.permute(1, 0, 2)


class Raw2Representation(nn.Module):
    def __init__(self, embedder, autoencoder, config):
        super().__init__()

        self.embedder = embedder
        self.autoencoder = autoencoder
        self.config = config

    def forward(self, x, x_emb, x_cal_emb):
        # Prepare inputs and send to autoencoder
        rnn_input = self.embedder(x, x_emb, x_cal_emb)
        rnn_output = self.autoencoder(rnn_input)

        return rnn_input, rnn_output


def create_model(config):
    # for item_id, dept_id, cat_id, store_id, state_id respectively
    embedding_sizes = [(3049 + 1, 50), (7 + 1, 4), (3 + 1, 2), (10 + 1, 5), (3 + 1, 2)]
    cal_embedding_sizes = (31, 16)
    num_features = 12 + sum([j for i, j in embedding_sizes]) + cal_embedding_sizes[1] * 2

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

    autoenc = Autoencoder(num_features, config)
    model = Raw2Representation(model_embedder, autoenc, config)
    model.to(config.device)

    return model
