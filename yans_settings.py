import torch.nn as nn
import torch.optim as optim

hyperparams = {
    "learning_rate": 0.02,
    "number_of_epochs": 20,
    "batch_size": 10,
    "rnn_layer_size": 100,
    "rnn_layer_number": 1,
    "bidirectional": False,
    "cell_type": "RNN_TANH",
    "dropout_rate": 0.5,
    "optimizer": optim.SGD,
    "loss_function": nn.CrossEntropyLoss,
    "activation": nn.Tanh
}