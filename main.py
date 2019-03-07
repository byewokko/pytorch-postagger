import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

torch.manual_seed(123)

import argparse
import sys
import reader
from time import time

import util

'''
Hyperparameters
'''

hparams = {
    # size of the hidden layer
    "rnn_layer_size": 128,

    # numbers of the training epochs
    "number_of_epochs": 20,

    # mini-batch size
    "batch_size": 128,

    # learning rate
    "learning_rate": 0.01,

    # dropout
    "dropout_rate": 0.5,

    # number of hidden layers
    "rnn_layer_number": 1,

    # length of input sentences
    # (shorter sentences are padded to match this length)
    # (do not change it)
    "input_len": 50
}

'''
Path settings
'''

root_dir = "/home/m17/hruska/PycharmProjects/pytorch-postagger"

emb_file = "embeddings/glove.txt" #pre-trained embeddings
data_dir = "english"
tag_file = "UDtags.txt"

emb_file = os.path.join(root_dir, emb_file)
tag_file = os.path.join(root_dir, tag_file)
train_file = os.path.join(root_dir, data_dir, "train.txt")
dev_file = os.path.join(root_dir, data_dir, "dev.txt")
test_file = os.path.join(root_dir, data_dir, "test.txt")


class Tagger(nn.Module):
    def __init__(self, input_len, embedding, rnn_layer_size, rnn_layer_number, tagset_size, batch_size, **kwargs):
        super(Tagger, self).__init__()

        self.input_len = input_len
        self.embedding = embedding
        self.emb_size = embedding.shape[1]
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_number = rnn_layer_number
        self.batch_size = batch_size
        self.tagset_size = tagset_size

        self.__build_model()

    def __build_model(self):
        self.word_emb = nn.Embedding.from_pretrained(self.embedding)
        self.lstm = nn.LSTM(self.emb_size,
                            hidden_size=self.rnn_layer_size,
                            num_layers=self.rnn_layer_number,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)
        self.hidden = self.init_hidden()
        # DROPOUT PLUGS IN HERE
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(2 * self.rnn_layer_size, self.tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size=None):
        """
        Initialize the hidden layers as random tensors
        :return:
        """
        if not batch_size:
            batch_size = self.batch_size

        hidden = torch.randn(2*self.rnn_layer_number, batch_size, self.rnn_layer_size)
        cell = torch.randn(2*self.rnn_layer_number, batch_size, self.rnn_layer_size)
        return hidden, cell

    def forward(self, X, lengths):
        self.hidden = self.init_hidden()
        batch_size, sent_len = X.size()
        X = self.word_emb(X)
        # X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        X, self.hidden = self.lstm(X, self.hidden)
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = self.dropout(X)
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.dense(X)
        X = self.softmax(X)
        X = X.view(-1, self.tagset_size)
        Y_h = X
        return Y_h


def train_model(model, train_data, dev_data, n_epochs, loss_function, optimizer, batch_size, **kwargs):

    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=8)
    devloader = torch.utils.data.DataLoader(dev_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=8)

    train_batches = len(trainloader)
    dev_batches = len(devloader)


    for epoch in range(n_epochs):

        print("epoch", epoch+1)

        for batch_n, (X, Y, sent_len) in enumerate(trainloader):
            print("training   |{}|".format(util.loadbar(batch_n/(train_batches-1))), end="\r")
            sys.stdout.flush()

            if len(sent_len) < model.batch_size:
                continue
            max_sent = max(sent_len)

            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(sent_len))

            Y_h = model(X, sent_len)

            train_loss = loss_function(Y_h, Y.view(-1))
            train_loss.backward()
            optimizer.step()


        print()

        for batch_n, (X, Y, sent_len) in enumerate(devloader):
            print("validation |{}|".format(util.loadbar(batch_n /(dev_batches-1))), end="\r")
            sys.stdout.flush()

            if len(sent_len) < model.batch_size:
                continue
            max_sent = max(sent_len)

            model.zero_grad()
            model.hidden = model.init_hidden()

            Y_h = model(X, sent_len)

            with torch.no_grad():
                dev_loss = loss_function(Y_h, Y.view(-1))


        print()
        print("train loss", train_loss.item(), "val loss", dev_loss.item())


def test_model(model, x, y, loss_function):
    """

    :param model:
    :param x:
    :param y:
    :param loss_function:
    :return:
    """
    with torch.no_grad():
        tag_scores = model(x, y)
    return loss, accuracy

#######################################

def main():
    # TODO: make dataprep more readable
    """
    " Data preparation
    """

    # First we read the word embeddings file
    # Go take a look how the file is structured!
    # This function returns a word-to-index dictionary and the embedding tensor
    word2i, _, embeddings = util.load_embeddings(emb_file)
    print("emb loaded")

    tag2i, _ = util.load_postags(tag_file)
    print("tags loaded")

    # Read in data and create tensors
    # _x is the input tokens and _y is the labels
    """train_x, train_y, train_y_list = util.prepare_data(train_file, word2i, tag2i, hparams["input_len"])
    dev_x, dev_y, dev_y_list = util.prepare_data(dev_file, word2i, tag2i, hparams["input_len"])
    test_x, test_y, test_y_list = util.prepare_data(test_file, word2i, tag2i, hparams["input_len"])"""
    train_data = util.prepare_data(train_file, word2i, tag2i, hparams["input_len"])
    dev_data = util.prepare_data(dev_file, word2i, tag2i, hparams["input_len"])
    test_data = util.prepare_data(test_file, word2i, tag2i, hparams["input_len"])
    print("data loaded")

    tagset_size = len(tag2i)  # Numbers of tags

    #print(train_x.shape, train_y.shape, file=sys.stderr)
    #print(embeddings.shape, file=sys.stderr)

    """
    a = nn.Embedding.from_pretrained(embeddings)
    b = nn.LSTM(50, 100)
    c, d = b(a(train_x[:10]))
    print(c.shape)
    print(d)

    quit()"""

    model = Tagger(embedding=embeddings,
                   tagset_size=tagset_size,
                   **hparams)

    train_model(model,
                train_data=train_data,
                dev_data=dev_data,
                n_epochs=20,
                loss_function=nn.NLLLoss(),
                optimizer=optim.Adam(model.parameters(), lr=0.01),
                **hparams)

    test_model(model,
               (test_x, test_y))

if __name__ == "__main__":
    main()