import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

torch.manual_seed(123)

import argparse
import sys
from time import time

import util
import tagger

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
    # if you increase the number of hidden layers
    # you can add dropout in between them
    "rnn_layer_number": 2,

    # length of input sentences
    # (shorter sentences are padded to match this length)
    # (do not change it)
    "input_len": 50,

    # optimizer type
    "optimizer": optim.Adam,

    # loss function
    "loss_function": nn.NLLLoss(), # weight parameter
    "loss_function": nn.NLLLoss(),

    # tagger type
    #"tagger": tagger.LSTMTagger,
    "tagger": tagger.RNNTagger,

    #"activation": nn.ReLU,
    "activation": nn.LogSoftmax,
    #"activation": nn.Sigmoid(),

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

# TODO allow random embeddings
def train_model(model, train_data, dev_data, number_of_epochs, loss_function, optimizer,
                learning_rate, batch_size, **kwargs):

    # initialize the optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # initialize DataLoaders that will yield data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=8)
    dev_loader = torch.utils.data.DataLoader(dev_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=8)

    n_train_batches = len(train_loader)
    n_dev_batches = len(dev_loader)

    for epoch in range(number_of_epochs):

        print("epoch", epoch+1)

        for batch_n, (X, Y, sent_len) in enumerate(train_loader):
            print("training   |{}|".format(util.loadbar(batch_n/(n_train_batches-1))), end="\r")
            sys.stdout.flush()

            # if len(sent_len) < model.batch_size:
            #     continue
            max_sent = max(sent_len)

            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(sent_len))

            Y_h = model(X, sent_len)

            train_loss = loss_function(Y_h, Y.view(-1))
            train_loss.backward()
            optimizer.step()


        print()

        for batch_n, (X, Y, sent_len) in enumerate(dev_loader):
            print("validation |{}|".format(util.loadbar(batch_n /(n_dev_batches-1))), end="\r")
            sys.stdout.flush()

            # if len(sent_len) < model.batch_size:
            #     continue
            max_sent = max(sent_len)

            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(sent_len))

            Y_h = model(X, sent_len)

            with torch.no_grad():
                dev_loss = loss_function(Y_h, Y.view(-1))


        print()
        print("train loss", train_loss.item(), "val loss", dev_loss.item())


def test_model(model, test_data, loss_function, batch_size):


    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=8)
    raise NotImplementedError

    return loss, accuracy

#######################################

def main():
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
    train_data = util.prepare_data(train_file, word2i, tag2i, hparams["input_len"])
    dev_data = util.prepare_data(dev_file, word2i, tag2i, hparams["input_len"])
    test_data = util.prepare_data(test_file, word2i, tag2i, hparams["input_len"])
    print("data loaded")

    tagset_size = len(tag2i)

    model = hparams["tagger"](embedding=embeddings,
                              tagset_size=tagset_size,
                              **hparams)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    quit()

    train_model(model,
                train_data=train_data,
                dev_data=dev_data,
                **hparams)

    #test_model(model, test_data=test_data)

if __name__ == "__main__":
    main()