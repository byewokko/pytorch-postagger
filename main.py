import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import util
import tagger
import dataprep

torch.manual_seed(123)

'''
Hyperparameters
'''

hparams = {
    # size of the hidden layer
    "rnn_layer_size": 128,

    # numbers of the training epochs
    "number_of_epochs": 5,

    # mini-batch size
    "batch_size": 256,

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
    #"loss_function": nn.NLLLoss(), # weight parameter
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
                learning_rate, batch_size, tagset_size, **kwargs):

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
    total_iter = number_of_epochs*(n_train_batches + n_dev_batches)

    timer = util.Timer()
    timer.start()

    print("epoch\ttr_loss\tvl_loss\ttr_acc\tvl_acc\ttr_f1\tvl_f1")

    for epoch in range(number_of_epochs):

        train_confm = util.ConfusionMatrix(tagset_size)

        for batch_n, (X, Y, sent_len) in enumerate(train_loader):
            print("Epoch {:>3d}: Training   |{}| {}".format(epoch+1,
                                                            util.loadbar(batch_n/(n_train_batches-1)),
                                                            timer.remaining(total_iter)), end="\r")
            sys.stdout.flush()

            max_sent = max(sent_len)

            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(sent_len))

            Y_h = model(X, sent_len)
            pred_tags = Y_h.max(dim=1)[1]
            train_confm.add(pred_tags, Y)

            train_loss = loss_function(Y_h, Y.view(-1))
            train_loss.backward()
            optimizer.step()

            timer.tick()

        dev_confm = util.ConfusionMatrix(tagset_size)

        for batch_n, (X, Y, sent_len) in enumerate(dev_loader):
            print("Epoch {:>3d}: Validation |{}| {}".format(epoch+1,
                                                            util.loadbar(batch_n/(n_dev_batches-1)),
                                                            timer.remaining(total_iter)), end="\r")
            sys.stdout.flush()

            max_sent = max(sent_len)

            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(sent_len))

            Y_h = model(X, sent_len)
            pred_tags = Y_h.max(dim=1)[1]
            dev_confm.add(pred_tags, Y)

            with torch.no_grad():
                dev_loss = loss_function(Y_h, Y.view(-1))


            timer.tick()

        print("\x1b[2K", end="")
        print("{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
            epoch+1, train_loss, dev_loss, train_confm.accuracy(), dev_confm.accuracy(),
            train_confm.f_score().mean(), dev_confm.f_score().mean()))


def predict(model, data, loss_function, batch_size, tagset_size, class_dict, **kwargs):

    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)
    n_batches = len(dataloader)

    confm = util.ConfusionMatrix(tagset_size)

    for batch_n, (X, Y, sent_len) in enumerate(dataloader):
        print("Predicting |{}|".format(util.loadbar(batch_n / (n_batches - 1))), end="\r")
        sys.stdout.flush()

        max_sent = max(sent_len)

        model.hidden = model.init_hidden(batch_size=len(sent_len))

        Y_h = model(X, sent_len)
        with torch.no_grad():
            loss = loss_function(Y_h, Y.view(-1))

        pred_tags = Y_h.max(dim=1)[1]
        confm.add(pred_tags, Y)

    print("\x1b[2K", end="")
    confm.print_stats(class_dict)

#######################################

def main():
    """
    " Data preparation
    """

    # First we read the word embeddings file
    # Go take a look how the file is structured!
    # This function returns a word-to-index dictionary and the embedding tensor
    word2i, _, embeddings = dataprep.load_embeddings(emb_file)
    print("emb loaded")

    tag2i, i2tag = dataprep.load_postags(tag_file)
    print("tags loaded")

    # Read in data and create tensors
    train_data = dataprep.prepare_data(train_file, word2i, tag2i, hparams["input_len"])
    dev_data = dataprep.prepare_data(dev_file, word2i, tag2i, hparams["input_len"])
    test_data = dataprep.prepare_data(test_file, word2i, tag2i, hparams["input_len"])
    print("data loaded")

    hparams["tagset_size"] = len(tag2i)

    model = hparams["tagger"](embedding=embeddings,
                              **hparams)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_model(model,
                train_data=train_data,
                dev_data=dev_data,
                class_dict=i2tag,
                **hparams)

    predict(model, data=dev_data,
                class_dict=i2tag, **hparams)

if __name__ == "__main__":
    main()