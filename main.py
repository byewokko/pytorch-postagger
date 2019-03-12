import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import util
import tagger
import datautil

from util import stderr_print

torch.manual_seed(123)

evaluate_on_test_data = True

hyperparams = {
    # tagger type
    #"tagger": tagger.LSTMTagger,
    "tagger": tagger.RNNTagger,

    # learning rate
    "learning_rate": 0.01,

    # numbers of the training epochs
    "number_of_epochs": 5,

    # mini-batch size
    "batch_size": 256,

    # size of the hidden layer
    "rnn_layer_size": 128,

    # number of hidden layers
    "rnn_layer_number": 1,

    # bidirectionality of rnn layer
    "bidirectional": True,

    # RNN cell type
    "cell_type": "RNN_TANH",
    #"cell_type": "LSTM",
    #"cell_type": "PEEP",  # Peephole cell (VG task only)

    # dropout
    "dropout_rate": 0.5,

    # optimizer type
    "optimizer": optim.Adam,
    #"optimizer": optim.Adagrad,

    # loss function
    #"loss_function": nn.NLLLoss,
    "loss_function": nn.CrossEntropyLoss,
    #"loss_function": "NLL",
    #"loss_function": "CrossEntropy",

    # activation function
    "activation": nn.LogSoftmax,
}

dataparams = {
    # if you want your model to be saved,
    # specify output_dir
    "output_dir": None,

    # data directory
    "data_dir": "english",
    #"data_dir": "/local/course/ml/2019/assignment4/english",

    "train_file": "train.txt",
    "dev_file": "dev.txt",
    "test_file": "test.txt",

    # embedding file
    #"emb_file": "embeddings/glove.txt",
    #"emb_file": "/nobackup/tmp/glove/english.glove.tiny.txt",
    #"emb_file": "/nobackup/tmp/glove/multilingual_embeddings.en.txt",
    "emb_file": "/nobackup/tmp/glove/english.glove.6B.50d.txt",

    # padding token
    "padding_token": "<PAD>",

    # length of input sentences
    # (shorter sentences are padded to match this length)
    # (do not change it)
    "input_len": 50,

    "tag_file": "UDtags.txt"
}

dataparams["train_file"] = os.path.join(dataparams["data_dir"], dataparams["train_file"])
dataparams["dev_file"] = os.path.join(dataparams["data_dir"], dataparams["dev_file"])
dataparams["test_file"] = os.path.join(dataparams["data_dir"], dataparams["test_file"])


# TODO allow random embeddings
def train(model, train_data, dev_data, number_of_epochs, batch_size, tagset_size, loss_function, optimizer,
          learning_rate, **kwargs):

    # initialize the optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # initialize DataLoaders that will yield data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               collate_fn=datautil.pad_sort_batch)
    dev_loader = torch.utils.data.DataLoader(dev_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             collate_fn=datautil.pad_sort_batch)

    n_train_batches = len(train_loader)
    n_dev_batches = len(dev_loader)
    total_iter = number_of_epochs*(n_train_batches + n_dev_batches)

    timer = util.Timer()
    timer.start()

    print("epoch\ttr_loss\tvl_loss\ttr_acc\tvl_acc\ttr_f1\tvl_f1")

    for epoch in range(number_of_epochs):

        train_confm = util.ConfusionMatrix(tagset_size, ignore_index=0)

        for batch_n, (X, Y, L) in enumerate(train_loader):
            stderr_print("Epoch {:>3d}: Training   |{}| {}".format(epoch+1,
                                                            util.loadbar(batch_n/(n_train_batches-1)),
                                                            timer.remaining(total_iter)), end="\r")
            sys.stdout.flush()

            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(L))

            Y_h = model(X, L)
            pred_tags = Y_h.max(dim=1)[1]
            train_confm.add(pred_tags, Y)

            train_loss = loss_function(Y_h, Y.view(-1))
            train_loss.backward()
            optimizer.step()

            timer.tick()

        dev_confm = util.ConfusionMatrix(tagset_size, ignore_index=0)

        for batch_n, (X, Y, L) in enumerate(dev_loader):
            stderr_print("Epoch {:>3d}: Validation |{}| {}".format(epoch+1,
                                                            util.loadbar(batch_n/(n_dev_batches-1)),
                                                            timer.remaining(total_iter)), end="\r")
            sys.stdout.flush()

            max_sent = max(L)

            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(L))

            Y_h = model(X, L)
            pred_tags = Y_h.max(dim=1)[1]
            dev_confm.add(pred_tags, Y)

            with torch.no_grad():
                dev_loss = loss_function(Y_h, Y.view(-1))


            timer.tick()

        stderr_print("\x1b[2K", end="")
        print("{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
            epoch+1, train_loss, dev_loss, train_confm.accuracy(), dev_confm.accuracy(),
            train_confm.f_score().mean(), dev_confm.f_score().mean()))

        # TODO: append to training log

    # TODO: revert to best model


def predict(model, data, loss_function, batch_size, tagset_size, class_dict, **kwargs):

    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             collate_fn=datautil.pad_sort_batch)
    n_batches = len(dataloader)

    confm = util.ConfusionMatrix(tagset_size, ignore_index=0)

    for batch_n, (X, Y, sent_len) in enumerate(dataloader):
        stderr_print("Predicting |{}|".format(util.loadbar(batch_n / (n_batches - 1))), end="\r")
        sys.stdout.flush()

        max_sent = max(sent_len)

        model.init_hidden(batch_size=len(sent_len))

        Y_h = model(X, sent_len)
        with torch.no_grad():
            loss = loss_function(Y_h, Y.view(-1))

        pred_tags = Y_h.max(dim=1)[1]
        confm.add(pred_tags, Y)

    stderr_print("\x1b[2K", end="")
    confm.print_class_stats(class_dict)
    confm.matrix_to_csv(class_dict, "confmat.csv")

#######################################

def main():

    """
    Data preparation
    """

    # First we read the word embeddings file
    # This function returns a word-to-index dictionary and the embedding tensor
    stderr_print("Loading embeddings ... ", end="")
    word2i, _, embeddings = datautil.load_embeddings(dataparams["emb_file"])
    stderr_print("DONE")

    # Load and index POS tag list
    stderr_print("Loading tagset ... ", end="")
    tag2i, i2tag = datautil.load_postags(dataparams["tag_file"])
    hyperparams["tagset_size"] = len(tag2i)
    hyperparams["padding_id"] = tag2i[dataparams["padding_token"]]
    stderr_print("DONE")

    # Read and index data, create tensors
    # Each dataset is a tuple: (input_tensor, targets_tensor, sentence_length_tensor)
    stderr_print("Loading datasets ... ", end="")
    train_data = datautil.prepare_data(dataparams["train_file"], word2i, tag2i, dataparams["input_len"])
    dev_data = datautil.prepare_data(dataparams["dev_file"], word2i, tag2i, dataparams["input_len"])
    test_data = datautil.prepare_data(dataparams["test_file"], word2i, tag2i, dataparams["input_len"])
    stderr_print("DONE")

    """
    Model preparation
    """
    hyperparams["loss_function"] = hyperparams["loss_function"](ignore_index=tag2i[dataparams["padding_token"]])

    model = tagger.RNNTagger(embedding_tensor=embeddings, **hyperparams)

    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train(model,
          train_data=train_data,
          dev_data=dev_data,
          class_dict=i2tag,
          **hyperparams)

    predict(model, data=dev_data,
            class_dict=i2tag,
            **hyperparams)

    if evaluate_on_test_data:
        predict(model, data=dev_data, class_dict=i2tag, **hyperparams)

if __name__ == "__main__":
    main()