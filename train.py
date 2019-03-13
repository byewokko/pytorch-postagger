import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import util
import tagger
import datautil

from util import stderr_print

torch.manual_seed(123)

evaluate_on_test_data = True

# The hyperparameter dictionary contains parameters related to the network

hyperparams = {
    # learning rate
    "learning_rate": 0.01,

    # numbers of the training epochs
    "number_of_epochs": 8,

    # mini-batch size
    "batch_size": 256,

    # size of the hidden layer
    "rnn_layer_size": 128,

    # number of hidden layers
    "rnn_layer_number": 1,

    # bidirectionality of rnn layer
    "bidirectional": False,

    # RNN cell type
    "cell_type": "RNN_TANH",  # Basic RNN
    #"cell_type": "LSTM",
    #"cell_type": "PEEP",      # Peephole cell (VG task only)

    # dropout
    "dropout_rate": 0.2,

    # optimizer type
    #"optimizer": optim.Adam,
    "optimizer": optim.Adagrad,

    # loss function
    "loss_function": nn.NLLLoss,
    #"loss_function": nn.CrossEntropyLoss,

    # activation function
    "activation": nn.LogSoftmax,
}

# The dataparameter dictionary contains parameters related to the data.
# If you want to experiment with a language different than English,
# change the data_dir and the emb_file parameters accordingly.

dataparams = {
    # if you want your model to be saved,
    # specify output_dir
    #"output_dir": None,
    "output_dir": "out",

    # data directory
    "data_dir": "/nobackup/tmp/ml2019/datasets/english",

    "train_file": "train.txt",
    "dev_file": "dev.txt",
    "test_file": "test.txt",

    # embedding file
    "emb_file": "/nobackup/tmp/ml2019/glove/english.glove.tiny.txt",
    #"emb_file": "/nobackup/tmp/ml2019/glove/english.glove.6B.50d.txt",

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

TIMESTAMP = util.timestamp()


def train(model, train_data, dev_data, number_of_epochs, batch_size, tagset_size, loss_function, optimizer,
          learning_rate, output_dir=None, **kwargs):

    # Initialize the optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Initialize DataLoaders that will yield data in batches
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

    training_log = []

    timer = util.Timer()
    timer.start()

    print("epoch\ttr_loss\tvl_loss\ttr_acc\tvl_acc\ttr_f1\tvl_f1")

    for epoch in range(1, number_of_epochs + 1):

        train_confm = util.ConfusionMatrix(tagset_size, ignore_index=0)

        # Training minibatch loop
        for batch_n, (X, Y, L) in enumerate(train_loader):
            stderr_print("Epoch {:>3d}: Training   |{}| {}".format(epoch,
                                                            util.loadbar(batch_n/(n_train_batches-1)),
                                                            timer.remaining(total_iter)), end="\r")

            # Reset the gradient descent and the hidden layers
            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(L))

            # Pass the input X through the network.
            # We also need to pass the lengths vector L
            # since the sentences have different lengths
            Y_h = model(X, L)

            # Here we get the indices of the highest scoring predictions
            # and add them to a confusion matrix
            pred_tags = Y_h.max(dim=1)[1]
            train_confm.add(pred_tags, Y)

            # We compute the loss and update the weights with gradient descent
            train_loss = loss_function(Y_h, Y.view(-1))
            train_loss.backward()
            optimizer.step()

            timer.tick()

        dev_confm = util.ConfusionMatrix(tagset_size, ignore_index=0)

        # Validation minibatch loop
        for batch_n, (X, Y, L) in enumerate(dev_loader):
            stderr_print("Epoch {:>3d}: Validation |{}| {}".format(epoch,
                                                            util.loadbar(batch_n/(n_dev_batches-1)),
                                                            timer.remaining(total_iter)), end="\r")

            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(L))

            Y_h = model(X, L)
            pred_tags = Y_h.max(dim=1)[1]
            dev_confm.add(pred_tags, Y)

            with torch.no_grad():
                dev_loss = loss_function(Y_h, Y.view(-1))


            timer.tick()

        stderr_print("\x1b[2K", end="")

        results = {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
                   "train_acc": train_confm.accuracy(), "dev_acc": dev_confm.accuracy(),
                   "train_f1": train_confm.f_score(mean=True), "dev_f1": dev_confm.f_score(mean=True)}
        training_log.append(results)
        print("{epoch:d}\t{train_loss:.4f}\t{dev_loss:.4f}\t{train_acc:.4f}\t"
              "{dev_acc:.4f}\t{train_f1:.4f}\t{dev_f1:.4f}".format(**results))

    print("Training finished in {:02d}:{:02d}:{:02d}.".format(*timer.since_start()))

    # TODO: revert to best model


def predict(model, data, loss_function, batch_size, tagset_size, class_dict, output_dir=None, **kwargs):

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
    if output_dir is not None:
        confm.matrix_to_csv(class_dict, f"{output_dir}/{TIMESTAMP}-confmat.csv")

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
          **hyperparams,
          **dataparams)

    predict(model, data=dev_data,
            class_dict=i2tag,
            **hyperparams,
            **dataparams)

    if evaluate_on_test_data:
        predict(model, data=dev_data, class_dict=i2tag, **hyperparams)

if __name__ == "__main__":
    main()