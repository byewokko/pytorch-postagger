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

# torch.manual_seed(123)

evaluate_on_test_data = False

# The hyperparameter dictionary contains parameters related to the network

hyperparams = {
    # learning rate
    "learning_rate": 0.01,
    # "learning_rate": 1,

    # numbers of the training epochs
    "number_of_epochs": 2,

    # mini-batch size
    "batch_size": 512,

    # size of the hidden layer
    "rnn_layer_size": 100,

    # number of hidden layers
    "rnn_layer_number": 1,

    # bidirectionality of rnn layer
    "bidirectional": False,

    # RNN cell type
    "cell_type": "RNN_TANH",  # Basic RNN
    #"cell_type": "LSTM",
    # "cell_type": "PEEP",      # Peephole cell (VG task only)

    # dropout
    "dropout_rate": 0.4,

    # optimizer type

    "optimizer": optim.Adam,
    # "optimizer": optim.SGD,
    # "optimizer": optim.Adadelta,
    # "optimizer": optim.RMSprop,
    # "optimizer": optim.Adagrad,

    # loss function
    # "loss_function": nn.NLLLoss,
    "loss_function": nn.CrossEntropyLoss,

    # activation function
    # "activation": nn.Softmax,
    # "activation": nn.LogSoftmax,
    "activation": nn.ReLU,
    # "activation": nn.Tanh,
    # "activation": nn.Sigmoid,
}

# The dataparameter dictionary contains parameters related to the data.
# If you want to experiment with a language different than English,
# change the data_dir and the emb_file parameters accordingly.

dataparams = {
    # directory for output files: trained models, logs and stats
    "output_dir": "out",
    "save_log": False,
    "save_conf_matrix": False,

    # data directory
    "data_dir": "/nobackup/tmp/ml2019/datasets/english",

    "train_file": "train.txt",
    "dev_file": "dev.txt",
    "test_file": "test.txt",

    # language
    "language": "english",

    # embedding file
    "emb_file": "/nobackup/tmp/ml2019/glove/english.glove.tiny.txt",
    # "emb_file": "/nobackup/tmp/ml2019/glove/english.glove.6B.50d.txt",

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


def train(model, train_loader, dev_loader, number_of_epochs, loss_function, optimizer,
          learning_rate, output_dir, conf_matrix, **kwargs):

    # Initialize the optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Set up variables for logging and timing
    n_train_batches = len(train_loader)
    n_dev_batches = len(dev_loader)
    total_iter = number_of_epochs * (n_train_batches + n_dev_batches)

    training_log = []
    best_model = (None, None)

    train_confm = conf_matrix.copy()
    dev_confm = conf_matrix.copy()

    timer = util.Timer()
    timer.start()

    print("Training started.")
    print()
    print("epoch\ttr_loss\tva_loss\ttr_acc\tva_acc\ttr_f1\tva_f1")

    for epoch in range(1, number_of_epochs + 1):

        # Switch model to training mode
        model = model.train()
        train_confm.reset()
        train_loss = 0

        # Training minibatch loop
        for batch_n, (X, Y, L) in enumerate(train_loader):
            stderr_print("Epoch {:>3d}: Training   |{}| {}".format(epoch,
                                                                   util.loadbar(batch_n / (n_train_batches - 1)),
                                                                   timer.remaining(total_iter)), end="\r")

            # Reset the gradient descent and the hidden layers
            model.zero_grad()
            model.hidden = model.init_hidden(batch_size=len(L))

            # Pass the input X through the network.
            # We also need to pass the lengths vector L
            # since the sentences have different lengths
            Y_h = model(X, L)

            pred_tags = Y_h.max(dim=1)[1]
            train_confm.add(pred_tags, Y)

            # We compute the loss and update the weights with gradient descent
            loss = loss_function(Y_h, Y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss

            timer.tick()

        train_loss /= len(train_loader.dataset)
        stderr_print("\x1b[2K", end="")

        # Switch model to evaluation mode
        model = model.eval()
        dev_confm.reset()

        # Validation minibatch loop
        # It has the same flow as the training loop above, with the exception
        # of using torch.no_grad() to prevent modifying the weights
        dev_loss = batch_predict(model, dev_loader, loss_function, dev_confm,
                                 loadtext="Evaluating dev",
                                 timer=timer,
                                 mean_loss=True)

        # Record the results
        results = {"epoch": epoch, "train_loss": train_loss.item(), "dev_loss": dev_loss.item(),
                   "train_acc": train_confm.accuracy().item(), "dev_acc": dev_confm.accuracy().item(),
                   "train_f1": train_confm.f_score(mean=True).item(), "dev_f1": dev_confm.f_score(mean=True).item()}
        training_log.append(results)
        print("{epoch:d}\t{train_loss:.5f}\t{dev_loss:.5f}\t{train_acc:.4f}\t"
              "{dev_acc:.4f}\t{train_f1:.4f}\t{dev_f1:.4f}".format(**results))

        # Save the current model if has the lowest validation loss
        if best_model[0] is None or best_model[0] > results["dev_loss"]:
            torch.save(model, f"{output_dir}/{TIMESTAMP}.check")
            best_model = (results["dev_loss"], epoch)

    print()
    print("Training finished in {:02d}:{:02d}:{:02d}.".format(*timer.since_start()))

    # Load the best model
    if best_model[1] != epoch:
        print(f"Loading model with the lowest validation loss (Epoch {best_model[1]}).")
        model = torch.load(f"{output_dir}/{TIMESTAMP}.check")

    # Clean up checkpoint file
    os.remove(f"{output_dir}/{TIMESTAMP}.check")

    return model, training_log


def batch_predict(model, data_loader, loss_function, conf_matrix,
                  loadtext="Evaluating", timer=None, mean_loss=False, **kwargs):
    n_batches = len(data_loader)
    loss = 0

    model = model.eval()
    with torch.no_grad():
        for batch_n, (X, Y, sent_len) in enumerate(data_loader):
            stderr_print("{} |{}|".format(loadtext, util.loadbar(batch_n / (n_batches - 1))), end="\r")
            sys.stdout.flush()

            model.init_hidden(batch_size=len(sent_len))

            Y_h = model(X, sent_len)
            loss += loss_function(Y_h, Y.view(-1))

            pred_tags = Y_h.max(dim=1)[1]
            conf_matrix.add(pred_tags, Y)

            if timer is not None:
                timer.tick()

    stderr_print("\x1b[2K", end="")

    if mean_loss:
        return loss/len(data_loader.dataset)
    return loss


#######################################

def main():
    # Prepare data

    # First we read the word embeddings file
    # This function returns a word-to-index dictionary and the embedding tensor
    stderr_print("Loading embeddings ... ", end="")
    word2i, _, embeddings = datautil.load_embeddings(dataparams["emb_file"])
    stderr_print("DONE")

    # Load and index POS tag list
    stderr_print("Loading tagset ... ", end="")
    tag2i, i2tag = datautil.load_postags(dataparams["tag_file"])
    hyperparams["tagset_size"] = len(tag2i)
    hyperparams["padding_id"] = 0
    stderr_print("DONE")

    # Read and index datasets, create tensors
    # Each dataset is a tuple: (input_tensor, targets_tensor, sentence_length_tensor)
    stderr_print("Loading datasets ... ", end="")
    train_data = datautil.prepare_data(dataparams["train_file"], word2i, tag2i, dataparams["input_len"])
    dev_data = datautil.prepare_data(dataparams["dev_file"], word2i, tag2i, dataparams["input_len"])
    test_data = datautil.prepare_data(dataparams["test_file"], word2i, tag2i, dataparams["input_len"])

    # Create dataloaders for batch processing
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=hyperparams["batch_size"],
                                               shuffle=True,
                                               num_workers=8,
                                               collate_fn=datautil.pad_sort_batch)
    dev_loader = torch.utils.data.DataLoader(dev_data,
                                             batch_size=hyperparams["batch_size"],
                                             shuffle=False,
                                             num_workers=8,
                                             collate_fn=datautil.pad_sort_batch)
    test_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=hyperparams["batch_size"],
                                             shuffle=False,
                                             num_workers=8,
                                             collate_fn=datautil.pad_sort_batch)
    stderr_print("DONE")

    # Set up the model
    hyperparams["loss_function"] = hyperparams["loss_function"](ignore_index=0)
    model = tagger.RNNTagger(embedding_tensor=embeddings, **hyperparams)
    print()
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Set up the confusion matrix to record our predictions
    conf_matrix = util.ConfusionMatrix(hyperparams["tagset_size"],
                                       ignore_index=0,
                                       class_dict=i2tag)

    # Train the model
    model, training_log = train(model,
                                train_loader=train_loader,
                                dev_loader=dev_loader,
                                conf_matrix=conf_matrix,
                                **hyperparams,
                                **dataparams)

    torch.save({"model": model,
                "emb_file": dataparams["emb_file"],
                "tag_file": dataparams["tag_file"],
                "padding_emb": embeddings[0],
                "unknown_emb": embeddings[1],
                "language": dataparams["language"]},
               f"{dataparams['output_dir']}/{TIMESTAMP}.model")

    if dataparams["save_log"]:
        util.dictlist_to_csv(training_log, f"{dataparams['output_dir']}/{TIMESTAMP}-log.csv")

    # Evaluate model on dev data
    print()
    print("Evaluating on dev data:")
    conf_matrix.reset()
    loss = batch_predict(model, data_loader=dev_loader, conf_matrix=conf_matrix, mean_loss=True, **hyperparams)

    conf_matrix.print_class_stats()
    print(f"Dev set accuracy: {conf_matrix.accuracy():.4f}")
    print(f"Dev set mean loss: {loss:8g}")
    if dataparams["save_conf_matrix"]:
        conf_matrix.matrix_to_csv(f"{dataparams['output_dir']}/{TIMESTAMP}-confmat-dev.csv")

    # Evaluate model on test data
    if evaluate_on_test_data:
        print()
        print("Evaluating on test data:")
        conf_matrix.reset()
        loss = batch_predict(model, data_loader=test_loader, conf_matrix=conf_matrix, mean_loss=True, **hyperparams)

        conf_matrix.print_class_stats()
        print(f"Test set accuracy: {conf_matrix.accuracy():.4f}")
        print(f"Test set mean loss: {loss:8g}")
        if dataparams["save_conf_matrix"]:
            conf_matrix.matrix_to_csv(f"{dataparams['output_dir']}/{TIMESTAMP}-confmat-test.csv")


if __name__ == "__main__":
    main()
