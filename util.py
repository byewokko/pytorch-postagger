import re
import codecs
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_embeddings(filename):
    """
    Read text file with embeddings, return a {word: index} dict,
    an {index: word} dict and embeddings FloatTensor
    :param filename:
    :return (word2ind, ind2word, embeddings):
    """
    word2ind = {}
    ind2word = {}
    embeddings = []
    with open(filename, "r") as f:
        for line in f:
            word, *emb_str = line.strip().split()
            ind2word[len(word2ind)] = word
            word2ind[word] = len(word2ind)
            embeddings.append([float(s) for s in emb_str])

    return word2ind, ind2word, torch.FloatTensor(embeddings)

def vocab_from_traindata(traindata, emb_size, unk_rate=0.05):
    """
    Extract vocabulary from traindata, return a {word: index} dict,
    an {index: word} dict and create a random embeddings FloatTensor
    :param traindata:
    :param unk_rate: fraction of words that will be treated as unknown
    :return (word2ind, ind2word, embeddings):
    """
    word2ind = {}
    ind2word = {}
    embeddings = []

    ###

    return word2ind, ind2word, torch.FloatTensor(embeddings)



def load_postags(filename):
    """
    Read text file with POS tags, return a {tag: index} dict
    plus an inverse dict
    :param filename:
    :return (tag2ind, ind2tag):
    """
    tag2ind = {}
    ind2tag = {}
    with open(filename, "r") as f:
        for line in f:
            word, *emb_str = line.strip().split()
            ind2tag[len(tag2ind)] = word
            tag2ind[word] = len(tag2ind)

    return tag2ind, ind2tag

def normalize_line(line):
    """

    :param line:
    :return:
    """
    line = line.strip()
    line = re.sub("([0-9][0-9.,]*)", "0", line)  # Replace any number token by 0
    return line

def prepare_sent(line, word2i, tag2i, sent_maxlength):
    # for any input preparation
    return

def prepare_data(filename, word2i, tag2i, sent_maxlength):
    """
    Load data and convert into tensors
    :param filename:
    :param word2i:
    :param tag2i:
    :param sent_maxlength:
    :return (X_pad, Y_pad, Y):
    """

    data = []

    with open(filename, "r") as f:
        for line in f:
            x = []
            y = []
            line = normalize_line(line)
            for token in line.split(" "):
                word, tag = token.split("_")
                if word in word2i:
                    x.append(word2i[word])
                elif word.lower() in word2i:
                    x.append(word2i[word.lower()])
                else:
                    x.append(word2i["<unk>"])
                y.append(tag2i[tag])

            data.append((F.pad(torch.LongTensor(x),
                               pad = (0, (sent_maxlength - len(x))),
                               mode = "constant",
                               value = word2i["<P>"]),
                        F.pad(torch.LongTensor(y),
                               (0, (sent_maxlength - len(y))),
                               mode = "constant",
                               value = tag2i["<P>"]),
                        len(x)))

        #return torch.stack(X_pad), torch.stack(Y_pad), Y
        return data


def sort_batch(X, Y, L):
    L_sorted, idx_sorted = L.sort(0, descending=True)
    X_sorted = X[idx_sorted]
    Y_sorted = Y[idx_sorted]
    return X_sorted, Y_sorted, L_sorted

def pad_sort_batch(batch):

    return batch_sorted

def loadbar(percent, n_blocks=10):
    blocks = [b for b in "-▏▎▍▌▋▊▉█"]
    whole = percent * n_blocks
    part = (whole - int(whole)) * len(blocks)
    #whole = int(whole)
    return int(whole)*"█" + blocks[int(part)] + int(n_blocks - int(whole) - 1)*"-"