import re
import torch
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
    line = re.sub("([0-9][0-9.,]*)", "0", line)
    return line


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

            x_padded = F.pad(torch.LongTensor(x),
                             pad=(0, (sent_maxlength - len(x))),
                             mode="constant",
                             value=word2i["<P>"])
            y_padded = F.pad(torch.LongTensor(y),
                             pad=(0, (sent_maxlength - len(y))),
                             mode="constant",
                             value=tag2i["<P>"])
            data.append((x_padded, y_padded, len(x)))

        return data


def sort_batch(X, Y, L):
    L_sorted, idx_sorted = L.sort(0, descending=True)
    X_sorted = X[idx_sorted]
    Y_sorted = Y[idx_sorted]
    return X_sorted, Y_sorted, L_sorted


def pad_sort_batch(batch):

    return batch_sorted