from nltk import tokenize
import re
import torch
import torch.nn.functional as F


def load_embeddings(filename, padding_token="<PAD>", unknown_token="<UNK>"):
    """
    Read text file with embeddings, return a {word: index} dict,
    an {index: word} dict and embeddings FloatTensor
    :param filename:
    :return (word2ind, ind2word, embeddings):
    """
    word2ind = {padding_token: 0, unknown_token: 1}
    ind2word = {0: padding_token, 1: unknown_token}
    embeddings = [None, None]

    with open(filename, "r") as f:
        for line in f:
            word, *emb_str = line.strip().split()
            vector = [float(s) for s in emb_str]
            if word == padding_token:
                embeddings[0] = torch.FloatTensor(vector)
            elif word == unknown_token:
                embeddings[1] = torch.FloatTensor(vector)
            else:
                ind2word[len(word2ind)] = word
                word2ind[word] = len(word2ind)
                embeddings.append(torch.FloatTensor(vector))

    if embeddings[0] is None:
        embeddings[0] = torch.zeros(len(embeddings[2]))
    if embeddings[1] is None:
        embeddings[1] = torch.randn(len(embeddings[2]))

    return word2ind, ind2word, torch.stack(embeddings)


def load_postags(filename, padding_token="<PAD>"):
    """
    Read text file with POS tags, return a {tag: index} dict
    plus an inverse dict
    :param filename:
    :return (tag2ind, ind2tag):
    """
    tag2ind = {padding_token: 0}
    ind2tag = {0: padding_token}
    with open(filename, "r") as f:
        for line in f:
            word, *emb_str = line.strip().split()
            if word == padding_token:
                continue
            ind2tag[len(tag2ind)] = word
            tag2ind[word] = len(tag2ind)

    return tag2ind, ind2tag


def normalize_line(line):
    """
    Replace all digit sequences with "0"
    :param line:
    :return:
    """
    line = line.strip()
    line = re.sub("([0-9][0-9.,]*)", "0", line)
    return line


def prepare_raw_text(file_object, word2i, pad_id=0, unk_id=1, language="english"):
    X_words = []
    X = []
    L = []
    ids = []
    sent_maxlength = 0

    # Tokenize and convert words to ids
    for line in file_object:
        sents = tokenize.sent_tokenize(line.strip(), language=language)
        for sent in sents:
            ids.append(len(ids))
            words = tokenize.word_tokenize(sent, language=language)
            L.append(len(words))
            if len(words) > sent_maxlength:
                sent_maxlength = len(words)
            X_words.append(words)
            x = []
            for word in words:
                word = re.sub("([0-9][0-9.,]*)", "0", word)
                if word in word2i:
                    x.append(word2i[word])
                elif word.lower() in word2i:
                    x.append(word2i[word.lower()])
                else:
                    x.append(unk_id)
            X.append(x)

    # Padding to the length of the longest sentence
    X_padded = []
    for x in X:
        x_padded = F.pad(torch.LongTensor(x),
                         pad=(0, (sent_maxlength - len(x))),
                         mode="constant",
                         value=pad_id)
        X_padded.append(x_padded)

    return torch.LongTensor(ids), torch.stack(X_padded), torch.LongTensor(L), X_words


def pair_words_with_tags(X_words, Y_h, L, i2tag):
    text_out = []
    for words, preds, length in zip(X_words, Y_h, L):
        sent_out = []
        for i in range(length):
            word = words[i]
            tag = i2tag[preds[i].item()]
            sent_out.append((word, tag))
        text_out.append(sent_out)
    return text_out


def print_words_with_tags(paired):
    for sent in paired:
        print(" ".join(["_".join(pair) for pair in sent]))


def prepare_data(filename, word2i, tag2i, sent_maxlength, padding=None):
    """
    Load data and convert into tensors
    :param filename:
    :param word2i:
    :param tag2i:
    :param sent_maxlength:
    :return (X_pad, Y_pad, Y):
    """
    pad_i = 0
    unk_i = 1
    data = []
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
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
                    x.append(unk_i)
                y.append(tag2i[tag])

            if padding is None:
                data.append((torch.LongTensor(x), torch.LongTensor(y), len(x)))
            else:
                x_padded = F.pad(torch.LongTensor(x),
                                 pad=(0, (sent_maxlength - len(x))),
                                 mode="constant",
                                 value=pad_i)
                y_padded = F.pad(torch.LongTensor(y),
                                 pad=(0, (sent_maxlength - len(y))),
                                 mode="constant",
                                 value=pad_i)
                data.append((x_padded, y_padded, len(x)))

        return data


def sort_batch(X, Y, L, descending=True):
    """
    Sort batch according to length L.
    :param X:
    :param Y:
    :param L:
    :return:
    """
    L_sorted, idx_sorted = L.sort(0, descending=descending)
    X_sorted = X[idx_sorted]
    Y_sorted = Y[idx_sorted]

    return X_sorted, Y_sorted, L_sorted


def pad_sort_batch(batch, padding_value=0):
    """
    Collate function that pads the current batch.
    :param batch:
    :param pad_value:
    :return:
    """
    # batch_item = (x, y, length)
    X = []
    Y = []
    L = []
    max_length = max(map(lambda i: i[2], batch))

    for x, y, l in batch:
        X.append(F.pad(x,
                       pad=(0, (max_length - l)),
                       mode="constant",
                       value=padding_value))
        Y.append(F.pad(y,
                       pad=(0, (max_length - l)),
                       mode="constant",
                       value=padding_value))
        L.append(l)

    X = torch.stack(X)
    Y = torch.stack(Y)
    L = torch.LongTensor(L)

    batch_sorted = sort_batch(X, Y, L)

    return batch_sorted
