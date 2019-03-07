from typing import TextIO

import nltk

def readPOStags(tag_path):
    label2idx = {'<P>': 0}
    idx = 1
    with open(file, "r") as fin:
        for line in fin:
            line = line.strip()
            label2idx[line] = idx
            idx += 1
    return label2idx

def read_data(file_path):
    words, tags = [], []
    with open(file, "r") as fin:
        for line in fin:
            for word_pos in line.split(" "):
                for w, t in word_pos.split("_"):
                    words.append(w)
                    tags.append(t)
