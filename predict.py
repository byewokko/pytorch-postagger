import argparse
import nltk.tokenize
import sys
import torch

import datautil
import util
from util import stderr_print

parser = argparse.ArgumentParser(description="Predict POS tags.")
parser.add_argument("-m", "--model-file")
parser.add_argument("-f", "--input-file", default=None)

args = parser.parse_args()


def predict(model, x):
    Y = None
    with torch.no_grad():
        pass
        # predict

    return Y


def main():

    # parse args (file/stdin, )
    input_file = args.input_file
    model_file = args.model_file

    # load model
    stderr_print(f"Loading embeddings from {input_file} ... ", end="")
    loaded = torch.load(model_file)
    model = loaded["model"]
    emb_file = loaded["emb_file"]
    tag_file = loaded["tag_file"]
    padding_id = loaded["padding_id"]
    stderr_print("DONE")

    # First we read the word embeddings file
    # This function returns a word-to-index dictionary and the embedding tensor
    stderr_print(f"Loading embeddings from {emb_file} ... ", end="")
    word2i, _, embeddings = datautil.load_embeddings(emb_file)
    stderr_print("DONE")

    # Load and index POS tag list
    stderr_print(f"Loading tagset from {tag_file} ... ", end="")
    tag2i, i2tag = datautil.load_postags(tag_file)
    tagset_size = len(tag2i)
    stderr_print("DONE")

    # read input text
    if input_file is not None:
        fin = open(input_file, "r")
    else:
        fin = sys.stdin

    # move this to a new function, return a generator
    for line in fin:
        sent = nltk.tokenize.sent_tokenize(line.strip())

    # sent_tokenize if necessary

    # convert sentence to tensor

    # predict

    # convert output tensor to text

    return


if __name__ == "__main__":
    main()