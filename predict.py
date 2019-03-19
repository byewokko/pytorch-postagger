import argparse
import nltk.tokenize
import sys
import torch

from warnings import warn

import datautil
import util
from util import stderr_print

parser = argparse.ArgumentParser(description="Predict POS tags.")
parser.add_argument("-m", "--model-file")
parser.add_argument("-f", "--input-file", default=None)
parser.add_argument("-l", "--language", default=None)

args = parser.parse_args()


def predict(model, X, L):

    model = model.eval()

    stderr_print("Predicting ... ", end="")

    model.init_hidden(batch_size=L.size(0))

    Y_h = model(X, L)

    predictions = Y_h.max(dim=1)[1]

    stderr_print("DONE")

    return predictions


def main():
    """
    Load a trained model from file (-m) and use it to PoS-tag input file.
    If no file is specified, read from stdin.
    :return:
    """

    # parse args (file/stdin, )
    input_file = args.input_file
    model_file = args.model_file
    language = args.language

    # load model
    stderr_print(f"Loading model from {model_file} ... ")
    loaded = torch.load(model_file)

    try:
        model = loaded["model"]
    except KeyError:
        raise Exception("Failed to load model.")

    try:
        emb_file = loaded["emb_file"]
        stderr_print(f"Embedding file: {emb_file}")
    except KeyError:
        raise Exception("No embedding file specified.")

    try:
        tag_file = loaded["tag_file"]
        stderr_print(f"Tag file: {tag_file}")
    except KeyError:
        raise Exception("No tag file specified.")

    try:
        padding_emb = loaded["padding_emb"]
        stderr_print(f"Padding embedding loaded.")
    except KeyError:
        padding_emb = None
        warn(f"No padding embedding specified, defaulting to embedding[0].")

    try:
        unknown_emb = loaded["unknown_emb"]
        stderr_print(f"'Unknown' embedding loaded.")
    except KeyError:
        unknown_emb = None
        warn(f"No 'unknown' embedding specified, defaulting to embedding[1].")

    if language is None:
        try:
            language = loaded["language"]
            stderr_print(f"Language: {language}")
        except KeyError:
            language = "english"
            warn(f"No language specified, defaulting to english.")

    stderr_print("DONE")

    # First we read the word embeddings file
    # This function returns a word-to-index dictionary and the embedding tensor
    stderr_print(f"Loading embeddings from {emb_file} ... ", end="")
    word2i, _, embeddings = datautil.load_embeddings(emb_file)
    if padding_emb is not None:
        embeddings[0] = padding_emb
    if unknown_emb is not None:
        embeddings[1] = unknown_emb
    stderr_print("DONE")

    # Load and index POS tag list
    stderr_print(f"Loading tagset from {tag_file} ... ", end="")
    tag2i, i2tag = datautil.load_postags(tag_file)
    tagset_size = len(tag2i)
    stderr_print("DONE")

    # Read input text from file
    # Read from stdin if no file (-f) is specified
    if input_file is not None:
        stderr_print(f"Reading from {input_file}...")
        fin = open(input_file, "r")
    else:
        stderr_print("Reading from standard input...")
        fin = sys.stdin

    sent_ids, X, L, X_words = datautil.prepare_raw_text(fin, word2i, pad_id=0, unk_id=1, language=language)
    sent_ids, X, L = datautil.sort_batch(sent_ids, X, L)

    if input_file is not None:
        fin.close()

    # Predict
    Y_h = predict(model, X, L)

    # Reshape flattened output tensor, match tag labels
    # and pair them with input words
    Y_h = Y_h.view(len(X_words), -1)
    Y_h, L, sent_ids = datautil.sort_batch(Y_h, L, sent_ids, descending=False)
    paired = datautil.pair_words_with_tags(X_words, Y_h, L, i2tag)

    # Print to output in the word_TAG format
    datautil.print_words_with_tags(paired)


if __name__ == "__main__":
    main()