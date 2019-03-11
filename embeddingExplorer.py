import argparse
import re

from embeddingSpace import EmbeddingSpace

parser = argparse.ArgumentParser(description="A simple word embedding explorer")
parser.add_argument("emb_file", action="store", default="embeddings/glove.txt", help="Path to the embedding file")
parser.add_argument("-k", action="store", dest="top_k", type=int, default=10, help="Number of results to display")
args = parser.parse_args()

def parse_and_add(expr, embedding_space):
    # initialize output tensor with zeros
    result = embedding_space.get_empty()
    # set operator to addition
    operator = 1

    # split input on spaces
    for w in expr.strip().split(" "):
        # if the word is "+" or "-", we set the operator accordingly
        if w == "+":
            operator = 1
        elif w == "-":
            operator = -1
        elif w == "*":
            operator = "mult"
        elif w == "/":
            operator = "div"

        # otherwise we look up the embedding of the word
        # and add/subtract it to the result
        else:
            try:
                wemb = embedding_space[w]
            except KeyError:
                print("'{}' is not in corpus.".format(w))
                return None
            if operator == "mult":
                result *= wemb
            elif operator == "div":
                result /= wemb
            else:
                result += operator * wemb

    return result[None, :]


def main():

    embs = EmbeddingSpace(args.emb_file)

    expr = input("Type an expression:\n>>> ").strip()
    while expr:
        emb_point = parse_and_add(expr, embs)
        if emb_point is not None:
            results = embs.fetch_k_closest(emb_point, k=args.top_k)
            print("COSSIM\tWORD")
            for distance, word in results:
                print("{:.4f}\t{:s}".format(distance, word))

        print()
        expr = input("Type an expression:\n>>> ").strip()


if __name__ == "__main__":
    main()

