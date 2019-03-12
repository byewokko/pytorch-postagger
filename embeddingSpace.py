import sys
import torch

from datautil import load_embeddings

class EmbeddingSpace():
    """
    A wrapper for word embedding tensor and word-index dictionary.
    Allows for nearest words lookup.
    """

    def __init__(self, emb_file):
        sys.stderr.write(f"Loading embeddings from {emb_file} ... ")
        sys.stderr.flush()
        self.word2i, self.i2word, self.emb_space = load_embeddings(emb_file)
        sys.stderr.write("DONE!\n")
        self.vocab_size, self.emb_size = self.emb_space.size()
        sys.stderr.write("-"*len("Vocabulary size:{:>8d}\n".format(self.vocab_size)) + "\n")
        sys.stderr.write("Vocabulary size:{:>8d}\n".format(self.vocab_size))
        sys.stderr.write("Embedding size: {:>8d}\n".format(self.emb_size))
        sys.stderr.write("-"*len("Vocabulary size:{:>8d}\n".format(self.vocab_size)) + "\n")
        sys.stderr.flush()

    def __getitem__(self, key):
        """
        Returns embedding vector for a given word
        :param key: word
        :return:
        """
        if key in self.word2i:
            return self.emb_space[self.word2i[key]]
        else:
            raise KeyError(key)

    def get_empty(self):
        """
        Retruns an all-zeros embedding vector
        :return:
        """
        return torch.zeros(self.emb_size)

    def compute_distances(self, emb_vector):
        """
        Computes the cosine distance between a given vector
        and all the words in the embedding space
        :param emb_vector: nn.Tensor
        :return dists: nn.Tensor
        """
        sys.stderr.write(f"Computing distances ... ")
        sys.stderr.flush()

        # transform 1-dim tensor into 2-dim
        if emb_vector.dim() == 1:
            emb_vector = emb_vector[None, :]

        # compute cosine distance using matrix multiplication
        p_norm = emb_vector / emb_vector.norm(dim=1)[:, None]
        s_norm = self.emb_space / self.emb_space.norm(dim=1)[:, None]
        dists = torch.mm(p_norm, s_norm.transpose(0, 1))
        sys.stderr.write("DONE!\n")
        sys.stderr.flush()

        return dists

    def fetch_k_closest(self, emb_vector, k=10):
        """
        Fetches k closest words to a given embedding vector.
        Returns a list of (cos_distance, word) tuples.
        :param emb_vector: nn.Tensor
        :param k: int
        :return results: list
        """
        dists = self.compute_distances(emb_vector)
        dist, ind = torch.topk(dists, k, largest=True, sorted=True)
        return [(d.item(), self.i2word[i.item()]) for (d, i) in zip(dist[0], ind[0])]


def main():

    emb_file = sys.argv[1]

    # Initialize EmbeddingSpace. This takes time. You can import the EmbeddingSpace class into an
    # interactive environment to prevent reloading the embedding file every time you want to run
    # a computation.
    embedding = EmbeddingSpace(emb_file)

    dog = embedding["dog"]
    dog_similar = embedding.fetch_k_closest(dog)
    print(dog_similar)


if __name__ == "__main__":
    main()

