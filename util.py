import re
import time
import torch
import torch.nn.functional as F


class Timer():
    def __init__(self, n_last=0):
        self.times = None
        self.last = None
        self.n_last = n_last

    def start(self):
        self.last = time.time()
        self.times = []

    def tick(self):
        now = time.time()
        self.times.append(now - self.last)
        self.last = now

    def get_average(self):
        return sum(self.times[-self.n_last:]) / len(self.times[-self.n_last:])

    def remaining(self, total_n):
        if len(self.times) == 0:
            return ""
        n = total_n - len(self.times)
        t = self.get_average() * n
        h, t = int(t//3600), t%3600
        m, s = int(t//60), int(t%60)
        return f"{h:d}:{m:02d}:{s:02d} remaining"


class ConfusionMatrix():
    """
    Dimension 0 is predictions, dimension 1 is targets
    """
    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        self.matrix = torch.zeros((n_classes, n_classes), dtype=torch.float)
        self.ignore_index = ignore_index
        self.filter = torch.LongTensor([i for i in range(n_classes) if i is not ignore_index])

    def __repr__(self):
        return repr(self.matrix.int())

    def __str__(self):
        return str(self.matrix.int())

    def add(self, predictions, targets):
        if predictions.numel() != targets.numel():
            raise Exception("The dimensions of matrices do not match.")
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        for i in range(predictions.numel()):
            self.matrix[predictions[i], targets[i]] += 1

    def accuracy(self):
        matrix = self.matrix.index_select(0,self.filter).index_select(1,self.filter)
        return matrix.diag().sum() / matrix.sum()

    def precision(self):
        matrix = self.matrix.index_select(0, self.filter).index_select(1, self.filter)
        return matrix.diag() / matrix.sum(dim=0)

    def recall(self):
        matrix = self.matrix.index_select(0, self.filter).index_select(1, self.filter)
        return matrix.diag() / matrix.sum(dim=1)

    def f_score(self, b2=1):
        return (1 + b2) * self.precision() * self.recall() / (b2 * self.precision() + self.recall())

    def print_stats(self, class_dict, fscore_b2=1):
        precision = self.precision()
        recall = self.recall()
        f_score = self.f_score(b2=fscore_b2)
        headline = f"Class\tPrec.\tRecall\tF-score (b2={fscore_b2})"
        print(headline)
        print(len(headline)*"-")
        for i, j in enumerate(item.item() for item in self.filter):
            print("{:s}\t{:.4f}\t{:.4f}\t{:.4f}".format(class_dict[j], precision[i], recall[i], f_score[i]))
        print(len(headline)*"-")
        print("Mean\t{:.4f}\t{:.4f}\t{:.4f}".format(precision.mean(), recall.mean(), f_score.mean()))

    def matrix_to_csv(self, class_dict, filename):
        with open(filename, "w") as csv:
            print(",".join([""] + [class_dict[i] for i in range(self.n_classes)]), file=csv)
            for i in range(self.n_classes):
                print(",".join([class_dict[i]] + [str(int(self.matrix[i, j])) for j in range(self.n_classes)]),
                      file=csv)



def loadbar(percent, n_blocks=15):
    percent = min(percent, 0.999999999)
    blocks = [b for b in "▏▎▍▌▋▊▉█"]
    whole = percent * n_blocks
    part = (whole - int(whole)) * len(blocks)
    #whole = int(whole)
    return int(whole)*"█" + blocks[int(part)] + int(n_blocks - int(whole) - 1)*"-"
