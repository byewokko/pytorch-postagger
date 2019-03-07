import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

class RNNTagger(nn.Module):
    def __init__(self, input_len, embedding, rnn_layer_size, rnn_layer_number, tagset_size, batch_size, **kwargs):
        super(RNNTagger, self).__init__()

        self.input_len = input_len
        self.embedding = embedding
        self.emb_size = embedding.shape[1]
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_number = rnn_layer_number
        self.batch_size = batch_size
        self.tagset_size = tagset_size

        self.__build_model()

    def __build_model(self):
        self.word_emb = nn.Embedding.from_pretrained(self.embedding)
        self.lstm = nn.RNN(self.emb_size,
                            hidden_size=self.rnn_layer_size,
                            num_layers=self.rnn_layer_number,
                            batch_first=True)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(self.rnn_layer_size, self.tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size=None):
        """
        Initialize the hidden layers as random tensors
        :return:
        """
        if not batch_size:
            batch_size = self.batch_size

        hidden = torch.randn(self.rnn_layer_number, batch_size, self.rnn_layer_size)
        return hidden

    def forward(self, X, lengths):

        X = self.word_emb(X)
        # X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        X, self.hidden = self.lstm(X, self.hidden)
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # here we reshape the output of the BiLSTM
        X = X.contiguous()
        X = X.view(-1, X.shape[2])


        X = self.dropout(X)


        X = self.dense(X)
        X = self.softmax(X)
        X = X.view(-1, self.tagset_size)
        Y_h = X
        return Y_h



class LSTMTagger(nn.Module):
    def __init__(self, input_len, embedding, rnn_layer_size, rnn_layer_number, tagset_size, batch_size, **kwargs):
        super(LSTMTagger, self).__init__()

        self.input_len = input_len
        self.embedding = embedding
        self.emb_size = embedding.shape[1]
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_number = rnn_layer_number
        self.batch_size = batch_size
        self.tagset_size = tagset_size

        self.__build_model()

    def __build_model(self):
        self.word_emb = nn.Embedding.from_pretrained(self.embedding)
        self.lstm = nn.LSTM(self.emb_size,
                            hidden_size=self.rnn_layer_size,
                            num_layers=self.rnn_layer_number,
                            batch_first=True,
                            bidirectional=True)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(2 * self.rnn_layer_size, self.tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size=None):
        """
        Initialize the hidden layers as random tensors
        :return:
        """
        if not batch_size:
            batch_size = self.batch_size

        hidden = torch.randn(2*self.rnn_layer_number, batch_size, self.rnn_layer_size)
        cell = torch.randn(2*self.rnn_layer_number, batch_size, self.rnn_layer_size)
        return hidden, cell

    def forward(self, X, lengths):

        X = self.word_emb(X)
        # X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        X, self.hidden = self.lstm(X, self.hidden)
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # here we reshape the output of the BiLSTM
        X = X.contiguous()
        X = X.view(-1, X.shape[2])


        X = self.dropout(X)


        X = self.dense(X)
        X = self.softmax(X)
        X = X.view(-1, self.tagset_size)
        Y_h = X
        return Y_h




