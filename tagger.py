import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


class RNNTagger(nn.Module):
    def __init__(self, embedding_tensor, rnn_layer_size, rnn_layer_number,
                 tagset_size, batch_size, activation, dropout_rate, bidirectional, cell_type, **kwargs):
        super(RNNTagger, self).__init__()

        #self.input_len = input_len
        self.embedding = embedding_tensor
        self.emb_size = embedding_tensor.size(1)
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_number = rnn_layer_number
        #self.batch_size = batch_size
        self.tagset_size = tagset_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.bidirectional = bidirectional
        self.cell_type = cell_type

        self.__build_model()

    def __build_model(self):
        self.embedding_layer = nn.Embedding.from_pretrained(self.embedding)
        self.rnn_layer = nn.RNN(input_size=self.emb_size,
                                hidden_size=self.rnn_layer_size,
                                num_layers=self.rnn_layer_number,
                                bidirectional=self.bidirectional,
                                batch_first=True)
        #self.hidden_layer = self.init_hidden(self.batch_size)
        self.hidden_layer = None
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.dense_layer = nn.Linear(self.rnn_layer_size, self.tagset_size)
        self.activation_layer = self.activation(dim=1)
        #self.activation_layer = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        """
        Initialize the hidden layers as random tensors.
        The number of hidden layers depends on
        :return:
        """

        if self.bidirectional:
            ##################
            # YOUR CODE HERE #
            ##################
            #raise NotImplementedError
            rnn_layer_number = 2 * self.rnn_layer_number

        else:
            rnn_layer_number = self.rnn_layer_number

        if self.cell_type == "BasicRNN":
            hidden = torch.randn(rnn_layer_number, batch_size, self.rnn_layer_size)
            self.hidden_layer = hidden
        elif self.cell_type == "LSTM":
            ##################
            # YOUR CODE HERE #
            ##################
            #raise NotImplementedError
            hidden = torch.randn(rnn_layer_number, batch_size, self.rnn_layer_size)
            cell = torch.randn(rnn_layer_number, batch_size, self.rnn_layer_size)
            self.hidden_layer = hidden, cell

    def forward(self, X, lengths):

        X = self.embedding_layer(X)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        X, self.hidden_layer = self.rnn_layer(X, self.hidden_layer)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # here we reshape the output of the BiLSTM
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        #X = self.dropout_layer(X)

        X = self.dense_layer(X)
        #X = self.activation_layer(X)
        X = X.view(-1, self.tagset_size)
        Y_h = X
        return Y_h


class LSTMTagger(nn.Module):
    def __init__(self, input_len, embedding,
                 rnn_layer_size, rnn_layer_number, tagset_size,
                 batch_size, **kwargs):
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

        #X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        X = self.word_emb(X)
        #print(X.size())
        X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        #print(X.data.size())
        X, self.hidden = self.lstm(X, self.hidden)
        #print(X.data.size())
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        #print(X.size())
        # here we reshape the output of the BiLSTM
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        #print(X.size())
        #quit()


        X = self.dropout(X)


        X = self.dense(X)
        X = self.softmax(X)
        X = X.view(-1, self.tagset_size)
        Y_h = X
        return Y_h




