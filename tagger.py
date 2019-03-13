import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class PeepholeCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PeepholeCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_ii = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.w_hi = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.w_ci = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.w_if = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.w_hf = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.w_cf = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.w_ic = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.w_hc = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.w_io = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.w_ho = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.w_cyo = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        self.b_ii = nn.Parameter(torch.zeros(hidden_size))
        self.b_hi = nn.Parameter(torch.zeros(hidden_size))
        self.b_ci = nn.Parameter(torch.zeros(hidden_size))
        self.b_cf = nn.Parameter(torch.zeros(hidden_size))
        self.b_if = nn.Parameter(torch.zeros(hidden_size))
        self.b_hf = nn.Parameter(torch.zeros(hidden_size))
        self.b_ic = nn.Parameter(torch.zeros(hidden_size))
        self.b_hc = nn.Parameter(torch.zeros(hidden_size))
        self.b_io = nn.Parameter(torch.zeros(hidden_size))
        self.b_ho = nn.Parameter(torch.zeros(hidden_size))
        self.b_cyo = nn.Parameter(torch.zeros(hidden_size))

    def peephole_cell(self, input, hx = None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)

        hx, cx = hx

        ingate = F.linear(input, self.w_ii, self.b_ii) + F.linear(hx, self.w_hi, self.b_hi) + F.linear(cx, self.w_ci, self.b_ci)
        forgetgate = F.linear(input, self.w_if, self.b_if) + F.linear(hx, self.w_hf, self.b_hf) + F.linear(cx, self.w_cf, self.b_cf)
        cellgate = F.linear(input, self.w_ic, self.b_ic) + F.linear(hx, self.w_hc, self.b_hc)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)

        cy = (forgetgate * cx) + (ingate * cellgate)

        outgate = F.linear(input, self.w_io, self.b_io) + F.linear(hx, self.w_ho, self.b_ho) + F.linear(cy, self.w_cyo, self.b_cyo)
        outgate = torch.sigmoid(outgate)

        hy = outgate * torch.tanh(cy)

        return hy, cy

    def forward(self, X, hx = None):
        hx = torch.randn(X.size(1), self.hidden_size)
        cx = torch.randn(X.size(1), self.hidden_size)
        out = []
        for i in X:
            hx, cx = self.peephole_cell(i, (hx, cx))
            out.append(hx)
        X = torch.stack(out)
        return X


class RNNTagger(nn.Module):
    def __init__(self, embedding_tensor, rnn_layer_size, rnn_layer_number,
                 tagset_size, activation, dropout_rate, bidirectional, cell_type, **kwargs):
        super(RNNTagger, self).__init__()

        self.embedding = embedding_tensor
        self.embedding_dim = embedding_tensor.size(1)
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_number = rnn_layer_number
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.tagset_size = tagset_size

        if bidirectional:
            self.rnn_out_size = 2 * rnn_layer_size
        else:
            self.rnn_out_size = rnn_layer_size

        self.__build_model()

    def __build_model(self):
        self.embedding_layer = nn.Embedding.from_pretrained(self.embedding)
        if self.cell_type == "PEEP":
            self.peephole_cell = PeepholeCell(self.embedding_dim, self.rnn_layer_size)
        else:
            self.rnn_layer = nn.RNNBase(mode=self.cell_type,
                                        input_size=self.embedding_dim,
                                        hidden_size=self.rnn_layer_size,
                                        num_layers=self.rnn_layer_number,
                                        bidirectional=self.bidirectional,
                                        batch_first=True)
        self.hidden_layer = None
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.dense_layer = nn.Linear(self.rnn_out_size, self.tagset_size)
        self.activation_layer = self.activation(dim=1)

    def init_hidden(self, batch_size):
        """
        Initialize the hidden layers as random tensors.
        The number of hidden layers depends on RNN (bi)directionality and cell type.
        """
        if self.bidirectional:
            ##################
            # YOUR CODE HERE
            # A bidirectional RNN has more hidden layers than a regular one-way RNN.
            # For each "left-to-right" layer there is one "right-to-left" layer.
            ##################
            #raise NotImplementedError
            rnn_layer_number = 2 * self.rnn_layer_number
        else:
            rnn_layer_number = self.rnn_layer_number

        if self.cell_type == "RNN_TANH":
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
        """
        This function specifies how the input gets transformed as it passes forward
        through the network. Here we plug in the layers defined in the __build_model
        function above.
        """

        # At this moment the input is a batch of word sequences, where each word is
        # represented by its index (an integer). So the X input is a 2-dimensional tensor
        # with shape (batch_size, max_sequence_length).

        # To get a better representation of the words, we pass the input through the embedding
        # layer, which replaces each word index by its corresponding embedding vector.

        X = self.embedding_layer(X)
        #raise NotImplementedError("Insert embedding layer here.")

        # X is now a 3-dimensional tensor with shape (batch_size, max_sequence_length,
        # embedding_size).

        # Now we pass X to the recurrent layer.

        if self.cell_type == "PEEP":
            # Only used in VG task
            # We need to loop through the input row-by-row instead of processing the whole
            # batch tensor. (Because otherwise we would need to weave our peephole cell into the PyTorch
            # RNN framework and that would be very difficult.) We continually collect the output
            # in a list and then stack it back into a tensor.

            X = self.peephole_cell(X)

        else:
            #
            X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
            #raise NotImplementedError("Insert RNN layer here")
            X, self.hidden_layer = self.rnn_layer(X, self.hidden_layer)
            X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # Now we reshape the output of the BiLSTM
        # First it needs to me made contiguous
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # Add dropout
        X = self.dropout_layer(X)

        #raise NotImplementedError("Insert dense layer here.")
        X = self.dense_layer(X)

        #raise NotImplementedError("Insert activation layer here.")
        X = self.activation_layer(X)
        X = X.view(-1, self.tagset_size)
        Y_h = X

        return Y_h

