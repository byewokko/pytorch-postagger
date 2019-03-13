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

        # Define the trainable parameters.
        # All the parameter matrices have dimension 'out_size x in_size'.
        
        # Weights for ingate
        self.w_ii = nn.Parameter(torch.zeros(hidden_size, input_size))   # from input
        self.w_hi = nn.Parameter(torch.zeros(hidden_size, hidden_size))  # from hidden
        self.w_ci = nn.Parameter(torch.zeros(hidden_size, hidden_size))  # from old cell state

        # Weights for forgetgate
        self.w_if = nn.Parameter(torch.zeros(hidden_size, input_size))   # from input
        self.w_hf = nn.Parameter(torch.zeros(hidden_size, hidden_size))  # from hidden
        self.w_cf = nn.Parameter(torch.zeros(hidden_size, hidden_size))  # from old cell state

        # Weights for proposed cell state
        self.w_ic = nn.Parameter(torch.zeros(hidden_size, input_size))   # from input
        self.w_hc = nn.Parameter(torch.zeros(hidden_size, hidden_size))  # from hidden

        # Weights for outgate
        self.w_io = nn.Parameter(torch.zeros(hidden_size, input_size))   # from input
        self.w_ho = nn.Parameter(torch.zeros(hidden_size, hidden_size))  # from hidden
        self.w_cyo = nn.Parameter(torch.zeros(hidden_size, hidden_size)) # from new cell state

        # Bias vectors have dimension 'out_size'.
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

        # NOTE: The following line computes the ingate without the peephole connection.
        # You will have to add the peephole connection yourself.
        ingate = F.linear(input, self.w_ii, self.b_ii) + F.linear(hx, self.w_hi, self.b_hi)
        ingate = torch.sigmoid(ingate)

        raise NotImplementedError("Calculate forgetgate here. Follow the structure of the ingate.")

        # NOTE: The following line computes the proposed_cellstate without the peephole connection.
        # You will have to add the peephole connection yourself.
        proposed_cellstate = F.linear(input, self.w_ic, self.b_ic) + F.linear(hx, self.w_hc, self.b_hc)
        proposed_cellstate = torch.tanh(proposed_cellstate)

        cy = (forgetgate * cx) + (ingate * proposed_cellstate)

        raise NotImplementedError("Calculate outgate here. Follow the structure of the gates above.")

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

            raise NotImplementedError

        else:
            rnn_layer_number = self.rnn_layer_number

        if self.cell_type == "RNN_TANH":
            hidden = torch.randn(rnn_layer_number, batch_size, self.rnn_layer_size)
            self.hidden_layer = hidden

        elif self.cell_type == "LSTM":
            ##################
            # YOUR CODE HERE
            # LSTM layer uses one more layer tier than basic RNN layer, in which
            # it stores the memory state of the cells. The nn.LSTM class handles the
            # hidden and cell layer in one tuple.
            ##################

            raise NotImplementedError

    def forward(self, X, lengths):
        """
        This function specifies how the input gets transformed as it passes forward
        through the network. Here we plug in the layers defined in the __build_model
        function above.
        """

        # At this moment the input is a batch of word sequences, where each word is
        # represented by its index (an integer). So the X input is a 2-dimensional tensor
        # with shape (batch_size, sequence_length).

        # To get a better representation of the words, we pass the input through the embedding
        # layer, which replaces each word index by its corresponding embedding vector.

        raise NotImplementedError("Insert embedding layer here.")

        # X is now a 3-dimensional tensor with shape (batch_size, sequence_length,
        # embedding_size).

        # Now we pass X to the recurrent layer.

        if self.cell_type == "PEEP":
            # Only used in VG task
            X = self.peephole_cell(X)

        else:
            # This part looks more complex than it really is. Since we process our data in
            # minibatches and the sentences have varying lengths, they have been all right-padded
            # with zeros to evenly fit into a tensor. We don't want the RNN layer to waste its power
            # on processing zeros, so here we pack the padded sequences into one, skipping all
            # the padding zeros. The lengths vector gives us the information about what the useful
            # data is and what is the padding.
            # After passing through the RNN layer we unpack the sequences and re-pad them.

            X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)

            raise NotImplementedError("Insert RNN layer here")

            X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # The shape of the output of the RNN layer is (batch_size, max_sentence_len, rnn_layer_size).
        # To be able to pass the tensor on to the dense (linear) layer, we need to reshape it by
        # collapsing the first two dimensions. Use the .view() method to do that.
        # The new shape of the output should be (batch_size * max_sentence_len, rnn_layer_size).

        # However, first the tensor needs to me made contiguous. This does not affect its shape nor
        # contents, it only reorganizes the underlying data structure. It is a necessary step before
        # reshaping the tensor. Simply call the .contiguous() method on the tensor.

        raise NotImplementedError("Make the X tensor contiguous.")

        raise NotImplementedError("Reshape the X tensor using view().")

        # The last step is passing the tensor through the dense layer and the activation function.
        # The shape of the output will be (batch_size * sequence_length, tagset_size).

        raise NotImplementedError("Insert dense layer here.")

        raise NotImplementedError("Insert activation layer here.")

        # Finally, we flatten the output for easier loss analysis.
        X = X.view(-1, self.tagset_size)
        Y_h = X

        return Y_h

