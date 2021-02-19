
class RoICaptionHead(nn.Module):
    """
    ResNe(X)t RoI head + captioning
    """

    def __init__(
        self,
        config
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].
        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            rnn (function): which tpye of rnn to be used (options: nn.rnn, nn.gru, nn.lstm)
        """
        super(RoICaptionHead, self).__init__()
            dim_in = config.DIM_IN
        vocab_size = config.VOCAB_SIZE
        dropout_rate = config.DROPOUT
        rnn_dims = config.DROPOUT
        rnn_type = config.RNN_TYPE
        self.num_timesteps = config.TIME_STEPS

        self.dropout = nn.Dropout(dropout_rate)

        # Perform FC before rnn
        self.rnn_projection = nn.Linear(dim_in, rnn_dims, bias=True)

        # Word Embeddings
        self.embedding = nn.Embedding(vocab_size, rnn_dims)
        self.start_vector = nn.Parameter(torch.randn((rnn_dims,1),requires_grad=True))

        # rnn
        if rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=rnn_dims, hidden_size=rnn_dims)
        elif rnn_type.lower() == 'rnn'
            self.rnn = nn.RNN(input_size=rnn_dims, hidden_size=rnn_dims)
        else:
            raise NotImplementedError(rnn_type + " not supported")

        # Perform FC on rnn output to probability distributions
        self.rnn_output_fc = nn.Linear(rnn_dims, vocab_size, bias=True)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input_vectors, caption_target=None):
        """
            inputs:
                input_vectors: torch.tensor Shape: [total_boxes_across_batch, dim_in]
                caption_target: torch.LongTensor Shape: [total_boxes_across_batch, sequence_length] (sequence_length = 12)
            output:
                probability_vectors: torch.tensor Shape: [total_boxes_across_batch, sequence_length, vocab_size]
        """
        assert input_vectors is not None
        if caption_target is not None:
            assert self.training

        # Perform dropout.
        x = self.dropout(input_vectors.unsqueeze(0))

        rnn_input = self.start_vector.T.unsqueeze(0).repeat(1, x.shape[0], 1)
        rnn_hidden = self.rnn_projection(x).unsqueeze(0)

        if self.training:
            rnn_input = torch.cat(rnn_input, self.embedding(caption_target).permute(1,0,2))
            rnn_out, _ = self.rnn(rnn_input, rnn_hidden)
            probs = self.rnn_output_fc(rnn_out)
            probs_predicts = self.softmax(probs.permute(1,0,2))
        else:
            probs = []
            for i in range(self.num_timesteps):   
                rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)
                probs.append(self.rnn_output_fc(rnn_out))
                labels = probs[-1].max(2)
                rnn_input = self.embedding(labels)

            probs_predicts = self.softmax(torch.cat(probs, axis=0).permute(1,0,2))

        return probs_predicts

