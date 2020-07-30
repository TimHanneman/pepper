import torch
import torch.nn as nn


class TransducerGRU(nn.Module):
    def __init__(self, image_channels, image_features, LSTM_layers, hidden_size, num_classes, bidirectional=True):
        super(TransducerGRU, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = LSTM_layers
        self.num_classes = num_classes
        self.LSTM_encoder = nn.LSTM(image_features,
                                    hidden_size,
                                    num_layers=self.num_layers,
                                    bidirectional=bidirectional,
                                    batch_first=True)
        self.LSTM_decoder = nn.LSTM(2 * hidden_size,
                                    hidden_size,
                                    num_layers=self.num_layers,
                                    bidirectional=bidirectional,
                                    batch_first=True)
        # self.LSTM_encoder.flatten_parameters()
        # self.LSTM_decoder.flatten_parameters()
        self.dense1 = nn.Linear(self.hidden_size * 2, self.num_classes)
        # self.dense2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, hidden, cell_state):
        hidden = hidden.transpose(0, 1).contiguous()
        #cell_state = torch.zeros_like(hidden)
        cell_state = cell_state.transpose(0,1).contiguous()
        #
        self.LSTM_encoder.flatten_parameters()
        #pytorch docs, example
        x_out, (hidden_out, cell_state_out) = self.LSTM_encoder(x, (hidden, cell_state))
        self.LSTM_decoder.flatten_parameters()
        x_out, (hidden_final, cell_state_final) = self.LSTM_decoder(x_out, (hidden_out, cell_state_out))

        x_out = self.dense1(x_out)
        # x = self.dense2(x)
        # if self.bidirectional:
        #     output_rnn = output_rnn.contiguous()
        #     output_rnn = output_rnn.view(output_rnn.size(0), output_rnn.size(1), 2, -1) \
        #         .sum(2).view(output_rnn.size(0), output_rnn.size(1), -1)

        hidden_final = hidden_final.transpose(0, 1).contiguous()
        #
        cell_state_final = cell_state_final.transpose(0,1).contiguous()
        return x_out, hidden_final , cell_state_final

    def init_hidden(self, batch_size, num_layers, bidirectional=True):
        num_directions = 1
        if bidirectional:
            num_directions = 2

        return torch.zeros(batch_size, num_directions * num_layers, self.hidden_size)