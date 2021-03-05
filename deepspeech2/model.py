# MIT License
#
# Copyright (c) 2021 Soohwan Kim.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from deepspeech2.convolution import DeepSpeech2Extractor
from deepspeech2.modules import Linear, LayerNorm
from deepspeech2.rnn import BNReluRNN


class DeepSpeech2(nn.Module):
    """
    Deep Speech2 model with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1512.02595

    Args:
        input_dim (int): dimension of input vector
        num_classes (int): number of classfication
        rnn_type (str, optional): type of RNN cell (default: gru)
        num_rnn_layers (int, optional): number of recurrent layers (default: 5)
        rnn_hidden_dim (int): the number of features in the hidden state `h`
        dropout_p (float, optional): dropout probability (default: 0.1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        activation (str): type of activation function (default: hardtanh)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: output
        - **output**: tensor containing the encoded features of the input sequence
    """
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            rnn_type='gru',
            num_rnn_layers: int = 5,
            rnn_hidden_dim: int = 512,
            dropout_p: float = 0.1,
            bidirectional: bool = True,
            activation: str = 'hardtanh',
            device: torch.device = 'cuda',
    ):
        super(DeepSpeech2, self).__init__()
        self.device = device
        self.conv = DeepSpeech2Extractor(input_dim, activation=activation)
        self.rnn_layers = nn.ModuleList()
        rnn_output_size = rnn_hidden_dim << 1 if bidirectional else rnn_hidden_dim

        for idx in range(num_rnn_layers):
            self.rnn_layers.append(
                BNReluRNN(
                    input_size=self.conv.get_output_dim() if idx == 0 else rnn_output_size,
                    hidden_state_dim=rnn_hidden_dim,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    dropout_p=dropout_p,
                )
            )

        self.fc = nn.Sequential(
            LayerNorm(rnn_output_size),
            Linear(rnn_output_size, num_classes, bias=False),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  ctc training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor):
            * predicted_log_prob (torch.FloatTensor)s: Log probability of model predictions.
            * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``
        """
        outputs, output_lengths = self.conv(inputs, input_lengths)
        outputs = outputs.permute(1, 0, 2).contiguous()

        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)

        outputs = self.fc(outputs.transpose(0, 1)).log_softmax(dim=-1)

        return outputs, output_lengths
