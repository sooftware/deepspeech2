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

from deepspeech2 import DeepSpeech2

batch_size = 3
sequence_length = 14321
dimension = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

model = DeepSpeech2(num_classes=10, input_dim=dimension).to(device)

criterion = nn.CTCLoss(blank=3, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

for i in range(10):
    inputs = torch.rand(batch_size, sequence_length, dimension).to(device)
    input_lengths = torch.IntTensor([12345, 12300, 12000])
    targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                                [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                                [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
    target_lengths = torch.LongTensor([9, 8, 7])
    outputs, output_lengths = model(inputs, input_lengths)

    loss = criterion(outputs.transpose(0, 1), targets[:, 1:], output_lengths, target_lengths)
    loss.backward()
    optimizer.step()
    print(loss)
