# deepspeech2
  
Released in 2015, Baidu Research's Deep Speech 2 model converts speech to text end to end from a normalized sound spectrogram to the sequence of characters. It consists of a few convolutional layers over both time and frequency, followed by gated recurrent unit (GRU) layers (modified with an additional batch normalization).
  
<img src="https://miro.medium.com/max/2116/1*D6mB5UY9p_0CwcaDXm9fig.png" height=400>  
  
This repository contains only model code, but you can train with deepspeech2 with [this repository](https://github.com/sooftware/kospeech).
  
## Installation
This project recommends Python 3.7 or higher.
We recommend creating a new virtual environment for this project (using virtual env or conda).
  
### Prerequisites
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.  
  
### Install from source
Currently we only support installation from source code using setuptools. Checkout the source code and run the
following commands:  
  
```
pip install -e .
```

## Usage

```python
import torch
import torch.nn as nn
from deepspeech2 import DeepSpeech2

batch_size, sequence_length, dim = 3, 12345, 80

cuda = torch.cuda.is_available()  
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(batch_size, sequence_length, dim).to(device)
input_lengths = torch.IntTensor([12345, 12300, 12000])

model = DeepSpeech2(num_classes=10, input_dim=dimension).to(device)

# Forward propagate
outputs = model(inputs, input_lengths)

# Recognize input speech
outputs = model.module.recognize(inputs, input_lengths)
```
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/conformer/issues) on github or   
contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
