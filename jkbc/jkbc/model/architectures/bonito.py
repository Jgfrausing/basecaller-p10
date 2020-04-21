# +
import math
import toml

from fastai.basics import *
import torch

import jkbc.types as t


# -

def model(window_size, device, definition: t.Union[dict, t.PathLike]):
    if type(definition) == dict:
        config = definition
    else:
        config = toml.load(definition)
    
    model = Model(config).to(device=device).half()
    
    test_input = torch.ones(1, 1, window_size, dtype=torch.float16, device=device)
    test_output = model(test_input)
    out_dimension = test_output.shape[1]
    out_scale = math.ceil(window_size/out_dimension)

    return model, out_scale


# +
activations = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}

class Model(nn.Module):
    """
    Model template for QuartzNet style architectures

    https://arxiv.org/pdf/1910.10261.pdf
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.stride = config['block'][0]['stride'][0]
        self.alphabet = config['labels']['labels']
        self.features = config['block'][-1]['filters']
        self.encoder = Encoder(config)
        self.decoder = Decoder(self.features, len(self.alphabet))
        if 'output_size' not in config or not config['output_size']:
            self.compressor = None
            print('No compression')
        else:
            self.compressor = self.compressor(1366, config['output_size'], 15)
            print('Using compression')

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if self.compressor:
            return self.compressor(decoded)
        return decoded
    
    def compressor(self, input_size, output_size, kernel_size, stride=1, dilation=1):
        padding = _get_padding(kernel_size, stride, dilation, len(self.alphabet))
        self.layer = nn.Sequential(
             nn.Conv1d(input_size, output_size, kernel_size, stride=stride, padding=padding)
            ,nn.BatchNorm1d(output_size)
            ,nn.ReLU()
        )

class Encoder(nn.Module):
    """
    Builds the model encoder
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        features = self.config['input']['features']
        activation = activations[self.config['encoder']['activation']]()
        encoder_layers = []

        for layer in self.config['block']:
            encoder_layers.append(
                Block(
                    features, layer['filters'], activation,
                    repeat=layer['repeat'], kernel_size=layer['kernel'],
                    stride=layer['stride'], dilation=layer['dilation'],
                    dropout=layer['dropout'], residual=layer['residual'],
                    separable=layer['separable'],
                )
            )

            features = layer['filters']

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder([x])


class TCSConv1d(nn.Module):
    """
    Time-Channel Separable 1D Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, separable=False):

        super(TCSConv1d, self).__init__()
        self.separable = separable

        if separable:
            self.depthwise = nn.Conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias, groups=in_channels
            )

            self.pointwise = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride,
                dilation=dilation, bias=bias, padding=0
            )
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )

    def forward(self, x):
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x


class Block(nn.Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False):

        super(Block, self).__init__()

        self.use_res = residual
        self.conv = nn.ModuleList()

        _in_channels = in_channels
        padding = _get_padding(kernel_size[0], stride[0], dilation[0], _in_channels)

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable
                )
            )

            self.conv.extend(self.get_activation(activation, dropout))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable
            )
        )

        # add the residual connection
        if self.use_res:
            self.residual = nn.Sequential(*self.get_tcs(in_channels, out_channels))

        # add the activation and dropout
        self.activation = nn.Sequential(*self.get_activation(activation, dropout))

    def get_activation(self, activation, dropout):
        return activation, nn.Dropout(p=dropout)

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def forward(self, x):
        _x = x[0]
        for layer in self.conv:
            _x = layer(_x)
        if self.use_res:
            _x += self.residual(x[0])
        return [self.activation(_x)]


class Decoder(Module):
    """
    Decoder
    """
    def __init__(self, features, classes):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(nn.Conv1d(features, classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.layers(x[-1])
        return x.transpose(1, 2)


# -

def _get_padding(kernel_size, stride, dilation, features):
    # https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
    if stride == 1 and kernel_size%2 == 0:
        raise ValueError(f"stride ({stride}) and kernel_size ({kernel_size}) cannot be padded to contain input size")

    #Dilation simulates an increased kernel size (where we ignore the zeros)
    #This means that kernel_size 5 and dilation 1 is similiar to kernel size 3 and dilation 2 regardig how much to pad
    features = features-1

    dilated_kernel_size = kernel_size+(kernel_size-1)*(dilation-1)
    padding = (dilated_kernel_size+features*stride-features)//2

    return padding
