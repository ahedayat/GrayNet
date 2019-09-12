import torch
import gc
import utils as util
import torch.nn as nn
import torch.nn.functional as F
from .graynet_utils import graynet_weight_init as weight_init


def conv1x1(in_channels, out_channels):
    """Two dimentional convoloution layer with kernel size of one
    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels

    Returns
    -------
    torch.nn.Conv2d:
        two dimentional convoloutional layer with kernel size of one
        and specified input and output channel
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


def conv3x3(in_channel, out_channels):
    """Two dimentional convoloution layer with kernel size of three
    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels

    Returns
    -------
    torch.nn.Conv2d:
        two dimentional convoloutional layer with kernel size of one
        and specified input and output channel
    """
    return nn.Conv2d(in_channel,
                     out_channels,
                     kernel_size=3,
                     padding=1,
                     stride=1)


def conv2x2(in_channel, out_channel):
    """Two dimentional convoloution layer with kernel size of three
    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels

    Returns
    -------
    torch.nn.Conv2d:
        two dimentional convoloutional layer with kernel size of one
        and specified input and output channel
    """
    return nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=2,
                     padding=1,
                     stride=1)


def upconv2x2(in_channels, out_channels, mode):
    """Two dimentional transposed convoloution layer with kernel size of two
    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    mode : str
        upsampling mode: "nearest-neighbour" or "transpose".

    Raises
    ------
    Exception
        if unkown mode is set for transposed convoloutional layer.

    Returns
    -------
    torch.nn.Conv2d:
        two dimentional transposed convoloutional layer with kernel size of two
    """
    upconv = None
    if mode == 'nearest-neighbour':
        upconv = nn.Sequential(
            nn.Upsample(mode='nearest', scale_factor=2),
            conv1x1(in_channels, out_channels))
    elif mode == 'transpose':
        upconv = nn.ConvTranspose2d(in_channels,
                                    out_channels,
                                    kernel_size=2,
                                    padding=0,
                                    stride=2)
    else:
        raise Exception('Unknown mode : {}'.format(mode))
    return upconv


class Encoder(nn.Module):
    """
    Encoder
    ...
    Attributes
    ----------
    in_channels : int
        num of input channels
    out_channel : int
        num of output channels
    block : torch.nn.Sequential
        contracting unit pytorch block
    pool : torch.nn.MaxPool2d or None
        pooling layer of end of contracting unit

    Methods
    -------
    weight_init() :
        initialize weights
    forward(x) :
        forward path of module
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        """
        Parameters
        ----------
        in_channels : int
            num of input channels
        out_channels : int
            num of output channels
        pooling : boolean, optional
        """
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            conv2x2(self.in_channels, self.out_channels),
            nn.ReLU()
        )
        self.pool = None
        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def weight_init(self):
        """Initializing module weights"""
        weight_init(self.block)

    def forward(self, x):
        """Forward path"""
        x = self.block(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class Decoder(nn.Module):
    """
    Decoder
    ...
    Attributes
    ----------
    in_channels : int
        num of input channels
    mid_channels : int

    out_channel : int
        num of output channels
    up_mode : str
        upsampling mode
    up_conv : upconv2x2
        upsampling layer
    block : ContractingUnit
        a contracting unit (ContractingUnit) without pooling layer

    Methods
    -------
    weight_init() :
        initialize weights
    forward(x) :
        forward path of module
    """

    def __init__(self, in_channels, out_channel, up_mode='nearest-neighbour'):
        """
        Parameters
        ----------
        in_channels : int
            num of input channels
        out_channel : int
            num of output channels
        up_mode : str, optional
            up-sampling mode
        """
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel

        self.up_mode = up_mode

        self.up_conv = upconv2x2(
            self.in_channels, self.out_channel, self.up_mode)
        self.block = conv3x3(self.out_channel, self.out_channel)

    def weight_init(self):
        """Initializing module weights"""
        weight_init(self.block)
        weight_init(self.up_conv)

    def forward(self, x):
        """Forward path"""
        x = self.up_conv(x)
        x = self.block(x)

        return x
