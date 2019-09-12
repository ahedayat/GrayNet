import utils as util
import gc
from .graynet_parts import *
from .graynet_utils import graynet_weight_init as weight_init


class GrayNet(nn.Module):
    """
    UNet Implementation
    ...
    Attributes
    ----------
    num_classes : int
        num of classes
    input_channels : int
        num of input channels 
    contracting_convs : list of [ContractingUnit]s
        list of encoders
    expanding_convs : list of [ExpandingUnit]s
        list of decoders 
    final_conv : conv1x1
        final convolution layer

    Methods
    -------
    weight_init() : 
        initialize parameters
    forward(x) :
        forward path of model
    """

    def __init__(self, input_channel=3, mid_channel=128, output_channel=1):
        """
        Prameters
        ---------
        output_channel : int, optional
            num of output channel
        mid_channel : int, optional
            num of channels after encoding input
        input_channel : int, optional
            num of input channels
        depth : int, optional 
            num of encoding and decoding levels
        second_layer_channels : int, optional
            num of channels of second layer
        """
        super(GrayNet, self).__init__()

        self.input_channel = input_channel
        self.mid_channel = mid_channel
        self.output_channel = output_channel
        self.encoder = Encoder(
            self.input_channel, self.mid_channel, pooling=True)
        self.decoder = Decoder(
            self.mid_channel, self.output_channel, up_mode='nearest-neighbour')

    def weight_init(self):
        """Initialize wieght"""
        for ix, m in enumerate(self.modules()):
            weight_init(m)

    def forward(self, x):
        """Forward path"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
