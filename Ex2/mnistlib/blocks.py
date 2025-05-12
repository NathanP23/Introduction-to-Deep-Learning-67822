# mnistlib/blocks.py
from torch import nn

def conv_bn_relu(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

def deconv_bn_relu(in_c, out_c, k=3, s=2, p=1, op=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, k, s, p, op, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )
