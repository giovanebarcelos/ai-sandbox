# GO1242-ConvolutionPara
# 1×1 convolution para ajustar dimensões
if in_channels != out_channels or stride != 1:
    identity = Conv2D(out_channels, 1, stride)(x)
