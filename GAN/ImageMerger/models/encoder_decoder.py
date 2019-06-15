import torch
import torch.nn as nn
import warnings


def custom_pad(kernel, pad):
    if pad == 'zero':
        return nn.ZeroPad2d(kernel)
    if pad == 'reflect':
        return nn.ReflectionPad2d(kernel)
    warnings.warn('Bad pad method')


class EncoderBlock(nn.Module):
    """
    This is class is a block in the pipe of the encoder.
    It has conv2d -> Normalization (optional) -> activation (optional) -> dropout (optional)
    """
    def __init__(self, filters_in, filters_out, kernel=4, stride=2, padding=1, activation=None, normalization='instance', dropout=0.):
        super(EncoderBlock, self).__init__()
        self.full = nn.Sequential(nn.Conv2d(filters_in, filters_out, kernel, stride, padding))
        if normalization is not None:
            if normalization == 'instance':
                self.full.add_module('norm', nn.InstanceNorm2d(filters_out))
            elif normalization == 'batch':
                self.full.add_module('norm', nn.BatchNorm2d(filters_out))
            else:
                warnings.warn('Unsupported normalization in EncoderBlock')

        if activation is not None:
            if activation == 'relu':
                self.full.add_module('relu', nn.ReLU(inplace=True))
            elif activation == 'lrelu':
                self.full.add_module('lrelu', nn.LeakyReLU(0.2, inplace=True))
            else:
                warnings.warn('Unsupported activation in EncoderBlock')
        if dropout > 0:
            self.full.add_module('dropout', nn.Dropout2d(p=dropout))

    def forward(self, x):
        return self.full(x)


class DecoderBlock(nn.Module):
    """
    This is class is a block in the pipe of the decoder.
    It has dropout (optional) -> conv2d -> Normalization (optional) -> activation (optional)
    """
    def __init__(self, filters_in, filters_out, kernel=4, stride=2, padding=1, activation=None, normalization='instance', dropout=0.):
        super(DecoderBlock, self).__init__()
        self.full = nn.Sequential()
        if dropout > 0:
            self.full.add_module('dropout', nn.Dropout2d(p=dropout))
        self.full.add_module('conv2d', nn.ConvTranspose2d(filters_in, filters_out, kernel, stride, padding))
        if normalization is not None:
            if normalization == 'instance':
                self.full.add_module('norm', nn.InstanceNorm2d(filters_out))
            elif normalization == 'batch':
                self.full.add_module('norm', nn.BatchNorm2d(filters_out))
            else:
                warnings.warn('Unsupported normalization in EncoderBlock')
        if activation is not None:
            if activation == 'relu':
                self.full.add_module('relu', nn.ReLU(inplace=True))
            elif activation == 'lrelu':
                self.full.add_module('lrelu', nn.LeakyReLU(0.2, inplace=True))
            else:
                warnings.warn('Unsupported activation in DecoderBlock')

    def forward(self, x):
        return self.full(x)


class E1(nn.Module):
    """
    This is the encoder of Image A path.
    """
    def __init__(self, input_nc, last_conv_nc, sep, input_size, depth, activation='lrelu', dropout=0., normalization='instance'):
        super(E1, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.sep = sep
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)
        self.depth = depth
        self.bottom_features = (self.last_conv_nc - self.sep) * self.feature_size * self.feature_size

        self.blocks = nn.ModuleDict()
        self.blocks[str(0)] = EncoderBlock(self.input_nc, 32, activation=activation, dropout=dropout, normalization=normalization)
        for d in range(1, depth-1):
            self.blocks[str(d)] = EncoderBlock(32*(2 ** min(4,d-1)), 32*(2 ** min(4,d)), activation=activation, dropout=dropout, normalization=normalization)
        self.blocks[str(depth-1)] = EncoderBlock(32 * (2 ** min(4,depth - 2)), last_conv_nc - sep, activation=activation, dropout=dropout, normalization=normalization)

    def forward(self, x, extract=None):
        x = [x]
        for d in range(self.depth):
            x.append(self.blocks[str(d)](x[-1]))

        if extract is None:
            return x[self.depth]
        return [x[d] if d in extract else None for d in range(self.depth + 1)]


class E2(nn.Module):
    """
    This is the encoder of Image B path.
    """
    def __init__(self, input_nc, sep, input_size, depth, activation='lrelu', dropout=0., normalization='instance'):
        super(E2, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.sep = sep
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)
        self.depth = depth
        self.bottom_features = self.sep * self.feature_size * self.feature_size

        self.blocks = nn.ModuleDict()
        self.blocks[str(0)] = EncoderBlock(self.input_nc, 32, activation=activation, dropout=dropout, normalization=normalization)
        for d in range(1, depth-1):
            self.blocks[str(d)] = EncoderBlock(32 * (2 ** min(3,d - 1)), 32 * (2 ** min(3,d)), activation=activation, dropout=dropout, normalization=normalization)
        self.blocks[str(depth-1)] = EncoderBlock(32 * (2 ** min(3,depth - 2)), sep, activation=activation, dropout=dropout, normalization=normalization)

    def forward(self, x, extract=None):
        x = [x]
        for d in range(self.depth):
            x.append(self.blocks[str(d)](x[-1]))

        if extract is None:
            return x[self.depth]
        return [x[d] if d in extract else None for d in range(self.depth+1)]


class EHeavy(nn.Module):
    """
    This is the encoder of Image A and B path for heavy mode.
    """
    def __init__(self, input_nc, last_conv_nc, input_size, depth, activation='lrelu', dropout=0., pad='reflect', normalization='instance'):
        super(EHeavy, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)
        self.depth = depth
        self.bottom_features = self.last_conv_nc * self.feature_size * self.feature_size

        self.blocks = nn.ModuleDict()
        self.blocks[str(0)] = nn.Sequential(
            custom_pad(1, pad),
            EncoderBlock(self.input_nc, 32, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout, normalization=normalization),
            custom_pad(1, pad),
            EncoderBlock(32, 32, kernel=4, stride=2, padding=0, activation=activation, dropout=dropout, normalization=normalization)
        )
        for d in range(1, depth-1):
            filters_in = min(last_conv_nc, 32 * (2 ** (d-1)))
            filters_out = min(last_conv_nc, 32 * (2 ** d))
            self.blocks[str(d)] = nn.Sequential(
                custom_pad(1, pad),
                EncoderBlock(filters_in, filters_out, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout, normalization=normalization),
                custom_pad(1, pad),
                EncoderBlock(filters_out, filters_out, kernel=4, stride=2, padding=0, activation=activation, dropout=dropout, normalization=normalization)
            )
        filters_in = min(last_conv_nc, 32 * (2 ** (depth - 2)))
        self.blocks[str(depth-1)] = nn.Sequential(
            custom_pad(1, pad),
            EncoderBlock(filters_in, last_conv_nc, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout, normalization=normalization),
            custom_pad(1, pad),
            EncoderBlock(last_conv_nc, last_conv_nc, kernel=4, stride=2, padding=0, activation=activation, dropout=dropout, normalization=normalization)
        )

    def forward(self, x, extract=None):
        x = [x]
        for d in range(self.depth):
            x.append(self.blocks[str(d)](x[-1]))

        if extract is None:
            return x[self.depth]
        return [x[d] if d in extract else None for d in range(self.depth + 1)]


class Decoder(nn.Module):
    """
    This is the decoder of Image C path.
    """
    def __init__(self, output_nc, last_conv_nc, input_size, depth, activation='relu', dropout=0., extract=None, normalization='instance'):
        super(Decoder, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.output_nc = output_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)
        self.depth = depth
        self.bottom_features = self.last_conv_nc * self.feature_size * self.feature_size
        self.extract = extract

        self.blocks = nn.ModuleDict()
        self.blocks[str(depth-1)] = DecoderBlock(last_conv_nc, 32 * (2 ** min(4,depth-2)), activation=activation, dropout=dropout, normalization=normalization)
        for d in reversed(range(1, depth-1)):
            self.blocks[str(d)] = DecoderBlock(32 * (2 ** min(4,d)), 32 * (2 ** min(4,d-1)), activation=activation, dropout=dropout, normalization=normalization)
        self.blocks[str(0)] = DecoderBlock(32, output_nc, activation=None, dropout=dropout, normalization=normalization)

        self.skip = nn.ModuleDict()
        if True or extract is not None and 0 in extract:  # added True since saved model has all the skips
            self.skip[str(0)] = nn.Conv2d(2 * output_nc, output_nc, kernel_size=1)
        for d in reversed(range(1, depth)):
            if True or extract is not None and d in extract:  # added True since saved model has all the skips
                self.skip[str(d)] = nn.Conv2d(2 * 32 * (2 ** min(4,d-1)), 32 * (2 ** min(4,d-1)), kernel_size=1)

        self.tanh = nn.Tanh()

    def forward(self, x, use_activation=True):
        if type(x) is not list:
            x = [None for _ in range(self.depth)] + [x]
        y = x[self.depth].view(-1, self.last_conv_nc, self.feature_size, self.feature_size)

        for d in reversed(range(0, self.depth)):
            y = self.blocks[str(d)](y)
            if x[d] is not None:
                y = self.skip[str(d)](torch.cat((y, x[d]), dim=1))

        if use_activation:
            return self.tanh(y)
        else:
            return y


class DecoderHeavy(nn.Module):
    """
    This is the decoder of Image C path for heavy mode.
    """
    def __init__(self, output_nc, last_conv_nc, input_size, depth, activation='relu', dropout=0., extract=None, pad='reflect', normalization='instance'):
        super(DecoderHeavy, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.output_nc = output_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)
        self.depth = depth

        self.blocks = nn.ModuleDict()
        self.blocks[str(depth-1)] = nn.Sequential(
            custom_pad(1, pad),
            EncoderBlock(last_conv_nc, last_conv_nc, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout, normalization=normalization),
            DecoderBlock(last_conv_nc, min(last_conv_nc, 32 * (2 ** (depth-2))), kernel=4, stride=2, padding=1, activation=activation, dropout=dropout, normalization=normalization)
        )
        for d in reversed(range(1, depth-1)):
            filters_in = min(last_conv_nc, 32 * (2 ** d))
            filters_out = min(last_conv_nc, 32 * (2 ** (d-1)))
            self.blocks[str(d)] = nn.Sequential(
                custom_pad(1, pad),
                EncoderBlock(filters_in, filters_in, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout, normalization=normalization),
                DecoderBlock(filters_in, filters_out, kernel=4, stride=2, padding=1, activation=activation, dropout=dropout, normalization=normalization)
            )
        self.blocks[str(0)] = nn.Sequential(
            custom_pad(1, pad),
            EncoderBlock(32, 32, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout, normalization=normalization),
            DecoderBlock(32, 32, kernel=4, stride=2, padding=1, activation=activation, dropout=dropout, normalization=normalization),
            custom_pad(1, pad),
            EncoderBlock(32, 32, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout, normalization=normalization),
            custom_pad(1, pad),
            EncoderBlock(32, output_nc, kernel=3, stride=1, padding=0, activation=None, dropout=dropout, normalization=normalization)
        )

        self.skip = nn.ModuleDict()
        for d in reversed(range(1, depth)):
            if extract is not None and d in extract:
                filters = 32 * (2 ** min(4,d-1))
                self.skip[str(d)] = nn.Sequential(
                    custom_pad(1, pad),
                    nn.Conv2d(2 * filters, filters, kernel_size=3, padding=0)
                )

        self.tanh = nn.Tanh()

    def forward(self, x, use_activation=True):
        if type(x) is not list:
            x = [None for _ in range(self.depth)] + [x]
        y = x[self.depth]
        for d in reversed(range(0, self.depth)):
            y = self.blocks[str(d)](y)
            if x[d] is not None:
                y = self.skip[str(d)](torch.cat((y, x[d]), dim=1))

        if use_activation:
            return self.tanh(y)
        else:
            return y


class DecoderMunit(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(DecoderMunit, self).__init__()
        from .munit import ResBlocks, Conv2dBlock

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class EncoderMunit(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(EncoderMunit, self).__init__()
        from .munit import Conv2dBlock

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(n_downsample):
            next_dim = min(style_dim, dim * 2)
            self.model += [Conv2dBlock(dim, next_dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim = next_dim
        #self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class TransformerBlock(nn.Module):
    """
    This is the transformer for heavy mode.
    """
    def __init__(self, filters_in, filters_out, activation, normalization='instance'):
        super(TransformerBlock, self).__init__()

        self.block1 = EncoderBlock(filters_in, filters_in, kernel=3, stride=1, padding=1, activation=activation, normalization=normalization)
        self.block2 = EncoderBlock(filters_in, filters_in, kernel=3, stride=1, padding=1, activation=activation, normalization=normalization)
        self.block3 = EncoderBlock(filters_in, filters_out, kernel=3, stride=1, padding=1, activation=activation, normalization=normalization)

    def forward(self, x):
        o1 = self.block1(x)
        o2 = self.block2(o1)
        o3 = self.block3(o2)
        return o3


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, e1_conv_nc, e2_conv_nc, last_conv_nc, input_size, depth, extract=None, normalization='instance', mask_input=False):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.e1_conv_nc = e1_conv_nc
        self.e2_conv_nc = e2_conv_nc
        self.last_conv_nc = last_conv_nc
        self.sep = 0
        self.input_size = input_size
        self.extract = extract
        self.mask_input = mask_input

        self.E_A = E1(self.input_nc, e1_conv_nc, 0, input_size, depth, normalization=normalization)
        self.E_B = E2(self.input_nc, e2_conv_nc, input_size, depth, normalization=normalization)
        self.Decoder = Decoder(output_nc, last_conv_nc, input_size, depth, extract=extract, normalization=normalization)
        self.Merger = nn.Sequential(
            nn.Linear(self.E_A.bottom_features + self.E_B.bottom_features, self.Decoder.bottom_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask_in, mode=None, use_activation=True):
        N, B, C, H, W = x.shape
        if mode is None:
            mask_in1 = mask_in[:, 0, 0, :, :].unsqueeze(1)
            mask_in2 = mask_in[:, 1, 0, :, :].unsqueeze(1)
        else:
            mask_in1 = mask_in[:, 0, :, :].unsqueeze(1)
            mask_in2 = mask_in[:, 0, :, :].unsqueeze(1)

        mask_in1 = torch.cat((mask_in1, 1 - mask_in1), dim=1)
        mask_in2 = torch.cat((mask_in2, 1 - mask_in2), dim=1)

        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]

        if mode is None or mode == 0:
            x1_A = x1 if not self.mask_input else \
                       torch.where(mask_in1[:, 1].unsqueeze(1).expand_as(x1) > 0.5, x1, torch.zeros_like(x1))
            x1_A = torch.cat((x1_A, mask_in1), dim=1)
            x2_B = torch.cat((x2, mask_in2), dim=1)
            e_x1_A = self.E_A(x1_A, extract=self.extract)
            e_x2_B = self.E_B(x2_B, extract=None)
            if self.extract is None:
                z1 = torch.cat((e_x1_A, e_x2_B), dim=1)
                z1 = self.Merger(z1.view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
            else:
                z1 = e_x1_A
                z1[-1] = torch.cat((e_x1_A[-1], e_x2_B), dim=1)
                z1[-1] = self.Merger(z1[-1].view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
            y1 = self.Decoder(z1, use_activation=use_activation)
        if mode is None or mode == 1:
            x2_A = x2 if not self.mask_input else \
                       torch.where(mask_in2[:, 1].unsqueeze(1).expand_as(x2) > 0.5, x2, torch.zeros_like(x2))
            x2_A = torch.cat((x2_A, mask_in2), dim=1)
            x1_B = torch.cat((x1, mask_in1), dim=1)
            e_x2_A = self.E_A(x2_A, extract=self.extract)
            e_x1_B = self.E_B(x1_B, extract=None)
            if self.extract is None:
                z2 = torch.cat((e_x2_A, e_x1_B), dim=1)
                z2 = self.Merger(z2.view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
            else:
                z2 = e_x2_A
                z2[-1] = torch.cat((e_x2_A[-1], e_x1_B), dim=1)
                z2[-1] = self.Merger(z2[-1].view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
            y2 = self.Decoder(z2, use_activation=use_activation)

        if mode is None:
            y = torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), dim=1)
        elif mode == 0:
            y = y1
        else:
            y = y2

        return y


class GeneratorHeavy(nn.Module):
    def __init__(self, input_nc, output_nc, e1_conv_nc, e2_conv_nc, last_conv_nc, input_size, depth, extract=None, pad='reflect', normalization='instance'):
        super(GeneratorHeavy, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.e1_conv_nc = e1_conv_nc
        self.e2_conv_nc = e2_conv_nc
        self.last_conv_nc = last_conv_nc
        self.sep = 0
        self.input_size = input_size
        self.extract = extract

        self.E_A = EHeavy(self.input_nc, e1_conv_nc, input_size, depth, pad=pad, normalization=normalization)
        self.E_B = EHeavy(self.input_nc, e2_conv_nc, input_size, depth, pad=pad, normalization=normalization)
        self.Decoder = DecoderHeavy(output_nc, last_conv_nc, input_size, depth, extract=extract, pad=pad, normalization=normalization)
        self.Merger = TransformerBlock(e1_conv_nc + e2_conv_nc, last_conv_nc, activation='relu', normalization=normalization)

    def forward(self, x, mask_in, mode=None, use_activation=True):
        N, B, C, H, W = x.shape
        if mode is None:
            mask_in1 = mask_in[:, 0, 0, :, :].unsqueeze(1)
            mask_in2 = mask_in[:, 1, 0, :, :].unsqueeze(1)
        else:
            mask_in1 = mask_in[:, 0, :, :].unsqueeze(1)
            mask_in2 = mask_in[:, 0, :, :].unsqueeze(1)

        mask_in1 = torch.cat((mask_in1, 1 - mask_in1), dim=1)
        mask_in2 = torch.cat((mask_in2, 1 - mask_in2), dim=1)

        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]

        if mode is None or mode == 0:
            x1_A = x1
            x1_A = torch.cat((x1_A, mask_in1), dim=1)
            x2_B = torch.cat((x2, mask_in2), dim=1)
            e_x1_A = self.E_A(x1_A, extract=self.extract)
            e_x2_B = self.E_B(x2_B, extract=None)
            if self.extract is None:
                z1 = torch.cat((e_x1_A, e_x2_B), dim=1)
                z1 = self.Merger(z1)
            else:
                z1 = e_x1_A
                z1[-1] = torch.cat((e_x1_A[-1], e_x2_B), dim=1)
                z1[-1] = self.Merger(z1[-1])
            y1 = self.Decoder(z1, use_activation=use_activation)
        if mode is None or mode == 1:
            x2_A = x2
            x2_A = torch.cat((x2_A, mask_in2), dim=1)
            x1_B = torch.cat((x1, mask_in1), dim=1)
            e_x2_A = self.E_A(x2_A, extract=self.extract)
            e_x1_B = self.E_B(x1_B, extract=None)
            if self.extract is None:
                z2 = torch.cat((e_x2_A, e_x1_B), dim=1)
                z2 = self.Merger(z2)
            else:
                z2 = e_x2_A
                z2[-1] = torch.cat((e_x2_A[-1], e_x1_B), dim=1)
                z2[-1] = self.Merger(z2[-1])
            y2 = self.Decoder(z2, use_activation=use_activation)

        if mode is None:
            y = torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), dim=1)
        elif mode == 0:
            y = y1
        else:
            y = y2

        return y


class GeneratorMunit(nn.Module):
    def __init__(self, input_nc, output_nc, e1_conv_nc, e2_conv_nc, last_conv_nc, input_size, depth, extract=None, normalization='instance', mask_input=False):
        super(GeneratorMunit, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.e1_conv_nc = e1_conv_nc
        self.e2_conv_nc = e2_conv_nc
        self.last_conv_nc = last_conv_nc
        self.sep = 0
        self.input_size = input_size
        self.extract = extract
        self.mask_input = mask_input

        #n_downsample, input_dim, dim, style_dim, norm, activ, pad_type)
        self.E_A = EncoderMunit(depth, input_nc, 32, e1_conv_nc, 'ln', 'lrelu', 'zero')
        self.E_B = EncoderMunit(depth, input_nc, 32, e2_conv_nc, 'ln', 'lrelu', 'zero')
        #n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero')
        self.Decoder = DecoderMunit(depth, 2, 2*last_conv_nc, output_nc, res_norm='ln', activ='relu', pad_type='zero')
        #self.Merger = TransformerBlock(e1_conv_nc + e2_conv_nc, last_conv_nc, activation='relu', normalization=normalization)

    def forward(self, x, mask_in, mode=None, use_activation=True):
        N, B, C, H, W = x.shape
        if mode is None:
            mask_in1 = mask_in[:, 0, 0, :, :].unsqueeze(1)
            mask_in2 = mask_in[:, 1, 0, :, :].unsqueeze(1)
        else:
            mask_in1 = mask_in[:, 0, :, :].unsqueeze(1)
            mask_in2 = mask_in[:, 0, :, :].unsqueeze(1)

        mask_in1 = torch.cat((mask_in1, 1 - mask_in1), dim=1)
        mask_in2 = torch.cat((mask_in2, 1 - mask_in2), dim=1)

        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]

        if mode is None or mode == 0:
            x1_A = x1 if not self.mask_input else \
                       torch.where(mask_in1[:,1].unsqueeze(1).expand_as(x1) > 0.5, x1, torch.zeros_like(x1))
            x1_A = torch.cat((x1_A, mask_in1), dim=1)
            x2_B = torch.cat((x2, mask_in2), dim=1)
            e_x1_A = self.E_A(x1_A)
            e_x2_B = self.E_B(x2_B)
            if self.extract is None:
                z1 = torch.cat((e_x1_A, e_x2_B), dim=1)
                #z1 = self.Merger(z1.view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
            else:
                z1 = e_x1_A
                z1[-1] = torch.cat((e_x1_A[-1], e_x2_B), dim=1)
                #z1[-1] = self.Merger(z1[-1].view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
            y1 = self.Decoder(z1)
        if mode is None or mode == 1:
            x2_A = x2 if not self.mask_input else \
                       torch.where(mask_in2[:,1].unsqueeze(1).expand_as(x2) > 0.5, x2, torch.zeros_like(x2))
            x2_A = torch.cat((x2_A, mask_in2), dim=1)
            x1_B = torch.cat((x1, mask_in1), dim=1)
            e_x2_A = self.E_A(x2_A)
            e_x1_B = self.E_B(x1_B)
            if self.extract is None:
                z2 = torch.cat((e_x2_A, e_x1_B), dim=1)
                #z2 = self.Merger(z2.view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
            else:
                z2 = e_x2_A
                z2[-1] = torch.cat((e_x2_A[-1], e_x1_B), dim=1)
                #z2[-1] = self.Merger(z2[-1].view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
            y2 = self.Decoder(z2)

        if mode is None:
            y = torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), dim=1)
        elif mode == 0:
            y = y1
        else:
            y = y2

        return y


class Discriminator(nn.Module):
    def __init__(self, input_nc, last_conv_nc, input_size, depth, normalization='instance'):
        super(Discriminator, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)

        self.E = E1(input_nc, last_conv_nc, 0, self.input_size, depth, activation='relu', normalization=normalization)
        self.linear = nn.Linear(last_conv_nc * self.feature_size * self.feature_size, 1)

    def forward(self, x):
        x = self.E(x)
        x = self.linear(x.view(-1, self.last_conv_nc * self.feature_size * self.feature_size))
        return x


class DiscriminatorReID(nn.Module):
    def __init__(self, input_nc, last_conv_nc, input_size, depth, out_features=512, dropout=0.1, normalization='instance'):
        super(DiscriminatorReID, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)
        self.out_features = out_features

        self.E = E1(input_nc, last_conv_nc, 0, self.input_size, depth, activation='relu', dropout=dropout, normalization=normalization)
        self.dropout = nn.Dropout2d(p=dropout)
        self.avg_pool = nn.AvgPool2d(self.feature_size)
        self.linear1 = nn.Linear(last_conv_nc, out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_activation=False):
        x = self.E(x)
        x = self.dropout(x)
        x = self.avg_pool(x)
        x = self.linear1(x.view(-1, self.last_conv_nc))
        if use_activation:
            x = self.sigmoid(x)
        return x


class Classifier200(nn.Module):
    def __init__(self, input_nc, last_conv_nc, input_size, depth):
        super(Classifier200, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)

        self.E = E1(input_nc, last_conv_nc, 0, self.input_size, depth)
        self.linear1 = nn.Linear(last_conv_nc * self.feature_size * self.feature_size, 1024)
        self.relu = nn.ReLU
        self.linear2 = nn.Linear(1024, 200)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.E(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
