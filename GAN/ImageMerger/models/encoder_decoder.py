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
    def __init__(self, filters_in, filters_out, kernel=4, stride=2, padding=1, activation=None, normalization='instance', dropout=0.):
        super(EncoderBlock, self).__init__()
        self.full = nn.Sequential()
        self.full.add_module('conv', nn.Conv2d(filters_in, filters_out, kernel, stride, padding))
        if normalization is not None:
            if normalization == 'instance':
                self.full.add_module('norm', nn.InstanceNorm2d(filters_out))
            if normalization == 'batch':
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
    def __init__(self, filters_in, filters_out, kernel=4, stride=2, padding=1, activation=None, normalization='instance', dropout=0.):
        super(DecoderBlock, self).__init__()
        self.full = nn.Sequential()
        if dropout > 0:
            self.full.add_module('dropout', nn.Dropout2d(p=dropout))
        self.full.add_module('conv2d', nn.ConvTranspose2d(filters_in, filters_out, kernel, stride, padding))
        if normalization is not None:
            if normalization == 'instance':
                self.full.add_module('norm', nn.InstanceNorm2d(filters_out))
            if normalization == 'batch':
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
    def __init__(self, input_nc, last_conv_nc, sep, input_size, depth, activation='lrelu', dropout=0.):
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
        self.blocks[str(0)] = EncoderBlock(self.input_nc, 32, activation=activation, dropout=dropout)
        for d in range(1, depth-1):
            self.blocks[str(d)] = EncoderBlock(32*(2 ** min(4,d-1)), 32*(2 ** min(4,d)), activation=activation, dropout=dropout)
        self.blocks[str(depth-1)] = EncoderBlock(32 * (2 ** min(4,depth - 2)), last_conv_nc - sep, activation=activation, dropout=dropout)

    def forward(self, x, extract=None):
        x = [x]
        for d in range(self.depth):
            x.append(self.blocks[str(d)](x[-1]))
            # print(x[-1].shape)

        #x[self.depth] = x[self.depth].view(-1, self.bottom_features)
        if extract is None:
            return x[self.depth]
        return [x[d] if d in extract else None for d in range(self.depth + 1)]


class E2(nn.Module):
    def __init__(self, input_nc, sep, input_size, depth, activation='lrelu', dropout=0.):
        super(E2, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.sep = sep
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)
        self.depth = depth
        self.bottom_features = self.sep * self.feature_size * self.feature_size

        self.blocks = nn.ModuleDict()
        self.blocks[str(0)] = EncoderBlock(self.input_nc, 32, activation=activation, dropout=dropout)
        for d in range(1, depth-1):
            self.blocks[str(d)] = EncoderBlock(32 * (2 ** min(3,d - 1)), 32 * (2 ** min(3,d)), activation=activation, dropout=dropout)
        self.blocks[str(depth-1)] = EncoderBlock(32 * (2 ** min(3,depth - 2)), sep, activation=activation, dropout=dropout)

    def forward(self, x, extract=None):
        x = [x]
        for d in range(self.depth):
            x.append(self.blocks[str(d)](x[-1]))
            # print(x[-1].shape)

        #x[self.depth] = x[self.depth].view(-1, self.bottom_features)
        if extract is None:
            return x[self.depth]
        return [x[d] if d in extract else None for d in range(self.depth+1)]


class EHeavy(nn.Module):
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
            # print(x[-1].shape)

        #x[self.depth] = x[self.depth].view(-1, self.bottom_features)
        if extract is None:
            return x[self.depth]
        return [x[d] if d in extract else None for d in range(self.depth + 1)]


class Decoder(nn.Module):
    def __init__(self, output_nc, last_conv_nc, input_size, depth, activation='relu', dropout=0., extract=None):
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
        self.blocks[str(depth-1)] = DecoderBlock(last_conv_nc, 32 * (2 ** min(4,depth-2)), activation=activation, dropout=dropout)
        for d in reversed(range(1, depth-1)):
            self.blocks[str(d)] = DecoderBlock(32 * (2 ** min(4,d)), 32 * (2 ** min(4,d-1)), activation=activation, dropout=dropout)
        self.blocks[str(0)] = DecoderBlock(32, output_nc, activation=None, dropout=dropout)

        self.skip = nn.ModuleDict()
        self.skip[str(0)] = nn.Conv2d(2 * output_nc, output_nc, kernel_size=1)
        for d in reversed(range(1, depth)):
            if extract is not None and d in extract:
                self.skip[str(d)] = nn.Conv2d(2 * 32 * (2 ** min(4,d-1)), 32 * (2 ** min(4,d-1)), kernel_size=1)

        self.tanh = nn.Tanh()

    def forward(self, x, use_activation=True):
        if type(x) is not list:
            x = [None for _ in range(self.depth)] + [x]
        y = x[self.depth].view(-1, self.last_conv_nc, self.feature_size, self.feature_size)

        for d in reversed(range(0, self.depth)):
            # print(d)
            # print(y.shape)
            y = self.blocks[str(d)](y)
            if x[d] is not None:
                y = self.skip[str(d)](torch.cat((y, x[d]), dim=1))

        if use_activation:
            return self.tanh(y)
        else:
            return y


class DecoderHeavy(nn.Module):
    def __init__(self, output_nc, last_conv_nc, input_size, depth, activation='relu', dropout=0., extract=None, pad='reflect'):
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
            EncoderBlock(last_conv_nc, last_conv_nc, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout),
            #nn.ReflectionPad2d(1),
            DecoderBlock(last_conv_nc, min(last_conv_nc, 32 * (2 ** (depth-2))), kernel=4, stride=2, padding=1, activation=activation, dropout=dropout)
        )
        for d in reversed(range(1, depth-1)):
            filters_in = min(last_conv_nc, 32 * (2 ** d))
            filters_out = min(last_conv_nc, 32 * (2 ** (d-1)))
            self.blocks[str(d)] = nn.Sequential(
                custom_pad(1, pad),
                EncoderBlock(filters_in, filters_in, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout),
                #nn.ReflectionPad2d(1),
                DecoderBlock(filters_in, filters_out, kernel=4, stride=2, padding=1, activation=activation, dropout=dropout)
            )
        self.blocks[str(0)] = nn.Sequential(
            custom_pad(1, pad),
            EncoderBlock(32, 32, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout),
            #nn.ReflectionPad2d(1),
            DecoderBlock(32, 32, kernel=4, stride=2, padding=1, activation=activation, dropout=dropout),
            custom_pad(1, pad),
            EncoderBlock(32, 32, kernel=3, stride=1, padding=0, activation=activation, dropout=dropout),
            custom_pad(1, pad),
            EncoderBlock(32, output_nc, kernel=3, stride=1, padding=0, activation=None, dropout=dropout)
        )

        self.skip = nn.ModuleDict()
        #self.skip[str(0)] = nn.Conv2d(2 * output_nc, output_nc, kernel_size=3, 0)
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
        #y = x[self.depth].view(-1, self.last_conv_nc, self.feature_size, self.feature_size)
        y = x[self.depth]
        for d in reversed(range(0, self.depth)):
            # print(d)
            # print(y.shape)
            y = self.blocks[str(d)](y)
            # print(y.shape)
            if x[d] is not None:
                # print(x[d].shape)
                y = self.skip[str(d)](torch.cat((y, x[d]), dim=1))

        if use_activation:
            return self.tanh(y)
        else:
            return y


class TransformerBlock(nn.Module):
    def __init__(self, filters_in, filters_out, activation):
        super(TransformerBlock, self).__init__()

        self.block1 = EncoderBlock(filters_in, filters_in, kernel=3, stride=1, padding=1, activation=activation)
        self.block2 = EncoderBlock(filters_in, filters_in, kernel=3, stride=1, padding=1, activation=activation)
        self.block3 = EncoderBlock(filters_in, filters_out, kernel=3, stride=1, padding=1, activation=activation)

    def forward(self, x):
        o1 = self.block1(x)
        o2 = self.block2(o1)
        o3 = self.block3(o2)
        return o3


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, e1_conv_nc, e2_conv_nc, last_conv_nc, input_size, depth, preprocess=False, extract=None):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.e1_conv_nc = e1_conv_nc
        self.e2_conv_nc = e2_conv_nc
        self.last_conv_nc = last_conv_nc
        self.sep = 0
        self.input_size = input_size
        self.preprocess = preprocess
        self.extract = extract

        self.E_A = E1(self.input_nc, e1_conv_nc, 0, input_size, depth)
        self.E_B = E2(self.input_nc, e2_conv_nc, input_size, depth)
        self.Decoder = Decoder(output_nc, last_conv_nc, input_size, depth, extract=extract)
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
            x1_A = x1
            x1_A = torch.cat((x1_A, mask_in1), dim=1)
            x2_B = torch.cat((x2, mask_in2), dim=1)
            #print(x1_A.shape)
            #print(x2_B.shape)
            e_x1_A = self.E_A(x1_A, extract=self.extract)
            e_x2_B = self.E_B(x2_B, extract=None)
            #print(e_x1_A.shape)
            #print(e_x2_B.shape)
            if self.extract is None:
                z1 = torch.cat((e_x1_A, e_x2_B), dim=1)
                z1 = self.Merger(z1.view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
            else:
                z1 = e_x1_A
                # print('before z')
                # print(e_x1_A[-1].shape)
                # print(e_x2_B.shape)
                z1[-1] = torch.cat((e_x1_A[-1], e_x2_B), dim=1)
                z1[-1] = self.Merger(z1[-1].view(-1, self.E_A.bottom_features + self.E_B.bottom_features))
                # print('after z')
            # print('decoding')
            y1 = self.Decoder(z1, use_activation=use_activation)
        if mode is None or mode == 1:
            x2_A = x2
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
    def __init__(self, input_nc, output_nc, e1_conv_nc, e2_conv_nc, last_conv_nc, input_size, depth, extract=None, pad='reflect'):
        super(GeneratorHeavy, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.e1_conv_nc = e1_conv_nc
        self.e2_conv_nc = e2_conv_nc
        self.last_conv_nc = last_conv_nc
        self.sep = 0
        self.input_size = input_size
        self.extract = extract

        self.E_A = EHeavy(self.input_nc, e1_conv_nc, input_size, depth, pad=pad)
        self.E_B = EHeavy(self.input_nc, e2_conv_nc, input_size, depth, pad=pad)
        self.Decoder = DecoderHeavy(output_nc, last_conv_nc, input_size, depth, extract=extract, pad=pad)
        self.Merger = TransformerBlock(e1_conv_nc + e2_conv_nc, last_conv_nc, activation='relu')

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
            # print(x1_A.shape)
            # print(x2_B.shape)
            e_x1_A = self.E_A(x1_A, extract=self.extract)
            e_x2_B = self.E_B(x2_B, extract=None)
            # print(e_x1_A.shape)
            # print(e_x2_B.shape)
            if self.extract is None:
                z1 = torch.cat((e_x1_A, e_x2_B), dim=1)
                z1 = self.Merger(z1)
                z1 = z1 + e_x2_B  # skip connection
            else:
                z1 = e_x1_A
                # print('before z')
                # print(e_x1_A[-1].shape)
                # print(e_x2_B.shape)
                z1[-1] = torch.cat((e_x1_A[-1], e_x2_B), dim=1)
                z1[-1] = self.Merger(z1[-1])
                z1[-1] = z1[-1] + e_x2_B  # skip connection
                # print('after z')
                # print(z1[-1].shape)
            # print('decoding')
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
                z2 = z2 + e_x1_B  # skip connection
            else:
                z2 = e_x2_A
                z2[-1] = torch.cat((e_x2_A[-1], e_x1_B), dim=1)
                z2[-1] = self.Merger(z2[-1])
                z2[-1] = z2[-1] + e_x1_B  # skip connection
            y2 = self.Decoder(z2, use_activation=use_activation)

        if mode is None:
            y = torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), dim=1)
        elif mode == 0:
            y = y1
        else:
            y = y2

        return y


class GeneratorDecoder(nn.Module):
    def __init__(self, in_features, output_nc, last_conv_nc, input_size, depth, extract=None):
        super(GeneratorDecoder, self).__init__()
        self.in_features = in_features
        self.output_nc = output_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.extract = extract

        self.Decoder = Decoder(output_nc, last_conv_nc, input_size, depth, extract=extract)
        self.Merger = nn.Sequential(
            nn.Linear(in_features, self.Decoder.bottom_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, e_x1, e_x2, mode=None, use_activation=True):

        if mode is None or mode == 0:
            if self.extract is None:
                e_x1_A = e_x1.clone()
                e_x2_B = e_x2.clone()
                z1 = torch.cat((e_x1_A, e_x2_B), dim=1)
                z1 = self.Merger(z1.view(-1, self.in_features))
            else:
                e_x1_A = [None if e_x1[i] is None else e_x1[i].clone() for i in range(len(e_x1))]
                e_x2_B = [None if e_x2[i] is None else e_x2[i].clone() for i in range(len(e_x2))]
                z1 = e_x1_A
                # print('before z')
                # print(e_x1_A[-1].shape)
                # print(e_x2_B.shape)
                z1[-1] = torch.cat((e_x1_A[-1], e_x2_B[-1]), dim=1)
                z1[-1] = self.Merger(z1[-1].view(-1, self.in_features))
                # print('after z')
            # print('decoding')
            y1 = self.Decoder(z1, use_activation=use_activation)
        if mode is None or mode == 1:

            if self.extract is None:
                e_x2_A = e_x2.clone()
                e_x1_B = e_x1.clone()
                z2 = torch.cat((e_x2_A, e_x1_B), dim=1)
                z2 = self.Merger(z2.view(-1, self.in_features))
            else:
                e_x2_A = [None if e_x2[i] is None else e_x2[i].clone() for i in range(len(e_x2))]
                e_x1_B = [None if e_x1[i] is None else e_x1[i].clone() for i in range(len(e_x1))]
                z2 = e_x2_A
                z2[-1] = torch.cat((e_x2_A[-1], e_x1_B[-1]), dim=1)
                z2[-1] = self.Merger(z2[-1].view(-1, self.in_features))
            y2 = self.Decoder(z2, use_activation=use_activation)

        if mode is None:
            y = torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), dim=1)
        elif mode == 0:
            y = y1
        else:
            y = y2

        return y


class GeneratorDecoderHeavy(nn.Module):
    def __init__(self, in_conv_nc, output_nc, last_conv_nc, input_size, depth, extract=None):
        super(GeneratorDecoderHeavy, self).__init__()
        self.in_conv_nc = in_conv_nc
        self.output_nc = output_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.extract = extract

        self.Decoder = DecoderHeavy(output_nc, last_conv_nc, input_size, depth, extract=extract)
        self.Merger = TransformerBlock(in_conv_nc, last_conv_nc, activation='relu')

    def forward(self, e_x1, e_x2, mode=None, use_activation=True):

        if mode is None or mode == 0:
            if self.extract is None:
                e_x1_A = e_x1.clone()
                e_x2_B = e_x2.clone()
                z1 = torch.cat((e_x1_A, e_x2_B), dim=1)
                z1 = self.Merger(z1)
            else:
                e_x1_A = [None if e_x1[i] is None else e_x1[i].clone() for i in range(len(e_x1))]
                e_x2_B = [None if e_x2[i] is None else e_x2[i].clone() for i in range(len(e_x2))]
                z1 = e_x1_A
                # print('before z')
                # print(e_x1_A[-1].shape)
                # print(e_x2_B.shape)
                z1[-1] = torch.cat((e_x1_A[-1], e_x2_B[-1]), dim=1)
                z1[-1] = self.Merger(z1[-1])
                # print('after z')
            # print('decoding')
            y1 = self.Decoder(z1, use_activation=use_activation)
        if mode is None or mode == 1:

            if self.extract is None:
                e_x2_A = e_x2.clone()
                e_x1_B = e_x1.clone()
                z2 = torch.cat((e_x2_A, e_x1_B), dim=1)
                z2 = self.Merger(z2)
            else:
                e_x2_A = [None if e_x2[i] is None else e_x2[i].clone() for i in range(len(e_x2))]
                e_x1_B = [None if e_x1[i] is None else e_x1[i].clone() for i in range(len(e_x1))]
                z2 = e_x2_A
                z2[-1] = torch.cat((e_x2_A[-1], e_x1_B[-1]), dim=1)
                z2[-1] = self.Merger(z2[-1])
            y2 = self.Decoder(z2, use_activation=use_activation)

        if mode is None:
            y = torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), dim=1)
        elif mode == 0:
            y = y1
        else:
            y = y2

        return y


class Disc(nn.Module):
    def __init__(self, input_nc, last_conv_nc, sep, input_size, depth):
        super(Disc, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.sep = sep
        self.feature_size = input_size // (2 ** depth)

        self.classify = nn.Sequential(
            nn.Linear((last_conv_nc - self.sep) * self.feature_size * self.feature_size, last_conv_nc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(last_conv_nc, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, (self.last_conv_nc - self.sep) * self.feature_size * self.feature_size)
        x = self.classify(x)
        x = x.view(-1)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc, last_conv_nc, input_size, depth):
        super(Discriminator, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)

        self.E = E1(input_nc, last_conv_nc, 0, self.input_size, depth)
        self.linear = nn.Linear(last_conv_nc * self.feature_size * self.feature_size, 1)
        #self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.E(x)
        x = self.linear(x.view(-1, self.last_conv_nc * self.feature_size * self.feature_size))
        #x = self.activation(x)
        return x


class DiscriminatorReID(nn.Module):
    def __init__(self, input_nc, last_conv_nc, input_size, depth, out_features=512, dropout=0.1):
        super(DiscriminatorReID, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)
        self.out_features = out_features

        self.E = E1(input_nc, last_conv_nc, 0, self.input_size, depth, activation='relu', dropout=dropout)
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


class DiscriminatorTriplet(nn.Module):
    def __init__(self, input_nc, last_conv_nc, input_size, depth):
        super(DiscriminatorTriplet, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)

        self.E = E1(input_nc, last_conv_nc, 0, self.input_size, depth)
        self.linear1 = nn.Linear(last_conv_nc * self.feature_size * self.feature_size, 256)
        #self.relu = nn.ReLU()
        #self.linear2 = nn.Linear(512, 256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_activation=False):
        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]
        x3 = x[:, 2, :, :, :]

        x1, x2, x3 = self.E(x1), self.E(x2), self.E(x3)
        x1, x2, x3 = self.linear1(x1), self.linear1(x2), self.linear1(x3)
        #x1, x2, x3 = self.relu(x1), self.relu(x2), self.relu(x3)
        #x1, x2, x3 = self.linear2(x1), self.linear2(x2), self.linear2(x3)
        if use_activation:
            x1, x2, x3 = self.sigmoid(x1), self.sigmoid(x2), self.sigmoid(x3)
        return torch.cat([x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)], dim=1)


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
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.E(x)
        x = self.relu(x)
        x = self.linear2(x)
        #x = self.softmax(x)
        return x


class AutoEncoder2(nn.Module):
    def __init__(self, input_nc, output_nc, last_conv_nc, sep, input_size, depth, preprocess):
        super(AutoEncoder2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.last_conv_nc = last_conv_nc
        self.sep = sep
        self.input_size = input_size
        self.preprocess = preprocess

        self.E = E1(self.input_nc, last_conv_nc, sep, input_size, depth)
        self.Decoder = Decoder(output_nc, last_conv_nc, input_size, depth)

    def forward(self, x, mask_in, mode=None):
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
            x1_A = torch.cat([x1_A, mask_in1], dim=1)
            x2_B = torch.cat([x2, mask_in2], dim=1)
            e_x1_A = self.E(x1_A)
            e_x2_B = self.E(x2_B)
            z1 = torch.cat([e_x1_A, e_x2_B], dim=1)
            y1 = self.Decoder(z1)
        if mode is None or mode == 1:
            x2_A = x2
            x2_A = torch.cat([x2_A, mask_in2], dim=1)
            x1_B = torch.cat([x1, mask_in1], dim=1)
            e_x2_A = self.E(x2_A)
            e_x1_B = self.E(x1_B)
            z2 = torch.cat([e_x2_A, e_x1_B], dim=1)
            y2 = self.Decoder(z2)

        if mode is None:
            y = torch.cat([y1.unsqueeze(1), y2.unsqueeze(1)], dim=1)
        elif mode == 0:
            y = y1
        else:
            y = y2

        return y
