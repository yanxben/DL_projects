import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, filters_in, filters_out, kernel=4, stride=2, padding=1, activation='lrelu', dropout=0.):
        super(EncoderBlock, self).__init__()
        self.full = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel, stride, padding),
            nn.InstanceNorm2d(filters_out),
            nn.LeakyReLU(0.2, inplace=True) if activation=='lrelu' else nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x):
        return self.full(x)


class DecoderBlock(nn.Module):
    def __init__(self, filters_in, filters_out, kernel=4, stride=2, padding=1, activation='relu', dropout=0.):
        super(DecoderBlock, self).__init__()
        self.full = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.ConvTranspose2d(filters_in, filters_out, kernel, stride, padding),
            nn.InstanceNorm2d(filters_out),
            nn.LeakyReLU(0.2, inplace=True) if activation == 'lrelu' else nn.ReLU(inplace=True)
        )

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

        self.blocks = nn.ModuleDict()
        self.blocks[str(0)] = EncoderBlock(self.input_nc, 32, activation=activation, dropout=dropout)
        for d in range(1, depth-1):
            self.blocks[str(d)] = EncoderBlock(32*(2 ** min(4,d-1)), 32*(2 ** min(4,d)), activation=activation, dropout=dropout)
        self.blocks[str(depth-1)] = EncoderBlock(32 * (2 ** min(4,depth - 1)), last_conv_nc - sep, activation=activation, dropout=dropout)

    def forward(self, x, extract=None):
        x = [x]
        for d in range(self.depth):
            x.append(self.blocks[str(d)](x[-1]))
            #print(x[-1].shape)

        x[self.depth] = x[self.depth].view(-1, (self.last_conv_nc - self.sep) * self.feature_size * self.feature_size)
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

        self.blocks = nn.ModuleDict()
        self.blocks[str(0)] = EncoderBlock(self.input_nc, 32, activation=activation, dropout=dropout)
        for d in range(1, depth-1):
            self.blocks[str(d)] = EncoderBlock(32 * (2 ** min(3,d - 1)), 32 * (2 ** min(3,d)), activation=activation, dropout=dropout)
        self.blocks[str(depth-1)] = EncoderBlock(32 * (2 ** min(3,depth - 2)), sep, activation=activation, dropout=dropout)

    def forward(self, x, extract=None):
        x = [x]
        for d in range(self.depth):
            x.append(self.blocks[str(d)](x[-1]))
            #print(x[-1].shape)

        x[self.depth] = x[self.depth].view(-1, self.sep * self.feature_size * self.feature_size)
        if extract is None:
            return x[self.depth]
        return [x[d] if d in extract else None for d in range(self.depth+1)]


class Decoder(nn.Module):
    def __init__(self, output_nc, last_conv_nc, input_size, depth, activation='relu', dropout=0.):
        super(Decoder, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.output_nc = output_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)
        self.depth = depth

        self.blocks = nn.ModuleDict()
        self.blocks[str(depth-1)] = DecoderBlock(last_conv_nc, 32 * (2 ** min(4,depth-2)), activation=activation, dropout=dropout)
        for d in reversed(range(1, depth-1)):
            self.blocks[str(d)] = DecoderBlock(32 * (2 ** min(4,d)), 32 * (2 ** min(4,d-1)), activation=activation, dropout=dropout)
        self.blocks[str(0)] = DecoderBlock(32, output_nc, activation=activation, dropout=dropout)

        self.skip = nn.ModuleDict()
        self.skip[str(0)] = nn.Conv2d(2 * output_nc, output_nc, kernel_size=1)
        for d in reversed(range(1, depth)):
            self.skip[str(d)] = nn.Conv2d(2 * 32 * (2 ** min(4,d-1)), 32 * (2 ** min(4,d-1)), kernel_size=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        if type(x) is not list:
            x = [None for _ in range(self.depth)] + [x]
        y = x[self.depth].view(-1, self.last_conv_nc, self.feature_size, self.feature_size)

        for d in reversed(range(0, self.depth)):
            y = self.blocks[str(d)](y)
            if x[d] is not None:
                y = self.skip[str(d)](torch.cat((y, x[d]), dim=1))

        return self.tanh(y)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, last_conv_nc, sep, input_size, depth, preprocess=False):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.last_conv_nc = last_conv_nc
        self.sep = sep
        self.input_size = input_size
        self.preprocess = preprocess

        self.E_A = E1(self.input_nc, last_conv_nc, sep, input_size, depth)
        self.E_B = E2(self.input_nc, sep, input_size, depth)
        self.Decoder = Decoder(output_nc, last_conv_nc, input_size, depth)

    def forward(self, x, mask_in, mode=None, extract=None):
        N, B, C, H, W = x.shape
        if mode is None:
            mask_in1 = mask_in[:, 0, :, :, :]
            mask_in2 = mask_in[:, 1, :, :, :]
        else:
            mask_in1 = mask_in
            mask_in2 = mask_in

        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]

        if mode is None or mode == 0:
            # if self.preprocess:
            #     x1_A = (1-mask_in1) * x1 # + mask_in1 * torch.mean(x1, dim=1, keepdim=True)
            # else:
            x1_A = x1
            x1_A = torch.cat([x1_A, mask_in1[:, 0, :, :].unsqueeze(1)], dim=1)
            x2_B = torch.cat([x2, mask_in2[:, 0, :, :].unsqueeze(1)], dim=1)
            #print(x1_A.shape)
            #print(x2_B.shape)
            e_x1_A = self.E_A(x1_A, extract=extract)
            e_x2_B = self.E_B(x2_B, extract=None)
            #print(e_x1_A.shape)
            #print(e_x2_B.shape)
            if extract is None:
                z1 = torch.cat([e_x1_A, e_x2_B], dim=1)
            else:
                z1 = e_x1_A
                z1[-1] = torch.cat([e_x1_A[-1], e_x2_B], dim=1)
            y1 = self.Decoder(z1)
        if mode is None or mode == 1:
            # if self.preprocess:
            #     x2_A = (1-mask_in2) * x2 # + mask_in2 * torch.mean(x2, dim=1, keepdim=True)
            # else:
            x2_A = x2
            x2_A = torch.cat([x2_A, mask_in2[:, 0, :, :].unsqueeze(1)], dim=1)
            x1_B = torch.cat([x1, mask_in1[:, 0, :, :].unsqueeze(1)], dim=1)
            e_x2_A = self.E_A(x2_A, extract=extract)
            e_x1_B = self.E_B(x1_B, extract=None)
            if extract is None:
                z2 = torch.cat([e_x2_A, e_x1_B], dim=1)
            else:
                z2 = e_x2_A
                z2[-1] = torch.cat([e_x2_A[-1], e_x1_B], dim=1)
            y2 = self.Decoder(z2)

        if mode is None:
            y = torch.cat([y1.unsqueeze(1), y2.unsqueeze(1)], dim=1)
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
        x = self.linear(x)
        #x = self.activation(x)
        return x


class DiscriminatorReID(nn.Module):
    def __init__(self, input_nc, last_conv_nc, input_size, depth, dropout=0.1):
        super(DiscriminatorReID, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)

        self.E = E1(input_nc, last_conv_nc, 0, self.input_size, depth, dropout=dropout)
        self.dropout = nn.Dropout2d(p=dropout)
        self.linear1 = nn.Linear(last_conv_nc * self.feature_size * self.feature_size, 512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_activation=False):
        x1 = self.E(x)
        x2 = self.dropout(x1)
        x3 = self.linear1(x2)
        if use_activation:
            x4 = self.sigmoid(x3)
            return x4
        return x3


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
            mask_in1 = mask_in[:, 0, :, :, :]
            mask_in2 = mask_in[:, 1, :, :, :]
        else:
            mask_in1 = mask_in
            mask_in2 = mask_in

        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]

        if mode is None or mode == 0:
            x1_A = x1
            x1_A = torch.cat([x1_A, mask_in1[:, 0, :, :].unsqueeze(1)], dim=1)
            x2_B = torch.cat([x2, mask_in2[:, 0, :, :].unsqueeze(1)], dim=1)
            e_x1_A = self.E(x1_A)
            e_x2_B = self.E(x2_B)
            z1 = torch.cat([e_x1_A, e_x2_B], dim=1)
            y1 = self.Decoder(z1)
        if mode is None or mode == 1:
            x2_A = x2
            x2_A = torch.cat([x2_A, mask_in2[:, 0, :, :].unsqueeze(1)], dim=1)
            x1_B = torch.cat([x1, mask_in1[:, 0, :, :].unsqueeze(1)], dim=1)
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
