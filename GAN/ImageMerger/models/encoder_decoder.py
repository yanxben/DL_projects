import torch
import torch.nn as nn

class E1(nn.Module):
    def __init__(self, input_nc, last_conv_nc, sep, input_size, depth):
        super(E1, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.sep = sep
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)

        if depth == 5:
            self.full = nn.Sequential(
                nn.Conv2d(self.input_nc, 32, 4, 2, 1),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, (last_conv_nc - self.sep), 4, 2, 1),
                nn.InstanceNorm2d(last_conv_nc - self.sep),
                nn.LeakyReLU(0.2, inplace=True)
            )
        if depth == 6:
            self.full = nn.Sequential(
                nn.Conv2d(self.input_nc, 32, 4, 2, 1),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, (last_conv_nc - self.sep), 4, 2, 1),
                nn.InstanceNorm2d(last_conv_nc - self.sep),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        x = self.full(x)
        x = x.view(-1, (self.last_conv_nc - self.sep) * self.feature_size * self.feature_size)
        return x


class E2(nn.Module):
    def __init__(self, input_nc, sep, input_size, depth):
        super(E2, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.sep = sep
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)

        if depth == 5:
            self.full = nn.Sequential(
                nn.Conv2d(self.input_nc, 32, 4, 2, 1),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, self.sep, 4, 2, 1),
                nn.InstanceNorm2d(self.sep),
                nn.LeakyReLU(0.2),
            )
        if depth == 6:
            self.full = nn.Sequential(
                nn.Conv2d(self.input_nc, 32, 4, 2, 1),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, self.sep, 4, 2, 1),
                nn.InstanceNorm2d(self.sep),
                nn.LeakyReLU(0.2),
            )

    def forward(self, x):
        x = self.full(x)
        x = x.view(-1, self.sep * self.feature_size * self.feature_size)
        return x


class Decoder(nn.Module):
    def __init__(self, output_nc, last_conv_nc, input_size, depth):
        super(Decoder, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.output_nc = output_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)

        if depth == 5:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(last_conv_nc, 256, 4, 2, 1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.InstanceNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, output_nc, 4, 2, 1),
                nn.Tanh()
            )
        if depth == 6:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(last_conv_nc, 512, 4, 2, 1),
                nn.InstanceNorm2d(512),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.InstanceNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, output_nc, 4, 2, 1),
                nn.Tanh()
            )

    def forward(self, x):
        x = x.view(-1, self.last_conv_nc, self.feature_size, self.feature_size)
        x = self.main(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, last_conv_nc, sep, input_size, depth, preprocess):
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
            if self.preprocess:
                x1_A = (1-mask_in1) * x1 # + mask_in1 * torch.mean(x1, dim=1, keepdim=True)
            else:
                x1_A = x1
            x1_A = torch.cat([x1_A, mask_in1[:, 0, :, :].unsqueeze(1)], dim=1)
            x2_B = torch.cat([x2, mask_in2[:, 0, :, :].unsqueeze(1)], dim=1)
            e_x1_A = self.E_A(x1_A)
            e_x2_B = self.E_B(x2_B)
            z1 = torch.cat([e_x1_A, e_x2_B], dim=1)
            y1 = self.Decoder(z1)
        if mode is None or mode == 1:
            if self.preprocess:
                x2_A = (1-mask_in2) * x2 # + mask_in2 * torch.mean(x2, dim=1, keepdim=True)
            else:
                x2_A = x2
            x2_A = torch.cat([x2_A, mask_in2[:, 0, :, :].unsqueeze(1)], dim=1)
            x1_B = torch.cat([x1, mask_in1[:, 0, :, :].unsqueeze(1)], dim=1)
            e_x2_A = self.E_A(x2_A)
            e_x1_B = self.E_B(x1_B)
            z2 = torch.cat([e_x2_A, e_x1_B], dim=1)
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

class DiscriminatorPair(nn.Module):
    def __init__(self, input_nc, last_conv_nc, input_size, depth):
        super(DiscriminatorPair, self).__init__()
        assert input_size // (2 ** depth) == input_size / (2 ** depth), 'Bad depth for input size'
        self.input_nc = input_nc
        self.last_conv_nc = last_conv_nc
        self.input_size = input_size
        self.feature_size = input_size // (2 ** depth)

        self.E = E1(input_nc, last_conv_nc, 0, self.input_size, depth)
        self.linear1 = nn.Linear(last_conv_nc * self.feature_size * self.feature_size, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, 256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_activation=True):
        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]

        x1, x2 = self.E(x1), self.E(x2)
        x1, x2 = self.linear1(x1), self.linear1(x2)
        x1, x2 = self.relu(x1), self.relu(x2)

        x1, x2 = self.linear2(x1), self.linear2(x2)
        if use_activation:
            x1, x2 = self.sigmoid(x1), self.sigmoid(x2)
        return torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)


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
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_activation=True):
        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]
        x3 = x[:, 2, :, :, :]

        x1, x2, x3 = self.E(x1), self.E(x2), self.E(x3)
        x1, x2, x3 = self.linear1(x1), self.linear1(x2), self.linear1(x3)
        #x1, x2, x3 = self.relu(x1), self.relu(x2), self.relu(x3)
        #x1, x2, x3 = self.linear2(x1), self.linear2(x2), self.linear2(x3)
        #if use_activation:
        #    x1, x2, x3 = self.sigmoid(x1), self.sigmoid(x2), self.sigmoid(x3)
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
