from torch import nn


class ENCODER(nn.Module):
    def __init__(self, features):
        super(ENCODER, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,features, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(features),
            nn.MaxPool2d(2),
            nn.Conv2d(features, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class DECODER(nn.Module):
    def __init__(self, features):
        super(DECODER, self).__init__()
        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512,256, 3, 1, 1),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x



class VGG(nn.Module):
    def __init__(self, features, num_classes=2):
        super(VGG, self).__init__()
        self.cnn = ENCODER(features)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        # print(x.shape)
        x = self.cnn(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


class VGG_AE(nn.Module):
    def __init__(self, features, num_classes=2):
        super(VGG_AE, self).__init__()
        self.encoder = ENCODER(features)
        self.decoder = DECODER(features)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x, m1, m2):
        # print(x.shape)
        x = self.encoder(x)
        if m1:
            a = self.avgpool(x)
            a = a.view(a.size(0), -1)
            a = self.classifier(a)
        else:
            m1 = None
        if m2:
            b = self.decoder(x)
        else:
            m2 = None

        return a, b