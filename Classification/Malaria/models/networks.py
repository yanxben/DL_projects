from torch import nn


class ENCODER(nn.Module):
    def __init__(self, features):
        super(ENCODER, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,features, 3, 1, 1),
            nn.ReLU(True),
            nn.InstanceNorm2d(features),
            # nn.LayerNorm((32, 128, 128)),
            nn.MaxPool2d(2),
            nn.Conv2d(features, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.InstanceNorm2d(64),
            # nn.LayerNorm((64, 64, 64)),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.InstanceNorm2d(128),
            # nn.LayerNorm((128, 32, 32)),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.InstanceNorm2d(256),
            # nn.LayerNorm((256, 16, 16)),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.LayerNorm((512, 8, 8)),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            # nn.Conv2d(512, 512, 4, 1, 0),
            # nn.ReLU(True),
            # nn.InstanceNorm2d(512),
            # # nn.MaxPool2d(2),
            # nn.Dropout2d(0.1)
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


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(True),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(True),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    def __init__(self, features, num_classes=2):
        super(VGG, self).__init__()
        self.cnn = ENCODER(features)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = MLP(512, 1024, num_classes, 0.3)

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
        self.classifier = MLP(512, 1024, num_classes, 0.3)

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