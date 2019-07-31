import torch
from torch import nn


class ENCODER(nn.Module):
    def __init__(self, features, version=1):
        super(ENCODER, self).__init__()
        if version==1:
            self.encoder = nn.Sequential(
                nn.Conv2d(1,features, 7, 1, 3),     #128x128
                nn.ReLU(True),
                nn.InstanceNorm2d(features),
                # nn.LayerNorm((32, 128, 128)),
                nn.MaxPool2d(2),                    #64x64
                nn.Conv2d(features, 64, 3, 1, 1),   #64x64
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                # nn.LayerNorm((64, 64, 64)),
                nn.MaxPool2d(2),                    #32x32
                nn.Dropout2d(0.1),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(128),
                # nn.LayerNorm((128, 32, 32)),
                nn.MaxPool2d(2),                    #16x16
                nn.Dropout2d(0.1),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(256),
                # nn.LayerNorm((256, 16, 16)),
                nn.MaxPool2d(2),                    #8x8
                nn.Dropout2d(0.2),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.ReLU(True),
                # nn.LayerNorm((512, 8, 8)),
                nn.InstanceNorm2d(512),
                nn.MaxPool2d(2),                    #4x4
                nn.Dropout2d(0.2),
                # nn.Conv2d(512, 512, 4, 1, 0),
                # nn.ReLU(True),
                # nn.InstanceNorm2d(512),
                # # nn.MaxPool2d(2),
                # nn.Dropout2d(0.1)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, features, 7, 1, (3,0)),    # 128x74/82/98/114/122/146
                nn.ReLU(True),                          # 128x68/76/92/108/116/140
                nn.InstanceNorm2d(features),
                # nn.LayerNorm((32, 128, 128)),
                nn.MaxPool2d(2),                        # 64x34/38/46/54/58/70
                nn.Conv2d(features, 64, 3, 1, (1,0)),   # 64x32/36/44/52/56/68
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                # nn.LayerNorm((64, 64, 64)),
                nn.MaxPool2d(2),                        # 32x16/18/22/26/28/34
                nn.Dropout2d(0.1),
                nn.Conv2d(64, 128, 3, 1, (1,0)),        # 32x14/16/20/24/26/32
                nn.ReLU(True),
                nn.InstanceNorm2d(128),
                # nn.LayerNorm((128, 32, 32)),
                nn.MaxPool2d(2),                        # 16x7/8/10/12/13/16
                nn.Dropout2d(0.1),
                nn.Conv2d(128, 256, 3, 1, (1,0)),       # 16x5/6/8/10/11/14
                nn.ReLU(True),
                nn.InstanceNorm2d(256),
                # nn.LayerNorm((256, 16, 16)),
                nn.MaxPool2d((2,1)),                    # 8x5/6/8/10/11/14
                nn.Dropout2d(0.2),
                nn.Conv2d(256, 512, 3, 1, (1,0)),       # 8x3/4/6/8/9/12
                nn.ReLU(True),
                # nn.LayerNorm((512, 8, 8)),
                nn.InstanceNorm2d(512),
                nn.MaxPool2d((2,1)),                    # 4x3/4/6/8/9/12
                nn.Dropout2d(0.2),
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
            nn.Linear(in_features, hidden_features[0]),
            nn.ReLU(True),
            nn.BatchNorm1d(hidden_features[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_features[0], hidden_features[1]),
            nn.ReLU(True),
            nn.BatchNorm1d(hidden_features[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_features[1], num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    def __init__(self, features, num_classes=2, version=1):
        super(VGG, self).__init__()
        self.cnn = ENCODER(features, version)
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool = nn.AvgPool2d((4, 4 if version==1 else 3), stride=(4,1), padding=0)
        self.classifier = MLP(512, [1024, 512], num_classes, dropout=0.5)

    def forward(self, x):
        # print(x.shape)
        x = self.cnn(x)
        # print(x.shape)
        # x = self.avgpool(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        assert x.shape[2] == 4
        x0 = self.avgpool(x).permute(0,2,3,1)
        a = x0.contiguous().view(-1, x0.shape[3])
        a = self.classifier(a)
        o = a.view(-1, x0.shape[2], a.shape[1]).squeeze(1)

        return o


class VGG_AE(nn.Module):
    def __init__(self, features, num_classes=2, version=1 ):
        super(VGG_AE, self).__init__()
        self.encoder = ENCODER(features, version=1)
        self.decoder = DECODER(features)
        self.avgpool = nn.AvgPool2d((4,4), stride=1)
        self.classifier = MLP(512, 1024, num_classes, 0.3)

    def forward(self, x, m1, m2):
        # print(x.shape)
        x = self.encoder(x)
        if m1:
            a = self.avgpool(x).squeeze(2)
            a = a.view(-1, a.shape[1])
            a = self.classifier(a)
            a = a.view(-1, a.shape[1], x.shape[3]).squeeze(2)
        else:
            m1 = None
        if m2:
            b = self.decoder(x)
        else:
            m2 = None

        return a, b
