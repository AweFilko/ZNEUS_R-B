import torch
import torch.nn as nn
import torch.nn.functional as F

def get_original_name(model):
    return model.__class__.__name__

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), #2D convolution layer
            nn.LeakyReLU(),
            nn.MaxPool2d(2),      # 112x112 -> downsampling layer

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),      # 56x56

            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),      # 28x28

            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.Dropout(float(cfg['model_hyperparams']['dropout'])),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # print(x.shape)
        return self.net(x)

class DeepCNN(nn.Module):
    def __init__(self, num_classes, cfg=None):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.gap = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = F.leaky_relu(self.bn5(self.conv5(x)))

        # print(x.shape)

        x = self.gap(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        return F.leaky_relu(x)

class ResNet(nn.Module):
    def __init__(self, num_classes=14, cfg= None):
        super().__init__()
        self.block1 = Block(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = Block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = Block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.block4 = Block(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.fc = nn.Linear(256 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = torch.flatten(x, 1)
        return self.fc(x)

class FENN(nn.Module):
     def __init__(self, input_dim = 6686, num_classes = 14, cfg = None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.LeakyReLU(),
            nn.Linear(256, num_classes)
         )

     def forward(self, x):
         return self.model(x)

