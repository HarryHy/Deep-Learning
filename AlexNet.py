import torch
from torch import nn
from torchstat import stat
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()      # b, 3, 224, 224
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),   # b, 64, 55, 55
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),    # b, 64, 27, 27

            nn.Conv2d(64, 192, kernel_size=5, padding=2),      # b, 192, 27, 27
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),   # b, 192, 13, 13

            nn.Conv2d(192, 384, kernel_size=3, padding=1),   # b, 384, 13, 13
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),   # b, 256, 13, 13
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),   # b, 256, 13, 13
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2))    # b, 256, 6, 6
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes))
    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

model = AlexNet(10)
stat(model, (3, 224, 224))
