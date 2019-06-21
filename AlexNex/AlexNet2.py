import torch
from torch import  nn
class AlexNet(nn.Module):
	def __init__(self, num_class):
		super(AlexNet, self).__init__()
		self.features == nn.Sequential(
			nn.Conv2d(3, 96, kernel_size = 11, stride=4, padding = 2),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(96, 256, kernel_size=5, padding=2),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(256, 384, kernel_size=3, padding=1),
			nn.ReLU(True),

			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.ReLU(True),

			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=3, stride=2)
			)

		self.classifier = nn.Sequential(
			nn.Linear(256*6*6, 4096),
			nn.ReLU(True),
			nn.Dropout(),

			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),

			nn.Linear(4096, num_class)
			)

	def forward(self, x):
		x = self.features(x)
		print(x.size())
		x = x.view(x.size(0), 256*6*6)
		x = self.classifier(x)
		return x

net = ALexNet(10)