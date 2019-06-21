VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M5', "FC1", "FC2", "FC"]
VGG19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M5', "FC1", "FC2", "FC"]

import torch
import torch.nn as nn
class VGGNet(nn.Module):
    def __init__(self, VGG, num_classes):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        layers = []
        in_channels = 3

        for args in VGG :
        	if args == 'M':
        		layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        	elif args =='M5':
        		layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        	elif args =='FC1'
        		layers.append(nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size =3, padding = 6, dilation = 6))
        		layers.append(nn.ReLU(True))
        	elif args =='FC2':
        		layers.append(nn.Conv2d(1024, 1024, kernel_size =1 ))
        		layers.append(nn.ReLU(True))
        	elif args = 'FC':
        		layers.append(nn.Conv2d(1024, self.num_classes, kernel_size=1))
        	else :
        		layers.append(nn.Conv2d(in_channels = in_channels, out_channels = args, kernel_size =3 , padding = 1))
        		layers.append(nn.ReLU(True))
        		in_channels = args

        self.vgg = nn.ModuleList(layers)

    def forward(self, x):
    	for layer in self.vgg:
    		x = layer(x)
    	out = x
    	return x


