import os
#把莫烦的mnist CNN改成CIFAR10 VGG16
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import time

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE =4
LR = 0.01              # learning rate

# prepare data for CNN
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,                                     # this is training data
    transform=transform,    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=True
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE , shuffle=True)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,download=True,transform=transform)
testloader = Data.DataLoader(test_data, batch_size=2000 , shuffle=False)


def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]
 
    for i in range(num_convs - 1):  # 定义后面的许多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
 
    net.append(nn.MaxPool2d(2, 2))  # 定义池化层
    return nn.Sequential(*net)
 
# 下面我们定义一个函数对这个 vgg block 进行堆叠
def vgg_stack(num_convs, channels):  # vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]#即in_channels
        out_c = c[1]#即out_channels
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)
vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512))) 
#vgg类
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )
 
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        return x
        x = self.fc(x)


cnn = CNN()
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    cnn =cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

def line_chart(x_epoch, y_acc, picName):
    plt.figure()#创建绘图对象
    plt.plot(x_epoch, y_acc, "b--", linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.ylim(0.00, 1.00)
    plt.xlabel("Epoch")            #X轴标签
    plt.ylabel("Accuracy")               #Y轴标签
    plt.title("VGG")          #图标题
    # plt.savefig(os.path.join(codeDirRoot, "log", "pic", "resnet50%s.png"%experimentSuffix))  # 保存图
    plt.savefig(picName)  # 保存图


# training and testing
accuracy1=np.linspace(0,0,1)
for epoch in range(EPOCH):
    print("start training")
    

    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
               #b_x,b_y就是VGGLONG里的img, label
                # 有的写作data，即data=(b_x, b_y),data已经被totensor转成[4x3x32x32的张量,长度4的张量]

               #在定义torch.utils.data.Dataset的子类时，必须重载的两个函数是__len__和__getitem__，
               #且在def__getitem__中打印出数据集的下标索引，返回对应的图像（Tensor，且自带数据类型所以如需引用其纯数据还要用.numpy函数）和标记
        
        b_x=Variable(b_x,requires_grad=True)
        
        output = cnn(b_x)            # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

     
        
    total_correct = 0
    total_images = 0

    with torch.no_grad():
          for data in testloader:
              images, labels = data
              outputs = cnn(images)
              _, predicted = torch.max(outputs.data, 1)
              total_images += labels.size(0)#.size(0)取的是labels这个张量四个维度的第一个的值,但并不知道其余三个维度代表什么
              total_correct += (predicted == labels).sum().item()
    model_accuracy = total_correct / total_images * 100
    print('In Epoch {0}, model accuracy on {1} test images is {2:.2f}% and the train loss is {3:.4f}'.format(epoch, total_images, model_accuracy, loss.data.numpy()))#官方52.20%
    accuracy1=np.append(accuracy1,model_accuracy)
epoch1=np.linspace(0,EPOCH,EPOCH+1)
line_chart(epoch1,accuracy1,"./png")




