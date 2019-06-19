#!/usr/bin/env python
# coding: utf-8

# In[28]:


import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable


# In[29]:


transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5], std=[0.5])])

data_train = datasets.MNIST(root = "./data", transform = transform, train=True, download=True)
data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)


# In[30]:


#数据装载， 现在我还不清楚什么情况 但这么写对的
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                                shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = 64,
                                               shuffle = True)


# In[31]:


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Conv2d(64,  128,kernel_size=3,stride=1,padding=1),
                                                     torch.nn.ReLU(),
                                                     torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128, 1024),
                                                      torch.nn.ReLU(),
                                                      torch.nn.Dropout(p=0.5),
                                                      torch.nn.Linear(1024, 10))
    def forward(self,  x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x


# In[32]:


model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# In[33]:


print(model)


# In[34]:


import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:





# In[37]:


total_step = len(data_loader_train)
for epoch in range(2):
    for i, (images, labels) in enumerate(data_loader_train):  
        # Move tensors to the configured device
        images = images
        labels = labels
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, 2, i+1, total_step, loss.item()))


# In[45]:


testing_correct = 0
for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
print("Test Accuracy is:{:.4f}".format(100*testing_correct/len(data_test)))


# In[48]:


data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                          batch_size = 4,
                                          shuffle = True)
X_test, y_test = next(iter(data_loader_test))
inputs = Variable(X_test)
pred = model(inputs)
_,pred = torch.max(pred, 1)

print("Predict Label is:", [ i for i in pred.data])
print("Real Label is:",[i for i in y_test])

img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
plt.imshow(img)


# In[ ]:




