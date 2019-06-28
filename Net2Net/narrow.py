import torch
x = torch.Tensor([[[1,2,3,4], [5,6,7,8], [9,10,11,12]],
                  [[21,22,23,24], [25,26,27,28], [29,210,211,212]]])
print(x.shape)
# torch.Size([2, 3, 4])
x.narrow(0, 1, 1)

'''
tensor([[[ 21.,  22.,  23.,  24.],
         [ 25.,  26.,  27.,  28.],
         [ 29., 210., 211., 212.]]])
'''

x = torch.Tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
x.narrow(0, 0, 2)

'''
tensor([[1., 2., 3., 4.],
        [5., 6., 7., 8.]])
'''

x.narrow(1, 0, 2)

'''
tensor([[ 1.,  2.],
        [ 5.,  6.],
        [ 9., 10.]])
'''
#torch.Tensor.narrow(dimension, start, length) → Tensor
#dimension (int) – 要进行缩小的维度
#start (int) – 开始维度索引
#length (int) – 缩小持续的长度
