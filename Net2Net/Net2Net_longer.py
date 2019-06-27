def Net2LongerNet(self, current_layer, new_length, noise=True):
	#增加卷积核的大小，比如从 3*3 to 5*5， 7*7
	#对全链接没有用 
    w1 = current_layer.weight.data
    half_length_increment = (new_length - w1.size(2)) / 2
    new_w1 = torch.FloatTensor(w1.size(0), w1.size(1), new_length, new_length).zero_()
    #3 4 维度把卷积核大小变成新的 
    new_w1.narrow(2, half_length_increment, w1.size(2)).narrow(3, half_length_increment, w1.size(3)).copy_(w1)
    #对新的weight 改变 copy 到中心 
    if noise:
        added_noise = np.random.normal(scale=5e-2 * new_w1.std(), size=list(new_w1.size()))
        new_w1 += torch.FloatTensor(added_noise).type_as(new_w1)
    new_w1.narrow(2, half_length_increment, w1.size(2)).narrow(3, half_length_increment, w1.size(3)).copy_(w1)
    current_layer.weight.data = new_w1
    current_layer.kernel_size = (new_length, new_length)
    current_layer.padding = ((new_length - 1) / 2, (new_length - 1) / 2)
