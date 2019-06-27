def Net2DeeperNet(self, current_layer_id, new_layer_fil_size=1, new_layer_type='Conv', noise=True):
    	#对于全链接 增加网络深度，
    	#对于卷积，增加卷积层数量
        #用一个对角矩阵初始化一个layer 
        current_layer = self.part1[current_layer_id]
        if "Linear" in new_layer_type:
            new_layer = nn.Linear(current_layer.out_features, current_layer.out_features)
            new_layer.weight.data.copy_(torch.eye(current_layer.out_features))
            #全链接 方法 用一个对角矩阵初始化一个layer 
            new_layer.bias.data.zero_()
            bnorm = nn.BatchNorm1d(current_layer.out_features)
            bnorm.weight.data.fill_(1)
            bnorm.bias.data.fill_(0)
            bnorm.running_mean.fill_(0)
            bnorm.running_var.fill_(1)
        elif "Conv" in new_layer_type:
            new_kernel_size = new_layer_fil_size
            new_layer = nn.Conv2d(current_layer.out_channels,        \
                                  current_layer.out_channels,        \
                                  kernel_size=new_kernel_size,       \
                                  padding=(new_kernel_size - 1) / 2)
            new_layer.weight.data.zero_()
            center = new_layer.kernel_size[0] // 2 + 1
            for i in range(0, current_layer.out_channels):
                new_layer.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, center - 1, 1).narrow(3, center - 1, 1).fill_(1)
                #对角矩阵初始化卷积层参数
            if noise:
                added_noise = np.random.normal(scale=5e-2 * new_layer.weight.data.std(), size=list(new_layer.weight.size()))
                new_layer.weight.data += torch.FloatTensor(added_noise).type_as(new_layer.weight.data)
            new_layer.bias.data.zero_()
            bnorm = nn.BatchNorm2d(new_layer.out_channels)
            bnorm.weight.data.fill_(1)
            bnorm.bias.data.fill_(0)
            bnorm.running_mean.fill_(0)
            bnorm.running_var.fill_(1)

        sub_part1 = list(self.part1.children())[0: current_layer_id + 3]
        sub_part2 = list(self.part1.children())[current_layer_id + 3:]
        sub_part1.append(new_layer)
        sub_part1.append(bnorm)
        sub_part1.append(nn.ReLU())
        sub_part1.extend(sub_part2)
        self.part1 = nn.Sequential(*sub_part1)
