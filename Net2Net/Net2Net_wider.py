    def Net2WiderNet(self, current_layer, next_layer, new_width, bnorm=None, noise=True):
    	#增加这一层神经网络数量 对于全链接来说
    	#对于卷积来说，增加卷积核的数量 Net2Net 

        # 
        w1 = current_layer.weight.data
        w2 = next_layer.weight.data
        b1 = current_layer.bias.data

        if 'Conv' in current_layer.__class__.__name__ or 'Linear' in next_layer.__class__.__name__:
            if 'Conv' in current_layer.__class__.__name__ and 'Linear' in next_layer.__class__.__name__:
                channel_length = int(np.sqrt(w2.size(1) / w1.size(0)))
                w2 = w2.view(w2.size(0), w2.size(1) / channel_length ** 2, channel_length, channel_length)
            old_width = w1.size(0)

            if 'Conv' in current_layer.__class__.__name__:
                new_w1 = torch.FloatTensor(new_width, w1.size(1), w1.size(2), w1.size(3))
                new_w2 = torch.FloatTensor(new_width, w2.size(0), w2.size(2), w2.size(3))
            else:
                new_w1 = torch.FloatTensor(new_width, w1.size(1))
                new_w2 = torch.FloatTensor(new_width, w2.size(0))
            new_b1 = torch.FloatTensor(new_width)

            if bnorm is not None:
                new_norm_mean = torch.FloatTensor(new_width)
                new_norm_var = torch.FloatTensor(new_width)
                if bnorm.affine:
                    new_norm_weight = torch.FloatTensor(new_width)
                    new_norm_bias = torch.FloatTensor(new_width)
                #初始化 一个 w2

            #net2net paper
            w2 = w2.transpose(0, 1)
            new_w1.narrow(0, 0, old_width).copy_(w1)
            new_w2.narrow(0, 0, old_width).copy_(w2)
            new_b1.narrow(0, 0, old_width).copy_(b1)

            if bnorm is not None:
                new_norm_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
                new_norm_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
                if bnorm.affine:
                    new_norm_weight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
                    new_norm_bias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

            #我们更新w1->new_w1 在旧的w1上进行随机挑选，把w1 weight 复制给new w1的节点 (见图)
            index_set = dict()
            for i in range(old_width, new_width):
                sampled_index = np.random.randint(0, old_width)
                if sampled_index in index_set:
                    index_set[sampled_index].append(i)
                else:
                    index_set[sampled_index] = [sampled_index]
                    index_set[sampled_index].append(i)
                new_w1.select(0, i).copy_(w1.select(0, sampled_index).clone())
                new_w2.select(0, i).copy_(w2.select(0, sampled_index).clone())
                new_b1[i] = b1[sampled_index]
                if bnorm is not None:
                    new_norm_mean[i] = bnorm.running_mean[sampled_index]
                    new_norm_var[i] = bnorm.running_var[sampled_index]
                    if bnorm.affine:
                        new_norm_weight[i] = bnorm.weight.data[sampled_index]
                        new_norm_bias[i] = bnorm.bias.data[sampled_index]
            for (index, d) in index_set.items():
                div_length = len(d)
                for next_layer_index in d:
                    new_w2[next_layer_index].div_(div_length)
            current_layer.out_channels = new_width
            next_layer.in_channels = new_width

            #peusdo-ensemble 随机noise 
            if noise:
                w1_added_noise = np.random.normal(scale=5e-2 * new_w1.std(), size=list(new_w1.size()))
                new_w1 += torch.FloatTensor(w1_added_noise).type_as(new_w1)
                w2_added_noise = np.random.normal(scale=5e-2 * new_w2.std(), size=list(new_w2.size()))
                new_w2 += torch.FloatTensor(w2_added_noise).type_as(new_w2)
            new_w1.narrow(0, 0, old_width).copy_(w1)
            new_w2.narrow(0, 0, old_width).copy_(w2)
            for (index, d) in index_set.items():
                div_length = len(d)
                new_w2[index].div_(div_length)

            w2.transpose_(0, 1)
            new_w2.transpose_(0, 1)
            current_layer.weight.data = new_w1
            current_layer.bias.data = new_b1

            if 'Conv' in current_layer.__class__.__name__ and 'Linear' in next_layer.__class__.__name__:
                next_layer.weight.data = new_w2.view(next_layer.weight.data.size(0), new_width * channel_length ** 2)
                next_layer.in_features = new_width * channel_length ** 2
            else:
                next_layer.weight.data = new_w2

            if bnorm is not None:
                bnorm.running_var = new_norm_var
                bnorm.running_mean = new_norm_mean
                if bnorm.affine:
                    bnorm.weight.data = new_norm_weight
                    bnorm.bias.data = new_norm_bias
