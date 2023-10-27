import numpy as np
import random
from attention import *

def get_sin_enc_table(n_position, embedding_dim):
    sinusoid_table = np.zeros((n_position, embedding_dim))
    # 遍历所有位置和维度，计算角度值
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle
    # 计算正弦和余弦值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数维
    # sinusoid_table 的维度是 [n_position, embedding_dim]

    # return torch.FloatTensor(sinusoid_table)  # 返回正弦位置编码表
    return sinusoid_table

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def X2conv(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU())

class DownsampleLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownsampleLayer, self).__init__()
        self.x2conv = X2conv(in_channel, out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        out_1 = self.x2conv(x)
        out = self.pool(out_1)
        return out_1, out

class UpSampleLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSampleLayer, self).__init__()
        self.x2conv = X2conv(in_channel, out_channel)
        self.upsample = nn.ConvTranspose2d \
            (in_channels=out_channel, out_channels=out_channel // 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x, out):
        x = self.x2conv(x)
        x = self.upsample(x)

        # x.shape中H W 应与 out.shape中的H W相同
        if (x.size(2) != out.size(2)) or (x.size(3) != out.size(3)):
            # 将右侧特征H W大小插值变为左侧特征H W大小
            x = F.interpolate(x, size=(out.size(2), out.size(3)),
                              mode="bilinear", align_corners=True)

        # Concatenate(在channel维度)
        cat_out = torch.cat([x, out], dim=1)
        return cat_out

class UNet(nn.Module):
    def __init__(self, num_classes = 3):
        super(UNet, self).__init__()
        self.head = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # 创建头部卷积层
        self.headr = Resnetbasic_Block(8, 16)

        # 下采样
        self.d1 = DownsampleLayer(16, 64)  # 16-64
        self.d1r = Resnetbasic_Block(64,64)
        self.d2 = DownsampleLayer(64, 128)  # 64-128
        self.d2r = Resnetbasic_Block(128, 128)
        self.d3 = DownsampleLayer(128, 256)  # 128-256
        self.d3r = Resnetbasic_Block(256, 256)
        self.d4 = DownsampleLayer(256, 512)  # 256-512
        self.d4r = Resnetbasic_Block(512, 512)

        # 上采样
        self.u1 = UpSampleLayer(512, 1024)  # 512-1024-512
        self.u1r = Resnetbasic_Block(1024, 1024)
        self.u2 = UpSampleLayer(1024, 512)  # 1024-512-256
        self.u2r = Resnetbasic_Block(512, 512)
        self.u3 = UpSampleLayer(512, 256)  # 512-256-128
        self.u3r = Resnetbasic_Block(256, 256)
        self.u4 = UpSampleLayer(256, 128)  # 256-128-64
        self.u4r = Resnetbasic_Block(128, 128)

        # 输出:经过一个二层3*3卷积 + 1个1*1卷积
        self.x2conv = X2conv(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # 最后一个卷积层的输出通道数为分割的类别数
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x ,t) :
        x = self.headr(self.head(x),t)

        out_1, out1 = self.d1(x)
        out1 = self.d1r(out1, t)
        out_1 = self.d1r(out_1, t)
        out_2, out2 = self.d2(out1)
        out2 = self.d2r(out2, t)
        out_2 = self.d2r(out_2, t)
        out_3, out3 = self.d3(out2)
        out3 = self.d3r(out3, t)
        out_3 = self.d3r(out_3, t)
        out_4, out4 = self.d4(out3)
        out4 = self.d4r(out4, t)
        out_4 = self.d4r(out_4, t)

        # 上采样层 拼接
        out5 = self.u1r(self.u1(out4, out_4),t)
        out6 = self.u2r(self.u2(out5, out_3),t)
        out7 = self.u3r(self.u3(out6, out_2),t)
        out8 = self.u4r(self.u4(out7, out_1),t)

        # 最后的三层卷积
        out = self.x2conv(out8)
        out = self.final_conv(out)
        # print(out.shape)
        return out

if __name__ == '__main__':
    n_position = 1000; embedding_dim = 512; num_indices = 16
    temb = get_sin_enc_table(n_position,embedding_dim)
    random_indices = random.sample(range(len(temb)), num_indices)

    t = torch.LongTensor(random_indices)

    temb_list = [temb[i] for i in random_indices]
    temb_array = np.array(temb_list)
    temb_Tensor = torch.FloatTensor(temb_array)
    # print(temb_Tensor.shape) #torch.Size([16, 512])
    model = UNet()
    x = torch.randn(16,1,28,28)
    y = model(x,temb_Tensor)
    print(y.shape)