import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        if in_ch >= 32:
            self.group_norm = nn.GroupNorm(32, in_ch)
        else:
            self.group_norm = nn.Identity()
        # self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h

class Resnetbasic_Block(nn.Module):
    def __init__(self, in_channes, out_channels):
        super().__init__()
        if in_channes >= 32:
            self.norm1 = nn.GroupNorm(32, in_channes)
        else:
            self.norm1 = nn.Identity()
        self.conv1 = nn.Conv2d(in_channes, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.Dropout = nn.Dropout(0.1)
        self.shortcut = nn.Conv2d(in_channes, out_channels, 1, stride=1, padding=0)

        self.timembedding = nn.Sequential(
            nn.Linear(512, out_channels),
            Swish(),
            nn.Linear(out_channels, out_channels),
        )

        self.attention = AttnBlock(out_channels)
        self.initialize()

    def initialize(self):
        for module in [self.conv1, self.conv2]:
            init.xavier_uniform_(module.weight)

    def forward(self, x, t):
        t = self.timembedding(t)
        _,c = t.shape
        t = t.view(-1,c, 1, 1)
        residual = x
        x = self.norm1(x)
        out = self.conv1(x)
        out = F.relu(self.bn1(out), inplace=True)
        out = out + t
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.Dropout(out)

        residual = self.shortcut(residual)

        out += residual
        out = self.attention(out)
        return out

if __name__ == '__main__':
    # model = AttnBlock(64)
    # x = torch.randn(16,64,28,28)
    # y = model(x)
    # print(y.shape)
    model = Resnetbasic_Block(64,128)
    x = torch.randn(16,64,28,28)
    t = torch.randn(16, 512)
    y = model(x,t)
    print(y.shape)

