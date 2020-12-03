import torch
from torch import nn
from torch.nn import functional as F
from model import common


class CARAFE(nn.Module):
    def __init__(self, in_channels, scale_factor, m_channels=48, k_encoder=3, k_up=5, conv=common.default_conv):
        super(CARAFE, self).__init__()
        self.k_up = k_up
        self.scale_factor = scale_factor
        # channel compressor
        self.compress = conv(in_channels, m_channels, 1)
        # content encoder
        self.encoder = conv(m_channels, (k_up*scale_factor)**2, k_encoder)
        self.shuffle = nn.PixelShuffle(scale_factor)
        # Extracts sliding local blocks from a batched input tensor.
        # k_up*k_up*channel x h*w
        self.unfold = nn.Unfold(kernel_size=self.k_up, stride=1, padding=self.k_up//2)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        batch, channel, h, w = x.size()
        # channel compressor
        y = self.relu(self.compress(x))  # batch x C x H x W -> batch x m_C x H x W
        # content encoder
        y = self.encoder(y)  # batch x m_C x H x W -> batch x (k_up*scale)**2 x H x W
        # batch x scale**2  x k_up**2 x H x W
        y = y.view(batch, self.scale_factor**2, self.k_up**2, h, w)
        y = F.softmax(y, dim=2)
        # batch x channel x k_up**2 x h x w
        x = self.unfold(x).view(batch, channel, self.k_up**2, h, w)
        out = []

        for i in range(self.scale_factor**2):
            # batch x 1 x k_up**2 x h x w , batch x channel x k_up**2 x h x w
            z = torch.mul(y[:, i, ...].unsqueeze(dim=1), x)
            # batch x channel x h x w
            out.append(torch.sum(z, dim=2))
        out = torch.cat(out, dim=1)
        out = self.shuffle(out)
        return out
