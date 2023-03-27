'''
Developer: Jiaxin Li
E-mail: 1319376761@qq.com
Github: https://github.com/chuqingi/Deep_Conv_TasNet
Description: End-to-end time-domain speech separation model
Reference:
[1] Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation
[2] An Empirical Study of Conv-TasNet
'''

import torch
import torch.nn as nn


class GlobalLayerNorm(nn.Module):

    # Calculate Global Layer Normalization 'gln'
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(1, self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(1, self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: n x N x T
        # mean, var: n x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        # n x N x T
        if self.elementwise_affine:
            x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.Module):

    # Calculate Cumulative Layer Normalization 'cln'
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: n x N x T
        # mean, var: n x 1 x T
        mean = torch.mean(x, 1, keepdim=True)
        var = torch.mean((x - mean) ** 2, 1, keepdim=True)
        # n x N x T
        if self.elementwise_affine:
            x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


def select_norm(norm, dim):
    if norm == 'gln':
        return GlobalLayerNorm(dim)
    if norm == 'cln':
        return CumulativeLayerNorm(dim)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Encoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.PReLU()
        )

    def forward(self, x):
        # x: n x Len => n x 1 x Len => n x N(out_channels) x T
        if x.dim() != 3:
            x = torch.unsqueeze(x, 1)
        x = self.sequential(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Decoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        )

    def forward(self, x):
        # n x N(in_channels) x T => n x 1 x Len => n x Len
        x = self.sequential(x)
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(nn.Module):

    #  Consider only residual links, no consider Skip-connection
    def __init__(self, in_channels, out_channels, kernel_size, dilation, norm, causal):
        super(Conv1D_Block, self).__init__()
        # conv 1 x 1
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.PReLu_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        # not causal don't need to padding, causal need to pad + 1 = kernel_size
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        # depthwise convolution
        self.Dw_conv = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation, groups=out_channels)
        self.PReLu_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        # pointwise convolution
        self.Pw_conv = nn.Conv1d(out_channels, in_channels, 1)
        self.causal = causal

    def forward(self, x):
        # x: n x B(in_channels) x T => n x B x T
        c = self.conv1x1(x)  # n x B_O x T
        c = self.PReLu_1(c)
        c = self.norm_1(c)
        c = self.Dw_conv(c)  # n x B_O x T
        # causal: n x B_O x (pad + T)
        # non-causal: # n x B_O x T
        if self.causal:
            c = c[:, :, :-self.pad]
        c = self.PReLu_2(c)
        c = self.norm_2(c)
        c = self.Pw_conv(c)  # n x B x T
        return x + c


class Separation(nn.Module):

    def __init__(self, R, X, B, H, P, norm, causal):
        super(Separation, self).__init__()
        self.separation = nn.ModuleList([])
        for i in range(R):
            for j in range(X):
                self.separation.append(Conv1D_Block(B, H, P, 2 ** j, norm, causal))

    def forward(self, x):
        # x: n x B x T => n x B x T
        for i in range(len(self.separation)):
            x = self.separation[i](x)
        return x


class DeepConvTasNet(nn.Module):
    '''
       N	   Number of filters in autoencoder
       L	   Length of the filters (in samples)
       B	   Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       H	   Number of channels in convolutional blocks
       P	   Kernel size in convolutional blocks
       X	   Number of convolutional blocks in each repeat
       R	   Number of repeats
       norm    The type of normalization(gln, cln, bn)
       causal  Two choice(causal or non-causal)
    '''
    def __init__(self, N=512, L=16, B=128, H=512, P=3, X=8, R=3, norm='gln', num_spks=2, activate='relu', causal=False):
        super(DeepConvTasNet, self).__init__()
        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation = active_f[activate]
        self.num_spks = num_spks
        # Encoder
        # n x 1 x Len => n x N x T
        self.encoder = Encoder(1, N, L, L // 2)
        # Layer Normalization of Separation
        # n x N x T => n x N x T
        self.LayerN_S = select_norm('cln', N)
        # Conv 1 x 1 of Separation
        # n x N x T => n x B x T
        self.BottleN_S = nn.Conv1d(N, B, 1)
        # Separation block
        # n x B x T => n x B x T
        self.separation = Separation(R, X, B, H, P, norm, causal)
        # Get masks
        # n x B x T => n x 2*N x T
        self.gen_masks = nn.Conv1d(B, num_spks * N, 1)
        # Decoder
        # n x N x T => n x 1 x Len
        self.decoder = Decoder(N, 1, L, L // 2)

    def forward(self, x):
        # n x 1 x Len => n x N x T
        w = self.encoder(x)
        # n x N x T => n x N x T
        e = self.LayerN_S(w)
        # n x N x T => n x B x T
        e = self.BottleN_S(e)
        # n x B x T => n x B x T
        e = self.separation(e)
        # n x B x T => n x num_spk*N x T
        m = self.gen_masks(e)
        # n x N x T x num_spks
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        # num_spks x n x N x T
        m = self.activation(torch.stack(m, dim=0))
        d = [w * m[i] for i in range(self.num_spks)]
        # num_spks x n x 1 x Len
        s = [self.decoder(d[i]) for i in range(self.num_spks)]
        return s


def check_parameters(net):
    #  Returns module parameters. Mb
    all_params = sum(param.numel() for param in net.parameters())
    return all_params / 10 ** 6


def test_net():
    x = torch.randn(1, 32000)
    net = DeepConvTasNet()
    s = net(x)
    print(net)
    print('Params: ', str(check_parameters(net)) + ' Mb')
    print(x.shape)
    print(s[0].shape)
    print(s[1].shape)


if __name__ == '__main__':
    test_net()
