import torch
import torch.nn as nn
from performer_pytorch import SelfAttention


class PointTransformerFeaturizer(nn.Module):
    def __init__(self, out_features=512, transposed_input: bool = False):
        super(PointTransformerFeaturizer, self).__init__()

        self.transposed_input = transposed_input

        self.out_features = out_features
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = FastSA_Layer(128)
        self.sa2 = FastSA_Layer(128)
        self.sa3 = FastSA_Layer(128)
        self.sa4 = FastSA_Layer(128)

        fuced_size = 512

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, fuced_size, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(fuced_size),
                                   nn.LeakyReLU(negative_slope=0.2))


        self.convs1 = nn.Linear(fuced_size * 2, self.out_features)


        self.relu = nn.ReLU()

    def forward(self, x):

        if not self.transposed_input:
            x = x.transpose(1, 2)

        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = x.max(2)[0]
        x_avg = x.mean(2)
        x_max_feature = x_max.view(batch_size, -1)
        x_avg_feature = x_avg.view(batch_size, -1)

        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1) # 1024 + 64
        x_global_feature = self.convs1(x_global_feature)
        return x_global_feature, 1



class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c
        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class FastSA_Layer(nn.Module):
    def __init__(self, channels):
        super(FastSA_Layer, self).__init__()
        self.attn = SelfAttention(
            dim=channels,
            heads=1,
            causal=False,
        )
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x_t = x.permute(0, 2, 1) # b, n, c
        x_r = self.attn(x_t)
        x_r = self.act(self.after_norm(self.trans_conv((x_t - x_r).permute(0, 2, 1))))
        x = x + x_r
        return x



if __name__ == '__main__':
    model = PointTransformerFeaturizer()
    x = torch.rand(2,512,3)
    print(model(x)[0].shape)