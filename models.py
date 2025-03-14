import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        bsz, seq_len, embed_dim = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)
        Q, K, V = [
            t.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        ]
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)

        out = (attention @ V)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        return self.out(out)

class FormulaEncoder(nn.Module):
    def __init__(self, dic_size=64):
        super(FormulaEncoder, self).__init__()

        size1 = 128
        size2 = 252
        size3 = 512
        size4 = 1024

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.norm1 = nn.LayerNorm(dic_size)
        self.dense1 = nn.Linear(dic_size, size1)
        self.dense2 = nn.Linear(size1, size2)
        self.dense3 = nn.Linear(size2, size3)
        self.dense4 = nn.Linear(size3, size4)
        self.dense5 = nn.Linear(size4, 1024)

    def forward(self, x):
        x = self.norm1(x)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.relu(self.dense4(x))
        x = self.relu(self.dense5(x))
        return x


class DescryptorEncoder(nn.Module):
    def __init__(self):
        super(DescryptorEncoder, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.dense1 = nn.Linear(32, 512)
        self.dense2 = nn.Linear(512, 1024)
        self.dense3 = nn.Linear(1024, 1024)

        self.attention = SelfAttention(embed_dim=16, num_heads=8)

    def forward(self, x):
        b = x.size(0)
        if b == 0:
            return torch.zeros(0, 16, 2, 1, device=x.device)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)

        x = self.attention(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = x.flatten(start_dim=1)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        return x

class ElectronAffinityRegressor(nn.Module):
    def __init__(self):
        super(ElectronAffinityRegressor, self).__init__()

        _in_layer = 2048
        first_layer = 1024
        second_layer = 512
        third_layer = 256
        fourth_layer = 128
        fifth_layer = 64
        sixth_layer = 32
        seventh_layer = 16
        _out = 1

        self.descryptor = DescryptorEncoder()
        self.formula = FormulaEncoder(dic_size=64)

        self.dense1 = nn.Linear(_in_layer, first_layer)
        self.dense2 = nn.Linear(first_layer, second_layer)
        self.dense3 = nn.Linear(second_layer, third_layer)
        self.dense4 = nn.Linear(third_layer, fourth_layer)
        self.dense5 = nn.Linear(fourth_layer, fifth_layer)
        self.dense6 = nn.Linear(fifth_layer, sixth_layer)
        self.dense7 = nn.Linear(sixth_layer, seventh_layer)
        self.dense8 = nn.Linear(seventh_layer, _out)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, inp):
        b = inp.size(0)
        if b == 0:
            return torch.zeros(0, 1, device=inp.device)

        y = inp[:, :64]
        y = self.formula(y)

        x = inp[:, 64:]
        x = x.reshape(b, 1, 8, 4)
        x = self.descryptor(x)
        x = x.reshape(b, -1)

        out = torch.cat((x, y), dim=1)

        out = self.relu(self.dense1(out))
        out = self.dropout(out)
        out = self.relu(self.dense2(out))
        out = self.dropout(out)
        out = self.relu(self.dense3(out))
        out = self.dropout(out)
        out = self.relu(self.dense4(out))
        out = self.dropout(out)
        out = self.relu(self.dense5(out))
        out = self.dropout(out)
        out = self.relu(self.dense6(out))
        out = self.dropout(out)
        out = self.relu(self.dense7(out))
        out = self.dropout(out)
        out = self.dense8(out)
        return out

