import torch
from torch import nn
from torch_geometric.nn import MessagePassing, HypergraphConv
from einops import rearrange
import math

# Hypergraph convolution with temperal attention.
class HCTA(nn.Module):
    def __init__(self, in_channels, n_joints, out_channels):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.hc = HypergraphConv(in_channels=in_channels, out_channels=out_channels) # return [nodes, outfeatures]
        self.ta = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4) # return [time, outfeatures]
        self.W_q = nn.Linear(n_joints * out_channels, out_channels)
        self.W_k = nn.Linear(n_joints * out_channels, out_channels)
        self.W_v = nn.Linear(n_joints * out_channels, out_channels)
        
        self.fc = nn.Linear(out_channels, n_joints * out_channels)

    def forward(self, x: torch.Tensor, HI=torch.tensor([[ 0,  1,  2,  3,  4,  0,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 22, 26, 27, 28, 29, 30, 31, 32,
         33, 34, 35, 36, 37, 21, 38, 39, 40, 41],
        [ 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,
          4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,
          7,  8,  8,  8,  8,  9,  9,  9,  9,  9]])) -> torch.Tensor:
        # x shape (Time, Nodes, Features)
        # print(x.shape, self.in_c, self.out_c)

        # print("HI", HI.get_device())
        # print("x", x.get_device())
        T, N, F = x.size()
        x = torch.stack([self.hc(g, HI) for g in x])
        
        x = rearrange(x, 't v c -> t (v c)')
        # print(x.shape)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        x, _ = self.ta(q, k, v)
        x = self.fc(x)
        x = x.view(T, N, -1)
        # Reshape back the output to match the batch size
        return x

class MyModel(nn.Module):
    def __init__(self, in_channels=3, num_class=40, n_joints=42, n_layers=3):
        super().__init__()

        layers = []
        last_dim = in_channels
        for i in range(n_layers):
            layers.append(
                HCTA(last_dim, n_joints, 16 * (i + 1))
            )
            last_dim = 16 * (i + 1)
        flatten_layer = nn.Flatten(0)
        
        layers.append(flatten_layer)

        fc = nn.Linear(last_dim*n_joints*8, num_class)
        nn.init.normal_(fc.weight, 0, math.sqrt(2. / num_class))
        layers.append(fc)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'n c t v m -> n t (v m) c')
        # print(x.shape)
        x = torch.stack([self.network(b) for b in x])
        return x