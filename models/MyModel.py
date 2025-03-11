import torch
from torch import nn
from torch_geometric.nn import MessagePassing, HypergraphConv
from einops import rearrange
import math
import numpy as np
from torchinfo import summary

import torch

def get_hi(batch_size, num_frames):
    vertices_per_graph = 42
    num_graphs = batch_size * num_frames
    edges_per_graph = 10
    
    # Given incidence matrix for one graph
    incidence_matrix_single = torch.tensor([
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=torch.long)
    
    row_indices, col_indices = incidence_matrix_single.nonzero(as_tuple=True)
    row_indices = row_indices.repeat(num_graphs) + torch.arange(num_graphs).repeat_interleave(len(row_indices)) * vertices_per_graph
    col_indices = col_indices.repeat(num_graphs) + torch.arange(num_graphs).repeat_interleave(len(col_indices)) * edges_per_graph
    
    return torch.stack([row_indices, col_indices])


# Hypergraph convolution with temperal attention.
class HCTA(nn.Module):
    def __init__(self, in_channels, n_joints, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.proj = nn.Linear(in_channels, out_channels)

        self.hc = HypergraphConv(in_channels=in_channels, out_channels=out_channels) # return [nodes, outfeatures]
        
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1,1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(8)

        
        embed_dim = n_joints * out_channels
        self.ta = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=6, batch_first=True) # return [time, outfeatures]

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.fc = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        # x shape (Time, Nodes, Features)
        # print(x.shape, self.in_c, self.out_c)

        # print("HI", HI.get_device())
        # print("x", x.get_device())
       #  print(x.shape)
        N, C, T, V, M = x.size()

        # Residual

        res = rearrange(x, 'n c t v m -> n c (v m) t')
        res = self.bn(res)
        res = rearrange(res, 'n c vm t -> n t vm c')
        res = self.proj(res)
        res = rearrange(res, 'n t vm c -> n t (vm c)')
        # print("Residue ", res.shape)
        

        x = rearrange(x, 'n c t v m -> (n t v m) c')
        
        x = self.hc(x, hi)
        # print("After hyperconv", x.shape)
        
        x = x.view(N, -1, T, V, M)
        x = rearrange(x, 'n c t v m -> n t (v m) c')
        # print("Before conv across node feature", x.shape)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv(x)
        x = self.bn2(x)
        # print("After conv", x.shape)
        # x = x.view(N, -1, T, V, M)
        x = rearrange(x, 'n t vm c -> n t (c vm)')
        
        # print("Input for attention", x.shape)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        x, _ = self.ta(q, k, v)
        # print("After temperal attention", x.shape)
        # x = x.view(N, T, -1, V, M)
        x = self.fc(x)
        # print("After FC", x.shape)
        x = self.act(x + res)
       
        # Reshape back the output to match the batch size
        x = x.view(N, -1, T, V, M)

        # print("Finish ", x.shape)
        return x

class MyModel(nn.Module):
    def __init__(self, in_channels=3, num_class=40, n_joints=42, n_layers=3, num_frames=8):
        super().__init__()

        # print(self.hi)
        self.l1=HCTA(3, n_joints, 4)
        self.l2=HCTA(4, n_joints, 4)
        self.l3=HCTA(4, n_joints, 4)
        self.l4=HCTA(4, n_joints, 4)
        self.l5=HCTA(4, n_joints, 4)
        # self.l4=HCTA(16, n_joints, 16)
        # self.l5=HCTA(16, n_joints, 16)
        # self.l6=HCTA(16, n_joints, 16)
        # self.l7=HCTA(16, n_joints, 16)
        # self.l8=HCTA(16, n_joints, 16)
        # self.l9=HCTA(16, n_joints, 16)
        # self.l10=HCTA(16, n_joints, 16)
       
        self.flat = nn.Flatten(1)
        self.fc = nn.Linear(4*n_joints*num_frames, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()
        
        hi = get_hi(N, T).to(x.device)
        x = self.l1(x, hi)
        x = self.l2(x, hi)
        x = self.l3(x, hi)
        x = self.l4(x, hi)
        x = self.l5(x, hi)
        # x = self.l6(x, hi)
        # x = self.l7(x, hi)
        # x = self.l8(x, hi)
        # x = self.l9(x, hi)
        # x = self.l10(x, hi)
        
        x = self.flat(x)
        x = self.fc(x)
       
        return x
    

if __name__ == "__main__":
    x = torch.rand((32, 3, 8, 21, 2))
    model = MyModel()
    print(summary(model, (32, 3, 8, 21, 2)))
    x = model(x)