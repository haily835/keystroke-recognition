import torch
from torch import nn
from torch_geometric.nn import MessagePassing, HypergraphConv
from einops import rearrange
import math
import numpy as np
from torchinfo import summary

def get_hi(batch_size, num_frames):
    # Number of vertices in each graph and number of graphs
    vertices_per_graph = 42
    num_graphs = batch_size * num_frames  # Example: 3 disconnected graphs
    edges = [
        {0, 1, 2, 3, 4},     # e1
        {0, 5, 6, 7, 8},     # e2
        {9, 10, 11, 12},     # e3
        {13, 14, 15, 16},    # e4
        {0, 17, 18, 19, 20}, # e5
        {21, 22, 23, 24, 25},# e6
        {22, 26, 27, 28, 29},# e7
        {30, 31, 32, 33},    # e8
        {34, 35, 36, 37},    # e9
        {21, 38, 39, 40, 41} # e10
    ]

    # Total number of vertices in all graphs
    total_vertices = vertices_per_graph * num_graphs
    # Number of edges (same for each graph)
    num_edges = len(edges)

    # Initialize the merged incidence matrix with all zeros
    incidence_matrix = np.zeros((total_vertices, num_edges * num_graphs), dtype=int)

    # Populate the incidence matrix for each graph
    for graph_index in range(num_graphs):
        # Shift the vertices for each graph
        vertex_offset = graph_index * vertices_per_graph
        
        for col, edge in enumerate(edges):
            for vertex in edge:
                # Adjust the vertex number by the graph offset
                incidence_matrix[vertex + vertex_offset, col + graph_index * num_edges] = 1

   
    # Define the sparse incidence matrix
    # Example from before
    incidence_matrix = incidence_matrix.T

    # Initialize lists to store row indices, column indices, and data
    row_indices = []
    col_indices = []
    data = []

    # Iterate through the incidence matrix and record the non-zero entries
    for i in range(incidence_matrix.shape[0]):  # Iterate over rows (vertices)
        for j in range(incidence_matrix.shape[1]):  # Iterate over columns (hyperedges)
            if incidence_matrix[i, j] == 1:
                row_indices.append(i)  # Store row index (vertex)
                col_indices.append(j)  # Store column index (hyperedge)
                data.append(1)          # Store the data (value)

    return torch.tensor([col_indices, row_indices])

# Hypergraph convolution with temperal attention.
class HCTA(nn.Module):
    def __init__(self, in_channels, n_joints, out_channels):
        super().__init__()
        self.hc = HypergraphConv(in_channels=in_channels, out_channels=out_channels) # return [nodes, outfeatures]
        self.conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1,1), stride=(1,8))
        embed_dim = n_joints * out_channels // 8
        self.ta = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True) # return [time, outfeatures]

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.proj = nn.Linear(in_channels, out_channels)
        
        self.fc = nn.Linear(embed_dim, embed_dim * 8)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        # x shape (Time, Nodes, Features)
        # print(x.shape, self.in_c, self.out_c)

        # print("HI", HI.get_device())
        # print("x", x.get_device())
        print(x.shape)
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
        x = self.conv(x)
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
    def __init__(self, in_channels=3, num_class=40, n_joints=42, n_layers=3, batch_size=32, num_frames=8):
        super().__init__()

        self.hi = get_hi(batch_size, num_frames)
        # print(self.hi)
        self.l1=HCTA(3, n_joints, 16)
        self.l2=HCTA(16, n_joints, 32)
        self.l3=HCTA(32, n_joints, 64)
        
        
        self.flat = nn.Flatten(1)
        self.fc = nn.Linear(64*n_joints*num_frames, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()
        
        hi = self.hi.to(x.device)
        x = self.l1(x, hi)
        # print("- L1 ", x.shape)
        x = self.l2(x, hi)
        # print("- L2 ", x.shape)
        x = self.l3(x, hi)
        # print("- L3 ", x.shape)
        
        x = self.flat(x)
        x = self.fc(x)
       
        return x
    

if __name__ == "__main__":
    x = torch.rand((32, 3, 8, 21, 2))
    model = MyModel()
    print(summary(model, (32, 3, 8, 21, 2)))
    x = model(x)