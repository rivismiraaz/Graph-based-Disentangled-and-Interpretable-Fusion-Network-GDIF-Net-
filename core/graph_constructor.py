import numpy as np
import torch
from torch_geometric.data import Data

def create_3d_grid_graph_pyg(dimensions, use_cuda=False):
    """
    Creates a PyTorch Geometric Data object representing a 3D grid graph.
    Each node is connected to its 26 neighbors.

    Args:
        dimensions (list or tuple): [width, height, depth] of the grid.
        use_cuda (bool): Whether to move tensors to the GPU.

    Returns:
        torch_geometric.data.Data: A graph data object with node positions `pos`
                                   and edge indices `edge_index`.
    """
    w, h, d = dimensions
    num_nodes = w * h * d
    
    # Create node positions (can be used as features if desired)
    x_coords = torch.arange(w, dtype=torch.float32)
    y_coords = torch.arange(h, dtype=torch.float32)
    z_coords = torch.arange(d, dtype=torch.float32)
    grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    pos = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    # Create edges (26-connectivity)
    # Map 3D coordinates to 1D index
    idx_map = torch.arange(num_nodes).view(w, h, d)
    
    edges = []
    for i in range(w):
        for j in range(h):
            for k in range(d):
                current_idx = idx_map[i, j, k]
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            
                            ni, nj, nk = i + di, j + dj, k + dk
                            
                            if 0 <= ni < w and 0 <= nj < h and 0 <= nk < d:
                                neighbor_idx = idx_map[ni, nj, nk]
                                edges.append([current_idx, neighbor_idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    graph = Data(pos=pos, edge_index=edge_index)
    
    if use_cuda and torch.cuda.is_available():
        return graph.to('cuda')
        
    return graph
