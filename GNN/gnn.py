import torch
import torch.nn as nn
import time as t

class GraphConv(nn.Module):

    def __init__(self, A, image_size, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.A = A
        self.num_nodes = image_size*image_size
        self.in_channels = in_channels
        self.linear = nn.Sequential(nn.Linear(self.num_nodes*in_channels, self.num_nodes*out_channels), nn.ReLU())

    def forward(self, x):
        num_batches = x.size(0)
        a = self.A.unsqueeze(0).expand(num_batches, -1, -1)
        h = x.view(num_batches, -1, self.in_channels)
        AH = torch.bmm(a, h).view(num_batches, -1)
        out = self.linear(AH)
        return out

class GNN(nn.Module):

    def __init__(self, image_size=28, input_channels=1):
        super(GNN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Adj = self.adjacency_matrix_for_rectangle_grid(image_size).to(self.device)
        self.gconv1 = GraphConv(Adj, image_size, input_channels, 3)
        self.gconv2 = GraphConv(Adj, image_size, 3, 1)
        self.gconv3 = GraphConv(Adj, image_size, 1, 1)
        self.fc = nn.Linear(image_size*image_size*1, 10, bias=False)

    def forward(self, x):
        x = self.gconv1(x)
        x = self.gconv2(x)
        x = self.gconv3(x)
        x = self.fc(x)
        return x

    def adjacency_matrix_cross_connection(self, N):
        """
        cross relationship in Matrix A
                    a_{i-1}{j}
                        |
        a_{i}{j-1} - a_{i}{j} - a_{i}{j+1}
                        |
                    a_{i+1}{j}
        """
        nodes = torch.arange(N*N).view(N, N)
        A = torch.eye(N*N)
        start = t.time()
        for i in range(N):
            for j in range(N):
                if j < N-1:
                    A[nodes[i, j], nodes[i, j+1]] = 1
                if j > 0:
                    A[nodes[i, j], nodes[i, j-1]] = 1
                if i < N-1:
                    A[nodes[i, j], nodes[i+1, j]] = 1
                if i > 0:
                    A[nodes[i, j], nodes[i-1, j]] = 1
        print("Runtime to generate an adjacency matrix: {:.4f} sec" .format(t.time()-start))
        return A

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gnn = GNN().to(device)
    print(gnn)