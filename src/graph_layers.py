import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv


class SConv(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.num_layers = 2
        self.convs = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(SplineConv(input_features, output_features, dim=2, kernel_size=5, aggr='max'))
            input_features = output_features
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        xs = [x]
        for conv in self.convs[:-1]:
            xs += [F.relu(conv(xs[-1], edge_index, edge_attr))]
        xs += [self.convs[-1](xs[-1], edge_index, edge_attr)]
        return xs[-1]


class SiameseSConvOnNodes(torch.nn.Module):
    def __init__(self, input_node_dim):
        super().__init__()
        self.num_node_features = input_node_dim
        self.mp_network = SConv(input_features=self.num_node_features, output_features=self.num_node_features)

    def forward(self, graph):
        old_features = graph.x
        result = self.mp_network(graph)
        graph.x = old_features + 0.1 * result
        return graph


class SiameseNodeFeaturesToEdgeFeatures(torch.nn.Module):
    def __init__(self, total_num_nodes):
        super().__init__()
        self.num_edge_features = total_num_nodes

    def forward(self, graph):
        return [self.vertex_attr_to_edge_attr(g) for g in graph.to_data_list()]

    @staticmethod
    def vertex_attr_to_edge_attr(graph):
        flat_edges = graph.edge_index.transpose(0, 1).reshape(-1)
        vertex_attrs = torch.index_select(graph.x, dim=0, index=flat_edges)
        new_shape = (graph.edge_index.shape[1], 2, vertex_attrs.shape[1])
        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose(0, 1)
        graph.edge_attr = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        return graph


class InnerProductWithWeightsAffinity(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.d = output_dim
        self.A = torch.nn.Linear(input_dim, output_dim)

    def _forward(self, X, Y, weights):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        coefficients = torch.tanh(self.A(weights))
        res = torch.matmul(X * coefficients, Y.transpose(0, 1))
        return torch.nn.functional.softplus(res) - 0.5

    def forward(self, Xs, Ys, Ws):
        return [self._forward(X, Y, W) for X, Y, W in zip(Xs, Ys, Ws)]
