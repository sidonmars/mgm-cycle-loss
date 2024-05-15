import itertools

import torch
import torch.nn as nn

from src import config
from src.backbone import VGG16_bn_final
from src.factorize_graph_matching import construct_aff_mat
from src.feature_align import feature_align
from src.gnn import GNNLayer
from src.graph_layers import InnerProductWithWeightsAffinity, SiameseNodeFeaturesToEdgeFeatures, SiameseSConvOnNodes
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.sinkhorn import Sinkhorn
from src.utils.pad_tensor import pad_tensor

FEATURE_PREFIXES = ('node_layers.', 'edge_layers.', 'final_layers.')


def normalize_over_channels(x):
    return x / torch.norm(x, dim=1, keepdim=True)


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class Model(VGG16_bn_final):
    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        node_dim = config.FEATURE_CHANNEL * 2
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=node_dim)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(total_num_nodes=node_dim)
        self.vertex_affinity = InnerProductWithWeightsAffinity(node_dim, node_dim)
        self.edge_affinity = InnerProductWithWeightsAffinity(node_dim, node_dim)
        self.rescale = config.RESCALE

        pools = [i for i, m in enumerate(self.node_layers) if isinstance(m, nn.MaxPool2d)]
        self.fine_split = pools[2]
        fine_channels = [m for m in self.node_layers[:self.fine_split] if isinstance(m, nn.Conv2d)][-1].out_channels
        self.fine_feature_proj = nn.Linear(fine_channels, config.FEATURE_CHANNEL, bias=False)
        nn.init.zeros_(self.fine_feature_proj.weight)

        self.mlp = nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.ReLU(), nn.Linear(1024, 256, bias=False))

        self.sinkhorn = Sinkhorn(max_iter=config.SK_ITER_NUM, tau=config.SK_TAU, epsilon=config.SK_EPSILON)
        self.gnn_layer = config.GNN_LAYER
        feat, emb = config.GNN_FEAT, config.SK_EMB
        for i in range(self.gnn_layer):
            if i == 0:
                layer = GNNLayer(1, 1, feat[i] + emb, feat[i], sk_channel=emb, sk_tau=config.SK_TAU,
                                 edge_emb=config.EDGE_EMB)
            else:
                layer = GNNLayer(feat[i - 1] + emb, feat[i - 1], feat[i] + emb, feat[i], sk_channel=emb,
                                 sk_tau=config.SK_TAU, edge_emb=config.EDGE_EMB)
            self.add_module('gnn_layer_{}'.format(i), layer)
        self.classifier = nn.Linear(feat[-1] + emb, 1)

    def forward(self, data_dict):
        images, points, n_points, graphs = data_dict['images'], data_dict['Ps'], data_dict['ns'], data_dict['pyg_graphs']
        global_list, graph_list, node_feature_list = [], [], []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            fine = self.node_layers[:self.fine_split](image)
            nodes = self.node_layers[self.fine_split:](fine)
            edges = self.edge_layers(nodes)
            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            U = U + self.fine_feature_proj(
                concat_features(feature_align(normalize_over_channels(fine), p, n_p, self.rescale), n_p))
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = torch.cat((U, F), dim=1)
            node_feature_list.append(node_features)

            graph.x = node_features
            graph = self.message_pass_node_features(graph)
            graph_list.append(self.build_edge_features_from_node_features(graph))
        data_dict['node_features'] = node_feature_list

        (g_1, g_2), = itertools.combinations(graph_list, 2)
        global_weights = normalize_over_channels(torch.cat(global_list, axis=-1))
        unary_affs = self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
        quadratic_affs = self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2],
                                            global_weights)
        quadratic_affs = [0.5 * x for x in quadratic_affs]

        kro_G, kro_H = data_dict['KGHs']
        Kp = torch.stack(pad_tensor(unary_affs), dim=0)
        Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
        K = construct_aff_mat(Ke, Kp, kro_G, kro_H)
        emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
        A = (K > 0).to(K.dtype)
        emb_K = K.unsqueeze(-1)
        for i in range(self.gnn_layer):
            emb_K, emb = getattr(self, 'gnn_layer_{}'.format(i))(A, emb_K, emb, n_points[0], n_points[1])
        v = self.classifier(emb)
        s = v.view(v.shape[0], points[1].shape[1], -1).transpose(1, 2)
        ss = self.sinkhorn(s, n_points[0], n_points[1], dummy_row=True)
        data_dict.update({'ds_mat': ss, 'match_logits': s, 'perm_mat': hungarian(ss, n_points[0], n_points[1])})
        return data_dict
