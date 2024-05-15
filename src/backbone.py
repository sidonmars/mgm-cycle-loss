import torch.nn as nn
from torchvision import models


class VGG16_bn_final(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone(pretrained)

    @staticmethod
    def get_backbone(pretrained=True):
        conv_layers = nn.Sequential(*list(models.vgg16_bn(pretrained=pretrained).features.children()))
        conv_list = node_list = edge_list = []
        cnt_m, cnt_r = 1, 0
        for module in conv_layers:
            if isinstance(module, nn.Conv2d):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]
            if cnt_m == 4 and cnt_r == 3 and isinstance(module, nn.Conv2d):
                node_list = conv_list
                conv_list = []
            elif cnt_m == 5 and cnt_r == 2 and isinstance(module, nn.Conv2d):
                edge_list = conv_list
                conv_list = []
        assert len(node_list) > 0 and len(edge_list) > 0
        final_layers = nn.Sequential(*conv_list, nn.AdaptiveMaxPool2d((1, 1), return_indices=False))
        return nn.Sequential(*node_list), nn.Sequential(*edge_list), final_layers
