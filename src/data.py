import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from PIL import Image
from torchvision import transforms

from src import config
from src.build_graphs import build_graphs
from src.factorize_graph_matching import kronecker_sparse
from src.sparse_torch import CSRMatrix3d
from src.sparse_torch.csx_matrix import CSCMatrix3d


class PascalVOC:
    def __init__(self, sets):
        assert sets in ('train', 'test'), 'No match found for dataset {}'.format(sets)
        self.classes = config.VOC_CLASSES
        self.sets = sets
        self.anno_path = Path(config.VOC_KPT_ANNO_DIR)
        self.img_path = Path(config.VOC_ROOT_DIR + 'JPEGImages')
        self.ori_anno_path = Path(config.VOC_ROOT_DIR + 'Annotations')
        self.obj_resize = config.RESCALE
        cache_file = Path(config.CACHE_PATH) / ('voc_db_' + sets + '.pkl')
        if cache_file.exists():
            with cache_file.open(mode='rb') as f:
                self.xml_list = pickle.load(f)
        else:
            cache_file.parent.mkdir(exist_ok=True, parents=True)
            with np.load(config.VOC_SET_SPLIT, allow_pickle=True) as f:
                self.xml_list = f[sets]
            self.filter_list()
            with cache_file.open(mode='wb') as f:
                pickle.dump(self.xml_list, f)

    def filter_list(self):
        for cls_id in range(len(self.classes)):
            to_del = []
            for xml_name in self.xml_list[cls_id]:
                xml_comps = xml_name.split('/')[-1].strip('.xml').split('_')
                ori_xml_name = '_'.join(xml_comps[:-1]) + '.xml'
                voc_idx = int(xml_comps[-1])
                xml_file = self.ori_anno_path / ori_xml_name
                assert xml_file.exists(), '{} does not exist.'.format(xml_file)
                obj = ET.parse(xml_file.open()).getroot().findall('object')[voc_idx - 1]
                flags = [obj.find(k) for k in ('difficult', 'occluded', 'truncated')]
                if any(f is not None and int(f.text) for f in flags):
                    to_del.append(xml_name)
                elif self.classes[cls_id] == 'person' and int(xml_comps[0]) > 2008:
                    to_del.append(xml_name)
            for x in to_del:
                self.xml_list[cls_id].remove(x)

    def get_anno_dict(self, xml_name, cls):
        xml_file = self.anno_path / xml_name
        assert xml_file.exists(), '{} does not exist.'.format(xml_file)
        root = ET.parse(xml_file.open()).getroot()
        img_file = self.img_path / (root.find('./image').text + '.jpg')
        bounds = root.find('./visible_bounds').attrib
        h, w = float(bounds['height']), float(bounds['width'])
        xmin, ymin = float(bounds['xmin']), float(bounds['ymin'])
        with Image.open(str(img_file)) as img:
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))
        keypoints = []
        for keypoint in root.findall('./keypoints/keypoint'):
            attr = keypoint.attrib
            attr['x'] = (float(attr['x']) - xmin) * self.obj_resize[0] / w
            attr['y'] = (float(attr['y']) - ymin) * self.obj_resize[1] / h
            keypoints.append(attr)
        return {'image': obj, 'keypoints': keypoints, 'cls': self.classes[cls]}


def normalize_image(image):
    return transforms.Normalize(config.NORM_MEANS, config.NORM_STD)(transforms.ToTensor()(image))


def to_pyg_graph(A, P):
    rescale = max(config.RESCALE)
    edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5
    edge_index = np.nonzero(A)
    edge_attr = np.clip(edge_feat[edge_index], 0, 1)
    return pyg.data.Data(x=torch.tensor(P / rescale).to(torch.float32),
                         edge_index=torch.tensor(np.array(edge_index), dtype=torch.long),
                         edge_attr=torch.tensor(edge_attr).to(torch.float32))


def make_view(image, points, names):
    points = points.float().cpu()
    valid = torch.isfinite(points).all(1) & ((points >= 0) & (points < 256)).all(1)
    points = points[valid]
    names = [name for name, keep in zip(names, valid.tolist()) if keep]
    if len(points) < 2:
        return None
    array = points.numpy()
    adjacency, g, h, _ = build_graphs(array, len(points), stg='tri', sym=True)
    return {'image': image.cpu(), 'points': points, 'names': names,
            'G': torch.from_numpy(g), 'H': torch.from_numpy(h), 'graph': to_pyg_graph(adjacency, array)}


def subset_view(view, names, rng):
    index = [view['names'].index(n) for n in names]
    rng.shuffle(index)
    return make_view(view['image'], view['points'][index], [view['names'][i] for i in index])


def voc_view(dataset, class_id, name):
    annotation = dataset.get_anno_dict(name, class_id)
    points = torch.tensor([(float(k['x']), float(k['y'])) for k in annotation['keypoints']],
                          dtype=torch.float32).reshape(-1, 2)
    return make_view(normalize_image(annotation['image']), points, [k['name'] for k in annotation['keypoints']])


def voc_pil(dataset, class_id, xml_name):
    annotation = dataset.get_anno_dict(xml_name, class_id)
    names, points = [], []
    for k in annotation['keypoints']:
        x, y = float(k['x']), float(k['y'])
        if 0 <= x < 256 and 0 <= y < 256:
            names.append(k['name']); points.append((x, y))
    return annotation['image'], torch.tensor(points, dtype=torch.float32).reshape(-1, 2), names


def pair_input(pair):
    views = (pair['left'], pair['right'])
    return {'Ps': [v['points'] for v in views], 'ns': [torch.tensor(len(v['points'])) for v in views],
            'Gs': [v['G'] for v in views], 'Hs': [v['H'] for v in views],
            'pyg_graphs': [v['graph'] for v in views], 'images': [v['image'] for v in views]}


def _pad_stack(tensors):
    max_shape = np.array([max(int(t.shape[i]) for t in tensors) for i in range(tensors[0].dim())])
    padded = []
    for t in tensors:
        pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
        pattern[::-2] = max_shape - np.array(t.shape)
        padded.append(F.pad(t, tuple(pattern.tolist()), 'constant', 0))
    return torch.stack(padded, 0)


def collate(pairs):
    items = [pair_input(p) for p in pairs]
    batch = {}
    for key in items[0]:
        per_graph = list(zip(*[item[key] for item in items]))
        if key == 'pyg_graphs':
            batch[key] = [pyg.data.Batch.from_data_list(list(g)) for g in per_graph]
        else:
            batch[key] = [_pad_stack(list(g)) for g in per_graph]
    (G1, G2), (H1, H2) = batch['Gs'], batch['Hs']
    K1G = CSRMatrix3d([kronecker_sparse(x, y).astype(np.float32) for x, y in zip(G2, G1)])
    K1H = CSRMatrix3d([kronecker_sparse(x, y).astype(np.float32) for x, y in zip(H2, H1)]).transpose()
    batch['KGHs'] = K1G, K1H
    batch['batch_size'] = len(pairs)
    return batch


def to_device(inputs, device):
    if device.type != 'cuda':
        return inputs
    if isinstance(inputs, (list, tuple)):
        return [to_device(x, device) for x in inputs]
    if isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    if isinstance(inputs, (torch.Tensor, CSRMatrix3d, CSCMatrix3d)):
        return inputs.cuda()
    if isinstance(inputs, (pyg.data.Data, pyg.data.Batch)):
        return inputs.to('cuda')
    return inputs


def model_input(pairs, device):
    batch = to_device(collate(pairs), device)
    batch['pyg_graphs'] = [g.clone() for g in batch['pyg_graphs']]
    return batch
