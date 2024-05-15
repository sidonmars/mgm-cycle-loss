RESCALE = (256, 256)
NORM_MEANS = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

FEATURE_CHANNEL = 512
SK_TAU = 0.05
SK_ITER_NUM = 20
SK_EPSILON = 1e-10
GNN_LAYER = 3
GNN_FEAT = [16, 16, 16]
SK_EMB = 1
EDGE_EMB = False

VOC_KPT_ANNO_DIR = 'data/voc2011_keypoints_Feb2012/annotations/'
VOC_ROOT_DIR = 'data/VOCtrainval_25-May-2011/TrainVal/VOCdevkit/VOC2011/'
VOC_SET_SPLIT = 'data/PascalVOC/voc2011_pairs.npz'
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']
CACHE_PATH = 'data/cache'

SPAIR_ROOT = 'data/SPair-71k'
