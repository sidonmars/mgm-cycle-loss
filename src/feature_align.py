import torch
from torch import Tensor


def feature_align(raw_feature: Tensor, P: Tensor, ns_t: Tensor, ori_size: tuple, device=None) -> Tensor:
    if device is None:
        device = raw_feature.device

    batch_num = raw_feature.shape[0]
    channel_num = raw_feature.shape[1]
    n_max = P.shape[1]

    ori_size = torch.tensor(ori_size, dtype=torch.float32, device=device)
    F = torch.zeros(batch_num, channel_num, n_max, dtype=torch.float32, device=device)
    for idx, feature in enumerate(raw_feature):
        n = ns_t[idx]
        feat_size = torch.as_tensor(feature.shape[1:3], dtype=torch.float32, device=device)
        _P = P[idx, 0:n]
        interp_2d(feature, _P, ori_size, feat_size, out=F[idx, :, 0:n])
    return F


def interp_2d(z: Tensor, P: Tensor, ori_size: Tensor, feat_size: Tensor, out=None, device=None) -> Tensor:
    if device is None:
        device = z.device

    step = ori_size / feat_size
    if out is None:
        out = torch.zeros(z.shape[0], P.shape[0], dtype=torch.float32, device=device)
    for i, p in enumerate(P):
        p = (p - step / 2) / ori_size * feat_size
        out[:, i] = bilinear_interpolate(z, p[0], p[1])

    return out


def bilinear_interpolate(im: Tensor, x: Tensor, y: Tensor, device=None):
    if device is None:
        device = im.device
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[2] - 1)
    x1 = torch.clamp(x1, 0, im.shape[2] - 1)
    y0 = torch.clamp(y0, 0, im.shape[1] - 1)
    y1 = torch.clamp(y1, 0, im.shape[1] - 1)

    x0 = x0.to(torch.int32).to(device)
    x1 = x1.to(torch.int32).to(device)
    y0 = y0.to(torch.int32).to(device)
    y1 = y1.to(torch.int32).to(device)

    Ia = im[:, y0, x0]
    Ib = im[:, y1, x0]
    Ic = im[:, y0, x1]
    Id = im[:, y1, x1]

    if x0 == x1:
        if x0 == 0:
            x0 -= 1
        else:
            x1 += 1
    if y0 == y1:
        if y0 == 0:
            y0 -= 1
        else:
            y1 += 1

    x0 = x0.to(torch.float32).to(device)
    x1 = x1.to(torch.float32).to(device)
    y0 = y0.to(torch.float32).to(device)
    y1 = y1.to(torch.float32).to(device)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out
