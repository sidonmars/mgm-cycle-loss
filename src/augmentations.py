import numpy as np
import torch
from PIL import Image, ImageEnhance

SIDE = 256
MIRROR_TOKENS = {'L': 'R', 'R': 'L', 'Left': 'Right', 'Right': 'Left'}
MIRROR = np.array([[-1., 0., SIDE], [0., 1., 0.], [0., 0., 1.]])


def mirror_name(name):
    return '_'.join(MIRROR_TOKENS.get(t, t) for t in name.split('_'))


def warp_pil_and_points(image, points, matrix):
    h = np.asarray(matrix, dtype=np.float64); inverse = np.linalg.inv(h); inverse /= inverse[2, 2]
    coeff = (inverse[0, 0], inverse[0, 1], inverse[0, 2], inverse[1, 0], inverse[1, 1], inverse[1, 2],
             inverse[2, 0], inverse[2, 1])
    transformed = image.transform(image.size, Image.Transform.PERSPECTIVE, coeff,
                                  resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    pts = points.detach().cpu().numpy().astype(np.float64)
    homogeneous = np.c_[pts, np.ones(len(pts))] @ h.T
    finite = np.isfinite(homogeneous).all(1) & (np.abs(homogeneous[:, 2]) > 1e-8)
    mapped = np.zeros((len(pts), 2), dtype=np.float64)
    mapped[finite] = homogeneous[finite, :2] / homogeneous[finite, 2, None]
    w, height = image.size
    valid = finite & (mapped[:, 0] >= 0) & (mapped[:, 0] < w) & (mapped[:, 1] >= 0) & (mapped[:, 1] < height)
    return transformed, torch.from_numpy(np.flatnonzero(valid)).long(), torch.tensor(mapped[valid], dtype=torch.float32)


def sample_tps_backward_map(rng, size, magnitude):
    from scipy.interpolate import RBFInterpolator
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]; grid = np.stack([xx, yy], axis=-1).astype(np.float64)
    xs = np.linspace(0, width - 1, 3); ys = np.linspace(0, height - 1, 3)
    control = np.array([(x, y) for y in ys for x in xs], dtype=float)
    offsets = np.array([[rng.uniform(-magnitude, magnitude), rng.uniform(-magnitude, magnitude)] for _ in control])
    displacement = RBFInterpolator(control, offsets, kernel='thin_plate_spline', degree=1)(
        grid.reshape(-1, 2)).reshape(height, width, 2)
    backward = grid + displacement
    dx_x = np.gradient(backward[:, :, 0], axis=1); dx_y = np.gradient(backward[:, :, 0], axis=0)
    dy_x = np.gradient(backward[:, :, 1], axis=1); dy_y = np.gradient(backward[:, :, 1], axis=0)
    if not np.isfinite(backward).all() or np.min(dx_x * dy_y - dx_y * dy_x) < .25:
        raise ValueError('folded or invalid nonlinear warp')
    return backward


def warp_backward_map_and_points(image, points, backward, iterations=40, tolerance=.2):
    from scipy.ndimage import map_coordinates
    arr = np.asarray(image.convert('RGB')); height, width = arr.shape[:2]
    backward = np.asarray(backward, dtype=np.float64)
    if backward.shape != (height, width, 2):
        raise ValueError('warp size mismatch')
    coords = [backward[:, :, 1], backward[:, :, 0]]
    warped = np.stack([map_coordinates(arr[:, :, c], coords, order=1, mode='constant', cval=0) for c in range(3)],
                      axis=-1).clip(0, 255).astype(np.uint8)
    src = points.detach().cpu().numpy().astype(np.float64); target = src.copy()
    for _ in range(iterations):
        query = [target[:, 1], target[:, 0]]
        sampled = np.stack([map_coordinates(backward[:, :, i], query, order=1, mode='nearest') for i in range(2)], axis=-1)
        target += src - sampled
    query = [target[:, 1], target[:, 0]]
    sampled = np.stack([map_coordinates(backward[:, :, i], query, order=1, mode='nearest') for i in range(2)], axis=-1)
    residual = np.linalg.norm(sampled - src, axis=1)
    valid = (np.isfinite(target).all(1) & (residual < tolerance) & (target[:, 0] >= 0) & (target[:, 0] < width - 1)
             & (target[:, 1] >= 0) & (target[:, 1] < height - 1) & (src[:, 0] >= 0) & (src[:, 0] < width)
             & (src[:, 1] >= 0) & (src[:, 1] < height))
    return Image.fromarray(warped), torch.from_numpy(np.flatnonzero(valid)).long(), torch.tensor(target[valid], dtype=torch.float32)


def tps(image, points, rng, magnitude=24., attempts=20):
    for _ in range(attempts):
        try:
            backward = sample_tps_backward_map(rng, image.size, magnitude)
        except ValueError:
            continue
        return warp_backward_map_and_points(image, points, backward)
    return image, torch.arange(len(points)), points.clone()


def photometric(image, rng):
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        if rng.random() < .8:
            image = enhancer(image).enhance(rng.uniform(.6, 1.4))
    if rng.random() < .2:
        image = image.convert('L').convert('RGB')
    return image


def known_view(image, points, rng, jitter=True):
    kept = torch.arange(len(points))
    current = points.float().clone()
    mirrored = rng.random() < .5
    if mirrored:
        image, idx, current = warp_pil_and_points(image, current, MIRROR)
        kept = kept[idx]
    image, idx, current = tps(image, current, rng)
    kept = kept[idx]
    if jitter:
        image = photometric(image, rng)
    return image, kept, current, mirrored


def view_map(source_names, kept, mirrored):
    result, names = torch.zeros(len(source_names), len(kept)), []
    index = {n: i for i, n in enumerate(source_names)}
    for j, k in enumerate(kept.tolist()):
        name = mirror_name(source_names[k]) if mirrored else source_names[k]
        names.append(name)
        if name in index:
            result[index[name], j] = 1
    return result, names
