import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import config
from src.data import PascalVOC, make_view, model_input, normalize_image, voc_view
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.partial import partial_assignment
from src.transport import log_partial_transport, transport_odds
from train import build_model


def filtered(view, common):
    ids = [i for i, n in enumerate(view['names']) if n in common]
    return make_view(view['image'], view['points'][ids], [view['names'][i] for i in ids])


def spair_pair(root, item):
    ids = [str(x) for x in item['kps_ids']]
    if len(ids) < 3:
        return None, 'spair_fewer_than_three', 0
    crops = []
    for side in ('src', 'trg'):
        im = Image.open(Path(root) / 'JPEGImages' / item['category'] / item[side + '_imname']).convert('RGB')
        box = tuple(map(float, item[side + '_bndbox']))
        im = im.resize((256, 256), Image.Resampling.BICUBIC, box=box)
        x0, y0, x1, y1 = box
        points = torch.tensor([[(x - x0) * 256 / (x1 - x0), (y - y0) * 256 / (y1 - y0)] for x, y in item[side + '_kps']],
                              dtype=torch.float32)
        crops.append((im, points))
    inside = [i for i in range(len(ids)) if all(0 <= float(c[1][i, 0]) < 256 and 0 <= float(c[1][i, 1]) < 256 for c in crops)]
    dropped = len(ids) - len(inside)
    if len(inside) < 3:
        return None, 'spair_fewer_than_three_in_box', dropped
    kept_ids = [ids[i] for i in inside]
    views = [make_view(normalize_image(im), points[inside], kept_ids) for im, points in crops]
    if any(v is None or len(v['names']) != len(kept_ids) for v in views):
        return None, 'spair_invalid_coordinates', dropped
    return {'class': item['category'], 'left': views[0], 'right': views[1]}, None, dropped


def main():
    p = argparse.ArgumentParser(description='Evaluate a checkpoint on SPair-71k and Pascal VOC (filtered and unfiltered).')
    p.add_argument('--checkpoint', required=True, help='dustbin_state.pt from train.py fit-dustbin (or last_state.pt from train.py)')
    p.add_argument('--output', required=True, help='directory for results.json')
    p.add_argument('--spair-root', default=config.SPAIR_ROOT)
    p.add_argument('--voc-pairs-per-class', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(device); model.load_state_dict(state['model']); model.eval()
    dustbin, bias = state['dustbin'].to(device), state.get('bias')
    if bias is None:
        print('no fitted dustbin in checkpoint: unfiltered uses Hungarian', flush=True)
    result = {'settings': vars(args), 'checkpoint_step': state['step'], 'partial_bias': bias}
    counts = defaultdict(lambda: defaultdict(list)); rejected = defaultdict(int)

    def run(pairs, task):
        for start in range(0, len(pairs), args.batch_size):
            chunk = pairs[start:start + args.batch_size]
            with torch.no_grad():
                out = model(model_input(chunk, device))
            for i, pair in enumerate(chunk):
                a, b = pair['left']['names'], pair['right']['names']; n, m = len(a), len(b)
                gt = torch.tensor([[x == y for y in b] for x in a])
                if task == 'voc_unfiltered' and bias is not None:
                    score = transport_odds(log_partial_transport(out['match_logits'][i, :n, :m], dustbin)).cpu()
                    pred = partial_assignment(score, bias).bool().cpu()
                else:
                    pred = hungarian(out['ds_mat'][i, :n, :m].cpu()).bool().cpu()
                tp = int((gt & pred).sum()); pr = int(pred.sum()); true = int(gt.sum())
                precision = tp / pr if pr else 0.; recall = tp / true if true else 0.
                f1 = 2 * tp / (pr + true) if pr + true else 0.
                counts[task][pair['class']].append([precision, recall, f1, tp, pr, true])
            if start % (args.batch_size * 100) == 0:
                print(task, start + len(chunk), '/', len(pairs), flush=True)

    files = sorted((Path(args.spair_root) / 'PairAnnotation' / 'test').glob('*.json'))
    pairs = []
    for index, path in enumerate(files):
        pair, reason, dropped = spair_pair(args.spair_root, json.loads(path.read_text()))
        if dropped:
            rejected['spair_keypoints_outside_box'] += dropped
        if pair is None:
            rejected[reason] += 1; continue
        pairs.append(pair)
        if len(pairs) >= args.batch_size * 20:
            run(pairs, 'spair_filtered'); pairs.clear()
            print('SPair files processed', index + 1, '/', len(files), flush=True)
    if pairs:
        run(pairs, 'spair_filtered')
    print('SPair skips', dict(rejected), flush=True); del pairs

    ds = PascalVOC('test'); cache = {}; pairs = []
    for ci, cls in enumerate(ds.classes):
        names = ds.xml_list[ci]; tries = got = 0
        while got < args.voc_pairs_per_class:
            tries += 1
            if tries > args.voc_pairs_per_class * 1000:
                raise RuntimeError('Unable to sample ' + cls)
            a, b = random.sample(names, 2)
            for name in (a, b):
                if (ci, name) not in cache:
                    cache[ci, name] = voc_view(ds, ci, name)
            left, right = cache[ci, a], cache[ci, b]
            if left is None or right is None or len(set(left['names']) & set(right['names'])) < 3:
                continue
            pairs.append({'class': cls, 'left': left, 'right': right}); got += 1
        print('VOC sampled', cls, got, 'draws', tries, flush=True)
    filtered_pairs = []
    for pair in pairs:
        common = set(pair['left']['names']) & set(pair['right']['names'])
        filtered_pairs.append({'class': pair['class'], 'left': filtered(pair['left'], common),
                               'right': filtered(pair['right'], common)})
    run(filtered_pairs, 'voc_filtered'); run(pairs, 'voc_unfiltered')

    for task, classes in counts.items():
        rows = {}
        for cls, values in sorted(classes.items()):
            x = np.asarray(values)
            rows[cls] = {'pairs': len(x), 'mean_pair_precision': float(x[:, 0].mean()),
                         'mean_pair_recall_accuracy': float(x[:, 1].mean()), 'mean_pair_f1': float(x[:, 2].mean()),
                         'true_positives': int(x[:, 3].sum()), 'predicted': int(x[:, 4].sum()),
                         'ground_truth': int(x[:, 5].sum())}
        result[task] = {'per_class': rows,
                        'macro_class_accuracy': float(np.mean([x['mean_pair_recall_accuracy'] for x in rows.values()])),
                        'macro_class_f1': float(np.mean([x['mean_pair_f1'] for x in rows.values()])),
                        'pairs': sum(x['pairs'] for x in rows.values())}
        print(task, 'pairs', result[task]['pairs'], 'macro accuracy', result[task]['macro_class_accuracy'],
              'macro F1', result[task]['macro_class_f1'], flush=True)
    result['skipped'] = dict(rejected)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
