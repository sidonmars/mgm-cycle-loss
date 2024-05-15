import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from src.augmentations import known_view, view_map
from src.data import PascalVOC, make_view, model_input, normalize_image, subset_view, voc_pil, voc_view
from src.loss import assignment, cycle_loss, known_view_loss
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.partial import partial_assignment
from src.model import FEATURE_PREFIXES, Model
from src.transport import ITERATIONS, TEMPERATURE, log_partial_transport, transport_odds


def build_model(device, matcher_seed=0):
    torch.manual_seed(matcher_seed)
    return Model().to(device)


def set_train_mode(model):
    model.train()
    for name in ('node_layers', 'edge_layers', 'final_layers'):
        getattr(model, name).eval()


def atomic_save(data, path):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


class GroupSampler:
    def __init__(self, dataset, graphs, min_common, known_views, seed):
        self.dataset, self.graphs, self.min_common, self.known_views = dataset, graphs, min_common, known_views
        self.images = {cls: {name: (ci, name) for name in dataset.xml_list[ci]} for ci, cls in enumerate(dataset.classes)}
        self.group_rng, self.view_rng = random.Random(seed), random.Random(seed + 1)
        self.classes = sorted(c for c, v in self.images.items() if len(v) >= graphs)
        self.cache = {}

    def load(self, cls, ident):
        if (cls, ident) not in self.cache:
            self.cache[cls, ident] = voc_pil(self.dataset, *self.images[cls][ident])
        return self.cache[cls, ident]

    def sample(self):
        for _ in range(1000):
            cls = self.group_rng.choice(self.classes)
            for _ in range(200):
                ids = self.group_rng.sample(sorted(self.images[cls]), self.graphs)
                raw = [self.load(cls, i) for i in ids]
                common = set.intersection(*(set(r[2]) for r in raw))
                if len(common) >= self.min_common:
                    return self.build(cls, ids, raw, sorted(common))
            self.classes.remove(cls)
        raise RuntimeError('no class yields groups')

    def build(self, cls, ids, raw, common):
        views, pils = [], []
        for image, points, names in raw:
            index = [names.index(n) for n in common]
            self.group_rng.shuffle(index)
            pts = points[index]; pils.append((image, pts))
            views.append(make_view(normalize_image(image), pts, [names[i] for i in index]))
        known = []
        for _ in range(self.known_views):
            for _ in range(10):
                source = self.view_rng.randrange(len(views))
                image, kept, mapped, mirrored = known_view(pils[source][0], pils[source][1], self.view_rng)
                if len(kept) < 3:
                    continue
                order = list(range(len(kept))); self.view_rng.shuffle(order)
                kept, mapped = kept[order], mapped[order]
                mapping, names = view_map(views[source]['names'], kept, mirrored)
                if mapping.sum() < 3:
                    continue
                view = make_view(normalize_image(image), mapped, names)
                if view is not None and len(view['names']) == len(kept):
                    views.append(view)
                    known.append((source, len(views) - 1, mapping))
                    break
        return {'class': cls, 'ids': ids, 'views': views, 'known': known, 'common': len(common)}


def forward_edges(model, group, edges, device, chunk, grad):
    out = {}
    for start in range(0, len(edges), chunk):
        part = edges[start:start + chunk]
        pairs = [{'left': group['views'][i], 'right': group['views'][j]} for i, j in part]
        with torch.set_grad_enabled(grad):
            result = model(model_input(pairs, device))
        n1 = [len(group['views'][i]['names']) for i, _ in part]; n2 = [len(group['views'][j]['names']) for _, j in part]
        f1 = torch.split(result['node_features'][0], n1); f2 = torch.split(result['node_features'][1], n2)
        for k, e in enumerate(part):
            out[e] = {'logits': result['match_logits'][k, :n1[k], :n2[k]], 'features': (f1[k], f2[k])}
    return out


def backbone_params(model):
    return [p for n, p in model.named_parameters() if n.startswith(FEATURE_PREFIXES)]


def add_train_arguments(p):
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--seed', type=int, default=123, help='group, view and noise sampling')
    p.add_argument('--matcher-seed', type=int, default=0, help='initialization of the random matcher')
    p.add_argument('--resume', default=None, help='checkpoint to continue from; --steps is the new total')
    p.add_argument('--checkpoint-every', type=int, default=500, help='also save a resumable checkpoint.pt every N steps')
    p.add_argument('--graphs', type=int, default=5)
    p.add_argument('--known-views', type=int, default=1)
    p.add_argument('--min-common', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--backbone-ratio', type=float, default=.01, help='backbone learning rate relative to --lr')
    p.add_argument('--cycle-start', type=int, default=500)
    p.add_argument('--cycle-ramp', type=int, default=500)
    p.add_argument('--cycle-weight', type=float, default=1.)
    p.add_argument('--contrastive-weight', type=float, default=.1, help='weight of the contrastive term of the known-view loss')
    p.add_argument('--ema', type=float, default=.99)
    p.add_argument('--support-up', type=int, default=3)
    p.add_argument('--contradict-min', type=int, default=2)
    p.add_argument('--samples', type=int, default=2)
    p.add_argument('--epsilon', type=float, default=.1)
    p.add_argument('--noise-scale', type=float, default=1.)
    p.add_argument('--chunk', type=int, default=5)


def train(args):
    torch.set_num_threads(4); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / 'settings.json').write_text(json.dumps(vars(args), indent=2) + '\n')

    model = build_model(device, args.matcher_seed)
    resumed, start_step = None, 1
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resumed['model']); start_step = resumed['step'] + 1
        if start_step > args.steps:
            raise SystemExit('--steps must exceed the resumed step %d' % resumed['step'])
    model.requires_grad_(True)
    dustbin = torch.nn.Parameter(torch.zeros((), device=device))
    matcher = [prm for n, prm in model.named_parameters() if not n.startswith(FEATURE_PREFIXES)]
    optimizer = torch.optim.Adam([{'params': matcher, 'lr': args.lr}, {'params': [dustbin], 'lr': 1e-3},
                                  {'params': backbone_params(model), 'lr': args.lr * args.backbone_ratio}])
    if resumed is not None:
        dustbin.data.copy_(resumed['dustbin'].to(device))
        optimizer.load_state_dict(resumed['optimizer'])
    backbone_init = [prm.detach().clone() for prm in backbone_params(model)]
    ema = copy.deepcopy(model).requires_grad_(False)

    sampler = GroupSampler(PascalVOC('train'), args.graphs, args.min_common, args.known_views, args.seed + start_step - 1)
    generator = torch.Generator(device=device).manual_seed(args.seed + start_step - 1)
    log = (out / 'training_metrics.jsonl').open('a' if resumed is not None else 'w'); started = time.time()
    for step in range(start_step, args.steps + 1):
        group = sampler.sample(); V = len(group['views'])
        edges = list(combinations(range(V), 2))
        known = {(i, j): m for i, j, m in group['known']}
        focal = [e for e in edges if e not in known]
        ramp = 0. if step <= args.cycle_start else min(1., (step - args.cycle_start) / max(args.cycle_ramp, 1))
        cycle_grads, stats = {}, {}
        if ramp > 0:
            set_train_mode(model)
            student = {e: v['logits'].detach() for e, v in forward_edges(model, group, edges, device, args.chunk, False).items()}
            ema.eval()
            reference = {e: v['logits'] for e, v in forward_edges(ema, group, edges, device, args.chunk, False).items()}
            base = {e: (known[e].to(device) if e in known else assignment(reference[e])) for e in edges}
            leaves = {e: student[e].clone().requires_grad_(True) for e in focal}
            surrogate, stats = cycle_loss(leaves, base, V, focal, support_up=args.support_up,
                                          contradict_min=args.contradict_min, epsilon=args.epsilon,
                                          samples=args.samples, noise_scale=args.noise_scale, generator=generator)
            surrogate.backward()
            cycle_grads = {e: leaves[e].grad * (args.cycle_weight * ramp) for e in focal if leaves[e].grad is not None}
        set_train_mode(model); optimizer.zero_grad(set_to_none=True)
        known_total = 0.
        for start in range(0, len(edges), args.chunk):
            part = edges[start:start + args.chunk]
            needed = [e for e in part if e in known or e in cycle_grads]
            if not needed:
                continue
            entries = forward_edges(model, group, needed, device, args.chunk, True)
            loss = dustbin * 0
            for e in needed:
                if e in known:
                    term = known_view_loss(model, entries[e], known[e], dustbin, args.contrastive_weight)
                    loss = loss + term / len(known); known_total += float(term) / len(known)
                if e in cycle_grads:
                    loss = loss + (entries[e]['logits'] * cycle_grads[e]).sum()
            loss.backward()
        grad_norm = math.sqrt(sum(float(prm.grad.pow(2).sum()) for prm in backbone_params(model) if prm.grad is not None))
        optimizer.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
        if step % 25 == 0 or step == 1:
            drift = math.sqrt(sum(float((prm - init).pow(2).sum()) for prm, init in zip(backbone_params(model), backbone_init)))
            scale = math.sqrt(sum(float(init.pow(2).sum()) for init in backbone_init))
            record = {'step': step, 'class': group['class'], 'views': V, 'known': len(known), 'common': group['common'],
                      'known_view_loss': known_total, 'cycle_ramp': ramp, 'dustbin': float(dustbin),
                      'backbone_grad_norm': grad_norm, 'backbone_relative_change': drift / scale,
                      'seconds': time.time() - started, **stats}
            log.write(json.dumps(record) + '\n'); log.flush()
            if step % 100 == 0 or step == 1:
                print(json.dumps(record), flush=True)
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step < args.steps:
            atomic_save({'model': model.state_dict(), 'dustbin': dustbin.detach().cpu(), 'settings': vars(args),
                         'step': step, 'optimizer': optimizer.state_dict()}, out / 'checkpoint.pt')
    atomic_save({'model': model.state_dict(), 'dustbin': dustbin.detach().cpu(), 'settings': vars(args),
                 'step': args.steps, 'optimizer': optimizer.state_dict()}, out / 'last_state.pt')


def sample_pairs(ds, image_sets, per_class, rng):
    cache, pairs = {}, []
    for ci, cls in enumerate(ds.classes):
        names = sorted(image_sets[ci]); got = tries = 0
        while got < per_class and len(names) > 1:
            tries += 1
            if tries > per_class * 1000:
                raise RuntimeError('Unable to sample ' + cls)
            a, b = rng.sample(names, 2)
            for name in (a, b):
                if (ci, name) not in cache:
                    cache[ci, name] = voc_view(ds, ci, name)
            left, right = cache[ci, a], cache[ci, b]
            if left is None or right is None or len(set(left['names']) & set(right['names'])) < 3:
                continue
            pairs.append({'class': cls, 'left': left, 'right': right}); got += 1
    return pairs


def logits_for(model, pairs, device, batch_size):
    out = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            logits = model(model_input(chunk, device))['match_logits']
            for i, p in enumerate(chunk):
                a, b = p['left']['names'], p['right']['names']
                out.append({'class': p['class'], 'logits': logits[i, :len(a), :len(b)].float().cpu(),
                            'gt': torch.tensor([[x == y for y in b] for x in a])})
    return out


def assignment_nll(logp, gt):
    matched = gt.any(1); unmatched_cols = ~gt.any(0); rows, cols = gt.nonzero(as_tuple=True)
    terms = torch.cat((logp[rows, cols], logp[:-1, -1][~matched], logp[-1, :-1][unmatched_cols]))
    return -terms.mean()


def macro_f1(records, dust, bias):
    per = defaultdict(list)
    for r in records:
        pred = partial_assignment(transport_odds(log_partial_transport(r['logits'], dust)), bias).bool()
        tp = int((pred & r['gt']).sum()); denominator = int(pred.sum()) + int(r['gt'].sum())
        per[r['class']].append(2 * tp / denominator if denominator else 0.)
    return float(np.mean([np.mean(v) for v in per.values()]))


def add_fit_dustbin_arguments(p):
    p.add_argument('--checkpoint', required=True, help='last_state.pt written by train.py')
    p.add_argument('--output', required=True, help='directory for dustbin_state.pt and dustbin_fit.json')
    p.add_argument('--fit-pairs-per-class', type=int, default=100)
    p.add_argument('--select-pairs-per-class', type=int, default=50)
    p.add_argument('--selection-image-fraction', type=float, default=.3)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=.05)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=7)


def fit_dustbin(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); rng = random.Random(args.seed)
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(device); model.load_state_dict(state['model']); model.eval()
    ds = PascalVOC('train'); fit_images, select_images = [], []
    for names in ds.xml_list:
        names = sorted(names); rng.shuffle(names); cut = max(2, round(len(names) * args.selection_image_fraction))
        select_images.append(names[:cut]); fit_images.append(names[cut:])
    fit = logits_for(model, sample_pairs(ds, fit_images, args.fit_pairs_per_class, rng), device, args.batch_size)
    select = logits_for(model, sample_pairs(ds, select_images, args.select_pairs_per_class, rng), device, args.batch_size)
    print('fit pairs', len(fit), 'selection pairs', len(select), flush=True)
    initial = float(state['dustbin']); dust = torch.nn.Parameter(torch.tensor(initial))
    opt = torch.optim.Adam([dust], lr=args.lr); history = []
    for step in range(1, args.steps + 1):
        chunk = rng.sample(fit, min(64, len(fit)))
        loss = torch.stack([assignment_nll(log_partial_transport(r['logits'], dust), r['gt']) for r in chunk]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); history.append({'step': step, 'loss': float(loss), 'dustbin': float(dust)})
        if step % 50 == 0:
            print('step', step, 'nll', round(float(loss), 4), 'dustbin', round(float(dust), 4), flush=True)
    dust = dust.detach()
    odds = np.concatenate([transport_odds(log_partial_transport(r['logits'], dust)).numpy().reshape(-1) for r in select])
    candidates = np.unique(np.concatenate((np.quantile(odds, np.linspace(.02, .98, 25)), [0.])))
    sweep = [{'bias': float(b), 'macro_f1': macro_f1(select, dust, float(b))} for b in candidates]
    best = max(sweep, key=lambda x: x['macro_f1'])
    print('selected bias', best['bias'], 'selection macro F1', best['macro_f1'], flush=True)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    atomic_save({'model': state['model'], 'dustbin': dust.clone(), 'bias': best['bias'], 'step': state['step'],
                 'settings': state['settings'], 'transport': {'temperature': TEMPERATURE, 'iterations': ITERATIONS}},
                out / 'dustbin_state.pt')
    (out / 'dustbin_fit.json').write_text(json.dumps({
        'settings': vars(args), 'source_checkpoint': args.checkpoint, 'initial_dustbin': initial,
        'fitted_dustbin': float(dust), 'selected': best, 'sweep': sweep, 'fit_pairs': len(fit),
        'selection_pairs': len(select), 'fit_history': history[::10]}, indent=2) + '\n')


def voc_groups(dataset, groups_per_class, size, seed):
    rng = random.Random(seed); groups, cache = [], {}
    for ci, cls in enumerate(dataset.classes):
        names = sorted(dataset.xml_list[ci]); made = 0
        for _ in range(groups_per_class * 50):
            if made >= groups_per_class or len(names) < size:
                break
            picked = rng.sample(names, size)
            for n in picked:
                if (ci, n) not in cache:
                    cache[ci, n] = voc_view(dataset, ci, n)
            views = [cache[ci, n] for n in picked]
            if any(v is None for v in views):
                continue
            common = sorted(set.intersection(*(set(v['names']) for v in views)))
            if len(common) < 3:
                continue
            views = [subset_view(v, common, rng) for v in views]
            if all(v is not None for v in views):
                groups.append({'class': cls, 'views': views}); made += 1
    return groups


def violation_rate(model, groups, device, batch_size=8):
    disagree = total = 0
    for g in groups:
        views = g['views']; edges = list(combinations(range(len(views)), 2))
        pairs = [{'left': views[i], 'right': views[j]} for i, j in edges]
        maps = {}
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                ds = model(model_input(pairs[start:start + batch_size], device))['ds_mat']
                for k, (i, j) in enumerate(edges[start:start + batch_size]):
                    maps[i, j] = hungarian(ds[k, :len(views[i]['names']), :len(views[j]['names'])].float().cpu()).float()
        get = lambda i, j: maps[i, j] if i < j else maps[j, i].T
        for i, j in edges:
            for k in range(len(views)):
                if k not in (i, j):
                    composed = get(i, k) @ get(k, j)
                    disagree += int((composed.argmax(1) != get(i, j).argmax(1)).sum()); total += composed.shape[0]
    return disagree / max(total, 1)


def add_select_seed_arguments(p):
    p.add_argument('--runs', nargs='+', required=True, help='run directories with last_state.pt')
    p.add_argument('--output', required=True, help='JSON report path')
    p.add_argument('--groups-per-class', type=int, default=10)
    p.add_argument('--size', type=int, default=4)
    p.add_argument('--seed', type=int, default=29)


def select_seed(args):
    torch.set_num_threads(4); device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    groups = voc_groups(PascalVOC('train'), args.groups_per_class, args.size, args.seed)
    print('groups', len(groups), flush=True)
    report = {}
    for run in map(Path, args.runs):
        model = build_model(device)
        model.load_state_dict(torch.load(run / 'last_state.pt', map_location=device, weights_only=False)['model'])
        model.eval()
        report[run.name] = {'violation_rate': violation_rate(model, groups, device), 'groups': len(groups)}
        print(run.name, 'violation rate %.4f' % report[run.name]['violation_rate'], flush=True)
    selected = min(report, key=lambda r: report[r]['violation_rate'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({'selected': selected, 'runs': report, 'settings': vars(args)}, indent=2) + '\n')
    print('selected', selected, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Self-supervised training, dustbin fitting and seed selection.')
    commands = parser.add_subparsers(dest='command', required=True)
    add_train_arguments(commands.add_parser('train', help='train the matcher'))
    add_fit_dustbin_arguments(commands.add_parser('fit-dustbin', help='fit the dustbin and partial-assignment bias'))
    add_select_seed_arguments(commands.add_parser('select-seed', help='pick a run by cycle violation'))
    args = parser.parse_args()
    {'train': train, 'fit-dustbin': fit_dustbin, 'select-seed': select_seed}[args.command](args)


if __name__ == '__main__':
    main()
