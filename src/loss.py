import math

import torch
import torch.nn.functional as F

from src.lap_solvers.hungarian import hungarian
from src.transport import log_partial_transport


def simclr_loss(z1, z2, temperature=0.5):
    B, N, D = z1.size()
    z1 = F.normalize(z1.view(B * N, D), dim=-1)
    z2 = F.normalize(z2.view(B * N, D), dim=-1)
    batch_size = z1.size(0)
    out = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


def known_view_loss(model, entry, mapping, dustbin, contrastive_weight=.1):
    mapping = mapping.to(entry['logits'].device)
    logp = log_partial_transport(entry['logits'], dustbin)
    rows, cols = mapping.nonzero(as_tuple=True)
    unmatched = ~mapping.bool().any(1)
    terms = torch.cat((logp[rows, cols], logp[:-1, -1][unmatched], logp[-1, :-1][~mapping.bool().any(0)]))
    a, b = entry['features']
    contrast = simclr_loss(model.mlp(a[rows])[None], model.mlp(b[cols])[None]) if len(rows) >= 2 else logp.sum() * 0
    return -terms.mean() + contrastive_weight * contrast


def assignment(scores):
    return hungarian(scores).to(scores)


def oriented(maps, i, j):
    return maps[i, j] if i < j else maps[j, i].transpose(0, 1)


def sign_row_gate(direction, supported, contradicted):
    raised, lowered = direction < 0, direction > 0
    row_ok = raised.any(1) & ~(raised & ~supported).any(1)
    keep = (raised & row_ok[:, None]) | (lowered & (row_ok[:, None] | contradicted))
    return direction * keep


def cycle_loss(scores, base, num_graphs, focal_edges, support_up=3, contradict_min=2, epsilon=.1, samples=1,
               noise_scale=1., generator=None):
    """Cycle loss, adapted from the direct-loss cycle surrogate of Cohen Indelman and Hazan,
    "Learning Constrained Structured Spaces with Application to Multi-Graph Matching" (2023).
    It is a more stable version of the CLUM loss.
    """
    if num_graphs < 3 or samples < 1 or epsilon <= 0:
        raise ValueError('invalid cycle loss configuration')
    loss = next(iter(scores.values())).sum() * 0
    stats = dict(violations=0., changed=0, raised=0, raised_kept=0, lowered=0, lowered_kept=0)
    beta = math.sqrt(.2 / (math.pi ** 2))
    for edge in focal_edges:
        score = scores[edge]; i, j = edge
        path_list = [oriented(base, i, k).to(score) @ oriented(base, k, j).to(score)
                     for k in range(num_graphs) if k not in edge]
        paths = sum(path_list)
        direct = base[edge].to(score) > 0
        supported = paths >= support_up
        contradicted = direct & (paths == 0) & (paths.max(1, keepdim=True).values >= contradict_min)
        for _ in range(samples):
            if noise_scale:
                u = torch.rand(score.shape, device=score.device, dtype=score.dtype, generator=generator).clamp_(1e-7, 1 - 1e-7)
                noise = noise_scale * (-beta * torch.log(-torch.log(u)) - beta * 0.5772156649015329)
            else:
                noise = torch.zeros_like(score)
            perturbed = score.detach() + noise
            pred = assignment(perturbed)
            penalty_matrix = sum(torch.where(pred < path, path, torch.zeros_like(path)) for path in path_list)
            plus = assignment(perturbed + epsilon * penalty_matrix)
            minus = assignment(perturbed - epsilon * penalty_matrix)
            direction = (minus - plus).detach()
            stats['raised'] += int((direction < 0).sum()); stats['lowered'] += int((direction > 0).sum())
            direction = sign_row_gate(direction, supported, contradicted)
            stats['raised_kept'] += int((direction < 0).sum()); stats['lowered_kept'] += int((direction > 0).sum())
            loss = loss + (score * direction).sum() / (len(focal_edges) * samples)
            stats['violations'] += float(penalty_matrix.sum())
            stats['changed'] += int(bool(direction.abs().sum()))
    return loss, stats
