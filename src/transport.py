import math

import torch

TEMPERATURE, ITERATIONS = .05, 50


def log_partial_transport(logits, dustbin_logit, temperature=TEMPERATURE, iterations=ITERATIONS):
    if logits.ndim != 2 or min(logits.shape) < 1 or temperature <= 0 or iterations < 1:
        raise ValueError('Expected nonempty 2D logits, positive temperature and iterations')
    n, m = logits.shape
    dust = torch.as_tensor(dustbin_logit, device=logits.device, dtype=logits.dtype)
    if dust.numel() != 1:
        raise ValueError('Dustbin logit must be scalar')
    upper = torch.cat((logits, dust.expand(n, 1)), dim=1)
    bottom = dust.expand(1, m + 1)
    z = torch.cat((upper, bottom), dim=0) / temperature
    log_total = math.log(n + m)
    log_mu = torch.cat((z.new_full((n,), -log_total), z.new_tensor([math.log(m) - log_total])))
    log_nu = torch.cat((z.new_full((m,), -log_total), z.new_tensor([math.log(n) - log_total])))
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(z + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(z + u[:, None], dim=0)
    return z + u[:, None] + v[None, :] + log_total


def transport_odds(logp):
    return logp[:-1, :-1] - logp[:-1, -1, None] - logp[-1, None, :-1]
