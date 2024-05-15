# Cycle-loss keypoint matching

Self-supervised training of an NGMv2 graph matcher on an ImageNet VGG16-BN for
sparse keypoint matching. No training loss uses a correspondence between
different images.

## Results

Three seeds, evaluated on the CLUM protocol (SPair-71k and Pascal VOC accuracy
on shared keypoints; VOC unfiltered is F1 with all keypoints):

| Run | SPair-71k | VOC filtered | VOC unfiltered F1 |
|---|---|---|---|
| seed 123 | 69.2 | 62.4 | 43.8 |
| seed 7 | 69.4 | 63.5 | 43.4 |
| seed 99 | 69.4 | 63.3 | 43.3 |
| **mean ± sd** | **69.4 ± 0.1** | **63.1 ± 0.6** | **43.5 ± 0.3** |

## Differences from the paper

The core of the method is the same as in the paper. We added a few ideas that
stabilize training and make it less sensitive to seed initialization:

1. The original paper uses keypoint position embeddings. We found that with
   more augmentations they are no longer needed.
2. The cycle loss is slightly different from the one in the CLUM paper. It uses
   Cohen Indelman and Hazan's formulation, which is mathematically near
   identical but smooths the gradient more, which leads to more stable training.
3. The cycle paths are built from an EMA copy of the matcher instead of the
   matcher being trained, again for more stable updates.

The results can be regenerated without these changes and stay within about a
point on each metric. Training is just more stable this way.

## Setup

Python 3.10, a CUDA GPU, and the CUDA toolkit (`nvcc`): two small CUDA
extensions in `src/extension` are compiled on first use.

```bash
pip install -r requirements.txt
```

torch-geometric and torch-spline-conv must match your torch and CUDA versions;
see their install instructions. ImageNet VGG16-BN weights are downloaded by
torchvision.

### Data

```
data/
  VOCtrainval_25-May-2011/TrainVal/VOCdevkit/VOC2011/   # Pascal VOC 2011
  voc2011_keypoints_Feb2012/annotations/                # Berkeley keypoint annotations
  PascalVOC/voc2011_pairs.npz                           # train/test object split
  SPair-71k/                                            # SPair-71k (evaluation only)
```

Paths are set in `src/config.py`. The filtered VOC object lists are cached in
`data/cache` on first use.

## Reproducing

```bash
./run.sh        # PY=/path/to/python ./run.sh to choose the interpreter
```

This trains seeds 123, 7 and 99 (6000 steps each; seed 123 trains 3000 steps
and continues to 6000 from that checkpoint), fits the dustbin, evaluates, picks
a seed and writes `runs/summary.txt`. The three seeds train in parallel on one
GPU; training takes about 5–6 hours and each evaluation about 1 hour. The
script can be rerun after an interruption: finished steps are skipped and
training resumes from its latest checkpoint.

Single steps:

```bash
python train.py train --output runs/s7 --seed 7 --matcher-seed 7
python train.py fit-dustbin --checkpoint runs/s7/last_state.pt --output runs/s7/dustbin
python evaluate.py --checkpoint runs/s7/dustbin/dustbin_state.pt --output runs/s7/eval
python train.py select-seed --runs runs/s123 runs/s7 runs/s99 --output runs/seed_selection.json
```

## Acknowledgements

We thank the authors of [BBGM](https://github.com/martius-lab/blackbox-deep-graph-matching)
and [NGMv2](https://github.com/Thinklab-SJTU/ThinkMatch). The graph-matching
components (NGMv2, SplineConv message passing, affinity layers, Sinkhorn and
Hungarian solvers, sparse CUDA kernels, VOC loading) are adapted from
[ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch).
