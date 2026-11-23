import os.path as osp

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import CitationFull, Coauthor, FacebookPagePage, Planetoid, WikiCS

from noise import make_noise
from utils import save_atomic

NOISE_SEED = 0
VAL_NOISE_RATE = 0.1


def load_planetoid(args):
    dataset = Planetoid(osp.join(args.data_dir, args.dataset), args.dataset, split='full')
    data = dataset[0]
    # the noisy labels are the same in every run, so generating them must not advance the random state of the run
    rng_state = np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    make_noise(data, args.noise_type, args.noise_rate, NOISE_SEED, args.noise_rate)
    np.random.set_state(rng_state[0])
    torch.set_rng_state(rng_state[1])
    torch.cuda.set_rng_state_all(rng_state[2])
    return T.NormalizeFeatures()(data), dataset.num_features, dataset.num_classes


def load_arxiv(args):
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.data_dir, transform=T.ToUndirected())
    data = dataset[0]
    data.y = data.y.reshape(-1)
    split = dataset.get_idx_split()
    for name, idx in (('train_mask', split['train']), ('val_mask', split['valid']), ('test_mask', split['test'])):
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        setattr(data, name, mask)
    cache = osp.join(args.data_dir, "arxiv_noise_{}_{}.pt".format(args.noise_type, args.noise_rate))
    if osp.exists(cache):
        data.y = torch.load(cache, map_location='cpu').clone()
    else:
        make_noise(data, args.noise_type, args.noise_rate, NOISE_SEED, VAL_NOISE_RATE)
        save_atomic(data.y.cpu(), cache)
    return data, data.x.shape[1], int(data.y.max()) + 1


def load_data(args):
    name = args.dataset
    if name in ('cora', 'citeseer', 'pubmed'):
        return load_planetoid(args)
    if name == 'arxiv':
        return load_arxiv(args)

    path = osp.join(args.data_dir, name)
    if name == 'wikics':
        dataset = WikiCS(path)
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
    else:
        if name == 'facebook':
            dataset = FacebookPagePage(path, pre_transform=T.NormalizeFeatures())
        elif name == 'physics':
            dataset = Coauthor(path, name='Physics')
        elif name == 'dblp':
            dataset = CitationFull(path, name='DBLP')
        split = T.RandomNodeSplit('random', num_train_per_class=200 if name == 'dblp' else 100,
                                  num_val=500, num_test=1000)
        data = split(dataset[0])
    make_noise(data, args.noise_type, args.noise_rate, NOISE_SEED, VAL_NOISE_RATE)
    return data, dataset.num_features, dataset.num_classes
