from math import inf

import numpy as np
import torch
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
from scipy import stats


def multiclass_noisify(y, P, random_state):
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    for idx in np.arange(y.shape[0]):
        flipped = flipper.multinomial(1, P[y[idx], :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y


def pair_noise(labels, rate, seed, num_classes):
    P = np.eye(num_classes)
    P[0, 0], P[0, 1] = 1. - rate, rate
    for i in range(1, num_classes - 1):
        P[i, i], P[i, i + 1] = 1. - rate, rate
    P[num_classes - 1, num_classes - 1], P[num_classes - 1, 0] = 1. - rate, rate
    return multiclass_noisify(labels.numpy(), P=P, random_state=seed)


def symmetric_noise(labels, rate, seed, num_classes):
    P = np.ones((num_classes, num_classes))
    P = (rate / (num_classes - 1)) * P
    P[0, 0] = 1. - rate
    for i in range(1, num_classes - 1):
        P[i, i] = 1. - rate
    P[num_classes - 1, num_classes - 1] = 1. - rate
    return multiclass_noisify(labels.numpy(), P=P, random_state=seed)


def instance_dependent_noise(features, labels, rate, seed, num_classes, norm_std=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    flip_distribution = stats.truncnorm((0 - rate) / norm_std, (1 - rate) / norm_std, loc=rate, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    W = np.random.randn(num_classes, features.shape[1], num_classes)
    W = torch.FloatTensor(W).to(device)

    P = []
    for i, (x, y) in enumerate(zip(features, labels)):
        x = x.to(device)
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    classes = [i for i in range(num_classes)]
    return np.array([np.random.choice(classes, p=P[i]) for i in range(labels.shape[0])])


def make_noise(data, noise_type, rate, seed, val_rate):
    num_classes = data.y.unique().shape[0]
    noisy = []
    for mask, r in ((data.train_mask, rate), (data.val_mask, val_rate)):
        labels = data.y[mask]
        if noise_type == 'sym':
            noisy.append(symmetric_noise(labels, r, seed, num_classes))
        elif noise_type == 'pair':
            noisy.append(pair_noise(labels, r, seed, num_classes))
        elif noise_type == 'idn':
            noisy.append(instance_dependent_noise(data.x[mask], labels, r, seed, num_classes))
    data.y[data.train_mask] = torch.from_numpy(noisy[0])
    data.y[data.val_mask] = torch.from_numpy(noisy[1])
