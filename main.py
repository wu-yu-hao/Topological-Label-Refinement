import argparse
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from data import load_data
from model import GCN
from utils import evaluate, load_ppr, node_difficulty, predict, training_scheduler, two_hop_edges

HYPERPARAMETERS = {
    'runs': 10, 'pretrain_epochs': 400, 'n_epoch': 500, 'lr': 0.01, 'weight_decay': 0.0005,
    'hidden': 64, 'dropout': 0.3, 'lam': 0.25, 'T': 400, 'label_smoothing': 0.0,
    'cbc_consistency_weight': 0.0, 'consistency_hops': 1, 'enhanced_epochs': 3,
    'refine_lambda': 0.5, 'refine_beta': 0.5, 'pseudo_loss_weight': 0.5,
    'refine_init_pretrained': True, 'cumulative_pseudo': True, 'confidence_weighted_pseudo': False,
}


def smooth_labels(labels, num_classes, smoothing):
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return (1 - smoothing) * one_hot + smoothing / num_classes


def supervised_loss(logits, labels, num_classes, smoothing):
    if smoothing > 0:
        return F.kl_div(F.log_softmax(logits, dim=1), smooth_labels(labels, num_classes, smoothing),
                        reduction='batchmean')
    return F.cross_entropy(logits, labels)


def consistency_loss(enhanced_model, model, data, train_subset, difficulty, edges, dropout):
    device = data.x.device
    eps = 1e-12

    was_training = model.training
    model.eval()
    with torch.no_grad():
        f_probs = torch.softmax(model(data, dropout), dim=1)
    if was_training:
        model.train()
    g_probs = torch.softmax(enhanced_model(data, dropout), dim=1)

    cb = difficulty[train_subset].clone()
    finite = torch.isfinite(cb)
    if finite.any():
        cb_min = cb[finite].min()
        cb_max = cb[finite].max()
        cb[~finite] = cb_max
    else:
        cb_min = torch.tensor(0.0, device=device)
        cb_max = torch.tensor(1.0, device=device)
    cb = (cb - cb_min) / (cb_max - cb_min + eps)

    g_i = g_probs[train_subset]
    f_i = f_probs[train_subset]
    kl_g_f = torch.sum(g_i * (torch.log(g_i + eps) - torch.log(f_i + eps)), dim=1)

    in_subset = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    in_subset[train_subset] = True
    src, dst = edges
    keep = in_subset[src]
    src = src[keep]
    dst = dst[keep]

    ppr = data.Pi[src, dst]
    g_src = g_probs[src]
    g_dst = g_probs[dst]
    kl_neighbors = torch.sum(g_src * (torch.log(g_src + eps) - torch.log(g_dst + eps)), dim=1)
    ppr_sum = torch.zeros(data.num_nodes, device=device)
    ppr_sum.index_add_(0, src, ppr)
    weights = ppr / (ppr_sum[src] + eps)
    neighbor_term = torch.zeros(data.num_nodes, device=device)
    neighbor_term.index_add_(0, src, weights * kl_neighbors)

    return torch.mean(cb * kl_g_f + (1.0 - cb) * neighbor_term[train_subset])


def initialize_refined_labels(data, num_classes, train_subset, f_probs, smoothing):
    r = torch.zeros((int(data.num_nodes), num_classes), device=data.x.device)
    if train_subset.numel() > 0:
        r[train_subset] = smooth_labels(data.y[train_subset], num_classes, smoothing)
    r[r.sum(dim=1) == 0] = f_probs[r.sum(dim=1) == 0]
    return r


def update_refined_labels(r, idx, f_probs, g_probs, beta, lam):
    blend = (1.0 - beta).unsqueeze(1) * f_probs[idx] + beta.unsqueeze(1) * g_probs[idx]
    r_new = (1.0 - lam) * r[idx] + lam * blend
    r[idx] = r_new / (r_new.sum(dim=1, keepdim=True) + 1e-12)
    return r


def split_by_agreement(pre_model, data, nodes, dropout):
    agree = predict(pre_model, data, dropout)[nodes] == data.y[nodes]
    return nodes[agree], nodes[~agree]


def run(args, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    data, num_features, num_classes = load_data(args)
    data = data.to(device)
    data.train_idx = data.train_mask.nonzero().reshape(-1,)
    data.Pi = load_ppr(args, data, device).to(device)

    difficulty = node_difficulty(args, data, device).to(device)
    difficulty[difficulty == 0] = float("inf")
    _, indices = torch.sort(difficulty[data.train_idx])
    sorted_trainset = data.train_idx[indices]
    edges = (data.edge_index[0], data.edge_index[1]) if args.consistency_hops == 1 else two_hop_edges(data)

    pre_model = GCN(num_features, num_classes, args.hidden)
    pre_model.to(device).reset_parameters()
    pre_optimizer = optim.Adam(pre_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.pretrain_epochs):
        pre_model.train()
        pre_optimizer.zero_grad()
        logits = pre_model(data, args.dropout)
        loss = supervised_loss(logits[data.train_idx], data.y[data.train_idx], num_classes, args.label_smoothing)
        loss.backward()
        pre_optimizer.step()

    model = GCN(num_features, num_classes, args.hidden)
    model.to(device).reset_parameters()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_subset, unconfident = split_by_agreement(
        pre_model, data, sorted_trainset[:int(args.lam * sorted_trainset.shape[0])], args.dropout)
    new_confident = torch.LongTensor([]).to(device)
    new_unconfident = unconfident
    size = args.lam

    with torch.no_grad():
        if args.refine_init_pretrained:
            pre_model.eval()
            f_probs = torch.softmax(pre_model(data, args.dropout), dim=1)
        else:
            f_probs = torch.softmax(model(data, args.dropout), dim=1)
    r = initialize_refined_labels(data, num_classes, train_subset, f_probs, args.label_smoothing)

    val_accs, test_accs = [], []
    for epoch in range(args.n_epoch):
        train_subset = torch.concat((train_subset, new_confident))

        enhanced_model = copy.deepcopy(model)
        enhanced_optimizer = optim.Adam(enhanced_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for _ in range(args.enhanced_epochs):
            enhanced_model.train()
            enhanced_optimizer.zero_grad()
            enhanced_logits = enhanced_model(data, args.dropout)
            enhanced_loss = F.cross_entropy(enhanced_logits[train_subset], data.y[train_subset])
            if args.cbc_consistency_weight > 0:
                enhanced_loss = enhanced_loss + args.cbc_consistency_weight * consistency_loss(
                    enhanced_model, model, data, train_subset, difficulty, edges, args.dropout)
            enhanced_loss.backward()
            enhanced_optimizer.step()
        enhanced_model.eval()
        with torch.no_grad():
            f_probs = torch.softmax(model(data, args.dropout), dim=1)
            g_probs = torch.softmax(enhanced_model(data, args.dropout), dim=1)

        pseudo_set = unconfident if args.cumulative_pseudo else new_unconfident
        if pseudo_set.numel() > 0:
            idx = torch.cat([train_subset, pseudo_set])
            beta = torch.zeros((idx.numel(),), device=device)
            beta[train_subset.numel():] = torch.tensor(args.refine_beta, device=device)
            r = update_refined_labels(r, idx, f_probs, g_probs, beta, torch.tensor(args.refine_lambda, device=device))

        model.train()
        optimizer.zero_grad()
        logits = model(data, args.dropout)
        loss = supervised_loss(logits[train_subset], data.y[train_subset], num_classes, args.label_smoothing)
        if pseudo_set.numel() > 0:
            refined = r[pseudo_set]
            if args.confidence_weighted_pseudo:
                w = refined.max(dim=1).values
                log_q = F.log_softmax(logits[pseudo_set], dim=1)
                kl = torch.sum(refined * (torch.log(refined + 1e-12) - log_q), dim=1)
                pseudo_loss = (w * kl).sum() / (w.sum() + 1e-12)
            else:
                pseudo_loss = F.kl_div(F.log_softmax(logits[pseudo_set], dim=1), refined, reduction='batchmean')
            loss = loss + args.pseudo_loss_weight * pseudo_loss
        loss.backward()
        optimizer.step()

        val_acc, test_acc = evaluate(model, data, args.dropout)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if epoch < args.T:
            new_size = training_scheduler(args.lam, epoch + 1, args.T)
            new_confident, new_unconfident = split_by_agreement(
                pre_model, data,
                sorted_trainset[int(size * sorted_trainset.shape[0]):int(new_size * sorted_trainset.shape[0])],
                args.dropout)
            if new_unconfident.numel() > 0:
                unconfident = torch.unique(torch.cat([unconfident, new_unconfident]))
            size = new_size
        else:
            new_confident = torch.LongTensor([]).to(device)
            new_unconfident = torch.LongTensor([]).to(device)

    return test_accs[-1] * 100, test_accs[int(np.argmax(val_accs))] * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cora', 'citeseer', 'pubmed', 'wikics', 'facebook', 'physics', 'dblp', 'arxiv'])
    parser.add_argument('--noise_type', type=str, required=True, choices=['sym', 'pair', 'idn'])
    parser.add_argument('--noise_rate', type=float, required=True)
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--pagerank_prob', type=float, default=0.85)
    for name, default in HYPERPARAMETERS.items():
        if isinstance(default, bool):
            parser.add_argument('--' + name, type=int, choices=[0, 1], default=int(default))
        else:
            parser.add_argument('--' + name, type=type(default), default=default)
    args = parser.parse_args()

    setting = "{}_{}".format(args.noise_type, args.noise_rate)
    print("{} {} {}".format(args.dataset, setting, {name: getattr(args, name) for name in HYPERPARAMETERS}),
          flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.out:
        open(args.out, 'w').close()
    latest, by_val = [], []
    for seed in range(args.runs):
        acc_latest, acc_by_val = run(args, seed, device)
        latest.append(acc_latest)
        by_val.append(acc_by_val)
        print("seed {}: last epoch {:.2f}, best validation epoch {:.2f}".format(seed, acc_latest, acc_by_val),
              flush=True)
        if args.out:
            with open(args.out, 'a') as f:
                f.write(json.dumps({'dataset': args.dataset, 'noise_type': args.noise_type,
                                    'noise_rate': args.noise_rate, 'seed': seed,
                                    'acc_latest': acc_latest, 'acc_by_val': acc_by_val}) + '\n')
    print("{} {}: last epoch {:.2f}±{:.2f}, best validation epoch {:.2f}±{:.2f}".format(
        args.dataset, setting, np.mean(latest), np.std(latest), np.mean(by_val), np.std(by_val)), flush=True)


if __name__ == '__main__':
    main()
