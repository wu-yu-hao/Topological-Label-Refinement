import os

import numpy as np
import torch


def training_scheduler(lam, t, T):
    return min(1, lam + (1 - lam) * t / T)


def evaluate(model, data, dropout):
    with torch.no_grad():
        model.eval()
        pred = torch.softmax(model(data, dropout), 1).argmax(1)
        val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return val_acc, test_acc


def predict(model, data, dropout):
    with torch.no_grad():
        model.eval()
        return torch.softmax(model(data, dropout), 1).argmax(1)


def save_atomic(obj, path):
    tmp = "{}.tmp.{}".format(path, os.getpid())
    torch.save(obj, tmp)
    os.replace(tmp, path)


def exact_ppr(data, pagerank_prob, device):
    pr_prob = 1 - pagerank_prob
    index = data.edge_index.cpu().numpy()
    A = np.zeros((data.num_nodes, data.num_nodes), dtype='int8')
    A[(index[0], index[1])] = 1
    A = torch.from_numpy(A).float()
    A_hat = A.to(device) + torch.eye(A.size(0)).to(device)
    D = torch.diag(torch.sum(A_hat, 1))
    D = D.inverse().sqrt()
    A_hat = torch.mm(torch.mm(D, A_hat), D)
    Pi = pr_prob * ((torch.eye(A.size(0)).to(device) - (1 - pr_prob) * A_hat).inverse())
    return Pi.cpu()


def approximate_ppr(data, device, alpha=0.1, iters=30, topk=128, batch=4096):
    N = int(data.num_nodes)
    ei = data.edge_index.to(device)
    self_loops = torch.arange(N, device=device)
    ei = torch.cat([ei, torch.stack([self_loops, self_loops])], dim=1)
    A = torch.sparse_coo_tensor(ei, torch.ones(ei.shape[1], device=device), (N, N)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    dinv_sqrt = deg.pow(-0.5)
    r, c = A.indices()
    A_hat = torch.sparse_coo_tensor(A.indices(), A.values() * dinv_sqrt[r] * dinv_sqrt[c], (N, N)).coalesce()
    At = A_hat.t().coalesce()

    rows, cols, vals = [], [], []
    for s in range(0, N, batch):
        e = min(s + batch, N)
        b = e - s
        X = torch.zeros(N, b, device=device)
        X[torch.arange(s, e, device=device), torch.arange(b, device=device)] = 1.0
        E0 = X.clone()
        for _ in range(iters):
            X = (1 - alpha) * torch.sparse.mm(At, X) + alpha * E0
        v, idx = torch.topk(X.t(), topk, dim=1)
        rows.append(torch.arange(s, e, device=device).unsqueeze(1).expand(-1, topk).reshape(-1).cpu())
        cols.append(idx.reshape(-1).cpu())
        vals.append(v.reshape(-1).cpu())
    index = torch.stack([torch.cat(rows), torch.cat(cols)])
    return torch.sparse_coo_tensor(index, torch.cat(vals), (N, N)).coalesce()


def load_ppr(args, data, device):
    ppr_file = "{}/{}_ppr.pt".format(args.data_dir, args.dataset)
    if os.path.exists(ppr_file):
        return torch.load(ppr_file, map_location='cpu')
    if args.dataset == 'arxiv':
        Pi = approximate_ppr(data, device)
    elif args.dataset in ('cora', 'citeseer', 'pubmed'):
        Pi = exact_ppr(data, args.pagerank_prob, device)
    else:
        Pi = exact_ppr(data, args.pagerank_prob, torch.device('cpu'))
    save_atomic(Pi, ppr_file)
    return Pi


def class_conditional_betweenness_centrality(data, rely_idx):
    num_classes = len(data.y.unique())
    Pi = data.Pi
    columns = []
    for c in range(num_classes):
        c_idx = torch.nonzero(data.y[rely_idx] == c).reshape(-1,)
        o_idx = torch.nonzero(data.y[rely_idx] != c).reshape(-1,)
        D = Pi[c_idx][:, o_idx]
        M = 1.0 / (D + 1e-16)
        A = Pi[c_idx, :]
        B = Pi[:, o_idx]
        MBt = M @ B.t()
        scores = (A * MBt).sum(0)
        columns.append(scores / c_idx.shape[0])
    CBC = torch.stack(columns, dim=1)
    CBC = CBC / (CBC.max(0).values - CBC.min(0).values)
    return CBC.sum(1)


def sparse_class_conditional_betweenness_centrality(Pi, y, train_idx):
    num_classes = int(y.max()) + 1
    N = Pi.shape[0]
    y_tr = y[train_idx]
    columns = []
    for c in range(num_classes):
        c_idx = train_idx[y_tr == c]
        o_idx = train_idx[y_tr != c]
        if c_idx.numel() == 0 or o_idx.numel() == 0:
            columns.append(torch.zeros(N))
            continue
        A = Pi.index_select(0, c_idx).to_dense()
        D = A[:, o_idx]
        M = 1.0 / (D + 1e-16)
        B = Pi.index_select(1, o_idx).coalesce()
        MBt_t = torch.sparse.mm(B, M.t())
        scores = (A * MBt_t.t()).sum(0)
        columns.append(scores / c_idx.shape[0])
    CBC = torch.stack(columns, dim=1)
    rng = CBC.max(0).values - CBC.min(0).values
    rng[rng == 0] = 1.0
    CBC = CBC / rng
    return CBC.sum(1)


def node_difficulty(args, data, device):
    if not data.Pi.is_sparse:
        return class_conditional_betweenness_centrality(data, data.train_idx)
    cache = "{}/{}_cbc_{}_{}.pt".format(args.data_dir, args.dataset, args.noise_type, args.noise_rate)
    if os.path.exists(cache):
        return torch.load(cache, map_location='cpu').to(device)
    rely = data.train_idx.cpu()
    if rely.numel() > 20000:
        g = torch.Generator().manual_seed(0)
        rely = rely[torch.randperm(rely.numel(), generator=g)[:20000]]
    score = sparse_class_conditional_betweenness_centrality(data.Pi.cpu(), data.y.cpu(), rely)
    save_atomic(score, cache)
    return score.to(device)


def two_hop_edges(data):
    N = int(data.num_nodes)
    ei = data.edge_index
    A = torch.sparse_coo_tensor(ei, torch.ones(ei.shape[1], device=ei.device), (N, N)).coalesce()
    A2 = torch.sparse.mm(A, A).coalesce()
    idx = A2.indices()
    return idx[0], idx[1]
