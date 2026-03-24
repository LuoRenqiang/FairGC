import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh

class SineEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.register_buffer('pe_base', self._get_pe_base(hidden_dim, max_len))
    def _get_pe_base(self, hidden_dim, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-np.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    def forward(self, x):
        n_eig = x.size(0)
        if n_eig > self.max_len:
            pe = self._get_pe_base(self.hidden_dim, n_eig).to(x.device)
        else:
            pe = self.pe_base[:n_eig, :]
        x = x.unsqueeze(1).expand(-1, self.hidden_dim)
        out = x * pe
        out = torch.clamp(out, -1e3, 1e3)
        return out

class FULayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, norm='batch'):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_type = norm
        if norm == 'layer':
            self.norm = nn.LayerNorm(out_dim)
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.Identity()
        self.activation = nn.ReLU()
    def forward(self, x, eig_feat):
        x1 = self.linear1(x)
        x2 = self.linear2(eig_feat)
        out = self.dropout(self.activation(x1 + x2))
        out = self.norm(out) if self.norm_type != 'none' else out
        return out

class EnhancedFUGNN(nn.Module):
    def __init__(self, nclass, nfeat, nlayer=2, hidden_dim=128, nheads=1,
                 tran_dropout=0.0, feat_dropout=0.3, prop_dropout=0.0, norm='batch'):
        super().__init__()
        self.nlayer = nlayer
        self.hidden_dim = hidden_dim
        self.norm = norm
        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(feat_dropout)
        )
        self.classify = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(tran_dropout),
            nn.Linear(hidden_dim // 2, nclass)
        )
        self.eignvalue_encoder = SineEncoding(hidden_dim, max_len=100)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(tran_dropout)
        )
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, dropout=tran_dropout, batch_first=True)
        self.oc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(tran_dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.layers = nn.ModuleList([
            FULayer(hidden_dim, hidden_dim, prop_dropout, norm)
            for _ in range(nlayer)
        ])
        self.dropout1 = nn.Dropout(tran_dropout)
        self.dropout2 = nn.Dropout(tran_dropout)
        self._reset_parameters()
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, eigvals, eigvecs, x):
        if eigvecs.shape[0] != x.shape[0]:
            with torch.no_grad():
                proj = torch.randn(eigvecs.shape[0], x.shape[0], device=x.device)
                proj = F.normalize(proj, dim=0)
            eigvecs_adapted = torch.matmul(proj.T, eigvecs)
            eigvecs = eigvecs_adapted
        h = self.feat_encoder(x)
        eig_emb = self.eignvalue_encoder(eigvals)
        attn_output, _ = self.mha(eig_emb.unsqueeze(0), eig_emb.unsqueeze(0), eig_emb.unsqueeze(0))
        attn_output = self.dropout1(attn_output.squeeze(0))
        eig_emb = eig_emb + self.oc(attn_output)
        eig_emb = self.dropout2(eig_emb)
        eig_emb = self.decoder(eig_emb)
        eig_feat = torch.matmul(eigvecs, eig_emb)
        for layer in self.layers:
            h = layer(h, eig_feat)
        out = self.classify(h)
        return F.log_softmax(out, dim=1)

class ImprovedGCond:
    def __init__(self, data, args, device='cpu', run_seed=None, sens_name="Gender"):
        self.data = data
        self.args = args
        self.device = device
        self.run_seed = run_seed or args.seed
        self.sens_name = sens_name
        total_nodes = data["feat"].shape[0]
        self.n_syn = max(10, int(total_nodes * args.reduction_rate))
        self.dataset_name = data.get("dataset_name", "unknown")
        feat_syn = self.sample_features()
        self.feat_syn = nn.Parameter(feat_syn.clone().to(device))
        self.labels_syn = self.generate_labels().to(device)
        self.sens_syn = self.generate_sensitive_attributes().to(device)
        self.optimizer = torch.optim.Adam([self.feat_syn], lr=args.lr_feat, weight_decay=1e-4)
    def sample_features(self):
        data = self.data
        train_mask = data["train_mask"]
        labels = data["labels"]
        train_idx = train_mask.nonzero().squeeze().cpu().numpy()
        train_labels = labels[train_mask].cpu().numpy()
        if len(train_idx) > 1e6:
            batch_size = 100000
            pos_idx = train_idx[train_labels == 1]
            neg_idx = train_idx[train_labels == 0]
            pos_ratio = len(pos_idx) / len(train_idx)
            n_pos_samples = int(self.n_syn * pos_ratio)
            n_neg_samples = self.n_syn - n_pos_samples
            rng = np.random.RandomState(self.run_seed)
            pos_sample = rng.choice(pos_idx, size=n_pos_samples, replace=True) if len(pos_idx) > 0 else []
            neg_sample = rng.choice(neg_idx, size=n_neg_samples, replace=True) if len(neg_idx) > 0 else []
            indices = np.concatenate([pos_sample, neg_sample])
        else:
            if len(train_idx) < 10:
                train_idx = np.arange(data["feat"].shape[0])
                train_labels = labels.cpu().numpy()
            indices = []
            unique_labels, label_counts = np.unique(train_labels, return_counts=True)
            if len(unique_labels) == 0:
                rng = np.random.RandomState(self.run_seed)
                indices = rng.choice(train_idx, size=self.n_syn, replace=True).tolist()
            elif len(unique_labels) == 1:
                rng = np.random.RandomState(self.run_seed)
                indices = rng.choice(train_idx, size=self.n_syn, replace=True).tolist()
            else:
                label_ratios = label_counts / len(train_labels)
                for i, c in enumerate(unique_labels):
                    class_mask = train_labels == c
                    class_idx = train_idx[class_mask]
                    if len(class_idx) == 0:
                        continue
                    n_samples = int(self.n_syn * label_ratios[i])
                    if i == len(unique_labels) - 1:
                        n_samples = self.n_syn - len(indices)
                    else:
                        n_samples = max(1, n_samples)
                    rng = np.random.RandomState(self.run_seed + c * 100)
                    sample = rng.choice(class_idx, size=n_samples, replace=len(class_idx) < n_samples)
                    indices.extend(sample.tolist())
        if len(indices) != self.n_syn:
            rng = np.random.RandomState(self.run_seed + 999)
            if len(indices) < self.n_syn:
                extra = rng.choice(train_idx, size=self.n_syn - len(indices), replace=True)
                indices.extend(extra.tolist())
            else:
                indices = indices[:self.n_syn]
        indices = np.array(indices)
        if len(indices) > 1000:
            batch_size = 500
            feat_batches = []
            for i in range(0, len(indices), batch_size):
                feat_batch = data["feat"][indices[i:i + batch_size]].clone()
                feat_batches.append(feat_batch)
            features = torch.cat(feat_batches, dim=0)
        else:
            features = data["feat"][indices].clone()
        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        return features
    def generate_labels(self):
        data = self.data
        train_mask = data["train_mask"]
        train_labels = data["labels"][train_mask].cpu().numpy()
        unique_labels, label_counts = np.unique(train_labels, return_counts=True)
        if len(unique_labels) == 0:
            return torch.zeros(self.n_syn, dtype=torch.long)
        if len(unique_labels) == 1:
            return torch.full((self.n_syn,), unique_labels[0], dtype=torch.long)
        label_ratios = label_counts / len(train_labels)
        labels_syn = []
        for i, c in enumerate(unique_labels):
            n_class = int(self.n_syn * label_ratios[i])
            if i == len(unique_labels) - 1:
                n_class = self.n_syn - len(labels_syn)
            labels_syn.extend([c] * n_class)
        return torch.LongTensor(labels_syn)
    def generate_sensitive_attributes(self):
        data = self.data
        train_mask = data["train_mask"]
        if len(data["sens_attrs"]) > 0:
            sens_attr = data["sens_attrs"][0]
            train_sens = sens_attr[train_mask].cpu().numpy()
            train_sens_valid = train_sens[train_sens >= 0]
            if len(train_sens_valid) == 0:
                all_sens = sens_attr.cpu().numpy()
                all_valid = all_sens[all_sens >= 0]
                if len(all_valid) == 0:
                    rng = np.random.RandomState(self.run_seed)
                    return torch.LongTensor(rng.randint(0, 7, (self.n_syn,)))
                unique_vals, counts = np.unique(all_valid, return_counts=True)
                rng = np.random.RandomState(self.run_seed)
                return torch.LongTensor(rng.choice(unique_vals, size=self.n_syn, p=counts/counts.sum(), replace=True))
            else:
                unique_sens, sens_counts = np.unique(train_sens_valid, return_counts=True)
                if len(unique_sens) == 0:
                    rng = np.random.RandomState(self.run_seed)
                    return torch.LongTensor(rng.randint(0, 7, (self.n_syn,)))
                if len(unique_sens) == 1:
                    return torch.full((self.n_syn,), unique_sens[0], dtype=torch.long)
                sens_ratios = sens_counts / len(train_sens_valid)
                sens_syn = []
                for i, s in enumerate(unique_sens):
                    n_sens = int(self.n_syn * sens_ratios[i])
                    if i == len(unique_sens) - 1:
                        n_sens = self.n_syn - len(sens_syn)
                    else:
                        n_sens = max(1, n_sens)
                    sens_syn.extend([s] * n_sens)
                if len(sens_syn) < self.n_syn:
                    largest_group_val = unique_sens[np.argmax(sens_counts)]
                    sens_syn.extend([largest_group_val] * (self.n_syn - len(sens_syn)))
                rng = np.random.RandomState(self.run_seed)
                sens_syn = np.array(sens_syn)
                rng.shuffle(sens_syn)
                return torch.LongTensor(sens_syn)
        else:
            rng = np.random.RandomState(self.run_seed)
            return torch.LongTensor(rng.randint(0, 7, (self.n_syn,)))
    def train_gcond(self):
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        grad_accum_steps = 4 if self.feat_syn.shape[1] > 5000 else 1
        for epoch in range(self.args.gcond_epochs):
            self.optimizer.zero_grad()
            model = nn.Sequential(
                nn.Linear(self.feat_syn.shape[1], min(128, self.feat_syn.shape[1] // 2)),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(min(128, self.feat_syn.shape[1] // 2), 2)
            ).to(self.device)
            loss_total = 0
            for i in range(grad_accum_steps):
                output = model(self.feat_syn)
                loss = F.nll_loss(F.log_softmax(output, dim=1), self.labels_syn) / grad_accum_steps
                loss_total += loss.item()
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
            if self.feat_syn.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.feat_syn], max_norm=1.0)
            self.optimizer.step()
            if loss_total < best_loss:
                best_loss = loss_total
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    def get_condensed_data(self):
        with torch.no_grad():
            adj_syn = build_adjacency_matrix_light({
                "feat": self.feat_syn.detach(),
                "labels": self.labels_syn.detach(),
                "sens": self.sens_syn.detach()
            }, self.device)
            if self.args.k == -1:
                dataset_k_map = {
                    "pokec-z_gender": 5, "pokec-z_region": 5,
                    "pokec-n_gender": 1, "pokec-n_region": 1,
                    "credit": 1, "aminer-l": 10
                }
                self.args.k = dataset_k_map.get(self.dataset_name.lower(), 3)
            try:
                eigvals_syn, eigvecs_syn = compute_graph_spectrum_light(adj_syn, n_eig=self.args.k if self.args.k != 0 else None, device=self.device)
            except Exception:
                n_nodes = self.feat_syn.shape[0]
                n_eig = self.args.k if self.args.k != 0 else max(1, n_nodes - 1)
                eigvals_syn = torch.ones(n_eig, device=self.device)
                eigvecs_syn = torch.eye(n_nodes, n_eig, device=self.device)
            return {
                "feat": self.feat_syn.detach(),
                "labels": self.labels_syn.detach(),
                "sens": self.sens_syn.detach(),
                "adj": adj_syn,
                "eigvals": eigvals_syn,
                "eigvecs": eigvecs_syn,
                "device": self.device,
                "k": self.args.k,
                "method": "gcond",
                "sens_name": self.sens_name
            }

def compute_graph_spectrum_light(adj, n_eig=None, device='cpu'):
    n_nodes = adj.shape[0]
    use_k = max(1, n_nodes - 1) if n_eig is None else min(n_eig, n_nodes - 1) if n_nodes > 1 else 1
    if isinstance(adj, torch.Tensor):
        adj_np = adj.cpu().numpy()
        adj_sparse = csr_matrix(adj_np) if adj_np.size > 1e7 else csc_matrix(adj_np)
    else:
        adj_sparse = adj if isinstance(adj, (csc_matrix, csr_matrix)) else csc_matrix(adj)
    degree = np.array(adj_sparse.sum(axis=1)).squeeze()
    degree[degree == 0] = 1e-8
    degree_inv_sqrt = 1.0 / np.sqrt(degree)
    D_inv_sqrt = diags(degree_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj_sparse @ D_inv_sqrt
    k = use_k
    eigvals, eigvecs = eigsh(adj_norm, k=k, which='LM', maxiter=2000, tol=1e-4)
    mag = np.abs(eigvals)
    sorted_idx = np.argsort(mag)[::-1]
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]
    eigvals = np.nan_to_num(eigvals, nan=1.0, posinf=1.0, neginf=-1.0)
    eigvecs = np.nan_to_num(eigvecs, nan=0.0, posinf=1.0, neginf=-1.0)
    if eigvecs.shape[0] != n_nodes:
        eigvecs_pad = np.zeros((n_nodes, k))
        eigvecs_pad[:min(eigvecs.shape[0], n_nodes), :min(eigvecs.shape[1], k)] = eigvecs[:n_nodes, :k]
        eigvecs = eigvecs_pad
    try:
        eigvals = torch.FloatTensor(eigvals).to(device)
        eigvecs = torch.FloatTensor(eigvecs).to(device)
    except:
        eigvals = torch.FloatTensor(eigvals).cpu()
        eigvecs = torch.FloatTensor(eigvecs).cpu()
    eigvecs = F.normalize(eigvecs, dim=0, p=2, eps=1e-8)
    return eigvals, eigvecs

def build_adjacency_matrix_light(data, device, max_nodes_for_dense=20000):
    if "adj_sparse" in data:
        return data["adj_sparse"]
    feat = data["feat"].cpu().numpy()
    n_nodes = feat.shape[0]
    if n_nodes > max_nodes_for_dense:
        k = min(10, n_nodes - 1)
        adj_sparse = csr_matrix((n_nodes, n_nodes), dtype=np.float32)
        batch_size = 2000
        n_batches = (n_nodes + batch_size - 1) // batch_size
        feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_nodes)
            feat_batch = feat[start:end]
            feat_batch_norm = feat_batch / (np.linalg.norm(feat_batch, axis=1, keepdims=True) + 1e-8)
            sim_batch = feat_batch_norm @ feat_norm.T
            for i in range(end - start):
                sim_vals = sim_batch[i]
                sim_vals[start + i] = -1
                top_k_idx = np.argsort(sim_vals)[-k:]
                adj_sparse[start + i, top_k_idx] = 1.0
        adj_sparse = (adj_sparse + adj_sparse.T) / 2
        return adj_sparse
    else:
        feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        sim_matrix = feat_norm @ feat_norm.T
        k = min(20, n_nodes - 1) if n_nodes > 1 else 1
        adj = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            if n_nodes == 1:
                adj[i, i] = 1.0
            else:
                sim_vals = sim_matrix[i].copy()
                sim_vals[i] = -1
                top_k_idx = np.argsort(sim_vals)[-k:]
                adj[i, top_k_idx] = 1.0
        adj = (adj + adj.T) / 2
        adj = torch.FloatTensor(adj).to(device)
        if len(adj.shape) == 1:
            adj = adj.reshape(-1, 1)
        if adj.shape[0] != adj.shape[1]:
            n = max(adj.shape)
            adj_square = torch.eye(n, device=adj.device)
            adj_square[:adj.shape[0], :adj.shape[1]] = adj
            adj = adj_square
        return adj

def train_and_evaluate_model(condensed_data, original_data, args, model_type='fugnn', run_seed=None):
    feat_syn = condensed_data["feat"]
    labels_syn = condensed_data["labels"]
    device = condensed_data["device"]
    sens_name = condensed_data.get("sens_name", "unknown")
    feat_syn_norm = (feat_syn - feat_syn.mean(dim=0)) / (feat_syn.std(dim=0) + 1e-8)
    n_syn = feat_syn.shape[0]
    rng = np.random.RandomState(run_seed or args.seed)
    indices = np.arange(n_syn)
    rng.shuffle(indices)
    n_train = max(min(10, n_syn - 5), int(n_syn * 0.8))
    train_mask = torch.zeros(n_syn, dtype=torch.bool).to(device)
    train_mask[indices[:n_train]] = True
    model = EnhancedFUGNN(
        nclass=2,
        nfeat=feat_syn_norm.shape[1],
        nlayer=min(args.nlayer, 2),
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        tran_dropout=args.tran_dropout,
        feat_dropout=args.feat_dropout,
        prop_dropout=args.prop_dropout,
        norm=args.norm
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=args.lr * 0.01)
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None
    best_auc = 0.0
    grad_accum_steps = 2 if feat_syn.shape[0] < 100 else 1
    for epoch in range(80):
        model.train()
        total_loss = 0
        for step in range(grad_accum_steps):
            optimizer.zero_grad()
            output = model(condensed_data["eigvals"], condensed_data["eigvecs"], feat_syn_norm)
            if epoch < 40:
                epsilon = 0.1
                n_classes = 2
                train_labels = labels_syn[train_mask]
                n_train_samples = train_labels.shape[0]
                smoothed_labels = torch.full((n_train_samples, n_classes), epsilon / (n_classes - 1), device=device)
                smoothed_labels.scatter_(1, train_labels.unsqueeze(1), 1 - epsilon)
                loss = -(smoothed_labels * output[train_mask]).sum(dim=1).mean()
            else:
                loss = F.nll_loss(output[train_mask], labels_syn[train_mask])
            loss = loss / grad_accum_steps
            total_loss += loss.item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            output_val = model(condensed_data["eigvals"], condensed_data["eigvecs"], feat_syn_norm)
            val_mask = ~train_mask
            if val_mask.sum() > 0:
                val_loss = F.nll_loss(output_val[val_mask], labels_syn[val_mask]).item()
                val_probs = torch.exp(output_val[val_mask]).cpu().numpy()
                val_labels = labels_syn[val_mask].cpu().numpy()
                if len(np.unique(val_labels)) > 1:
                    try:
                        val_auc = roc_auc_score(val_labels, val_probs[:, 1])
                        if val_auc > best_auc:
                            best_auc = val_auc
                            best_model_state = model.state_dict().copy()
                            patience_counter = 0
                    except:
                        val_auc = 0.5
            else:
                val_loss = total_loss
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            if best_model_state is None:
                best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return evaluate_model_light(model, original_data, 'fugnn', condensed_data)

def evaluate_model_light(model, original_data, model_type='fugnn', condensed_data=None):
    model.eval()
    device = original_data["device"]
    with torch.no_grad():
        test_mask = original_data["test_mask"]
        feat = original_data["feat"]
        labels = original_data["labels"]
        batch_size = 1000
        all_preds = []
        all_probs = []
        all_true = []
        test_idx = test_mask.nonzero().squeeze()
        if len(test_idx.shape) == 0:
            test_idx = test_idx.unsqueeze(0)
        n_batches = (len(test_idx) + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(test_idx))
            batch_indices = test_idx[start:end]
            feat_batch = feat[batch_indices].to(device)
            labels_batch = labels[batch_indices].cpu().numpy()
            feat_batch_norm = (feat_batch - feat_batch.mean(dim=0)) / (feat_batch.std(dim=0) + 1e-8)
            output_batch = model(condensed_data["eigvals"], condensed_data["eigvecs"], feat_batch_norm)
            preds_batch = output_batch.argmax(dim=1).cpu().numpy()
            probs_batch = torch.exp(output_batch).cpu().numpy()
            all_preds.extend(preds_batch)
            all_probs.extend(probs_batch)
            all_true.extend(labels_batch)
            del feat_batch, output_batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        y_true = np.array(all_true)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        acc = accuracy_score(y_true, y_pred)
        auc = 0.5
        if len(np.unique(y_true)) > 1 and y_prob.shape[1] == 2:
            try:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            except Exception:
                pass
        fairness_results = {}
        sens_name = condensed_data.get("sens_name", "unknown") if condensed_data else "unknown"
        for sens_idx, sens_name_in_data in enumerate(original_data["sens_names"]):
            if sens_name_in_data == sens_name or sens_name == "unknown":
                sens_attr = original_data["sens_attrs"][sens_idx]
                sens_test = []
                for batch_idx in range(n_batches):
                    start = batch_idx * batch_size
                    end = min((batch_idx + 1) * batch_size, len(test_idx))
                    sens_test.extend(sens_attr[test_idx[start:end]].cpu().numpy())
                sens_test = np.array(sens_test)
                eo, sp = compute_fairness_metrics(y_true, y_pred, sens_test)
                fairness_results[f"{sens_name.lower()}_eo"] = eo
                fairness_results[f"{sens_name.lower()}_sp"] = sp
                break
        return {
            "accuracy": acc,
            "auc": auc,
            "fairness": fairness_results,
            "y_true": y_true,
            "y_pred": y_pred,
            "k": condensed_data.get("k", -1) if condensed_data else -1,
            "method": condensed_data.get("method", "unknown") if condensed_data else "unknown",
            "sens_name": sens_name
        }

def compute_fairness_metrics(y_true, y_pred, sensitive):
    valid_mask = (sensitive >= 0)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    sensitive_valid = sensitive[valid_mask]
    if len(y_true_valid) == 0 or len(np.unique(sensitive_valid)) < 2:
        return 0.0, 0.0
    group_stats = []
    for s in np.unique(sensitive_valid):
        s_mask = (sensitive_valid == s)
        s_count = np.sum(s_mask)
        s_pos_mask = s_mask & (y_true_valid == 1)
        s_pos_count = np.sum(s_pos_mask)
        tpr = np.sum(y_pred_valid[s_pos_mask] == 1) / s_pos_count if s_pos_count > 0 else 0.0
        pred_pos_rate = np.mean(y_pred_valid[s_mask] == 1) if s_count > 0 else 0.0
        group_stats.append({'tpr': tpr, 'pred_pos_rate': pred_pos_rate})
    tpr_list = [stats['tpr'] for stats in group_stats]
    pos_rate_list = [stats['pred_pos_rate'] for stats in group_stats]
    eo = max([abs(tpr_list[i] - tpr_list[j]) for i in range(len(tpr_list)) for j in range(i + 1, len(tpr_list))])
    sp = max([abs(pos_rate_list[i] - pos_rate_list[j]) for i in range(len(pos_rate_list)) for j in range(i + 1, len(pos_rate_list))])
    return eo, sp

def load_pokec_data(dataset_name, device, data_dir):
    data_dir = os.path.join(data_dir, dataset_name)
    if not os.path.exists(data_dir):
        raise ValueError(f"{dataset_name} data directory does not exist: {data_dir}")
    feat = torch.load(os.path.join(data_dir, "feat.pt"), map_location=device)
    labels = torch.load(os.path.join(data_dir, "labels.pt"), map_location=device)
    n_nodes = feat.shape[0]
    train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, "test_idx.npy"))
    train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    sens_gender = torch.load(os.path.join(data_dir, "sens_gender.pt"), map_location=device)
    sens_region = torch.load(os.path.join(data_dir, "sens_region.pt"), map_location=device)
    data_gender = {
        "feat": feat,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "sens_attrs": [sens_gender],
        "sens_names": ["Gender"],
        "device": device,
        "dataset_name": f"{dataset_name}_gender"
    }
    data_region = {
        "feat": feat,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "sens_attrs": [sens_region],
        "sens_names": ["Region"],
        "device": device,
        "dataset_name": f"{dataset_name}_region"
    }
    return [data_gender, data_region]

def load_aminer_l_data(device, data_dir):
    data_dir = os.path.join(data_dir, "aminer_l")
    if not os.path.exists(data_dir):
        raise ValueError(f"aminer_l data directory does not exist: {data_dir}")
    labels = torch.load(os.path.join(data_dir, "labels.pt"), map_location=device)
    feat = torch.load(os.path.join(data_dir, "feat.pt"), map_location=device)
    n_nodes = feat.shape[0]
    train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, "test_idx.npy"))
    train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    sens_path = os.path.join(data_dir, "sens.pt")
    if os.path.exists(sens_path):
        sens_binary = torch.load(sens_path, map_location=device).clone()
    else:
        torch.manual_seed(42)
        sens_binary = torch.randint(0, 7, (n_nodes,), device=device)
    return [{
        "feat": feat,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "sens_attrs": [sens_binary],
        "sens_names": ["Academic"],
        "device": device,
        "dataset_name": "aminer-l"
    }]

def load_credit_data(device, data_dir):
    data_dir = os.path.join(data_dir, "credit")
    if not os.path.exists(data_dir):
        raise ValueError(f"credit data directory does not exist: {data_dir}")
    feat = torch.load(os.path.join(data_dir, "feat.pt"), map_location=device)
    labels = torch.load(os.path.join(data_dir, "labels.pt"), map_location=device)
    n_nodes = feat.shape[0]
    train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, "test_idx.npy"))
    train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    sens_path = os.path.join(data_dir, "sens.pt")
    if os.path.exists(sens_path):
        sens = torch.load(sens_path, map_location=device)
        sens_unique = torch.unique(sens)
        if len(sens_unique) > 2:
            sens_binary = (sens > torch.median(sens)).long()
        else:
            sens_binary = sens.long()
    else:
        torch.manual_seed(42)
        sens_binary = torch.randint(0, 2, (n_nodes,), device=device)
    return [{
        "feat": feat,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "sens_attrs": [sens_binary],
        "sens_names": ["Age"],
        "device": device,
        "dataset_name": "credit"
    }]

def load_dataset(dataset_name, device, data_dir):
    if dataset_name in ["pokec-n", "pokec-z"]:
        return load_pokec_data(dataset_name, device, data_dir)
    elif dataset_name == "aminer-l":
        return load_aminer_l_data(device, data_dir)
    elif dataset_name == "credit":
        return load_credit_data(device, data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")