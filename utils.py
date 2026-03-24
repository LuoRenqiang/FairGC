import scipy.sparse as sp
import numpy as np
import torch
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from scipy.spatial import distance_matrix
import random

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def build_relationship(x, thresh=0.7):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)),columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    return np.array(idx_map)

def feature_normalize(feat):
    if sp.issparse(feat):
        feat = feat.toarray()
    feat = np.array(feat, dtype=np.float32)
    rowsum = feat.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return feat / rowsum

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def fair_loss(output, sens):
    pred = F.softmax(output, dim=1)[:, 1]
    sens = sens.float()
    cov = torch.mean(pred * sens) - torch.mean(pred) * torch.mean(sens)
    sens_0 = (sens == 0)
    sens_1 = (sens == 1)
    acc_0 = (torch.argmax(output, dim=1)[sens_0] == torch.argmax(output, dim=1)[
        sens_0]).float().mean() if sens_0.sum() > 0 else 1.0
    acc_1 = (torch.argmax(output, dim=1)[sens_1] == torch.argmax(output, dim=1)[
        sens_1]).float().mean() if sens_1.sum() > 0 else 1.0
    acc_gap = torch.abs(acc_0 - acc_1)
    return torch.abs(cov) + acc_gap

def normalize_adj_tensor(adj, sparse=False):
    adj = adj + torch.eye(adj.size(0)).to(adj.device)
    row_sum = adj.sum(1)
    row_sum_inv = row_sum.pow(-0.5).flatten()
    row_sum_inv[torch.isinf(row_sum_inv)] = 0.
    adj_norm = row_sum_inv.view(-1, 1) * adj * row_sum_inv.view(1, -1)
    return adj_norm

def init_params(layer):
    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)

def load_pokec_n(data_path='data/pokec-n/'):
    node_file = "region_job_2.csv"
    edge_file = "region_job_2_relationship.txt"
    label_col = "I_am_working_in_field"
    sensitive_cols = ["gender", "region"]
    node_df = pd.read_csv(os.path.join(data_path, node_file))
    label_mapping = {label: label if label == -1 else 1 if label > 1 else 0 for label in node_df[label_col].unique()}
    node_df[label_col] = node_df[label_col].map(label_mapping)
    user2idx = {uid: i for i, uid in enumerate(node_df["user_id"].unique())}
    nnodes = len(user2idx)
    node_df["idx"] = node_df["user_id"].map(user2idx)
    valid_user_ids = set(user2idx.keys())
    sens_dict = {}
    for col in sensitive_cols:
        sens_vals = node_df[col].fillna(node_df[col].mode()[0]).values
        sens_dict[col] = sens_vals
    feature_cols = [col for col in node_df.columns if col not in ["user_id", label_col, "idx"] + sensitive_cols]
    continuous_cols = [col for col in feature_cols if
                       pd.api.types.is_numeric_dtype(node_df[col]) and len(node_df[col].dropna().unique()) > 10]
    cont_feat = node_df[continuous_cols].copy()
    for col in continuous_cols:
        cont_feat[col] = cont_feat[col].fillna(cont_feat[col].mean())
        cont_feat[col] = (cont_feat[col] - cont_feat[col].mean()) / (cont_feat[col].std() + 1e-6)
    categorical_cols = [col for col in feature_cols if
                        col not in continuous_cols and len(node_df[col].dropna().unique()) <= 10]
    cat_feat = node_df[categorical_cols].copy()
    for col in categorical_cols:
        cat_feat[col] = cat_feat[col].fillna(cat_feat[col].mode()[0])
    encoder = OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore")
    one_hot_feat = encoder.fit_transform(cat_feat) if categorical_cols else np.array([]).reshape(-1, 0)
    binary_cols = [col for col in feature_cols if
                   col not in continuous_cols + categorical_cols and len(node_df[col].dropna().unique()) == 2]
    bin_feat = node_df[binary_cols].copy()
    for col in binary_cols:
        bin_feat[col] = bin_feat[col].fillna(0).astype(int)
    n_samples = len(node_df)
    cont_arr = cont_feat.values if continuous_cols else np.empty((n_samples, 0))
    bin_arr = bin_feat.values if binary_cols else np.empty((n_samples, 0))
    feat = np.hstack([cont_arr, bin_arr, one_hot_feat]).astype(np.float32)
    labels = node_df[label_col].values
    valid_mask = labels != -1
    valid_idx = np.where(valid_mask)[0]
    idx_train, idx_temp = train_test_split(valid_idx, test_size=0.4, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
    edges = []
    with open(os.path.join(data_path, edge_file), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u, v = map(int, line.split())
            except ValueError:
                continue
            if u in valid_user_ids and v in valid_user_ids:
                i, j = user2idx[u], user2idx[v]
                edges.append((i, j))
                edges.append((j, i))

    if edges:
        edges_np = np.array(edges).T
        adj = sp.coo_matrix((np.ones(edges_np.shape[1]), (edges_np[0], edges_np[1])),shape=(nnodes, nnodes), dtype=np.float32)
    else:
        adj = sp.coo_matrix((nnodes, nnodes), dtype=np.float32)

    gender = sens_dict["gender"].reshape(-1, 1)
    region = sens_dict["region"].reshape(-1, 1)
    sens = np.hstack([gender, region])
    feat = feature_normalize(sp.csr_matrix(feat))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return feat, labels, sens, adj, idx_train, idx_val, idx_test

def load_pokec_z(data_path='data/pokec-z/'):
    node_file = "region_job.csv"
    edge_file = "region_job_relationship.txt"
    label_col = "I_am_working_in_field"
    sensitive_cols = ["gender", "region"]
    node_df = pd.read_csv(os.path.join(data_path, node_file))
    label_mapping = {label: label if label == -1 else 1 if label > 1 else 0 for label in node_df[label_col].unique()}
    node_df[label_col] = node_df[label_col].map(label_mapping)
    user2idx = {uid: i for i, uid in enumerate(node_df["user_id"].unique())}
    nnodes = len(user2idx)
    node_df["idx"] = node_df["user_id"].map(user2idx)
    valid_user_ids = set(user2idx.keys())
    sens_dict = {}
    for col in sensitive_cols:
        sens_vals = node_df[col].fillna(node_df[col].mode()[0]).values
        sens_dict[col] = sens_vals
    feature_cols = [col for col in node_df.columns if col not in ["user_id", label_col, "idx"] + sensitive_cols]
    continuous_cols = [col for col in feature_cols if
                       pd.api.types.is_numeric_dtype(node_df[col]) and len(node_df[col].dropna().unique()) > 10]
    cont_feat = node_df[continuous_cols].copy()
    for col in continuous_cols:
        cont_feat[col] = cont_feat[col].fillna(cont_feat[col].mean())
        cont_feat[col] = (cont_feat[col] - cont_feat[col].mean()) / (cont_feat[col].std() + 1e-6)
    categorical_cols = [col for col in feature_cols if
                        col not in continuous_cols and len(node_df[col].dropna().unique()) <= 10]
    cat_feat = node_df[categorical_cols].copy()
    for col in categorical_cols:
        cat_feat[col] = cat_feat[col].fillna(cat_feat[col].mode()[0])
    encoder = OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore")
    one_hot_feat = encoder.fit_transform(cat_feat) if categorical_cols else np.array([]).reshape(-1, 0)

    binary_cols = [col for col in feature_cols if
                   col not in continuous_cols + categorical_cols and len(node_df[col].dropna().unique()) == 2]
    bin_feat = node_df[binary_cols].copy()
    for col in binary_cols:
        bin_feat[col] = bin_feat[col].fillna(0).astype(int)
    n_samples = len(node_df)
    cont_arr = cont_feat.values if continuous_cols else np.empty((n_samples, 0))
    bin_arr = bin_feat.values if binary_cols else np.empty((n_samples, 0))
    feat = np.hstack([cont_arr, bin_arr, one_hot_feat]).astype(np.float32)
    labels = node_df[label_col].values
    valid_mask = labels != -1
    valid_idx = np.where(valid_mask)[0]
    idx_train, idx_temp = train_test_split(valid_idx, test_size=0.4, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
    edges = []
    with open(os.path.join(data_path, edge_file), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u, v = map(int, line.split())
            except ValueError:
                continue
            if u in valid_user_ids and v in valid_user_ids:
                i, j = user2idx[u], user2idx[v]
                edges.append((i, j))
                edges.append((j, i))

    if edges:
        edges_np = np.array(edges).T
        adj = sp.coo_matrix((np.ones(edges_np.shape[1]), (edges_np[0], edges_np[1])),
                           shape=(nnodes, nnodes), dtype=np.float32)
    else:
        adj = sp.coo_matrix((nnodes, nnodes), dtype=np.float32)

    gender = sens_dict["gender"].reshape(-1, 1)
    region = sens_dict["region"].reshape(-1, 1)
    sens = np.hstack([gender, region])
    feat = feature_normalize(sp.csr_matrix(feat))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return feat, labels, sens, adj, idx_train, idx_val, idx_test

def load_aminerl(data_path='data/aminer_l/'):
    seed = 42
    edges = []
    edgelist_path = os.path.join(data_path, 'edgelist_LCC.txt')
    if not os.path.exists(edgelist_path):
        raise FileNotFoundError(f"Edge file not found:{edgelist_path}")
    with open(edgelist_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u, v = map(int, line.split('\t'))
                edges.append((u, v))
                edges.append((v, u))
            except ValueError:
                continue
    raw_labels = []
    labels_path = os.path.join(data_path, 'labels_LCC.txt')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label file not found:{labels_path}")
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                _, label = line.split('\t')
                raw_labels.append(float(label))
            except ValueError:
                continue
    nnodes = len(raw_labels)
    raw_labels_np = np.array(raw_labels)
    labels = (raw_labels_np == 3).astype(int)
    sens = np.zeros(nnodes, dtype=np.float32)
    sen_path = os.path.join(data_path, 'sens_LCC.txt')
    if not os.path.exists(sen_path):
        raise FileNotFoundError(f"Sens file not found:{sen_path}")
    with open(sen_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                node_id, sen_val = line.split('\t')
                node_id = int(node_id)
                if 0 <= node_id < nnodes:
                    sens[node_id] = float(sen_val)
            except ValueError:
                continue
    feat_path = os.path.join(data_path, 'X_LCC.npz')
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found:{feat_path}")
    feat_npz = np.load(feat_path)
    feat = sp.csr_matrix(
        (feat_npz["data"], (feat_npz["row"], feat_npz["col"])),
        shape=(nnodes, np.max(feat_npz["col"]) + 1),
        dtype=np.float32
    ).todense()
    row_norms = np.linalg.norm(feat, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    feat = feat / row_norms
    feat = feat.astype(np.float32)
    idx = np.arange(nnodes)
    idx_train, idx_temp, _, _ = train_test_split(idx, labels, test_size=0.2, random_state=seed, stratify=labels)
    idx_val, idx_test, _, _ = train_test_split(idx_temp, labels[idx_temp], test_size=0.5, random_state=seed,stratify=labels[idx_temp])

    if edges:
        edges_np = np.array(edges).T
        adj = sp.coo_matrix((np.ones(edges_np.shape[1]), (edges_np[0], edges_np[1])),shape=(nnodes, nnodes), dtype=np.float32)
    else:
        adj = sp.coo_matrix((nnodes, nnodes), dtype=np.float32)

    adj = normalize(adj + sp.eye(adj.shape[0]))
    print(f"Data loading completed:")
    return feat, labels, sens, adj, idx_train, idx_val, idx_test

def load_credit(dataset_path='data/credit/'):
    os.makedirs(dataset_path, exist_ok=True)
    print(f'Loading credit dataset from {dataset_path}')
    csv_path = os.path.join(dataset_path, 'credit.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Credit does not exist: {csv_path}")
    idx_features_labels = pd.read_csv(csv_path)
    sens_attr = "Age"
    predict_attr = "NoDefaultNextMonth"
    header = list(idx_features_labels.columns)
    if predict_attr not in header:
        raise ValueError(f"The prediction attribute '{predict_attr}' does not exist")
    header.remove(predict_attr)

    if 'Single' in header:
        header.remove('Single')
    edges_txt_path = os.path.join(dataset_path, 'credit_edges.txt')

    if os.path.exists(edges_txt_path):
        edges_unordered = np.genfromtxt(edges_txt_path).astype(int)
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(edges_txt_path, edges_unordered)
    features = feature_normalize(idx_features_labels[header])
    labels = idx_features_labels[predict_attr].values

    if sens_attr in idx_features_labels.columns:
        sens = idx_features_labels[sens_attr].values
        age_median = np.median(sens)
        sens = (sens > age_median).astype(int)
    else:
        sens = np.zeros_like(labels)

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj)
    n_nodes = len(labels)
    indices = np.arange(n_nodes)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(0.6 * n_nodes)
    val_size = int(0.2 * n_nodes)
    idx_train = indices[:train_size]
    idx_val = indices[train_size:train_size + val_size]
    idx_test = indices[train_size + val_size:]
    return features, labels, sens, adj, idx_train, idx_val, idx_test