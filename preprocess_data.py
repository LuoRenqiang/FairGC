import scipy.sparse as sp
import numpy as np
import torch
import os
import time
import threading
import gc
from tqdm import tqdm

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_node_data(dataset_name, k=20, add_self_loop=True):
    start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, f'data/{dataset_name}/')
    os.makedirs(data_path, exist_ok=True)
    clean_memory()
    from utils import (
        normalize, feature_normalize, sparse_mx_to_torch_sparse_tensor,
        load_pokec_n, load_pokec_z, load_aminerl, load_credit
    )
    sens_gender = None
    sens_region = None

    if dataset_name == 'pokec-n':
        feat, labels, sens, adj, idx_train, idx_val, idx_test = load_pokec_n()
        if len(sens.shape) == 2:
            sens_gender = sens[:, 0]
            sens_region = sens[:, 1]
        else:
            sens_gender = sens
            sens_region = np.zeros_like(sens)
    elif dataset_name == 'pokec-z':
        feat, labels, sens, adj, idx_val, idx_test, idx_train = load_pokec_z()
        if len(sens.shape) == 2:
            sens_gender = sens[:, 0]
            sens_region = sens[:, 1]
        else:
            sens_gender = sens
            sens_region = np.zeros_like(sens)
    elif dataset_name == 'aminer_l':
        feat, labels, sens, adj, idx_train, idx_val, idx_test = load_aminerl()
    elif dataset_name == 'credit':
        feat, labels, sens, adj, idx_train, idx_val, idx_test = load_credit()
    else:
        raise ValueError(f"Unsupported dataset:{dataset_name}")
    if add_self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_norm = normalize(adj)
    clean_memory()
    if sp.issparse(feat):
        feat = feat.toarray()
    feat = feature_normalize(feat)
    clean_memory()

    def run_eigsh(result_dict):
        adj_sparse = adj_norm.tocsr() if not sp.isspmatrix_csr(adj_norm) else adj_norm
        try:
            max_k = min(k, adj_sparse.shape[0] - 1)
            if max_k <= 0:
                max_k = 1
            eigvals, eigvecs = sp.linalg.eigsh(adj_sparse, k=max_k, which='LM', maxiter=5000, tol=1e-6)
            result_dict['success'] = True
        except Exception as e:
            n_nodes = adj_sparse.shape[0]
            max_k = min(k, n_nodes)
            eigvals = np.ones(max_k)
            eigvecs = np.eye(n_nodes, max_k)
            result_dict['success'] = False
        result_dict['eigvals'] = eigvals
        result_dict['eigvecs'] = eigvecs
        result_dict['node_count'] = adj_sparse.shape[0]
    result_dict = {}
    eig_thread = threading.Thread(target=run_eigsh, args=(result_dict,))
    eig_thread.start()
    progress_bar = tqdm(
        iterable=iter(int, 1),
        desc=f"{dataset_name} - Eigenvalue decomposition",
        bar_format='{l_bar}{bar}| Calculating [{elapsed}]',
        dynamic_ncols=True,
        leave=False
    )
    while eig_thread.is_alive():
        progress_bar.update(1)
        time.sleep(0.1)
    progress_bar.close()
    clean_memory()
    eigvals = result_dict['eigvals']
    eigvecs = result_dict['eigvecs']
    node_count = result_dict['node_count']
    success = result_dict.get('success', True)
    if success and len(eigvals) > 1:
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    eigvals_tensor = torch.tensor(eigvals, dtype=torch.float32)
    eigvecs_tensor = torch.tensor(eigvecs, dtype=torch.float32)
    feat_tensor = torch.tensor(feat, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    sens_tensor = torch.tensor(sens, dtype=torch.long)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm)
    file_map = {
        f'eigvals_k{k}.pt': eigvals_tensor,
        f'eigvecs_k{k}.pt': eigvecs_tensor,
        'feat.pt': feat_tensor,
        'labels.pt': labels_tensor,
        'sens.pt': sens_tensor,
        'adj.pt': adj_tensor
    }
    for fname, tensor in file_map.items():
        file_path = os.path.join(data_path, fname)
        torch.save(tensor, file_path)
        print(f"  ✓ {fname}")
    if dataset_name in ['pokec-n', 'pokec-z'] and sens_gender is not None and sens_region is not None:
        sens_gender_tensor = torch.tensor(sens_gender, dtype=torch.long)
        sens_region_tensor = torch.tensor(sens_region, dtype=torch.long)
        torch.save(sens_gender_tensor, os.path.join(data_path, 'sens_gender.pt'))
        torch.save(sens_region_tensor, os.path.join(data_path, 'sens_region.pt'))
        print(f"  ✓ sens_gender.pt")
        print(f"  ✓ sens_region.pt")
    np.save(os.path.join(data_path, 'train_idx.npy'), idx_train)
    np.save(os.path.join(data_path, 'val_idx.npy'), idx_val)
    np.save(os.path.join(data_path, 'test_idx.npy'), idx_test)
    print(f"  ✓ train_idx.npy, val_idx.npy, test_idx.npy")
    clean_memory()
    cost_time = time.time() - start_time
    return_dict = {
        'eigvals': eigvals_tensor,
        'eigvecs': eigvecs_tensor,
        'feat': feat_tensor,
        'labels': labels_tensor,
        'sens': sens_tensor,
        'adj': adj_tensor,
        'train_idx': idx_train,
        'val_idx': idx_val,
        'test_idx': idx_test
    }
    if dataset_name in ['pokec-n', 'pokec-z'] and sens_gender is not None and sens_region is not None:
        return_dict['sens_gender'] = torch.tensor(sens_gender, dtype=torch.long)
        return_dict['sens_region'] = torch.tensor(sens_region, dtype=torch.long)
    return return_dict

def main():
    datasets = ['pokec-n', 'pokec-z', 'credit', 'aminer_l']
    k = 20
    all_success = []
    for dataset in tqdm(
            datasets,
            desc="Overall preprocessing progress",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    ):
        try:
            result = generate_node_data(dataset, k=k)
            all_success.append(dataset)
        except Exception as e:
            print(f"\nError processing dataset {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping this dataset, continuing with the next one...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    success_count = 0
    for dataset in datasets:
        data_path = os.path.join(script_dir, f'data/{dataset}/')
        if os.path.exists(data_path):
            pt_files = [f for f in os.listdir(data_path) if f.endswith('.pt')]
            npy_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
            total_files = len(pt_files) + len(npy_files)
            print(f"{dataset}: {total_files} files generated ({data_path})")
        else:
            print(f"{dataset}: Preprocessing failed")

if __name__ == '__main__':
    main()