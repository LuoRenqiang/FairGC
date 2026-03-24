import argparse
import torch
import numpy as np
import os
import time
from collections import defaultdict
from models import (
    ImprovedGCond, compute_graph_spectrum_light, build_adjacency_matrix_light,
    train_and_evaluate_model, load_dataset, evaluate_model_light, compute_fairness_metrics
)

def run_single_experiment(args, run_seed, device):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    data_list = load_dataset(args.dataset, device, args.data_dir)
    results = {}
    for data in data_list:
        sens_name = data["sens_names"][0]
        try:
            gcond = ImprovedGCond(data, args, device, run_seed=run_seed, sens_name=sens_name)
            gcond.train_gcond()
            condensed_data_gcond = gcond.get_condensed_data()
        except Exception as e:
            print(f"Training error: {e}")
            continue
        results_key = f'gcond_fugnn_{sens_name.lower()}'
        results[results_key] = train_and_evaluate_model(condensed_data_gcond, data, args, model_type='fugnn', run_seed=run_seed)
        results['original_data'] = data_list
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return results

def aggregate_results(all_results, args):
    aggregated = {}
    method_model_sens_combinations = set()
    for result in all_results:
        for key in result.keys():
            if key != 'original_data':
                method_model_sens_combinations.add(key)
    for combo in method_model_sens_combinations:
        parts = combo.split('_')
        method = parts[0] if len(parts) > 0 else 'unknown'
        model = parts[1] if len(parts) > 1 else 'unknown'
        sens = '_'.join(parts[2:]) if len(parts) >= 3 else 'unknown'
        accuracies = []
        aucs = []
        fairness_dict = defaultdict(list)
        for result in all_results:
            if combo in result:
                accuracies.append(result[combo]["accuracy"])
                aucs.append(result[combo]["auc"])
                for metric, value in result[combo]["fairness"].items():
                    fairness_dict[metric].append(value)
        if len(accuracies) > 0:
            aggregated[combo] = {
                "method": method,
                "model": model,
                "sens": sens,
                "accuracy": {"mean": np.mean(accuracies), "std": np.std(accuracies), "values": accuracies},
                "auc": {"mean": np.mean(aucs), "std": np.std(aucs), "values": aucs},
                "fairness": {metric: {"mean": np.mean(values), "std": np.std(values), "values": values} for metric, values in fairness_dict.items()}
            }
    return aggregated

def print_results(aggregated_results, args):
    print(f"\nExperiment Settings:")
    print(f"Dataset: {args.dataset}")
    print(f"Reduction Rate: {args.reduction_rate * 100:.2f}%")
    print(f"Number of Runs: {args.num_runs}")
    all_sens = sorted({aggregated_results[combo]['sens'] for combo in aggregated_results.keys() if combo != 'original_data'})
    for sens_name in all_sens:
        print(f"\nSensitive Attribute: {sens_name.upper()}")
        print("-" * 60)
        combo = f"gcond_fugnn_{sens_name}"
        if combo in aggregated_results:
            result = aggregated_results[combo]
            print(f"{'Metric':<20} {'Mean±Std':<45}")
            print("-" * 60)
            acc = result["accuracy"]
            print(f"{'Accuracy':<20} {acc['mean']:.4f}±{acc['std']:.4f} ")
            if result["fairness"]:
                print(f"\nFairness Metrics:")
                for metric, fair_val in result["fairness"].items():
                    print(f"{metric:<20} {fair_val['mean']:.4f}±{fair_val['std']:.4f}")
        else:
            print("No experimental results available")

def parse_args():
    parser = argparse.ArgumentParser(description='FairGC Experiment')
    parser.add_argument('--dataset', type=str, default='credit', choices=['pokec-n', 'pokec-z', 'aminer-l', 'credit'], help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--reduction_rate', type=float, default=0.1, help='Reduction rate (0.1=10% nodes)')
    parser.add_argument('--gcond_epochs', type=int, default=5000, help='GCond training epochs')
    parser.add_argument('--lr_feat', type=float, default=0.01, help='Feature learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--nlayer', type=int, default=2, help='layers')
    parser.add_argument('--nheads', type=int, default=1, help='attention heads')
    parser.add_argument('--tran_dropout', type=float, default=0.1, help='transformer dropout rate')
    parser.add_argument('--feat_dropout', type=float, default=0.3, help='feature dropout rate')
    parser.add_argument('--prop_dropout', type=float, default=0.1, help='propagation dropout rate')
    parser.add_argument('--norm', type=str, default='layer', choices=['batch', 'layer', 'none'], help='normalization type')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of experiment runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Computing device')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--k', type=int, default=-1, help='Number of principal eigenvalues')
    return parser.parse_args()

def main(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f'cuda:{args.gpu_id}')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    all_results = []
    start_time = time.time()
    for run_seed in range(args.seed, args.seed + args.num_runs):
        try:
            result = run_single_experiment(args, run_seed, device)
            all_results.append(result)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Experiment {run_seed - args.seed + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    if len(all_results) > 0:
        aggregated_results = aggregate_results(all_results, args)
        print_results(aggregated_results, args)
        total_time = time.time() - start_time
        print(f"\nTotal Experiment Time: {total_time / 60:.2f} minutes")
        print(f"\nFairGC Experiment Completed!")
    else:
        print(f"All experiment runs failed")
        total_time = time.time() - start_time
        print(f"\nTotal Experiment Time: {total_time / 60:.2f} minutes")
        print(f"\nNo valid experiment runs completed")

if __name__ == "__main__":
    args = parse_args()
    main(args)