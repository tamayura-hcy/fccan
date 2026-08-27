"""t-SNE visualization & training-log persistence."""
import os
import json
import shutil

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from util.data_utils import ensure_save_dir


def save_tsne_and_metrics(src_label, tgt_label, source_dataset, target_dataset,
                          iter_idx, tgt_encoder, classifier,
                          dataloader_s_test, dataloader_t_test,
                          dataset_size_src, dataset_size_tgt,
                          metrics, results_root='results', saves_dir=None):
    """Save t-SNE / metrics / heatmap for this run; also write a copy to saves_dir if given."""
    results_dir = os.path.join(results_root, f"{source_dataset}_to_{target_dataset}", f"iter{iter_idx}")
    os.makedirs(results_dir, exist_ok=True)
    if saves_dir:
        ensure_save_dir(saves_dir)

    # 1) Save metrics as json
    metrics_path = os.path.join(results_dir, "metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        if saves_dir:
            shutil.copy2(metrics_path, os.path.join(saves_dir, "metrics.json"))
    except Exception as e:
        print(f"[Warning] Failed to save metrics to {metrics_path}: {e}")

    # 2) Extract features for t-SNE (source + target test sets)
    tgt_encoder.eval()
    classifier.eval()
    feats = []
    labels = []
    domains = []  # 0=source, 1=target
    with torch.no_grad():
        # Source domain
        for inputs, lab in dataloader_s_test:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            feat = tgt_encoder(inputs)[0]
            feats.append(feat.cpu())
            labels.append(lab.cpu())
            domains.append(torch.zeros_like(lab.cpu()))  # 0=source
        # Target domain
        for inputs, lab in dataloader_t_test:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            feat = tgt_encoder(inputs)[0]
            feats.append(feat.cpu())
            labels.append(lab.cpu())
            domains.append(torch.ones_like(lab.cpu()))  # 1=target
    if not feats:
        return
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    domains = torch.cat(domains, dim=0)
    # Subsample if too many points
    max_points = 2000
    if feats.size(0) > max_points:
        idx = torch.randperm(feats.size(0))[:max_points]
        feats = feats[idx]
        labels = labels[idx]
    try:
        tsne = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto")
        emb = tsne.fit_transform(feats.numpy())
        plt.figure(figsize=(6, 6))
        num_classes = int(labels.max().item()) + 1
        colors = ['tab:red', 'tab:green', 'tab:blue']
        lab_np = labels.numpy()
        dom_np = domains.numpy()
        for c in range(num_classes):
            color = colors[c % len(colors)]
            # Source: circles
            mask_src = (lab_np == c) & (dom_np == 0)
            if mask_src.any():
                plt.scatter(emb[mask_src, 0], emb[mask_src, 1],
                            s=8, c=color, marker='o', alpha=0.7,
                            label=f"Src-{c}")
            # Target: triangles
            mask_tgt = (lab_np == c) & (dom_np == 1)
            if mask_tgt.any():
                plt.scatter(emb[mask_tgt, 0], emb[mask_tgt, 1],
                            s=12, c=color, marker='^', alpha=0.7,
                            label=f"Tgt-{c}")
        plt.legend(markerscale=1.5, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"t-SNE {source_dataset}->{target_dataset} iter{iter_idx}")
        tsne_path = os.path.join(results_dir, "tsne.png")
        plt.tight_layout()
        plt.savefig(tsne_path, dpi=200)
        plt.close()
        if saves_dir:
            shutil.copy2(tsne_path, os.path.join(saves_dir, "tsne.png"))
    except Exception as e:
        print(f"[Warning] Failed to compute/save t-SNE for {source_dataset}->{target_dataset} iter{iter_idx}: {e}")


def save_src_tsne(source_dataset, iter_idx, encoder, classifier,
                  dataloader_s_test, dataset_size_src,
                  dataloader_t_test=None,
                  results_root='results', saves_dir=None):
    """Plot t-SNE on the source test set; optionally overlay target points (triangles) to compare distributions."""
    results_dir = os.path.join(results_root, f"{source_dataset}_src_only", f"iter{iter_idx}")
    os.makedirs(results_dir, exist_ok=True)
    if saves_dir:
        ensure_save_dir(saves_dir)

    encoder.eval()
    classifier.eval()
    feats = []
    labels = []
    domains = []  # 0=source, 1=target
    with torch.no_grad():
        for inputs, lab in dataloader_s_test:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            feat = encoder(inputs)[0]
            feats.append(feat.cpu())
            labels.append(lab.cpu())
            domains.append(torch.zeros_like(lab.cpu()))
        if dataloader_t_test is not None:
            for inputs, lab in dataloader_t_test:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                feat = encoder(inputs)[0]
                feats.append(feat.cpu())
                labels.append(lab.cpu())
                domains.append(torch.ones_like(lab.cpu()))
    if not feats:
        return
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    domains = torch.cat(domains, dim=0)
    # Subsample if too many points
    max_points = 2000
    if feats.size(0) > max_points:
        idx = torch.randperm(feats.size(0))[:max_points]
        feats = feats[idx]
        labels = labels[idx]
        domains = domains[idx]
    try:
        tsne = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto")
        emb = tsne.fit_transform(feats.numpy())
        plt.figure(figsize=(6, 6))
        num_classes = int(labels.max().item()) + 1
        colors = ['tab:red', 'tab:green', 'tab:blue']
        lab_np = labels.numpy()
        dom_np = domains.numpy()
        for c in range(num_classes):
            color = colors[c % len(colors)]
            mask_src = (lab_np == c) & (dom_np == 0)
            if mask_src.any():
                plt.scatter(emb[mask_src, 0], emb[mask_src, 1],
                            s=8, c=color, marker='o', alpha=0.7,
                            label=f"Src-{c}")
            mask_tgt = (lab_np == c) & (dom_np == 1)
            if mask_tgt.any():
                plt.scatter(emb[mask_tgt, 0], emb[mask_tgt, 1],
                            s=12, c=color, marker='^', alpha=0.7,
                            label=f"Tgt-{c}")
        plt.legend(markerscale=1.5, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        title_suffix = " + Target(test)" if dataloader_t_test is not None else ""
        plt.title(f"t-SNE Source {source_dataset}{title_suffix} iter{iter_idx}")
        tsne_path = os.path.join(results_dir, "tsne.png")
        plt.tight_layout()
        plt.savefig(tsne_path, dpi=200)
        plt.close()
        if saves_dir:
            shutil.copy2(tsne_path, os.path.join(saves_dir, "tsne_src.png"))
    except Exception as e:
        print(f"[Warning] Failed to compute/save source-only t-SNE for {source_dataset} iter{iter_idx}: {e}")


def write_iter_training_log(mode_name, source_dataset, target_dataset, iter_idx,
                            src_final_acc, tgt_epoch_accs, save_name, results_root='results'):
    """Save one txt per iter: final source acc + per-epoch target accs."""
    log_dir = os.path.join(results_root, "training_logs", mode_name, f"{source_dataset}_to_{target_dataset}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"iter{iter_idx}.txt")
    lines = []
    lines.append(f"mode={mode_name}")
    lines.append(f"task={source_dataset}->{target_dataset}")
    lines.append(f"iter={iter_idx}")
    lines.append(f"save_dir={save_name}")
    lines.append(f"source_final_acc={src_final_acc:.6f}" if src_final_acc is not None else "source_final_acc=NA")
    if tgt_epoch_accs:
        lines.append("target_epoch_accs=" + ",".join([f"{x:.6f}" for x in tgt_epoch_accs]))
        lines.append(f"target_final_acc={tgt_epoch_accs[-1]:.6f}")
        lines.append(f"target_best_acc={max(tgt_epoch_accs):.6f}")
    else:
        lines.append("target_epoch_accs=NA")
        lines.append("target_final_acc=NA")
        lines.append("target_best_acc=NA")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [IterLog] saved: {log_path}")


def write_final_average_log(mode_name, source_dataset, target_dataset,
                            src_acc_list, tgt_epoch_acc_lists, results_root='results'):
    """Save one txt with the averaged result over all iters."""
    log_dir = os.path.join(results_root, "training_logs", mode_name, f"{source_dataset}_to_{target_dataset}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "final_avg.txt")

    lines = []
    lines.append(f"mode={mode_name}")
    lines.append(f"task={source_dataset}->{target_dataset}")
    lines.append(f"num_iters={len(tgt_epoch_acc_lists)}")
    if src_acc_list:
        lines.append(f"source_final_acc_mean={sum(src_acc_list)/len(src_acc_list):.6f}")
    else:
        lines.append("source_final_acc_mean=NA")

    # Align to the shortest epoch list for per-epoch means (avoids length mismatch on early abort)
    valid_lists = [x for x in tgt_epoch_acc_lists if x]
    if valid_lists:
        min_len = min(len(x) for x in valid_lists)
        epoch_means = []
        for i in range(min_len):
            epoch_means.append(sum(x[i] for x in valid_lists) / len(valid_lists))
        lines.append("target_epoch_acc_mean=" + ",".join([f"{x:.6f}" for x in epoch_means]))
        lines.append(f"target_final_acc_mean={epoch_means[-1]:.6f}")
        lines.append(f"target_best_acc_mean={max(epoch_means):.6f}")
    else:
        lines.append("target_epoch_acc_mean=NA")
        lines.append("target_final_acc_mean=NA")
        lines.append("target_best_acc_mean=NA")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [FinalLog] saved: {log_path}")
