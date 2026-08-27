"""Unified result export: writes a command's full results to txt when --save_result_txt is on.

Contains: the command line (reproducible), task / seed / iter / save dir, the main-metric
name (EMA / SWA / student), per-epoch student + EMA acc, and the final 10 full metrics.
"""
import os


def _fmt(v):
    return "NA" if v is None else "{:.6f}".format(float(v))


def write_full_result_txt(mode_name, source_dataset, target_dataset, iter_idx,
                          seed, save_name, cmd,
                          tgt_epoch_accs, ema_epoch_accs,
                          final_metric_name, final_metrics,
                          source_final_acc=None,
                          results_root='results'):
    """Write result file: {results_root}/result_logs/{mode}/{src}_to_{tgt}/iter{iter_idx}_seed{seed}.txt

    final_metrics: dict with at least the 10 keys acc/auc/recall/precision/f1/bacc/specificity/kappa/gmean/mcc.
    """
    log_dir = os.path.join(results_root, "result_logs", mode_name,
                           f"{source_dataset}_to_{target_dataset}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"iter{iter_idx}_seed{seed}.txt")

    lines = []
    lines.append("# FEA-Net result log")
    lines.append(f"mode={mode_name}")
    lines.append(f"task={source_dataset}->{target_dataset}")
    lines.append(f"iter={iter_idx}")
    lines.append(f"seed={seed}")
    lines.append(f"save_dir={save_name}")
    lines.append(f"cmd={cmd}")
    lines.append(f"final_metric={final_metric_name}")
    lines.append("")

    # ── per-epoch acc ──
    lines.append("== per-epoch acc ==")
    lines.append("epoch,student,ema")
    _n_ema = len(ema_epoch_accs) if ema_epoch_accs else 0
    _n = max(len(tgt_epoch_accs) if tgt_epoch_accs else 0, _n_ema)
    for i in range(_n):
        s = _fmt(tgt_epoch_accs[i]) if tgt_epoch_accs and i < len(tgt_epoch_accs) else "NA"
        e = _fmt(ema_epoch_accs[i]) if ema_epoch_accs and i < len(ema_epoch_accs) else "NA"
        lines.append(f"{i + 1},{s},{e}")
    lines.append("")

    # ── final metrics (main metric) ──
    lines.append(f"== final metrics ({final_metric_name}) ==")
    if final_metrics:
        for _k in ["acc", "auc", "recall", "precision", "f1",
                   "bacc", "specificity", "kappa", "gmean", "mcc"]:
            lines.append(f"{_k}={_fmt(final_metrics.get(_k))}")
    else:
        lines.append("final_metrics=NA")
    if source_final_acc is not None:
        lines.append(f"source_final_acc={_fmt(source_final_acc)}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [ResultLog] saved: {log_path}")
    return log_path
