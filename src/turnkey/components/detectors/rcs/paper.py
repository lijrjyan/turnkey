from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turnkey._internal.hf_deps import import_torch as import_hf_torch
from turnkey.runtime_providers import HFLastTokenHiddenStateProvider, LastTokenHiddenStateConfig
from turnkey.schema import DetectorDecision, Sample

from .training import (
    _HiddenStateProvider,
    _RCSTrainExample,
    _paper_training_example_summary,
    _sample_from_train_example,
    _split_paper_examples,
)


_RCS_PAPER_SCORING_STATE_SCHEMA = "rcs_paper_scoring_state/v1"
_RCS_PAPER_THRESHOLD_GRID_SIZE = 200
_RCS_LAYER_SELECTION_AUTO = "auto_svm_silhouette_ratio"
_RCS_LAYER_SELECTION_OFFICIAL_SWEEP = "official_sweep"


def _import_torch():
    return import_hf_torch(
        error_message="rcs: mode=paper requires optional deps. Install with: pip install -e '.[hf]'"
    )


def _normalize_layer_selection_strategy(value: Any) -> str:
    if value is None:
        return _RCS_LAYER_SELECTION_AUTO
    if not isinstance(value, str):
        raise ValueError("rcs: layer_selection_strategy must be a string")
    strategy = value.strip().lower().replace("-", "_")
    aliases = {
        "auto": _RCS_LAYER_SELECTION_AUTO,
        "auto_svm": _RCS_LAYER_SELECTION_AUTO,
        "auto_svm_silhouette": _RCS_LAYER_SELECTION_AUTO,
        _RCS_LAYER_SELECTION_AUTO: _RCS_LAYER_SELECTION_AUTO,
        "official": _RCS_LAYER_SELECTION_OFFICIAL_SWEEP,
        "official_layer_sweep": _RCS_LAYER_SELECTION_OFFICIAL_SWEEP,
        "official_parity": _RCS_LAYER_SELECTION_OFFICIAL_SWEEP,
        _RCS_LAYER_SELECTION_OFFICIAL_SWEEP: _RCS_LAYER_SELECTION_OFFICIAL_SWEEP,
    }
    if strategy not in aliases:
        raise ValueError(
            "rcs: layer_selection_strategy must be "
            f"'{_RCS_LAYER_SELECTION_AUTO}' or '{_RCS_LAYER_SELECTION_OFFICIAL_SWEEP}'"
        )
    return aliases[strategy]


def _normalize_01(xs: list[float]) -> list[float]:
    if not xs:
        return []
    mn = min(xs)
    mx = max(xs)
    if not math.isfinite(mn) or not math.isfinite(mx):
        return [0.0 for _ in xs]
    if abs(mx - mn) <= 1e-12:
        return [0.0 for _ in xs]
    return [(float(x - mn) / float(mx - mn)) for x in xs]


def _svm_margin_torch(x, y, *, iters: int, seed: int) -> float:
    torch, _, F = _import_torch()

    if x.ndim != 2:
        raise ValueError("rcs: svm expects 2D features")
    n, d = x.shape
    if int(n) < 4:
        return 0.0

    x = x.to(dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True) + 1e-6)

    # Add bias via feature augmentation.
    xb = torch.cat([x, torch.ones((n, 1), dtype=x.dtype, device=x.device)], dim=1)
    y01 = y.to(dtype=torch.int64)
    ypm = torch.where(y01 > 0, torch.tensor(1.0, device=x.device), torch.tensor(-1.0, device=x.device))

    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))
    w = torch.zeros((d + 1,), dtype=torch.float32, device=x.device, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    reg = 1e-2
    for _ in range(max(10, int(iters))):
        opt.zero_grad(set_to_none=True)
        scores = xb @ w
        loss = F.relu(1.0 - ypm * scores).mean() + reg * (w[:-1] * w[:-1]).sum()
        loss.backward()
        opt.step()

        idx = torch.randint(0, n, (n,), generator=g, device=x.device)
        xb = xb[idx]
        ypm = ypm[idx]

    margin = 1.0 / (float(torch.linalg.vector_norm(w[:-1]).detach()) + 1e-8)
    return float(margin)


def _silhouette_two_clusters(x, y) -> float:
    torch, _, _ = _import_torch()

    if x.ndim != 2:
        raise ValueError("rcs: silhouette expects 2D features")
    n = int(x.shape[0])
    if n < 3:
        return 0.0
    y = y.to(dtype=torch.int64)
    if int((y == 0).sum()) < 2 or int((y == 1).sum()) < 2:
        return 0.0

    d = torch.cdist(x, x)
    out = []
    idxs = torch.arange(n, device=x.device)
    for i in range(n):
        same = (y == int(y[i])) & (idxs != i)
        other = y != int(y[i])
        if not bool(same.any()) or not bool(other.any()):
            out.append(torch.tensor(0.0, device=x.device))
            continue
        a = d[i][same].mean()
        b = d[i][other].mean()
        denom = torch.maximum(a, b)
        out.append(torch.where(denom > 1e-12, (b - a) / denom, torch.tensor(0.0, device=x.device)))
    return float(torch.stack(out).mean().item())


def _discriminative_ratio(x, y) -> float:
    torch, _, _ = _import_torch()

    y = y.to(dtype=torch.int64)
    xb = x[y == 0]
    xm = x[y == 1]
    if int(xb.shape[0]) < 2 or int(xm.shape[0]) < 2:
        return 0.0
    mu_b = xb.mean(dim=0)
    mu_m = xm.mean(dim=0)
    inter = torch.linalg.vector_norm(mu_b - mu_m)
    sigma_b = torch.linalg.vector_norm(xb - mu_b, dim=1).mean()
    sigma_m = torch.linalg.vector_norm(xm - mu_m, dim=1).mean()
    pooled = 0.5 * (sigma_b + sigma_m)
    return float((inter / (pooled + 1e-8)).item())


def _auto_select_layer(
    hidden_state_provider: _HiddenStateProvider,
    examples: list[_RCSTrainExample],
    *,
    max_samples: int,
    svm_iters: int,
    seed: int,
) -> tuple[int, dict[str, list[float]]]:
    torch, _, F = _import_torch()

    rng = random.Random(int(seed))
    pool = list(examples)
    rng.shuffle(pool)
    pool = pool[: max(4, int(max_samples))]

    ys = torch.tensor([0 if ex.is_benign else 1 for ex in pool], dtype=torch.int64)
    if int((ys == 0).sum()) < 2 or int((ys == 1).sum()) < 2:
        raise ValueError("rcs: auto layer selection needs >=2 benign and >=2 malicious examples")

    per_layer: list[list[Any]] = []
    n_layers = None
    for i, ex in enumerate(pool):
        s = _sample_from_train_example(ex, index=i)
        vecs = _last_token_by_layer_tensor(hidden_state_provider(s))
        if n_layers is None:
            n_layers = int(vecs.shape[0])
            per_layer = [[] for _ in range(n_layers)]
        for layer_idx in range(int(vecs.shape[0])):
            per_layer[layer_idx].append(vecs[layer_idx].cpu())

    assert n_layers is not None
    gammas: list[float] = []
    sils: list[float] = []
    ratios: list[float] = []
    for layer_idx in range(n_layers):
        x = torch.stack(per_layer[layer_idx], dim=0).to(dtype=torch.float32)
        x = F.normalize(x, p=2.0, dim=1)
        gammas.append(_svm_margin_torch(x, ys, iters=svm_iters, seed=seed + 1000 + layer_idx))
        sils.append(_silhouette_two_clusters(x, ys))
        ratios.append(_discriminative_ratio(x, ys))

    g_hat = _normalize_01(gammas)
    s_hat = _normalize_01(sils)
    r_hat = _normalize_01(ratios)
    composite = [(a + b + c) / 3.0 for a, b, c in zip(g_hat, s_hat, r_hat, strict=True)]
    best = max(range(n_layers), key=lambda i: composite[i])

    metrics = {"gamma": gammas, "silhouette": sils, "ratio": ratios, "composite": composite}
    return int(best), metrics


def _finite_float(value: float) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, 6)


def _layer_selection_diagnostics(
    *,
    selected_layer: int,
    metrics: dict[str, list[float]] | None,
) -> dict[str, Any]:
    if metrics is None:
        return {"strategy": "fixed", "selected_layer": int(selected_layer)}

    composite = metrics.get("composite", [])
    n_layers = len(composite)
    ranked = sorted(range(n_layers), key=lambda i: composite[i], reverse=True)
    top_layers: list[dict[str, Any]] = []
    for rank, layer_idx in enumerate(ranked[:8], start=1):
        top_layers.append(
            {
                "rank": rank,
                "layer": int(layer_idx),
                "composite": _finite_float(metrics["composite"][layer_idx]),
                "gamma": _finite_float(metrics["gamma"][layer_idx]),
                "silhouette": _finite_float(metrics["silhouette"][layer_idx]),
                "ratio": _finite_float(metrics["ratio"][layer_idx]),
            }
        )

    selected = None
    for row in top_layers:
        if row["layer"] == int(selected_layer):
            selected = dict(row)
            break
    if selected is None and 0 <= int(selected_layer) < n_layers:
        i = int(selected_layer)
        selected = {
            "rank": ranked.index(i) + 1,
            "layer": i,
            "composite": _finite_float(metrics["composite"][i]),
            "gamma": _finite_float(metrics["gamma"][i]),
            "silhouette": _finite_float(metrics["silhouette"][i]),
            "ratio": _finite_float(metrics["ratio"][i]),
        }

    return {
        "strategy": "auto_svm_silhouette_ratio",
        "selected_layer": int(selected_layer),
        "n_layers": int(n_layers),
        "top_layers": top_layers,
        "selected": selected,
        "truncated": n_layers > len(top_layers),
    }


def _metric_or_none(metrics: dict[str, Any], key: str) -> float | int | str | None:
    value = metrics.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _finite_float(value)
    return None


def _official_sweep_layer_selection_diagnostics(
    *,
    selected_layer: int,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    def rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
        metrics = row["metrics"]
        official = metrics.get("official_combined")
        objective = metrics.get("threshold_objective")
        fpr = metrics.get("fpr")
        return (
            float(official) if official is not None else float("-inf"),
            float(objective) if objective is not None else float("-inf"),
            -float(fpr) if fpr is not None else float("-inf"),
            -int(row["layer"]),
        )

    ranked = sorted(candidates, key=rank_key, reverse=True)
    top_layers: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[:8], start=1):
        metrics = row["metrics"]
        top_layers.append(
            {
                "rank": int(rank),
                "layer": int(row["layer"]),
                "official_combined": _metric_or_none(metrics, "official_combined"),
                "accuracy": _metric_or_none(metrics, "accuracy"),
                "f1": _metric_or_none(metrics, "f1"),
                "tpr": _metric_or_none(metrics, "tpr"),
                "fpr": _metric_or_none(metrics, "fpr"),
                "auroc": _metric_or_none(metrics, "auroc"),
                "auprc": _metric_or_none(metrics, "auprc"),
                "threshold": _metric_or_none(metrics, "threshold"),
                "threshold_objective": _metric_or_none(metrics, "threshold_objective"),
                "score_std": _metric_or_none(metrics, "score_std"),
            }
        )

    selected = None
    for rank, row in enumerate(ranked, start=1):
        if int(row["layer"]) == int(selected_layer):
            metrics = row["metrics"]
            selected = {
                "rank": int(rank),
                "layer": int(row["layer"]),
                "official_combined": _metric_or_none(metrics, "official_combined"),
                "accuracy": _metric_or_none(metrics, "accuracy"),
                "f1": _metric_or_none(metrics, "f1"),
                "tpr": _metric_or_none(metrics, "tpr"),
                "fpr": _metric_or_none(metrics, "fpr"),
                "auroc": _metric_or_none(metrics, "auroc"),
                "auprc": _metric_or_none(metrics, "auprc"),
                "threshold": _metric_or_none(metrics, "threshold"),
                "threshold_objective": _metric_or_none(metrics, "threshold_objective"),
                "score_std": _metric_or_none(metrics, "score_std"),
            }
            break

    ranking_split = None
    if candidates:
        ranking_split = candidates[0]["metrics"].get("ranking_split")
    return {
        "strategy": _RCS_LAYER_SELECTION_OFFICIAL_SWEEP,
        "reference_behavior": "layer_specific_projection_threshold_sweep",
        "ranking_metric": "0.5*accuracy + 0.3*auroc + 0.2*auprc",
        "ranking_split": ranking_split,
        "selected_layer": int(selected_layer),
        "n_layers": int(len(candidates)),
        "top_layers": top_layers,
        "selected": selected,
        "truncated": len(candidates) > len(top_layers),
    }


def _reference_shrinkage_intensity(*, n: int, dim: int, trace_cov: float, frobenius_sq: float) -> float:
    if n <= 1:
        return 1.0
    trace_sq = float(trace_cov) ** 2
    denominator = float(n + 2) * (float(frobenius_sq) - trace_sq / float(dim))
    if denominator <= 0.0:
        return 0.0
    numerator = (float(n - 2) / float(n)) * float(frobenius_sq) + trace_sq
    return min(1.0, max(0.0, numerator / denominator))


def _ledoit_wolf_cov(x, *, var_floor: float, min_samples: int = 50) -> tuple[Any, float]:
    torch, _, _ = _import_torch()

    n = int(x.shape[0])
    p = int(x.shape[1])
    if n <= 1 or n < int(min_samples):
        cov = torch.eye(p, dtype=torch.float32, device=x.device)
        return cov, 1.0

    x = x.to(dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    s = (x.T @ x) / float(n - 1)
    mu = torch.trace(s) / float(p)
    prior = torch.eye(p, dtype=s.dtype, device=s.device) * mu
    trace_cov = float(torch.trace(s).item())
    frobenius_sq = float(torch.sum(s * s).item())
    shrink = _reference_shrinkage_intensity(
        n=n,
        dim=p,
        trace_cov=trace_cov,
        frobenius_sq=frobenius_sq,
    )
    cov = (1.0 - shrink) * s + shrink * prior
    cov = cov.to(dtype=torch.float32)
    return cov, shrink


def _new_rcs_projection(*, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
    _, nn, _ = _import_torch()
    h1 = int(hidden_dim)
    h2 = h1 // 2

    class Proj(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), h1),
                nn.BatchNorm1d(h1),
                nn.ReLU(),
                nn.Dropout(p=float(dropout)),
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                nn.ReLU(),
                nn.Dropout(p=float(dropout)),
                nn.Linear(h2, int(output_dim)),
                nn.BatchNorm1d(int(output_dim)),
            )
            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        def forward(self, inp):  # type: ignore[override]
            return self.net(inp)

    return Proj()


def _fit_projection(
    x,
    y,
    dataset_ids,
    *,
    out_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    beta: float,
    md: float,
    ms: float,
    dropout: float,
    seed: int,
):
    torch, _, _ = _import_torch()

    if x.ndim != 2:
        raise ValueError("rcs: projection expects 2D features")
    n, d = x.shape
    if int(n) < 4:
        raise ValueError("rcs: need >=4 training samples for paper mode")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    random.seed(int(seed))

    g = _new_rcs_projection(input_dim=int(d), output_dim=int(out_dim), dropout=float(dropout)).to(x.device)
    opt = torch.optim.Adam(g.parameters(), lr=float(lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, int(epochs)),
        eta_min=float(lr) * 0.05,
    )

    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))

    idxs = torch.arange(n, device=x.device)
    effective_batch_size = min(max(1, int(batch_size)), int(n))
    max_patience = 15
    best_loss = float("inf")
    patience = 0
    for _ in range(max(1, int(epochs))):
        perm = idxs[torch.randperm(n, generator=gen, device=x.device)]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n - effective_batch_size + 1, effective_batch_size):
            batch = perm[start : start + effective_batch_size]
            xb = x[batch]
            yb = y[batch]
            db = dataset_ids[batch]

            zb = g(xb)
            loss, _dataset_loss, _toxicity_loss = _projection_contrastive_loss(
                zb,
                db,
                yb,
                margin_dataset=float(md),
                margin_toxicity=float(ms),
                dataset_weight=float(alpha),
                toxicity_weight=float(beta),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        if n_batches <= 0:
            continue
        avg_loss = epoch_loss / float(n_batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
        if patience >= max_patience:
            break

    g.eval()
    return g


def _projection_contrastive_loss(
    embeddings,
    dataset_labels,
    toxicity_labels,
    *,
    margin_dataset: float,
    margin_toxicity: float,
    dataset_weight: float,
    toxicity_weight: float,
):
    torch, _, F = _import_torch()

    device = embeddings.device
    batch_size = int(embeddings.shape[0])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    dataset_loss = torch.tensor(0.0, dtype=embeddings.dtype, device=device)
    unique_datasets = torch.unique(dataset_labels)
    if len(unique_datasets) > 1 and batch_size > 1:
        distances = torch.cdist(embeddings, embeddings, p=2)
        same = (dataset_labels.unsqueeze(0) == dataset_labels.unsqueeze(1)).to(dtype=embeddings.dtype)
        diff = 1.0 - same
        eye_mask = 1.0 - torch.eye(batch_size, dtype=embeddings.dtype, device=device)
        same = same * eye_mask
        diff = diff * eye_mask

        intra_dataset = (distances * same).sum() / same.sum() if bool(same.sum() > 0) else dataset_loss
        inter_dataset = (
            F.relu(torch.tensor(float(margin_dataset), dtype=embeddings.dtype, device=device) - distances) * diff
        )
        inter_dataset = inter_dataset.sum() / diff.sum() if bool(diff.sum() > 0) else dataset_loss
        dataset_loss = intra_dataset + inter_dataset

    toxicity_loss = torch.tensor(0.0, dtype=embeddings.dtype, device=device)
    unique_toxicity = torch.unique(toxicity_labels)
    if len(unique_toxicity) > 1:
        benign = embeddings[toxicity_labels == 0]
        malicious = embeddings[toxicity_labels == 1]
        if int(benign.shape[0]) > 0 and int(malicious.shape[0]) > 0:
            benign_centroid = benign.mean(dim=0)
            malicious_centroid = malicious.mean(dim=0)
            centroid_distance = F.pairwise_distance(
                benign_centroid.unsqueeze(0),
                malicious_centroid.unsqueeze(0),
            )
            toxicity_loss = F.relu(
                torch.tensor(float(margin_toxicity), dtype=embeddings.dtype, device=device) - centroid_distance
            ).mean()
            if int(benign.shape[0]) > 1:
                toxicity_loss = toxicity_loss + torch.linalg.vector_norm(
                    benign - benign_centroid.unsqueeze(0),
                    dim=1,
                ).mean()
            if int(malicious.shape[0]) > 1:
                toxicity_loss = toxicity_loss + torch.linalg.vector_norm(
                    malicious - malicious_centroid.unsqueeze(0),
                    dim=1,
                ).mean()

    total_loss = float(dataset_weight) * dataset_loss + float(toxicity_weight) * toxicity_loss
    return total_loss, dataset_loss, toxicity_loss


def _score_stats(scores: list[float]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    mean = sum(float(score) for score in scores) / float(len(scores))
    var = sum((float(score) - mean) ** 2 for score in scores) / float(len(scores))
    return mean, math.sqrt(var)


def _quantile_sorted(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, float(q))) * float(len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - float(lo)
    return float(sorted_values[lo]) * (1.0 - weight) + float(sorted_values[hi]) * weight


def _paper_threshold_candidates(
    *,
    scores: list[float],
    labels: list[int],
    grid_size: int,
) -> list[float]:
    benign_scores = [float(score) for score, label in zip(scores, labels, strict=True) if int(label) == 0]
    malicious_scores = [float(score) for score, label in zip(scores, labels, strict=True) if int(label) == 1]
    if not benign_scores or not malicious_scores:
        return [0.0]

    benign_mean, benign_std = _score_stats(benign_scores)
    malicious_mean, malicious_std = _score_stats(malicious_scores)
    separation = abs(malicious_mean - benign_mean) / (benign_std + malicious_std + 1e-8)
    if separation > 2.0:
        midpoint = (benign_mean + malicious_mean) / 2.0
        range_width = min(benign_std, malicious_std) * 2.0
        if range_width <= 1e-12:
            lo = hi = midpoint
        else:
            lo = midpoint - range_width
            hi = midpoint + range_width
    else:
        finite_scores = sorted(float(score) for score in scores if math.isfinite(float(score)))
        lo = _quantile_sorted(finite_scores, 0.05)
        hi = _quantile_sorted(finite_scores, 0.95)
        range_width = hi - lo
        lo -= 0.2 * range_width
        hi += 0.2 * range_width

    if abs(hi - lo) <= 1e-12:
        return [float(lo)]
    n = max(2, int(grid_size))
    step = (hi - lo) / float(n - 1)
    return [float(lo + float(i) * step) for i in range(n)]


def _threshold_objective(
    *,
    scores: list[float],
    labels: list[int],
    threshold: float,
    w_bal_acc: float,
    w_f1: float,
) -> tuple[float, float, float]:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels, strict=True):
        pred = bool(float(score) > float(threshold))
        if int(label) == 1 and pred:
            tp += 1
        elif int(label) == 1 and not pred:
            fn += 1
        elif int(label) == 0 and pred:
            fp += 1
        else:
            tn += 1
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tpr
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return float(w_bal_acc) * bal_acc + float(w_f1) * f1, float(bal_acc), float(f1)


def _binary_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = [float(score) for score, label in zip(scores, labels, strict=True) if int(label) == 1]
    negatives = [float(score) for score, label in zip(scores, labels, strict=True) if int(label) == 0]
    if not positives or not negatives:
        return None

    wins = 0.0
    total = float(len(positives) * len(negatives))
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return float(wins / total)


def _binary_average_precision(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(1 for label in labels if int(label) == 1)
    if positives <= 0 or positives >= len(labels):
        return None

    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: float(item[0]), reverse=True)
    seen_pos = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ranked, start=1):
        if int(label) != 1:
            continue
        seen_pos += 1
        precision_sum += float(seen_pos) / float(rank)
    return float(precision_sum / float(positives))


def _binary_score_metrics(
    *,
    scores: list[float],
    labels: list[int],
    threshold: float,
    w_bal_acc: float,
    w_f1: float,
) -> dict[str, float | int | None]:
    if len(scores) != len(labels) or not scores:
        return {
            "count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "f1": None,
            "tpr": None,
            "fpr": None,
            "auroc": None,
            "auprc": None,
            "threshold_objective": None,
            "official_combined": None,
            "score_std": None,
            "score_min": None,
            "score_max": None,
        }

    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels, strict=True):
        pred = bool(float(score) > float(threshold))
        if int(label) == 1 and pred:
            tp += 1
        elif int(label) == 1 and not pred:
            fn += 1
        elif int(label) == 0 and pred:
            fp += 1
        else:
            tn += 1

    count = len(scores)
    accuracy = (tp + tn) / float(count)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2.0 * precision * tpr / (precision + tpr)) if (precision + tpr) else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)
    objective = float(w_bal_acc) * balanced_accuracy + float(w_f1) * f1
    auroc = _binary_auc(scores, labels)
    auprc = _binary_average_precision(scores, labels)
    official_combined = 0.5 * accuracy + 0.3 * (auroc or 0.0) + 0.2 * (auprc or 0.0)
    _mean_score, score_std = _score_stats(scores)

    return {
        "count": int(count),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "auroc": float(auroc) if auroc is not None else None,
        "auprc": float(auprc) if auprc is not None else None,
        "threshold_objective": float(objective),
        "official_combined": float(official_combined),
        "score_std": float(score_std),
        "score_min": float(min(scores)),
        "score_max": float(max(scores)),
    }


def _calibrate_threshold(
    *,
    scores: list[float],
    labels: list[int],
    w_bal_acc: float,
    w_f1: float,
    grid_size: int = _RCS_PAPER_THRESHOLD_GRID_SIZE,
) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("rcs: calibrate needs non-empty scores/labels")

    if len(set(int(label) for label in labels)) < 2:
        return 0.0

    candidates = _paper_threshold_candidates(scores=scores, labels=labels, grid_size=grid_size)
    best_theta = float(candidates[0])
    best_obj, _best_bal_acc, _best_f1 = _threshold_objective(
        scores=scores,
        labels=labels,
        threshold=best_theta,
        w_bal_acc=w_bal_acc,
        w_f1=w_f1,
    )
    for theta in candidates[1:]:
        obj, _bal_acc, _f1 = _threshold_objective(
            scores=scores,
            labels=labels,
            threshold=float(theta),
            w_bal_acc=w_bal_acc,
            w_f1=w_f1,
        )
        if obj > best_obj + 1e-12:
            best_obj = obj
            best_theta = float(theta)
    return float(best_theta)


@dataclass
class _MCDGaussian:
    mean: Any
    cov: Any


@dataclass
class _RCSPaperScoringState:
    layer: int
    layer_diagnostics: dict[str, Any]
    projection: Any
    method: str
    threshold: float
    k: int
    mcd_benign: list[_MCDGaussian]
    mcd_malicious: list[_MCDGaussian]
    kcd_benign: Any | None
    kcd_malicious: Any | None


@dataclass
class _RCSPaperState:
    hidden_state_provider: _HiddenStateProvider
    device: Any | None
    scoring: _RCSPaperScoringState


def _save_rcs_paper_scoring_state(path: str | Path, state: _RCSPaperScoringState) -> None:
    torch, nn, _ = _import_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    linear_layers = [module for module in state.projection.net if isinstance(module, nn.Linear)]
    dropout_layers = [module for module in state.projection.net if isinstance(module, nn.Dropout)]
    if not linear_layers:
        raise ValueError("rcs: projection state has no linear layers")
    projection_meta = {
        "input_dim": int(linear_layers[0].in_features),
        "hidden_dim": int(linear_layers[0].out_features),
        "output_dim": int(linear_layers[-1].out_features),
        "dropout": float(dropout_layers[0].p) if dropout_layers else 0.3,
    }
    payload = {
        "schema_version": _RCS_PAPER_SCORING_STATE_SCHEMA,
        "layer": int(state.layer),
        "layer_diagnostics": dict(state.layer_diagnostics),
        "method": state.method,
        "threshold": float(state.threshold),
        "k": int(state.k),
        "projection_meta": projection_meta,
        "projection_state_dict": {key: value.detach().cpu() for key, value in state.projection.state_dict().items()},
        "mcd_benign": [
            {"mean": item.mean.detach().cpu(), "cov": item.cov.detach().cpu()} for item in state.mcd_benign
        ],
        "mcd_malicious": [
            {"mean": item.mean.detach().cpu(), "cov": item.cov.detach().cpu()} for item in state.mcd_malicious
        ],
        "kcd_benign": state.kcd_benign.detach().cpu() if state.kcd_benign is not None else None,
        "kcd_malicious": state.kcd_malicious.detach().cpu() if state.kcd_malicious is not None else None,
    }
    torch.save(payload, path)


def _load_rcs_paper_scoring_state(path: str | Path, *, map_location: str | None = "cpu") -> _RCSPaperScoringState:
    torch, _, _ = _import_torch()
    raw = torch.load(Path(path), map_location=map_location)
    if not isinstance(raw, dict) or raw.get("schema_version") != _RCS_PAPER_SCORING_STATE_SCHEMA:
        raise ValueError(f"rcs: unsupported scoring state schema in {path}")
    meta = raw.get("projection_meta")
    if not isinstance(meta, dict):
        raise ValueError("rcs: scoring state missing projection_meta")
    projection = _new_rcs_projection(
        input_dim=int(meta["input_dim"]),
        output_dim=int(meta["output_dim"]),
        hidden_dim=int(meta.get("hidden_dim", 512)),
        dropout=float(meta.get("dropout", 0.3)),
    )
    projection.load_state_dict(raw["projection_state_dict"])
    projection.eval()
    return _RCSPaperScoringState(
        layer=int(raw["layer"]),
        layer_diagnostics=dict(raw.get("layer_diagnostics") or {}),
        projection=projection,
        method=str(raw["method"]),
        threshold=float(raw["threshold"]),
        k=int(raw["k"]),
        mcd_benign=[
            _MCDGaussian(mean=item["mean"].to(dtype=torch.float32), cov=item["cov"].to(dtype=torch.float32))
            for item in raw.get("mcd_benign", [])
        ],
        mcd_malicious=[
            _MCDGaussian(mean=item["mean"].to(dtype=torch.float32), cov=item["cov"].to(dtype=torch.float32))
            for item in raw.get("mcd_malicious", [])
        ],
        kcd_benign=raw["kcd_benign"].to(dtype=torch.float32) if raw.get("kcd_benign") is not None else None,
        kcd_malicious=raw["kcd_malicious"].to(dtype=torch.float32) if raw.get("kcd_malicious") is not None else None,
    )


def _scoring_state_path_from_artifact(artifact: Any) -> Path:
    files = getattr(artifact, "files", None)
    if not isinstance(files, dict):
        raise ValueError("rcs_paper_v3 calibration artifact missing files")
    scoring_state = files.get("scoring_state")
    if not isinstance(scoring_state, dict):
        raise ValueError("rcs_paper_v3 calibration artifact missing files.scoring_state")
    path = scoring_state.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("rcs_paper_v3 calibration artifact missing files.scoring_state.path")
    return Path(path)


def _projection_device(projection: Any) -> Any | None:
    try:
        return next(projection.parameters()).device
    except StopIteration:
        return None
    except AttributeError:
        return None


def _last_token_by_layer_tensor(value: Any, *, device: Any | None = None):
    torch, _, _ = _import_torch()

    raw = getattr(value, "last_token_by_layer", value)
    if hasattr(raw, "detach"):
        tensor = raw.detach()
    else:
        tensor = torch.as_tensor(raw)
    tensor = tensor.to(dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device=device)
    if tensor.ndim != 2:
        raise ValueError("rcs: expected last_token_by_layer tensor with shape [layers, hidden_size]")
    return tensor


def _select_layer_vector(value: Any, *, layer: int, device: Any | None = None):
    vecs = _last_token_by_layer_tensor(value, device=device)
    if layer < 0 or layer >= int(vecs.shape[0]):
        raise ValueError(f"rcs: invalid layer={layer} (available 0..{int(vecs.shape[0]) - 1})")
    return vecs[layer]


def _extract_layer_features(
    hidden_state_provider: _HiddenStateProvider,
    examples: list[_RCSTrainExample],
    *,
    layer: int,
    device: Any | None,
):
    torch, _, _ = _import_torch()

    xs = []
    ys: list[int] = []
    ds: list[int] = []
    ds_map: dict[str, int] = {}
    for i, ex in enumerate(examples):
        ds_id = ds_map.setdefault(ex.dataset, len(ds_map))
        s = _sample_from_train_example(ex, index=i)
        xs.append(_select_layer_vector(hidden_state_provider(s), layer=layer, device=device))
        ys.append(0 if ex.is_benign else 1)
        ds.append(ds_id)

    x = torch.stack(xs, dim=0).to(dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.int64, device=x.device)
    d = torch.tensor(ds, dtype=torch.int64, device=x.device)
    return x, y, d


def _extract_all_layer_features(
    hidden_state_provider: _HiddenStateProvider,
    examples: list[_RCSTrainExample],
    *,
    device: Any | None,
):
    torch, _, _ = _import_torch()

    per_layer: list[list[Any]] = []
    ys: list[int] = []
    ds: list[int] = []
    ds_map: dict[str, int] = {}
    n_layers = None
    for i, ex in enumerate(examples):
        ds_id = ds_map.setdefault(ex.dataset, len(ds_map))
        sample = _sample_from_train_example(ex, index=i)
        vecs = _last_token_by_layer_tensor(hidden_state_provider(sample), device=device)
        if n_layers is None:
            n_layers = int(vecs.shape[0])
            per_layer = [[] for _ in range(n_layers)]
        elif int(vecs.shape[0]) != int(n_layers):
            raise ValueError("rcs: inconsistent layer count from hidden_state_provider")
        for layer_idx in range(int(vecs.shape[0])):
            per_layer[layer_idx].append(vecs[layer_idx].to(dtype=torch.float32))
        ys.append(0 if ex.is_benign else 1)
        ds.append(ds_id)

    if n_layers is None:
        raise ValueError("rcs: no hidden states extracted")
    x_by_layer = [torch.stack(layer_rows, dim=0).to(dtype=torch.float32) for layer_rows in per_layer]
    y = torch.tensor(ys, dtype=torch.int64, device=x_by_layer[0].device)
    d = torch.tensor(ds, dtype=torch.int64, device=x_by_layer[0].device)
    return x_by_layer, y, d


def _score_projected_features(
    z,
    *,
    method: str,
    mcd_benign: list[_MCDGaussian],
    mcd_malicious: list[_MCDGaussian],
    kcd_benign: Any | None,
    kcd_malicious: Any | None,
    k: int,
) -> list[float]:
    return [
        _score_paper(
            z[i],
            method=method,
            mcd_benign=mcd_benign,
            mcd_malicious=mcd_malicious,
            kcd_benign=kcd_benign,
            kcd_malicious=kcd_malicious,
            k=int(k),
        )
        for i in range(int(z.shape[0]))
    ]


def _fit_paper_layer_scoring_state(
    *,
    layer: int,
    x_train: Any,
    y_train: Any,
    d_train: Any,
    x_val: Any | None,
    y_val: Any | None,
    method: str,
    threshold: float,
    k: int,
    calibrate_threshold: bool,
    objective_bal_acc_weight: float,
    objective_f1_weight: float,
    projection_dim: int,
    projection_epochs: int,
    projection_batch_size: int,
    projection_lr: float,
    projection_alpha: float,
    projection_beta: float,
    projection_md: float,
    projection_ms: float,
    projection_dropout: float,
    var_floor: float,
    seed: int,
) -> tuple[_RCSPaperScoringState, dict[str, Any], list[float], list[int]]:
    torch, _, _ = _import_torch()

    proj = _fit_projection(
        x_train,
        y_train,
        d_train,
        out_dim=int(projection_dim),
        epochs=int(projection_epochs),
        batch_size=int(projection_batch_size),
        lr=float(projection_lr),
        alpha=float(projection_alpha),
        beta=float(projection_beta),
        md=float(projection_md),
        ms=float(projection_ms),
        dropout=float(projection_dropout),
        seed=int(seed),
    )

    with torch.inference_mode():
        z_train = proj(x_train)

    method = str(method).strip().lower()
    mcd_b: list[_MCDGaussian] = []
    mcd_m: list[_MCDGaussian] = []
    kcd_b = kcd_m = None
    effective_k = int(k)
    if method == "mcd":
        mcd_b, mcd_m = _fit_mcd(z_train, y_train, d_train, var_floor=float(var_floor))
    else:
        if effective_k <= 0:
            effective_k = 50
        kcd_b, kcd_m = _fit_kcd(z_train, y_train)

    train_scores = _score_projected_features(
        z_train,
        method=method,
        mcd_benign=mcd_b,
        mcd_malicious=mcd_m,
        kcd_benign=kcd_b,
        kcd_malicious=kcd_m,
        k=effective_k,
    )
    train_labels = [int(y) for y in y_train.tolist()]

    eval_scores = train_scores
    eval_labels = train_labels
    ranking_split = "train"
    effective_threshold = float(threshold)
    if x_val is not None and y_val is not None and int(x_val.shape[0]) > 0:
        with torch.inference_mode():
            z_val = proj(x_val)
        eval_scores = _score_projected_features(
            z_val,
            method=method,
            mcd_benign=mcd_b,
            mcd_malicious=mcd_m,
            kcd_benign=kcd_b,
            kcd_malicious=kcd_m,
            k=effective_k,
        )
        eval_labels = [int(y) for y in y_val.tolist()]
        ranking_split = "validation"
        if bool(calibrate_threshold):
            effective_threshold = _calibrate_threshold(
                scores=eval_scores,
                labels=eval_labels,
                w_bal_acc=float(objective_bal_acc_weight),
                w_f1=float(objective_f1_weight),
            )

    state = _RCSPaperScoringState(
        layer=int(layer),
        layer_diagnostics={"strategy": "fixed", "selected_layer": int(layer)},
        projection=proj,
        method=method,
        threshold=float(effective_threshold),
        k=int(effective_k),
        mcd_benign=mcd_b,
        mcd_malicious=mcd_m,
        kcd_benign=kcd_b,
        kcd_malicious=kcd_m,
    )
    metrics = _binary_score_metrics(
        scores=eval_scores,
        labels=eval_labels,
        threshold=float(effective_threshold),
        w_bal_acc=float(objective_bal_acc_weight),
        w_f1=float(objective_f1_weight),
    )
    metrics["ranking_split"] = ranking_split
    metrics["threshold"] = float(effective_threshold)
    metrics["train_count"] = int(x_train.shape[0])
    metrics["evaluation_count"] = len(eval_scores)
    return state, metrics, eval_scores, eval_labels


def _fit_paper_scoring_state(
    *,
    hidden_state_provider: _HiddenStateProvider,
    device: Any | None,
    examples: list[_RCSTrainExample],
    method: str,
    threshold: float,
    k: int,
    val_ratio: float,
    calibrate_threshold: bool,
    objective_bal_acc_weight: float,
    objective_f1_weight: float,
    layer: int | None,
    auto_layer_max_samples: int,
    auto_layer_svm_iters: int,
    projection_dim: int,
    projection_epochs: int,
    projection_batch_size: int,
    projection_lr: float,
    projection_alpha: float,
    projection_beta: float,
    projection_md: float,
    projection_ms: float,
    projection_dropout: float,
    var_floor: float,
    seed: int,
    layer_selection_strategy: str = _RCS_LAYER_SELECTION_AUTO,
) -> _RCSPaperScoringState:
    train, val = _split_paper_examples(examples, val_ratio=val_ratio, seed=seed)
    if len(train) < 4:
        raise ValueError("rcs: mode=paper requires >=4 training examples after split")

    strategy = _normalize_layer_selection_strategy(layer_selection_strategy)
    if layer is None and strategy == _RCS_LAYER_SELECTION_OFFICIAL_SWEEP:
        x_train_by_layer, y_train, d_train = _extract_all_layer_features(hidden_state_provider, train, device=device)
        x_val_by_layer: list[Any] = []
        y_val = None
        if val:
            x_val_by_layer, y_val, _d_val = _extract_all_layer_features(hidden_state_provider, val, device=device)
            if len(x_val_by_layer) != len(x_train_by_layer):
                raise ValueError("rcs: validation hidden states have different layer count")

        candidates: list[dict[str, Any]] = []
        for layer_idx, x_train in enumerate(x_train_by_layer):
            x_val = x_val_by_layer[layer_idx] if x_val_by_layer else None
            state, metrics, _scores, _labels = _fit_paper_layer_scoring_state(
                layer=layer_idx,
                x_train=x_train,
                y_train=y_train,
                d_train=d_train,
                x_val=x_val,
                y_val=y_val,
                method=method,
                threshold=float(threshold),
                k=int(k),
                calibrate_threshold=bool(calibrate_threshold),
                objective_bal_acc_weight=float(objective_bal_acc_weight),
                objective_f1_weight=float(objective_f1_weight),
                projection_dim=int(projection_dim),
                projection_epochs=int(projection_epochs),
                projection_batch_size=int(projection_batch_size),
                projection_lr=float(projection_lr),
                projection_alpha=float(projection_alpha),
                projection_beta=float(projection_beta),
                projection_md=float(projection_md),
                projection_ms=float(projection_ms),
                projection_dropout=float(projection_dropout),
                var_floor=float(var_floor),
                seed=int(seed) + layer_idx,
            )
            candidates.append({"layer": layer_idx, "state": state, "metrics": metrics})

        def rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
            metrics = row["metrics"]
            official = metrics.get("official_combined")
            objective = metrics.get("threshold_objective")
            fpr = metrics.get("fpr")
            return (
                float(official) if official is not None else float("-inf"),
                float(objective) if objective is not None else float("-inf"),
                -float(fpr) if fpr is not None else float("-inf"),
                -int(row["layer"]),
            )

        best = max(candidates, key=rank_key)
        selected_layer = int(best["layer"])
        scoring = best["state"]
        scoring.layer_diagnostics = _official_sweep_layer_selection_diagnostics(
            selected_layer=selected_layer,
            candidates=candidates,
        )
        return scoring

    if layer is None:
        selected_layer, layer_metrics = _auto_select_layer(
            hidden_state_provider,
            train,
            max_samples=auto_layer_max_samples,
            svm_iters=auto_layer_svm_iters,
            seed=seed,
        )
    else:
        selected_layer = int(layer)
        layer_metrics = None

    x_train, y_train, d_train = _extract_layer_features(
        hidden_state_provider,
        train,
        layer=selected_layer,
        device=device,
    )
    x_val = None
    y_val = None
    if val:
        x_val, y_val, _d_val = _extract_layer_features(
            hidden_state_provider,
            val,
            layer=selected_layer,
            device=device,
        )

    scoring, _metrics, _scores, _labels = _fit_paper_layer_scoring_state(
        layer=int(selected_layer),
        x_train=x_train,
        y_train=y_train,
        d_train=d_train,
        x_val=x_val,
        y_val=y_val,
        method=method,
        threshold=float(threshold),
        k=int(k),
        calibrate_threshold=bool(calibrate_threshold),
        objective_bal_acc_weight=float(objective_bal_acc_weight),
        objective_f1_weight=float(objective_f1_weight),
        projection_dim=int(projection_dim),
        projection_epochs=int(projection_epochs),
        projection_batch_size=int(projection_batch_size),
        projection_lr=float(projection_lr),
        projection_alpha=float(projection_alpha),
        projection_beta=float(projection_beta),
        projection_md=float(projection_md),
        projection_ms=float(projection_ms),
        projection_dropout=float(projection_dropout),
        var_floor=float(var_floor),
        seed=int(seed),
    )
    scoring.layer_diagnostics = _layer_selection_diagnostics(
        selected_layer=int(selected_layer),
        metrics=layer_metrics,
    )
    return scoring


def _fit_mcd(z, y, d, *, var_floor: float) -> tuple[list[_MCDGaussian], list[_MCDGaussian]]:
    torch, _, _ = _import_torch()

    benign: list[_MCDGaussian] = []
    malicious: list[_MCDGaussian] = []
    for cls, out in [(0, benign), (1, malicious)]:
        for ds_id in sorted(set(int(x) for x in d.tolist())):
            mask = (y == cls) & (d == ds_id)
            pts = z[mask]
            if int(pts.shape[0]) <= 1:
                continue
            mean = pts.mean(dim=0)
            cov, _shrink = _ledoit_wolf_cov(pts, var_floor=var_floor)
            out.append(_MCDGaussian(mean=mean, cov=cov))
    if not benign or not malicious:
        raise ValueError("rcs: mcd requires both benign and malicious gaussians")
    return benign, malicious


def _fit_kcd(z, y) -> tuple[Any, Any]:
    torch, _, F = _import_torch()

    zb = z[y == 0]
    zm = z[y == 1]
    if int(zb.shape[0]) <= 0 or int(zm.shape[0]) <= 0:
        raise ValueError("rcs: kcd requires both benign and malicious points")
    zb = F.normalize(zb.to(dtype=torch.float32), p=2.0, dim=1)
    zm = F.normalize(zm.to(dtype=torch.float32), p=2.0, dim=1)
    return zb, zm


def _score_paper(
    z,
    *,
    method: str,
    mcd_benign: list[_MCDGaussian],
    mcd_malicious: list[_MCDGaussian],
    kcd_benign: Any | None,
    kcd_malicious: Any | None,
    k: int,
) -> float:
    torch, _, _ = _import_torch()

    method = str(method).strip().lower()
    if method == "mcd":
        def mdist(g: _MCDGaussian) -> Any:
            mean = g.mean.to(device=z.device, dtype=torch.float32)
            diff = z.to(dtype=torch.float32) - mean
            cov = g.cov.to(device=z.device, dtype=torch.float32)
            sol = torch.linalg.solve(cov, diff)
            dist_sq = (diff * sol).sum()
            return torch.sqrt(torch.clamp(dist_sq, min=0.0))

        d_b = torch.stack([mdist(g) for g in mcd_benign]).min()
        d_m = torch.stack([mdist(g) for g in mcd_malicious]).min()
        return float((d_b - d_m).item())

    assert kcd_benign is not None and kcd_malicious is not None
    z_unit = z / (torch.linalg.vector_norm(z) + 1e-8)
    kcd_benign = kcd_benign.to(device=z_unit.device, dtype=torch.float32)
    kcd_malicious = kcd_malicious.to(device=z_unit.device, dtype=torch.float32)
    sim_b = kcd_benign @ z_unit
    sim_m = kcd_malicious @ z_unit
    kb = min(max(1, int(k)), int(sim_b.shape[0]))
    km = min(max(1, int(k)), int(sim_m.shape[0]))
    top_b = torch.topk(sim_b, kb, largest=True).values[-1]
    top_m = torch.topk(sim_m, km, largest=True).values[-1]
    d_b = torch.sqrt(torch.clamp(2.0 - 2.0 * top_b, min=0.0))
    d_m = torch.sqrt(torch.clamp(2.0 - 2.0 * top_m, min=0.0))
    return float((d_b - d_m).item())


def _score_paper_state(state: _RCSPaperScoringState, z) -> float:
    return _score_paper(
        z,
        method=state.method,
        mcd_benign=state.mcd_benign,
        mcd_malicious=state.mcd_malicious,
        kcd_benign=state.kcd_benign,
        kcd_malicious=state.kcd_malicious,
        k=state.k,
    )


def _decide_paper_from_hidden_states(
    state: _RCSPaperScoringState,
    last_token_by_layer: Any,
    sample: Sample,
    *,
    device: Any | None,
) -> DetectorDecision:
    torch, _, _ = _import_torch()

    if device is None:
        device = _projection_device(state.projection)
    x = _select_layer_vector(last_token_by_layer, layer=state.layer, device=device).unsqueeze(0)
    with torch.inference_mode():
        z = state.projection(x)[0]
    score = _score_paper_state(state, z)
    block = bool(float(score) > float(state.threshold))
    reason = (
        f"rcs(mode=paper, method={state.method}, layer={state.layer}, score={float(score):.4f}, "
        f"threshold={float(state.threshold):.4f}, n_images={len(sample.images)})"
    )
    return DetectorDecision(
        block=block,
        score=float(score),
        reason=reason,
        diagnostics={"rcs_layer_selection": state.layer_diagnostics},
    )


def _decide_paper(state: _RCSPaperState, sample: Sample) -> DetectorDecision:
    return _decide_paper_from_hidden_states(
        state.scoring,
        state.hidden_state_provider(sample),
        sample,
        device=state.device,
    )


def _hidden_state_config_for_paper_detector(detector: Any) -> LastTokenHiddenStateConfig:
    return LastTokenHiddenStateConfig(
        model_id=detector._paper_model_id,
        revision=detector._paper_revision,
        device=detector._paper_device,
        torch_dtype=detector._paper_torch_dtype,
        trust_remote_code=detector._paper_trust_remote_code,
        local_files_only=detector._paper_local_files_only,
        token_env=detector._paper_token_env,
        image_token=str(detector.image_token),
        auto_insert_image_tokens=bool(detector.auto_insert_image_tokens),
        token_strategy=detector._paper_hidden_state_token_strategy,
        include_embedding_layer=detector._paper_include_embedding_layer,
        model_family=detector._paper_hidden_state_model_family,
        max_length=detector._paper_hidden_state_max_length,
    )


def _score_paper_examples(
    state: _RCSPaperScoringState,
    hidden_state_provider: _HiddenStateProvider,
    examples: list[_RCSTrainExample],
) -> tuple[list[float], list[int]]:
    torch, _, _ = _import_torch()
    device = _projection_device(state.projection)
    scores: list[float] = []
    labels: list[int] = []
    for i, ex in enumerate(examples):
        sample = _sample_from_train_example(ex, index=i)
        x = _select_layer_vector(hidden_state_provider(sample), layer=state.layer, device=device)
        with torch.inference_mode():
            z = state.projection(x.unsqueeze(0))[0]
        scores.append(_score_paper_state(state, z))
        labels.append(0 if ex.is_benign else 1)
    return scores, labels


def _model_param_str(model: dict[str, Any], key: str, default: str | None = None) -> str | None:
    value = model.get(key, default)
    return value if isinstance(value, str) else default


def _model_param_int(model: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = model.get(key, default)
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    return default


def calibrate_rcs_paper_from_config(
    cfg: Any,
    *,
    artifact_path: Path,
    source_config_path: str | None,
    command: list[str] | None,
) -> tuple[Any, dict[str, Any]]:
    from turnkey.calibration import (
        CALIBRATION_REPORT_SCHEMA,
        CalibrationArtifact,
        CalibrationDataManifest,
        CalibrationTargetModel,
        calibration_artifact_identity,
        detector_config_hash,
        score_distribution_summary,
        _read_git_revision,
        _source_config_identity,
    )

    from turnkey.components.detectors.rcs import RCSDetector

    detector = RCSDetector(**cfg.detector.params)
    manifest = detector.manifest(name=cfg.detector.name)
    provider = HFLastTokenHiddenStateProvider(_hidden_state_config_for_paper_detector(detector))

    examples = list(detector._paper_examples)
    train, val = _split_paper_examples(
        examples,
        val_ratio=float(detector.val_ratio),
        seed=int(detector.seed),
    )
    scoring = _fit_paper_scoring_state(
        hidden_state_provider=provider.last_token_by_layer,
        device=None,
        examples=examples,
        method=detector.method,
        threshold=float(detector.threshold),
        k=int(detector.k),
        val_ratio=float(detector.val_ratio),
        calibrate_threshold=bool(detector.calibrate_threshold),
        objective_bal_acc_weight=float(detector.objective_bal_acc_weight),
        objective_f1_weight=float(detector.objective_f1_weight),
        layer=detector.layer,
        auto_layer_max_samples=int(detector.auto_layer_max_samples),
        auto_layer_svm_iters=int(detector.auto_layer_svm_iters),
        projection_dim=int(detector.projection_dim),
        projection_epochs=int(detector.projection_epochs),
        projection_batch_size=int(detector.projection_batch_size),
        projection_lr=float(detector.projection_lr),
        projection_alpha=float(detector.projection_alpha),
        projection_beta=float(detector.projection_beta),
        projection_md=float(detector.projection_md),
        projection_ms=float(detector.projection_ms),
        projection_dropout=float(detector.projection_dropout),
        var_floor=float(detector.var_floor),
        seed=int(detector.seed),
        layer_selection_strategy=detector.layer_selection_strategy,
    )
    detector.threshold = float(scoring.threshold)
    detector.k = int(scoring.k)

    state_path = artifact_path.with_suffix(".state.pt").resolve()
    _save_rcs_paper_scoring_state(state_path, scoring)

    all_scores, all_labels = _score_paper_examples(scoring, provider.last_token_by_layer, examples)
    train_scores, _ = _score_paper_examples(scoring, provider.last_token_by_layer, train)
    val_scores, _ = _score_paper_examples(scoring, provider.last_token_by_layer, val) if val else ([], [])
    benign_scores = [score for score, label in zip(all_scores, all_labels, strict=True) if label == 0]
    harmful_scores = [score for score, label in zip(all_scores, all_labels, strict=True) if label == 1]

    benign_count = sum(1 for ex in examples if ex.is_benign)
    harmful_count = len(examples) - benign_count
    source_type = "jsonl" if detector.train_jsonl else "inline_prototypes"
    source = detector.train_jsonl or "inline_prototypes"
    source_identity = (
        calibration_artifact_identity(detector.train_jsonl)
        if detector.train_jsonl is not None and Path(detector.train_jsonl).exists()
        else {"source_type": source_type}
    )
    method = {
        "reproduction_scope": "paper_aligned_bounded_reproduction",
        "procedure_id": "rcs_paper_fit_load",
        "reference_sources": [
            "https://arxiv.org/abs/2512.12069",
            "https://github.com/sarendis56/Jailbreak_Detection_RCS",
        ],
        "calibration_rule": "held_out_training_split_weighted_balanced_accuracy_f1",
    }
    operating_point = {
        "method": scoring.method,
        "selected_layer": scoring.layer,
        "layer_selection_strategy": scoring.layer_diagnostics.get("strategy"),
        "k": scoring.k,
        "threshold": scoring.threshold,
        "score_rule": "block_when_score_gt_threshold",
        "calibrate_threshold": bool(detector.calibrate_threshold),
        "threshold_objective": {
            "balanced_accuracy_weight": float(detector.objective_bal_acc_weight),
            "f1_weight": float(detector.objective_f1_weight),
            "grid_size": _RCS_PAPER_THRESHOLD_GRID_SIZE,
        },
        "validation_split": {
            "source": "explicit_train_jsonl_split" if any(ex.split is not None for ex in examples) else "held_out_training_split",
            "val_ratio": float(detector.val_ratio),
            "seed": int(detector.seed),
            "train_count": len(train),
            "validation_count": len(val),
        },
        "projection": {
            "dim": int(detector.projection_dim),
            "epochs": int(detector.projection_epochs),
            "batch_size": int(detector.projection_batch_size),
            "lr": float(detector.projection_lr),
            "alpha": float(detector.projection_alpha),
            "beta": float(detector.projection_beta),
            "md": float(detector.projection_md),
            "ms": float(detector.projection_ms),
            "dropout": float(detector.projection_dropout),
        },
    }

    artifact = CalibrationArtifact(
        detector_name=cfg.detector.name,
        detector_version=manifest.version,
        artifact_kind="rcs_paper_scoring_state",
        target_model=CalibrationTargetModel(
            model_id=cfg.model.model_id,
            backend=cfg.model.backend,
            revision=cfg.model.revision,
        ),
        detector_config_hash=detector_config_hash(cfg.detector.params),
        method=method,
        calibration_data=CalibrationDataManifest(
            source=str(source),
            count=len(examples),
            benign_count=benign_count,
            harmful_count=harmful_count,
            split=(
                "explicit_train_jsonl_split"
                if any(ex.split is not None for ex in examples)
                else "held_out_training_split" if val else "training_only"
            ),
            seed=int(detector.seed),
            identity=source_identity,
        ),
        threshold=float(scoring.threshold),
        operating_point=operating_point,
        score_summary={
            "all": score_distribution_summary(all_scores),
            "train": score_distribution_summary(train_scores),
            "validation": score_distribution_summary(val_scores),
            "benign": score_distribution_summary(benign_scores),
            "harmful": score_distribution_summary(harmful_scores),
        },
        files={"scoring_state": calibration_artifact_identity(state_path)},
        code=_read_git_revision(),
        metadata={
            "detector_manifest": manifest.to_dict(),
            "training_examples": _paper_training_example_summary(
                examples=examples,
                source=str(source),
                source_type=source_type,
                require_balanced_train=bool(detector.require_balanced_train),
                prototype_image_count=int(detector.prototype_image_count),
                prototype_image_path=detector.prototype_image_path,
            ),
            "layer_selection": scoring.layer_diagnostics,
            "hidden_state_model": _hidden_state_config_for_paper_detector(detector).__dict__,
        },
    )
    report = {
        "schema_version": CALIBRATION_REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": list(command) if command is not None else None,
            "source_config": _source_config_identity(source_config_path),
        },
        "detector": {
            "name": cfg.detector.name,
            "params": dict(cfg.detector.params),
            "manifest": manifest.to_dict(),
        },
        "target_model": artifact.target_model.to_dict(),
        "method": method,
        "calibration_data": artifact.calibration_data.to_dict(),
        "operating_point": operating_point,
        "score_summary": {name: summary.to_dict() for name, summary in artifact.score_summary.items()},
        "files": dict(artifact.files),
    }
    return artifact, report
