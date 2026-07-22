from __future__ import annotations

from turnkey.schema import Sample


def _toy_feature_vector(sample: Sample) -> list[float]:
    text = sample.prompt.lower()
    unsafe = 1.0 if "unsafe_placeholder" in text else 0.0
    redacted = 1.0 if "redacted" in text else 0.0
    length = min(1.0, max(0.0, float(len(sample.prompt)) / 256.0))
    n_images = min(1.0, max(0.0, float(len(sample.images)) / 4.0))
    qmarks = min(1.0, float(sample.prompt.count("?")) / 4.0)
    return [unsafe, redacted, length, n_images, qmarks]


def _mean(vs: list[list[float]]) -> list[float]:
    if not vs:
        raise ValueError("rcs: empty vectors")
    d = len(vs[0])
    m = [0.0] * d
    for v in vs:
        if len(v) != d:
            raise ValueError("rcs: inconsistent vector dims")
        for i in range(d):
            m[i] += float(v[i])
    n = float(len(vs))
    return [x / n for x in m]


def _var_diag(vs: list[list[float]], mu: list[float], *, var_floor: float = 1e-3) -> list[float]:
    d = len(mu)
    out = [0.0] * d
    if len(vs) <= 1:
        return [float(var_floor)] * d
    for v in vs:
        for i in range(d):
            diff = float(v[i]) - float(mu[i])
            out[i] += diff * diff
    denom = float(len(vs) - 1)
    vars_ = [x / denom for x in out]
    return [max(float(var_floor), float(v)) for v in vars_]


def _mahalanobis_diag(x: list[float], mu: list[float], var: list[float], *, eps: float = 1e-8) -> float:
    if not (len(x) == len(mu) == len(var)):
        raise ValueError("rcs: invalid dims")
    acc = 0.0
    for i in range(len(x)):
        diff = float(x[i]) - float(mu[i])
        acc += (diff * diff) / (float(var[i]) + eps)
    return float(acc)


def _l2(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("rcs: invalid dims")
    acc = 0.0
    for i in range(len(x)):
        diff = float(x[i]) - float(y[i])
        acc += diff * diff
    return float(acc) ** 0.5


def _knn_mean_distance(x: list[float], pts: list[list[float]], *, k: int) -> float:
    if not pts:
        raise ValueError("rcs: empty points")
    k = max(1, int(k))
    dists = sorted(_l2(x, p) for p in pts)
    top = dists[:k]
    return float(sum(top) / float(len(top)))
