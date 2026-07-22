from __future__ import annotations

import pytest


try:
    import torch
except Exception:  # noqa: BLE001
    pytest.skip("RCS paper-form tests require torch (install with `.[hf]`).", allow_module_level=True)


from turnkey.components.detectors.rcs import (
    _MCDGaussian,
    _RCSPaperScoringState,
    _RCSTrainExample,
    _calibrate_threshold,
    _decide_paper_from_hidden_states,
    _extract_layer_features,
    _fit_paper_scoring_state,
    _fit_projection,
    _ledoit_wolf_cov,
    _load_rcs_paper_scoring_state,
    _save_rcs_paper_scoring_state,
    _projection_contrastive_loss,
    _reference_shrinkage_intensity,
    _score_paper,
    _score_paper_state,
)
from turnkey.schema import Sample


def test_ledoit_wolf_cov_matches_reference_formula() -> None:
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(50, 10, generator=rng)
    cov_t, shrink = _ledoit_wolf_cov(x, var_floor=1e-3)
    assert cov_t.shape == (10, 10)
    assert 0.0 <= float(shrink) <= 1.0

    x64 = x.to(dtype=torch.float64)
    centered = x64 - x64.mean(dim=0, keepdim=True)
    sample_cov = (centered.T @ centered) / float(x.shape[0] - 1)
    trace_cov = float(torch.trace(sample_cov).item())
    frobenius_sq = float(torch.sum(sample_cov * sample_cov).item())
    expected_shrink = _reference_shrinkage_intensity(
        n=x.shape[0],
        dim=x.shape[1],
        trace_cov=trace_cov,
        frobenius_sq=frobenius_sq,
    )
    target = torch.eye(x.shape[1], dtype=torch.float64) * (trace_cov / float(x.shape[1]))
    expected_cov = (1.0 - expected_shrink) * sample_cov + expected_shrink * target

    assert float(shrink) == pytest.approx(expected_shrink, abs=1e-6)
    assert float(torch.max(torch.abs(cov_t - expected_cov.to(dtype=torch.float32)))) < 1e-5


def test_ledoit_wolf_cov_uses_reference_identity_fallback_for_small_clusters() -> None:
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    cov_t, shrink = _ledoit_wolf_cov(x, var_floor=1e-3)

    assert shrink == 1.0
    assert torch.allclose(cov_t, torch.eye(2), atol=1e-6)


def test_kcd_score_sign_on_synthetic_unit_sphere() -> None:
    # benign points near +x, malicious points near -x
    benign = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    malicious = torch.tensor([[-1.0, 0.0], [-0.9, -0.1]])
    benign = benign / torch.linalg.vector_norm(benign, dim=1, keepdim=True)
    malicious = malicious / torch.linalg.vector_norm(malicious, dim=1, keepdim=True)

    z_b = torch.tensor([1.0, 0.0])
    z_m = torch.tensor([-1.0, 0.0])

    s_b = _score_paper(
        z_b,
        method="kcd",
        mcd_benign=[],
        mcd_malicious=[],
        kcd_benign=benign,
        kcd_malicious=malicious,
        k=1,
    )
    s_m = _score_paper(
        z_m,
        method="kcd",
        mcd_benign=[],
        mcd_malicious=[],
        kcd_benign=benign,
        kcd_malicious=malicious,
        k=1,
    )
    assert s_b < 0.0
    assert s_m > 0.0


def test_kcd_score_matches_reference_kth_normalized_euclidean_distance() -> None:
    benign = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=torch.float32)
    malicious = torch.tensor([[-1.0, 0.0], [-0.8, -0.6], [0.0, -1.0]], dtype=torch.float32)
    z = torch.tensor([0.7, 0.7], dtype=torch.float32)

    benign = benign / torch.linalg.vector_norm(benign, dim=1, keepdim=True)
    malicious = malicious / torch.linalg.vector_norm(malicious, dim=1, keepdim=True)
    z_unit = z / torch.linalg.vector_norm(z)

    score = _score_paper(
        z,
        method="kcd",
        mcd_benign=[],
        mcd_malicious=[],
        kcd_benign=benign,
        kcd_malicious=malicious,
        k=2,
    )

    benign_dist = torch.linalg.vector_norm(benign - z_unit.unsqueeze(0), dim=1)
    malicious_dist = torch.linalg.vector_norm(malicious - z_unit.unsqueeze(0), dim=1)
    expected = float(torch.topk(benign_dist, 2, largest=False).values[-1])
    expected -= float(torch.topk(malicious_dist, 2, largest=False).values[-1])

    assert score == pytest.approx(expected, abs=1e-6)


def test_mcd_score_sign_on_isotropic_gaussians() -> None:
    eye = torch.eye(2)
    benign = [_MCDGaussian(mean=torch.tensor([1.0, 0.0]), cov=eye)]
    malicious = [_MCDGaussian(mean=torch.tensor([-1.0, 0.0]), cov=eye)]

    z_b = torch.tensor([1.0, 0.0])
    z_m = torch.tensor([-1.0, 0.0])

    s_b = _score_paper(
        z_b,
        method="mcd",
        mcd_benign=benign,
        mcd_malicious=malicious,
        kcd_benign=None,
        kcd_malicious=None,
        k=1,
    )
    s_m = _score_paper(
        z_m,
        method="mcd",
        mcd_benign=benign,
        mcd_malicious=malicious,
        kcd_benign=None,
        kcd_malicious=None,
        k=1,
    )
    assert s_b < 0.0
    assert s_m > 0.0


def test_mcd_score_uses_reference_mahalanobis_distance_not_squared_distance() -> None:
    cov = torch.diag(torch.tensor([4.0, 1.0]))
    benign = [_MCDGaussian(mean=torch.tensor([0.0, 0.0]), cov=cov)]
    malicious = [_MCDGaussian(mean=torch.tensor([3.0, 0.0]), cov=cov)]
    z = torch.tensor([1.0, 0.0])

    score = _score_paper(
        z,
        method="mcd",
        mcd_benign=benign,
        mcd_malicious=malicious,
        kcd_benign=None,
        kcd_malicious=None,
        k=1,
    )

    d_b = ((z[0] - 0.0) ** 2 / 4.0).sqrt()
    d_m = ((z[0] - 3.0) ** 2 / 4.0).sqrt()
    assert score == pytest.approx(float(d_b - d_m), abs=1e-6)


def test_threshold_calibration_prefers_separating_cut() -> None:
    scores = [-2.0, -1.0, 1.0, 2.0]
    labels = [0, 0, 1, 1]
    theta = _calibrate_threshold(scores=scores, labels=labels, w_bal_acc=0.5, w_f1=0.5)
    assert -1.1 < float(theta) < 1.1


def test_threshold_calibration_uses_paper_weighted_objective_grid() -> None:
    scores = [-0.4, 0.1, 0.2, 0.9]
    labels = [0, 1, 0, 1]

    theta = _calibrate_threshold(scores=scores, labels=labels, w_bal_acc=0.8, w_f1=0.2)

    assert min(scores) <= theta <= max(scores)


def test_projection_contrastive_loss_matches_reference_objective_terms() -> None:
    embeddings = torch.tensor(
        [
            [2.0, 0.0],
            [1.0, 1.0],
            [-2.0, 0.0],
            [-1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    dataset_labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    toxicity_labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    total, dataset_loss, toxicity_loss = _projection_contrastive_loss(
        embeddings,
        dataset_labels,
        toxicity_labels,
        margin_dataset=1.0,
        margin_toxicity=2.0,
        dataset_weight=1.0,
        toxicity_weight=5.0,
    )

    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    distances = torch.cdist(normalized, normalized, p=2)
    same = dataset_labels[:, None] == dataset_labels[None, :]
    diag = torch.eye(4, dtype=torch.bool)
    same = same & ~diag
    other = (~same) & ~diag
    expected_dataset = distances[same].mean() + torch.relu(torch.tensor(1.0) - distances[other]).mean()

    benign = normalized[toxicity_labels == 0]
    malicious = normalized[toxicity_labels == 1]
    benign_centroid = benign.mean(dim=0)
    malicious_centroid = malicious.mean(dim=0)
    expected_toxicity = torch.relu(
        torch.tensor(2.0) - torch.nn.functional.pairwise_distance(
            benign_centroid.unsqueeze(0),
            malicious_centroid.unsqueeze(0),
        )
    ).mean()
    expected_toxicity += torch.linalg.vector_norm(benign - benign_centroid.unsqueeze(0), dim=1).mean()
    expected_toxicity += torch.linalg.vector_norm(malicious - malicious_centroid.unsqueeze(0), dim=1).mean()

    assert float(dataset_loss) == pytest.approx(float(expected_dataset), abs=1e-6)
    assert float(toxicity_loss) == pytest.approx(float(expected_toxicity), abs=1e-6)
    assert float(total) == pytest.approx(float(expected_dataset + 5.0 * expected_toxicity), abs=1e-6)


def test_projection_network_matches_reference_shape_and_batchnorm() -> None:
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.1],
            [-1.0, 0.0, 0.0],
            [-0.8, -0.2, -0.1],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    d = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    projection = _fit_projection(
        x,
        y,
        d,
        out_dim=2,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        alpha=1.0,
        beta=5.0,
        md=1.0,
        ms=2.0,
        dropout=0.3,
        seed=7,
    )

    layers = list(projection.net)
    assert [layer.__class__.__name__ for layer in layers] == [
        "Linear",
        "BatchNorm1d",
        "ReLU",
        "Dropout",
        "Linear",
        "BatchNorm1d",
        "ReLU",
        "Dropout",
        "Linear",
        "BatchNorm1d",
    ]
    assert layers[0].out_features == 512
    assert layers[4].out_features == 256
    assert layers[8].out_features == 2
    with torch.inference_mode():
        assert projection(x).shape == (4, 2)


def test_extract_layer_features_uses_hidden_state_provider_boundary() -> None:
    calls: list[str] = []

    def provider(sample: Sample):
        calls.append(sample.sample_id)
        sign = 1.0 if sample.is_benign else -1.0
        return torch.tensor([[0.0, sign, 0.0], [sign, 0.5, 1.0]], dtype=torch.float32)

    examples = _paper_examples()
    x, y, d = _extract_layer_features(
        provider,
        examples,
        layer=1,
        device=torch.device("cpu"),
    )

    assert calls == ["rcs-train-0", "rcs-train-1", "rcs-train-2", "rcs-train-3"]
    assert x.shape == (4, 3)
    assert y.tolist() == [0, 0, 1, 1]
    assert d.tolist() == [0, 0, 1, 1]


def test_fit_scoring_state_and_decide_from_hidden_state_tensor() -> None:
    def provider(sample: Sample):
        sign = 1.0 if sample.is_benign else -1.0
        offset = 0.2 if sample.sample_id.endswith("1") else 0.0
        return torch.tensor(
            [
                [0.0, sign + offset, 0.0],
                [sign + offset, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )

    state = _fit_paper_scoring_state(
        hidden_state_provider=provider,
        device=torch.device("cpu"),
        examples=_paper_examples(),
        method="kcd",
        threshold=0.0,
        k=1,
        val_ratio=0.0,
        calibrate_threshold=False,
        objective_bal_acc_weight=0.5,
        objective_f1_weight=0.5,
        layer=1,
        auto_layer_max_samples=4,
        auto_layer_svm_iters=10,
        projection_dim=2,
        projection_epochs=1,
        projection_batch_size=4,
        projection_lr=1e-3,
        projection_alpha=1.0,
        projection_beta=1.0,
        projection_md=1.0,
        projection_ms=1.0,
        projection_dropout=0.0,
        var_floor=1e-3,
        seed=0,
    )

    sample = Sample(sample_id="eval", behavior_id="b", is_benign=True, prompt="benign eval")
    decision = _decide_paper_from_hidden_states(
        state,
        provider(sample),
        sample,
        device=torch.device("cpu"),
    )

    assert state.layer == 1
    assert state.method == "kcd"
    assert state.layer_diagnostics == {"strategy": "fixed", "selected_layer": 1}
    assert decision.reason is not None
    assert "rcs(mode=paper, method=kcd, layer=1" in decision.reason
    assert decision.diagnostics["rcs_layer_selection"] == {"strategy": "fixed", "selected_layer": 1}
    assert isinstance(decision.block, bool)
    assert isinstance(decision.score, float)


def test_paper_scoring_state_round_trips(tmp_path) -> None:
    state = _RCSPaperScoringState(
        layer=1,
        layer_diagnostics={"strategy": "fixed", "selected_layer": 1},
        projection=_fit_projection(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.1, 0.0],
                    [-1.0, 0.0, 0.0],
                    [-0.8, -0.1, 0.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor([0, 0, 1, 1], dtype=torch.int64),
            torch.tensor([0, 0, 1, 1], dtype=torch.int64),
            out_dim=2,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            alpha=1.0,
            beta=1.0,
            md=1.0,
            ms=1.0,
            dropout=0.0,
            seed=11,
        ),
        method="kcd",
        threshold=0.25,
        k=1,
        mcd_benign=[],
        mcd_malicious=[],
        kcd_benign=torch.tensor([[1.0, 0.0], [0.9, 0.1]], dtype=torch.float32),
        kcd_malicious=torch.tensor([[-1.0, 0.0], [-0.9, -0.1]], dtype=torch.float32),
    )
    path = tmp_path / "rcs.state.pt"

    _save_rcs_paper_scoring_state(path, state)
    loaded = _load_rcs_paper_scoring_state(path)

    z = torch.tensor([1.0, 0.0], dtype=torch.float32)
    assert loaded.layer == state.layer
    assert loaded.threshold == state.threshold
    assert loaded.k == state.k
    assert _score_paper_state(loaded, z) == pytest.approx(_score_paper_state(state, z), abs=1e-6)


def test_auto_layer_selection_persists_ranked_diagnostics() -> None:
    def provider(sample: Sample):
        sign = 1.0 if sample.is_benign else -1.0
        offset = 0.2 if sample.sample_id.endswith("1") else 0.0
        return torch.tensor(
            [
                [0.1 * sign, 0.0, 0.0],
                [sign + offset, 0.5, 1.0],
                [0.0, 0.2 * sign, 0.0],
            ],
            dtype=torch.float32,
        )

    state = _fit_paper_scoring_state(
        hidden_state_provider=provider,
        device=torch.device("cpu"),
        examples=_paper_examples(),
        method="kcd",
        threshold=0.0,
        k=1,
        val_ratio=0.0,
        calibrate_threshold=False,
        objective_bal_acc_weight=0.5,
        objective_f1_weight=0.5,
        layer=None,
        auto_layer_max_samples=4,
        auto_layer_svm_iters=10,
        projection_dim=2,
        projection_epochs=1,
        projection_batch_size=4,
        projection_lr=1e-3,
        projection_alpha=1.0,
        projection_beta=1.0,
        projection_md=1.0,
        projection_ms=1.0,
        projection_dropout=0.0,
        var_floor=1e-3,
        seed=0,
    )

    diagnostics = state.layer_diagnostics
    assert diagnostics["strategy"] == "auto_svm_silhouette_ratio"
    assert diagnostics["selected_layer"] == state.layer
    assert diagnostics["n_layers"] == 3
    assert diagnostics["top_layers"]
    assert diagnostics["top_layers"][0]["rank"] == 1
    assert {"rank", "layer", "composite", "gamma", "silhouette", "ratio"} <= set(diagnostics["top_layers"][0])


def test_official_sweep_trains_layer_specific_states_and_ranks_validation_metrics() -> None:
    def provider(sample: Sample):
        sign = 1.0 if sample.is_benign else -1.0
        offset = 0.1 if sample.prompt.endswith("1") else 0.0
        return torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [4.0 * sign + offset, sign, 0.0],
            ],
            dtype=torch.float32,
        )

    state = _fit_paper_scoring_state(
        hidden_state_provider=provider,
        device=torch.device("cpu"),
        examples=_paper_examples_with_validation(),
        method="kcd",
        threshold=0.0,
        k=1,
        val_ratio=0.5,
        calibrate_threshold=True,
        objective_bal_acc_weight=0.8,
        objective_f1_weight=0.2,
        layer=None,
        layer_selection_strategy="official_sweep",
        auto_layer_max_samples=4,
        auto_layer_svm_iters=10,
        projection_dim=2,
        projection_epochs=12,
        projection_batch_size=4,
        projection_lr=1e-2,
        projection_alpha=1.0,
        projection_beta=5.0,
        projection_md=1.0,
        projection_ms=2.0,
        projection_dropout=0.0,
        var_floor=1e-3,
        seed=0,
    )

    diagnostics = state.layer_diagnostics
    assert state.layer == 1
    assert diagnostics["strategy"] == "official_sweep"
    assert diagnostics["reference_behavior"] == "layer_specific_projection_threshold_sweep"
    assert diagnostics["ranking_metric"] == "0.5*accuracy + 0.3*auroc + 0.2*auprc"
    assert diagnostics["ranking_split"] == "validation"
    assert diagnostics["selected_layer"] == 1
    assert diagnostics["n_layers"] == 2
    assert diagnostics["selected"]["rank"] == 1
    assert diagnostics["top_layers"][0]["layer"] == 1
    assert diagnostics["top_layers"][0]["official_combined"] is not None
    assert isinstance(state.threshold, float)


def _paper_examples() -> list[_RCSTrainExample]:
    return [
        _RCSTrainExample(prompt="benign 0", images=(), is_benign=True, dataset="benign"),
        _RCSTrainExample(prompt="benign 1", images=(), is_benign=True, dataset="benign"),
        _RCSTrainExample(prompt="malicious 0", images=(), is_benign=False, dataset="malicious"),
        _RCSTrainExample(prompt="malicious 1", images=(), is_benign=False, dataset="malicious"),
    ]


def _paper_examples_with_validation() -> list[_RCSTrainExample]:
    return [
        _RCSTrainExample(prompt="benign train 0", images=(), is_benign=True, dataset="benign", split="train"),
        _RCSTrainExample(prompt="benign train 1", images=(), is_benign=True, dataset="benign", split="train"),
        _RCSTrainExample(prompt="malicious train 0", images=(), is_benign=False, dataset="malicious", split="train"),
        _RCSTrainExample(prompt="malicious train 1", images=(), is_benign=False, dataset="malicious", split="train"),
        _RCSTrainExample(prompt="benign validation 0", images=(), is_benign=True, dataset="benign", split="validation"),
        _RCSTrainExample(prompt="benign validation 1", images=(), is_benign=True, dataset="benign", split="validation"),
        _RCSTrainExample(
            prompt="malicious validation 0",
            images=(),
            is_benign=False,
            dataset="malicious",
            split="validation",
        ),
        _RCSTrainExample(
            prompt="malicious validation 1",
            images=(),
            is_benign=False,
            dataset="malicious",
            split="validation",
        ),
    ]
