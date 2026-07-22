from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from turnkey.components.detectors.base import Detector, DetectorManifest
from turnkey.runtime_providers import (
    HFLastTokenHiddenStateProvider,
    LastTokenHiddenStateRequest,
    LastTokenHiddenStateRequestProvider,
    LastTokenHiddenStateResult,
    hidden_state_effective_max_length,
    normalize_hidden_state_max_length,
    normalize_hidden_state_model_family,
    normalize_hidden_state_token_strategy,
)
from turnkey.methods import MethodContext
from turnkey.policy import NextPolicy, Outcome, Policy, PolicyRequest
from turnkey.schema import DetectorDecision, ImageInput, Sample

from . import paper as _paper
from .paper import (
    _RCS_LAYER_SELECTION_AUTO,
    _MCDGaussian,
    _RCSPaperScoringState,
    _calibrate_threshold,
    _decide_paper_from_hidden_states,
    _extract_layer_features,
    _fit_paper_scoring_state,
    _fit_projection,
    _hidden_state_config_for_paper_detector,
    _ledoit_wolf_cov,
    _load_rcs_paper_scoring_state,
    _model_param_int,
    _model_param_str,
    _projection_contrastive_loss,
    _reference_shrinkage_intensity,
    _save_rcs_paper_scoring_state,
    _score_paper,
    _score_paper_state,
    _scoring_state_path_from_artifact,
    _normalize_layer_selection_strategy,
)
from .toy import _knn_mean_distance, _mahalanobis_diag, _mean, _toy_feature_vector, _var_diag
from .training import (
    _RCSTrainExample,
    _load_train_examples_jsonl,
    _paper_examples_from_params,
    _paper_training_example_summary,
    _split_paper_examples,
)


def calibrate_rcs_paper_from_config(
    cfg: Any,
    *,
    artifact_path: Any,
    source_config_path: str | None,
    command: list[str] | None,
) -> tuple[Any, dict[str, Any]]:
    # Keep the historic monkeypatch target turnkey.components.detectors.rcs.HFLastTokenHiddenStateProvider working.
    _paper.HFLastTokenHiddenStateProvider = HFLastTokenHiddenStateProvider
    return _paper.calibrate_rcs_paper_from_config(
        cfg,
        artifact_path=artifact_path,
        source_config_path=source_config_path,
        command=command,
    )


@dataclass
class RCSDetector(Detector):
    """
    Representational Contrastive Scoring (RCS).

    Paper: arXiv:2512.12069 (LVLM jailbreak detection with contrastive scoring on safety-critical representations).

    Modes:
      - mode=toy (EXTRA): CI-safe skeleton that mimics the shape of contrastive scoring without LVLM hidden states.
      - mode=paper: paper-form pipeline with real hidden-state extraction + learned projection + MCD/KCD scoring.
    """

    mode: str = "toy"  # toy | paper
    method: str = "mcd"  # mcd | kcd
    threshold: float = 0.0
    k: int = 40
    prototype_image_count: int = 0
    prototype_image_path: str | None = None
    benign_prompts: list[str] = field(
        default_factory=lambda: [
            "Say hello in one short sentence.",
            "Explain the water cycle in one sentence.",
            "Describe the image in one short sentence.",
        ]
    )
    malicious_prompts: list[str] = field(
        default_factory=lambda: [
            "UNSAFE_PLACEHOLDER: request redacted (fixture).",
        ]
    )

    # paper-form params
    model: dict[str, Any] | None = None
    train_jsonl: str | None = None
    require_balanced_train: bool = False
    val_ratio: float = 0.2
    calibrate_threshold: bool = True
    objective_bal_acc_weight: float = 0.8
    objective_f1_weight: float = 0.2

    layer: int | None = None
    layer_selection_strategy: str = _RCS_LAYER_SELECTION_AUTO
    auto_layer_max_samples: int = 64
    auto_layer_svm_iters: int = 200

    projection_dim: int = 256
    projection_epochs: int = 100
    projection_batch_size: int = 64
    projection_lr: float = 1e-3
    projection_alpha: float = 1.0
    projection_beta: float = 5.0
    projection_md: float = 1.0
    projection_ms: float = 2.0
    projection_dropout: float = 0.3
    seed: int = 0

    image_token: str = "<image>"
    auto_insert_image_tokens: bool = True
    hidden_state_token_strategy: str = "last_token"
    include_embedding_layer: bool = False
    hidden_state_model_family: str = "generic"
    hidden_state_max_length: int | None = None
    var_floor: float = 1e-3

    _benign_vecs: list[list[float]] = field(init=False, repr=False)
    _malicious_vecs: list[list[float]] = field(init=False, repr=False)
    _mu_b: list[float] = field(init=False, repr=False)
    _mu_m: list[float] = field(init=False, repr=False)
    _var_b: list[float] = field(init=False, repr=False)
    _var_m: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        method = str(self.method).strip().lower()
        if mode not in {"toy", "paper"}:
            raise ValueError("rcs: mode must be 'toy' or 'paper'")
        if method not in {"mcd", "kcd"}:
            raise ValueError("rcs: method must be 'mcd' or 'kcd'")
        if not self.benign_prompts or not self.malicious_prompts:
            raise ValueError("rcs: benign_prompts and malicious_prompts must be non-empty")

        if mode == "paper":
            self._init_policy_paper(method)
            return

        proto_images = tuple(
            ImageInput(path=f"__proto_image_{i}__") for i in range(max(0, int(self.prototype_image_count)))
        )

        self._benign_vecs = [
            _toy_feature_vector(
                Sample(
                    sample_id=f"b{i}",
                    behavior_id="rcs:benign",
                    is_benign=True,
                    prompt=p,
                    images=proto_images,
                )
            )
            for i, p in enumerate(self.benign_prompts)
        ]
        self._malicious_vecs = [
            _toy_feature_vector(
                Sample(
                    sample_id=f"m{i}",
                    behavior_id="rcs:mal",
                    is_benign=False,
                    prompt=p,
                    images=proto_images,
                )
            )
            for i, p in enumerate(self.malicious_prompts)
        ]

        self._mu_b = _mean(self._benign_vecs)
        self._mu_m = _mean(self._malicious_vecs)
        self._var_b = _var_diag(self._benign_vecs, self._mu_b)
        self._var_m = _var_diag(self._malicious_vecs, self._mu_m)

    def _init_policy_paper(self, method: str) -> None:
        if method not in {"mcd", "kcd"}:
            raise ValueError("rcs_paper_v3: method must be 'mcd' or 'kcd'")
        if not isinstance(self.model, dict):
            raise ValueError("rcs_paper_v3 requires params.model (dict)")

        model_id = self.model.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("rcs_paper_v3: params.model must include model_id (str)")

        self.mode = "paper"
        self.method = method
        self.layer_selection_strategy = _normalize_layer_selection_strategy(self.layer_selection_strategy)
        self._paper_model_id = model_id
        self._paper_revision = _model_param_str(self.model, "revision")
        self._paper_device = _model_param_str(self.model, "device", "auto") or "auto"
        self._paper_torch_dtype = _model_param_str(self.model, "torch_dtype", "auto") or "auto"
        self._paper_trust_remote_code = bool(self.model.get("trust_remote_code", False))
        self._paper_local_files_only = bool(self.model.get("local_files_only", False))
        self._paper_token_env = _model_param_str(self.model, "token_env", "HF_TOKEN") or "HF_TOKEN"
        token_strategy = (
            _model_param_str(self.model, "hidden_state_token_strategy")
            or _model_param_str(self.model, "token_strategy")
            or self.hidden_state_token_strategy
        )
        self._paper_hidden_state_token_strategy = normalize_hidden_state_token_strategy(token_strategy)
        self._paper_include_embedding_layer = bool(
            self.model.get("include_embedding_layer", self.include_embedding_layer)
        )
        model_family = (
            _model_param_str(self.model, "hidden_state_model_family")
            or _model_param_str(self.model, "model_family")
            or self.hidden_state_model_family
        )
        self._paper_hidden_state_model_family = normalize_hidden_state_model_family(model_family)
        max_length = _model_param_int(self.model, "hidden_state_max_length")
        if max_length is None:
            max_length = _model_param_int(self.model, "max_length", self.hidden_state_max_length)
        self._paper_hidden_state_max_length = hidden_state_effective_max_length(
            model_family=self._paper_hidden_state_model_family,
            max_length=normalize_hidden_state_max_length(max_length),
        )
        self._paper_examples = _paper_examples_from_params(
            train_jsonl=self.train_jsonl,
            require_balanced_train=bool(self.require_balanced_train),
            prototype_image_count=int(self.prototype_image_count),
            prototype_image_path=self.prototype_image_path,
            benign_prompts=self.benign_prompts,
            malicious_prompts=self.malicious_prompts,
        )
    def manifest(self, *, name: str | None = None) -> DetectorManifest:
        if str(self.mode).strip().lower() == "paper":
            return self._paper_manifest(name=name)
        return DetectorManifest(
            name=name or "rcs_toy_v3",
            version="v3-toy",
            required_inputs=("sample", "prompt", "images"),
            reproducibility={
                "mode": "toy",
                "method": self.method,
                "threshold": self.threshold,
                "k": self.k,
                "prototype_image_count": self.prototype_image_count,
                "benign_prompts": list(self.benign_prompts),
                "malicious_prompts": list(self.malicious_prompts),
                "var_floor": self.var_floor,
            },
        )

    def method_providers(self) -> tuple[LastTokenHiddenStateRequestProvider, ...]:
        if str(self.mode).strip().lower() == "paper":
            return (LastTokenHiddenStateRequestProvider(),)
        return ()

    def policy(self, *, calibration_artifact: Any | None = None) -> Policy:
        if str(self.mode).strip().lower() == "paper":
            return RCSPaperPolicy(self, calibration_artifact=calibration_artifact)
        return super().policy(calibration_artifact=calibration_artifact)

    def _paper_manifest(self, *, name: str | None = None) -> DetectorManifest:
        training_examples = _paper_training_example_summary(
            examples=self._paper_examples,
            source=self.train_jsonl or "inline_prototypes",
            source_type="jsonl" if self.train_jsonl else "inline_prototypes",
            require_balanced_train=bool(self.require_balanced_train),
            prototype_image_count=int(self.prototype_image_count),
            prototype_image_path=self.prototype_image_path,
        )
        return DetectorManifest(
            name=name or "rcs_paper_v3",
            version="v3-paper",
            required_inputs=(
                "sample",
                "prompt",
                "images",
            ),
            reproducibility={
                "mode": "paper",
                "method": self.method,
                "threshold": self.threshold,
                "k": self.k,
                "model": {
                    "model_id": self._paper_model_id,
                    "revision": self._paper_revision,
                    "device": self._paper_device,
                    "torch_dtype": self._paper_torch_dtype,
                    "trust_remote_code": self._paper_trust_remote_code,
                    "local_files_only": self._paper_local_files_only,
                    "token_env": self._paper_token_env,
                    "hidden_state_token_strategy": self._paper_hidden_state_token_strategy,
                    "include_embedding_layer": self._paper_include_embedding_layer,
                    "hidden_state_model_family": self._paper_hidden_state_model_family,
                    "hidden_state_max_length": self._paper_hidden_state_max_length,
                },
                "training_examples": training_examples,
                "layer": self.layer,
                "layer_selection_strategy": self.layer_selection_strategy,
                "auto_layer_max_samples": self.auto_layer_max_samples,
                "auto_layer_svm_iters": self.auto_layer_svm_iters,
                "projection": {
                    "dim": self.projection_dim,
                    "epochs": self.projection_epochs,
                    "batch_size": self.projection_batch_size,
                    "lr": self.projection_lr,
                    "alpha": self.projection_alpha,
                    "beta": self.projection_beta,
                    "md": self.projection_md,
                    "ms": self.projection_ms,
                    "dropout": self.projection_dropout,
                },
                "calibration": {
                    "val_ratio": self.val_ratio,
                    "calibrate_threshold": self.calibrate_threshold,
                    "objective_bal_acc_weight": self.objective_bal_acc_weight,
                    "objective_f1_weight": self.objective_f1_weight,
                },
                "image_token": self.image_token,
                "auto_insert_image_tokens": self.auto_insert_image_tokens,
                "var_floor": self.var_floor,
                "seed": self.seed,
            },
        )

    def decide(self, sample: Sample) -> DetectorDecision:
        if str(self.mode).strip().lower() == "paper":
            raise RuntimeError("rcs: mode=paper requires execution through RCSPaperPolicy")

        x = _toy_feature_vector(sample)

        method = str(self.method).strip().lower()
        if method == "mcd":
            d_b = _mahalanobis_diag(x, self._mu_b, self._var_b)
            d_m = _mahalanobis_diag(x, self._mu_m, self._var_m)
        else:
            d_b = _knn_mean_distance(x, self._benign_vecs, k=self.k)
            d_m = _knn_mean_distance(x, self._malicious_vecs, k=self.k)

        score = float(d_b - d_m)
        block = bool(score > float(self.threshold))
        reason = (
            f"rcs(mode={self.mode}, method={self.method}, score={score:.4f}, threshold={float(self.threshold):.4f}, "
            f"n_images={len(sample.images)})"
        )
        return DetectorDecision(block=block, score=score, reason=reason)

@dataclass
class RCSPaperPolicy:
    detector: RCSDetector
    calibration_artifact: Any | None = None
    _scoring: _RCSPaperScoringState | None = field(default=None, init=False, repr=False)

    def apply(
        self,
        request: PolicyRequest,
        call_next: NextPolicy,
        context: MethodContext,
    ) -> Outcome:
        scoring = self._ensure_scoring(context)
        hidden_states = self._hidden_states(context, request.sample)
        decision = _decide_paper_from_hidden_states(
            scoring,
            hidden_states,
            request.sample,
            device=None,
        )
        if decision.block:
            return Outcome.blocked(
                request.target,
                score=decision.score,
                reason=decision.reason,
                diagnostics=decision.diagnostics,
            )
        outcome = call_next(request)
        return replace(
            outcome,
            score=outcome.score if outcome.score is not None else decision.score,
            reason=outcome.reason if outcome.reason is not None else decision.reason,
            diagnostics={**decision.diagnostics, **outcome.diagnostics},
        )

    def _ensure_scoring(self, context: MethodContext) -> _RCSPaperScoringState:
        if self._scoring is not None:
            return self._scoring
        if self.calibration_artifact is not None:
            state_path = _scoring_state_path_from_artifact(self.calibration_artifact)
            scoring = _load_rcs_paper_scoring_state(state_path)
        else:
            scoring = _fit_paper_scoring_state(
                hidden_state_provider=lambda sample: self._hidden_states(context, sample),
                device=None,
                examples=self.detector._paper_examples,
                method=self.detector.method,
                threshold=float(self.detector.threshold),
                k=int(self.detector.k),
                val_ratio=float(self.detector.val_ratio),
                calibrate_threshold=bool(self.detector.calibrate_threshold),
                objective_bal_acc_weight=float(self.detector.objective_bal_acc_weight),
                objective_f1_weight=float(self.detector.objective_f1_weight),
                layer=self.detector.layer,
                auto_layer_max_samples=int(self.detector.auto_layer_max_samples),
                auto_layer_svm_iters=int(self.detector.auto_layer_svm_iters),
                projection_dim=int(self.detector.projection_dim),
                projection_epochs=int(self.detector.projection_epochs),
                projection_batch_size=int(self.detector.projection_batch_size),
                projection_lr=float(self.detector.projection_lr),
                projection_alpha=float(self.detector.projection_alpha),
                projection_beta=float(self.detector.projection_beta),
                projection_md=float(self.detector.projection_md),
                projection_ms=float(self.detector.projection_ms),
                projection_dropout=float(self.detector.projection_dropout),
                var_floor=float(self.detector.var_floor),
                seed=int(self.detector.seed),
                layer_selection_strategy=self.detector.layer_selection_strategy,
            )
        self.detector.threshold = float(scoring.threshold)
        self.detector.k = int(scoring.k)
        self._scoring = scoring
        return scoring

    def _hidden_states(self, context: MethodContext, sample: Sample) -> LastTokenHiddenStateResult:
        return context.get(
            LastTokenHiddenStateRequest.from_sample(
                _hidden_state_config_for_paper_detector(self.detector),
                sample,
            )
        )


__all__ = [
    "RCSPaperPolicy",
    "RCSDetector",
    "calibrate_rcs_paper_from_config",
    "_RCSTrainExample",
    "_load_train_examples_jsonl",
    "_split_paper_examples",
    "_MCDGaussian",
    "_RCSPaperScoringState",
    "_calibrate_threshold",
    "_decide_paper_from_hidden_states",
    "_extract_layer_features",
    "_fit_paper_scoring_state",
    "_fit_projection",
    "_hidden_state_config_for_paper_detector",
    "_ledoit_wolf_cov",
    "_load_rcs_paper_scoring_state",
    "_projection_contrastive_loss",
    "_reference_shrinkage_intensity",
    "_save_rcs_paper_scoring_state",
    "_score_paper",
    "_score_paper_state",
]
