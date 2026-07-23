#!/usr/bin/env python3
"""
Reproduce JailGuard (text detector).

Reference repo: references/repos/JailGuard
Reference entrypoint (text): references/repos/JailGuard/JailGuard/main_txt.py

This repo provides two modes:
- paper-openai: query a victim model via `openai_compat` and compute divergence; uses `spacy`
  similarity if requested (paper default).
- extra-echo: CI-safe shape reproduction (no external model calls; responses=variants).

For a reference-style text run, pass:
  --similarity spacy --policy-pool PI,TI,TL --policy-probs 0.24,0.52,0.24
  --punctuation-backend textaugment

Safety note:
- Do NOT commit raw jailbreak artifacts or harmful prompts. Prefer placeholders/redaction.
"""

from __future__ import annotations

import argparse
import json

from turnkey.components.backends import load_backend
from turnkey.components.detectors.jailguard import JailGuardDetector
from turnkey.config import ModelConfig
from turnkey.methods import MethodContext
from turnkey.policy import Generate, PolicyChain, PolicyRequest, TargetSession
from turnkey.schema import Sample

_MUTATORS = ("RR", "RI", "TR", "TI", "RD", "SR", "PI", "TL", "PL")


class _BackendGenerateProvider:
    request_type = Generate
    model_forwards_per_call = 1

    def __init__(self, backend) -> None:  # noqa: ANN001
        self.backend = backend

    def provide(self, request: Generate):  # noqa: ANN201
        return self.backend.generate(
            prompt=request.prompt,
            images=request.images,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


def _csv_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated non-empty list")
    return values


def _float_csv_tuple(raw: str) -> tuple[float, ...]:
    try:
        return tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["paper-openai", "extra-echo"], default="extra-echo")
    p.add_argument("--prompt", default="Explain the water cycle in one sentence.")
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--n-variants", type=int, default=8)
    p.add_argument("--mutator", default="PL", choices=_MUTATORS)
    p.add_argument("--char-rate", type=float, default=0.005)
    p.add_argument("--similarity", choices=["spacy", "bow"], default="bow")
    p.add_argument("--spacy-model", default="en_core_web_md")
    p.add_argument("--policy-pool", type=_csv_tuple, default=("PI", "RI", "RD"))
    p.add_argument("--policy-probs", type=_float_csv_tuple, default=(0.34, 0.33, 0.33))
    p.add_argument("--punctuation-backend", choices=["simple", "textaugment"], default="simple")
    p.add_argument("--synonym-level", type=int, default=20)
    p.add_argument("--translation-level", type=int, default=10)
    p.add_argument("--translation-source-lang", default="en")
    p.add_argument("--translation-target-langs", type=_csv_tuple, default=("ru", "fr", "de", "el", "id", "it", "ja", "ko", "la", "pl"))
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

    # openai_compat params (paper-openai)
    p.add_argument("--base-url", default="https://api.openai.com")
    p.add_argument("--model-id", default="gpt-3.5-turbo-1106")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")

    args = p.parse_args(argv)

    response_mode = "echo_prompt" if args.mode == "extra-echo" else "backend"
    det = JailGuardDetector(
        n_variants=int(args.n_variants),
        mutator=args.mutator,
        char_rate=float(args.char_rate),
        threshold=float(args.threshold),
        similarity=args.similarity,
        spacy_model=args.spacy_model,
        response_mode=response_mode,
        policy_pool=args.policy_pool,
        policy_probs=args.policy_probs,
        punctuation_backend=args.punctuation_backend,
        synonym_level=int(args.synonym_level),
        translation_level=int(args.translation_level),
        translation_source_lang=args.translation_source_lang,
        translation_target_langs=args.translation_target_langs,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        seed=int(args.seed),
    )

    sample = Sample(sample_id="jg-demo-0", behavior_id="jailguard:demo", is_benign=True, prompt=args.prompt)
    if response_mode == "echo_prompt":
        decision = det.decide(sample)
        result = {"block": decision.block, "score": decision.score, "reason": decision.reason}
    else:
        backend = load_backend(
            ModelConfig(
                backend="openai_compat",
                model_id=args.model_id,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                timeout_s=300.0,
            )
        )
        provider = _BackendGenerateProvider(backend)
        request = PolicyRequest(
            sample=sample,
            target=Generate(
                prompt=sample.prompt,
                images=sample.images,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
            ),
        )
        with MethodContext((provider,)) as context:
            outcome = PolicyChain((det.policy(),)).run(request, TargetSession(context))
        result = {"block": outcome.action == "block", "score": outcome.score, "reason": outcome.reason}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
