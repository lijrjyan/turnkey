from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Generic, Protocol, TypeVar, cast
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from turnkey.runtime_events import EventRecorder, request_summary, result_summary


ValueT = TypeVar("ValueT")
RequestT = TypeVar("RequestT", bound="Request[Any]")


class Request(Generic[ValueT]):
    """Base class for typed, hashable runtime requests."""


class Provider(Protocol[RequestT]):
    request_type: type[RequestT]

    def provide(self, request: RequestT) -> Any:
        ...


@dataclass(frozen=True)
class RequestUse:
    request_type: str
    provider_type: str
    cache_hit: bool
    scope: str | None
    duration_s: float
    model_forwards: int


@dataclass(frozen=True)
class LoadedEntrypoint:
    value: Any
    source: dict[str, Any]


class MethodContext:
    def __init__(
        self,
        providers: Iterable[Provider[Any]] = (),
        *,
        event_recorder: EventRecorder | None = None,
    ) -> None:
        self._providers: dict[type[Request[Any]], Provider[Any]] = {}
        for provider in providers:
            request_type = getattr(provider, "request_type", None)
            if not isinstance(request_type, type) or not issubclass(request_type, Request):
                raise TypeError("method providers must declare a Request subclass as request_type")
            if request_type in self._providers:
                raise ValueError(f"duplicate provider for {_type_name(request_type)}")
            self._providers[request_type] = provider

        self._cache: dict[Request[Any], Any] = {}
        self._uses: list[RequestUse] = []
        self._materialized_providers: list[Provider[Any]] = []
        self._scope: str | None = None
        self._case_id: str | None = None
        self._parent_event_id: str | None = None
        self._event_recorder = event_recorder
        self._closed = False

    @property
    def uses(self) -> tuple[RequestUse, ...]:
        return tuple(self._uses)

    def get(self, request: Request[ValueT]) -> ValueT:
        if self._closed:
            raise RuntimeError("method context is closed")
        try:
            hash(request)
        except TypeError as exc:
            raise TypeError("method requests must be hashable value objects") from exc

        request_type = type(request)
        provider = self._providers.get(request_type)
        if provider is None:
            available = ", ".join(
                sorted(_type_name(available_type) for available_type in self._providers)
            ) or "<none>"
            raise LookupError(
                f"no provider for {_type_name(request_type)}; available request types: {available}"
            )

        provider_type = _type_name(type(provider))
        cache_hit = request in self._cache
        if cache_hit:
            value = self._cache[request]
            if self._event_recorder is not None:
                with self._request_event(
                    request_type=request_type,
                    provider_type=provider_type,
                    cache_hit=True,
                    model_forwards=0,
                ) as event:
                    event.request = request_summary(request)
                    event.result = result_summary(value)
            self._uses.append(
                RequestUse(
                    request_type=_type_name(request_type),
                    provider_type=provider_type,
                    cache_hit=True,
                    scope=self._scope,
                    duration_s=0.0,
                    model_forwards=0,
                )
            )
            return cast(ValueT, value)

        if not any(materialized is provider for materialized in self._materialized_providers):
            self._materialized_providers.append(provider)
        model_forwards = getattr(provider, "model_forwards_per_call", 0)
        if isinstance(model_forwards, bool) or not isinstance(model_forwards, int) or model_forwards < 0:
            raise TypeError("method provider model_forwards_per_call must be a non-negative integer")
        started = perf_counter()
        if self._event_recorder is None:
            value = provider.provide(request)
        else:
            with self._request_event(
                request_type=request_type,
                provider_type=provider_type,
                cache_hit=False,
                model_forwards=model_forwards,
            ) as event:
                event.request = request_summary(request)
                value = provider.provide(request)
                event.result = result_summary(value)
        duration_s = perf_counter() - started
        self._cache[request] = value
        self._uses.append(
            RequestUse(
                request_type=_type_name(request_type),
                provider_type=provider_type,
                cache_hit=False,
                scope=self._scope,
                duration_s=duration_s,
                model_forwards=model_forwards,
            )
        )
        return cast(ValueT, value)

    @contextmanager
    def scope(
        self,
        name: str | None = None,
        *,
        case_id: str | None = None,
        pass_name: str | None = None,
        parent_event_id: str | None = None,
    ) -> Iterator[None]:
        effective_pass = pass_name if pass_name is not None else name
        if not effective_pass:
            raise ValueError("method request scope must be non-empty")
        previous = (self._scope, self._case_id, self._parent_event_id)
        self._scope = effective_pass
        self._case_id = case_id
        self._parent_event_id = parent_event_id
        try:
            yield
        finally:
            self._scope, self._case_id, self._parent_event_id = previous

    def _request_event(
        self,
        *,
        request_type: type[Request[Any]],
        provider_type: str,
        cache_hit: bool,
        model_forwards: int,
    ):
        if self._event_recorder is None or self._case_id is None or self._scope is None:
            raise RuntimeError("runtime event recording requires case and pass scope")
        return self._event_recorder.span(
            case_id=self._case_id,
            pass_name=self._scope,
            kind="request",
            name=_type_name(request_type),
            parent_event_id=self._parent_event_id,
            provider=provider_type,
            cache_hit=cache_hit,
            model_forwards=model_forwards,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for provider in reversed(self._materialized_providers):
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # cleanup must continue for remaining providers
                    errors.append(exc)
        if errors:
            noun = "provider" if len(errors) == 1 else "providers"
            raise RuntimeError(f"failed to close {len(errors)} method {noun}") from errors[0]

    def __enter__(self) -> "MethodContext":
        if self._closed:
            raise RuntimeError("method context is closed")
        return self

    def __exit__(
        self,
        _exc_type: object,
        exc: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if exc is None:
                raise
            _attach_secondary_failure(
                exc,
                label="method provider cleanup",
                secondary=close_error,
            )


def load_entrypoint(spec: str) -> Any:
    return load_entrypoint_with_identity(spec).value


def load_entrypoint_with_identity(spec: str) -> LoadedEntrypoint:
    source, object_path = split_entrypoint(spec)

    source_path = Path(source).expanduser()
    if source.endswith(".py") or source_path.is_file():
        module, source_identity = _load_file_module(source_path, object_path=object_path)
    else:
        module, source_identity = _load_module(source, object_path=object_path)

    value: Any = module
    for attribute in object_path.split("."):
        if not attribute:
            raise ValueError("method entrypoint object path contains an empty attribute")
        try:
            value = getattr(value, attribute)
        except AttributeError as exc:
            raise AttributeError(f"entrypoint {spec!r} has no object {object_path!r}") from exc
    return LoadedEntrypoint(value=value, source=source_identity)


def split_entrypoint(spec: str) -> tuple[str, str]:
    source, separator, object_path = spec.partition(":")
    if not separator or not source or not object_path:
        raise ValueError("method entrypoint must use module:object or path.py:object")
    if any(not attribute for attribute in object_path.split(".")):
        raise ValueError("method entrypoint object path contains an empty attribute")
    return source, object_path


def is_entrypoint(spec: str) -> bool:
    try:
        split_entrypoint(spec)
    except ValueError:
        return False
    return True


def _load_file_module(
    path: Path,
    *,
    object_path: str,
) -> tuple[Any, dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"method entrypoint file does not exist: {path}")

    source = resolved.read_bytes()
    path_digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    source_sha256 = hashlib.sha256(source).hexdigest()
    module_name = f"_turnkey_external_{path_digest}_{source_sha256[:16]}"

    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load method entrypoint file: {resolved}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(resolved), "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module, {
        "kind": "file",
        "path": str(resolved),
        "sha256": source_sha256,
        "bytes": len(source),
        "object": object_path,
    }


def _load_module(
    module_name: str,
    *,
    object_path: str,
) -> tuple[Any, dict[str, Any]]:
    source_attribute = "__turnkey_entrypoint_source_identity__"
    existing = sys.modules.get(module_name)
    cached_identity = getattr(existing, source_attribute, None)
    if isinstance(cached_identity, dict):
        return existing, {**cached_identity, "object": object_path}

    identity: dict[str, Any] = {
        "kind": "module",
        "module": module_name,
    }
    module_spec = importlib.util.find_spec(module_name)
    origin = module_spec.origin if module_spec is not None else None
    if not isinstance(origin, str) or not Path(origin).is_file():
        raise RuntimeError(
            f"local module entrypoint {module_name!r} has no verifiable source file"
        )
    path = Path(origin).resolve()
    top_level_package = module_name.partition(".")[0]
    distributions = importlib.metadata.packages_distributions().get(top_level_package, ())
    distribution_identity = _distribution_identity_for_path(
        path,
        distributions,
        module_name=module_name,
    )
    if distribution_identity is not None:
        identity.update(distribution_identity)
        module = importlib.import_module(module_name)
    else:
        if existing is not None:
            raise RuntimeError(
                f"local module entrypoint {module_name!r} was imported before Turnkey "
                "could record its source identity; use path.py:object or restart the process"
            )
        source = path.read_bytes()
        identity.update(
            {
                "path": str(path),
                "sha256": hashlib.sha256(source).hexdigest(),
                "bytes": len(source),
            }
        )
        if path.suffix.lower() in {".py", ".pyw"}:
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            try:
                exec(compile(source, str(path), "exec"), module.__dict__)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
        else:
            module = importlib.import_module(module_name)
    setattr(module, source_attribute, dict(identity))
    return module, {**identity, "object": object_path}


def _distribution_identity_for_path(
    path: Path,
    distribution_names: Iterable[str],
    *,
    module_name: str,
) -> dict[str, str] | None:
    for distribution_name in sorted(set(distribution_names)):
        distribution = importlib.metadata.distribution(distribution_name)
        for distribution_file in distribution.files or ():
            located = Path(distribution.locate_file(distribution_file)).resolve()
            if located == path:
                return {
                    "distribution": distribution_name,
                    "version": distribution.version,
                }
        if _editable_distribution_contains(
            distribution,
            path,
            module_name=module_name,
        ):
            return {
                "distribution": distribution_name,
                "version": distribution.version,
            }
    return None


def _editable_distribution_contains(
    distribution: importlib.metadata.Distribution,
    path: Path,
    *,
    module_name: str,
) -> bool:
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        return False
    try:
        direct_url = json.loads(direct_url_text)
    except (json.JSONDecodeError, TypeError):
        return False
    if direct_url.get("dir_info", {}).get("editable") is not True:
        return False
    parsed_url = urlparse(direct_url.get("url", ""))
    if parsed_url.scheme != "file" or parsed_url.netloc not in {"", "localhost"}:
        return False
    editable_root = Path(url2pathname(unquote(parsed_url.path))).resolve()
    for distribution_file in distribution.files or ():
        if Path(distribution_file).suffix.lower() != ".pth":
            continue
        pth_path = Path(distribution.locate_file(distribution_file)).resolve()
        for raw_line in pth_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(("import ", "import\t")):
                continue
            declared_root = Path(line)
            if not declared_root.is_absolute():
                declared_root = pth_path.parent / declared_root
            declared_root = declared_root.resolve()
            if not (
                declared_root == editable_root or editable_root in declared_root.parents
            ):
                continue
            module_path = declared_root.joinpath(*module_name.split("."))
            candidates = {
                Path(f"{module_path}{suffix}").resolve()
                for suffix in importlib.machinery.all_suffixes()
            }
            candidates.update(
                (module_path / f"__init__{suffix}").resolve()
                for suffix in importlib.machinery.all_suffixes()
            )
            if path in candidates:
                return True
    return False


def _attach_secondary_failure(
    primary: BaseException,
    *,
    label: str,
    secondary: BaseException,
) -> None:
    failures = tuple(getattr(primary, "__turnkey_secondary_failures__", ()))
    try:
        setattr(
            primary,
            "__turnkey_secondary_failures__",
            (*failures, (label, secondary)),
        )
    except (AttributeError, TypeError):
        pass
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(f"{label} also failed: {secondary!r}")


def _type_name(value_type: type[Any]) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"
