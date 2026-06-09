"""OpenAI image generation backend.

Exposes OpenAI's ``gpt-image-2`` model at three quality tiers as an
:class:`ImageGenProvider` implementation. The tiers are implemented as
three virtual model IDs so the ``hermes tools`` model picker and the
``image_gen.model`` config key behave like any other multi-model backend:

    gpt-image-2-low     ~15s   fastest, good for iteration
    gpt-image-2-medium  ~40s   default — balanced
    gpt-image-2-high    ~2min  slowest, highest fidelity

All three hit the same underlying API model (``gpt-image-2`` by default)
with a different ``quality`` parameter. Output is base64 JSON → saved under
``$HERMES_HOME/cache/images/``.

Selection precedence (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``

The underlying API model and base URL are also configurable:

* ``OPENAI_IMAGE_API_MODEL`` or ``image_gen.openai.api_model``
* ``OPENAI_IMAGE_BASE_URL`` / ``image_gen.openai.base_url`` /
  ``OPENAI_BASE_URL``
* ``OPENAI_IMAGE_API_MODE`` or ``image_gen.openai.api_mode`` (``images`` or
  ``responses``)
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
#
# All three IDs resolve to the same underlying API model with a different
# ``quality`` setting. ``api_model`` is what gets sent to OpenAI;
# ``quality`` is the knob that changes generation time and output fidelity.

API_MODEL = "gpt-image-2"
API_MODE_IMAGES = "images"
API_MODE_RESPONSES = "responses"
RESPONSES_MODEL = "gpt-5.5"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

_SUPPORTED_REFERENCE_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}
_REFERENCE_IMAGE_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_EDIT_IMAGE_MAX_BYTES = 50 * 1024 * 1024
_RESPONSES_INPUT_IMAGE_MAX_BYTES = 20 * 1024 * 1024


def _load_openai_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _clean_str(value: Any) -> Optional[str]:
    """Return a stripped non-empty string, or ``None``."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _openai_subconfig(cfg: Dict[str, Any]) -> Dict[str, Any]:
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    return openai_cfg if isinstance(openai_cfg, dict) else {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_openai_config()
    openai_cfg = _openai_subconfig(cfg)
    candidate: Optional[str] = None
    value = openai_cfg.get("model")
    if isinstance(value, str) and value in _MODELS:
        candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_api_model() -> str:
    """Resolve the actual model string sent to ``/images/generations``."""
    env_override = _clean_str(os.environ.get("OPENAI_IMAGE_API_MODEL"))
    if env_override:
        return env_override

    cfg = _load_openai_config()
    openai_cfg = _openai_subconfig(cfg)

    for value in (openai_cfg.get("api_model"), cfg.get("api_model")):
        cleaned = _clean_str(value)
        if cleaned:
            return cleaned

    return API_MODEL


def _resolve_base_url() -> Optional[str]:
    """Resolve an OpenAI-compatible image API base URL, if configured."""
    env_override = _clean_str(os.environ.get("OPENAI_IMAGE_BASE_URL"))
    if env_override:
        return env_override.rstrip("/")

    cfg = _load_openai_config()
    openai_cfg = _openai_subconfig(cfg)

    for value in (
        openai_cfg.get("base_url"),
        cfg.get("base_url"),
        os.environ.get("OPENAI_BASE_URL"),
    ):
        cleaned = _clean_str(value)
        if cleaned:
            return cleaned.rstrip("/")

    return None


def _resolve_api_mode() -> str:
    """Resolve whether to use ``/images/generations`` or Responses tools."""
    env_override = _clean_str(os.environ.get("OPENAI_IMAGE_API_MODE"))
    if env_override:
        mode = env_override.lower().replace("-", "_")
        if mode in {"response", "responses", "codex_responses"}:
            return API_MODE_RESPONSES
        return API_MODE_IMAGES

    cfg = _load_openai_config()
    openai_cfg = _openai_subconfig(cfg)

    for value in (openai_cfg.get("api_mode"), cfg.get("api_mode")):
        cleaned = _clean_str(value)
        if cleaned:
            mode = cleaned.lower().replace("-", "_")
            if mode in {"response", "responses", "codex_responses"}:
                return API_MODE_RESPONSES
            return API_MODE_IMAGES

    return API_MODE_IMAGES


def _resolve_responses_model() -> str:
    """Resolve the chat model that hosts the Responses image_generation tool."""
    env_override = _clean_str(os.environ.get("OPENAI_IMAGE_RESPONSES_MODEL"))
    if env_override:
        return env_override

    cfg = _load_openai_config()
    openai_cfg = _openai_subconfig(cfg)

    for value in (openai_cfg.get("responses_model"), cfg.get("responses_model")):
        cleaned = _clean_str(value)
        if cleaned:
            return cleaned

    return RESPONSES_MODEL


def _normalize_reference_image_paths(value: Any) -> List[Path]:
    """Validate local reference image paths and return canonical Path objects."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        return []

    paths: List[Path] = []
    seen = set()
    for item in raw_items:
        if not isinstance(item, str):
            continue
        raw = item.strip()
        if not raw:
            continue
        if raw.startswith("file://"):
            raw = unquote(raw[7:])
        path = Path(os.path.expanduser(raw))
        if not path.is_file():
            raise ValueError(f"Reference image does not exist or is not a file: {raw}")
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        size = resolved.stat().st_size
        if size <= 0:
            raise ValueError(f"Reference image is empty: {resolved}")
        if size > _EDIT_IMAGE_MAX_BYTES:
            raise ValueError(
                f"Reference image exceeds {_EDIT_IMAGE_MAX_BYTES // (1024 * 1024)}MB: {resolved}"
            )
        mime = _guess_reference_image_mime(resolved)
        if mime not in _SUPPORTED_REFERENCE_IMAGE_TYPES:
            raise ValueError(
                f"Unsupported reference image type for {resolved}: {mime}. "
                "Use PNG, JPEG, or WEBP."
            )
        paths.append(resolved)
    return paths[:16]


def _guess_reference_image_mime(path: Path) -> str:
    """Return an OpenAI-supported image MIME type for a local reference image."""
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed in _SUPPORTED_REFERENCE_IMAGE_TYPES:
        return guessed
    return _REFERENCE_IMAGE_EXT_TO_MIME.get(path.suffix.lower(), "application/octet-stream")


def _reference_image_to_data_url(path: Path) -> str:
    """Encode a local reference image as a Responses API input_image data URL."""
    size = path.stat().st_size
    if size > _RESPONSES_INPUT_IMAGE_MAX_BYTES:
        raise ValueError(
            f"Reference image exceeds Responses inline image cap "
            f"({_RESPONSES_INPUT_IMAGE_MAX_BYTES // (1024 * 1024)}MB): {path}"
        )
    mime = _guess_reference_image_mime(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_responses_payload(
    *,
    prompt: str,
    responses_model: str,
    api_model: str,
    size: str,
    quality: str,
    reference_image_data_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible Responses image_generation request."""
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for data_url in reference_image_data_urls or []:
        if isinstance(data_url, str) and data_url.startswith("data:image/"):
            content.append({"type": "input_image", "image_url": data_url})

    return {
        "model": responses_model,
        "store": False,
        "instructions": (
            "You are an assistant that must fulfill image generation requests "
            "by using the image_generation tool when provided."
        ),
        "input": [{
            "type": "message",
            "role": "user",
            "content": content,
        }],
        "tools": [{
            "type": "image_generation",
            "model": api_model,
            "size": size,
            "quality": quality,
            "output_format": "png",
            "background": "opaque",
            "partial_images": 1,
        }],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "image_generation"}],
        },
        "stream": False,
    }


def _extract_image_b64(value: Any) -> Optional[str]:
    """Return the newest image b64 embedded in a Responses payload."""
    found: Optional[str] = None
    if isinstance(value, dict):
        if value.get("type") == "image_generation_call":
            result = value.get("result")
            if isinstance(result, str) and result:
                found = result
        partial = value.get("partial_image_b64")
        if isinstance(partial, str) and partial:
            found = partial
        for child in value.values():
            nested = _extract_image_b64(child)
            if nested:
                found = nested
    elif isinstance(value, list):
        for child in value:
            nested = _extract_image_b64(child)
            if nested:
                found = nested
    return found


def _collect_responses_image_b64(
    *,
    api_key: str,
    base_url: Optional[str],
    prompt: str,
    responses_model: str,
    api_model: str,
    size: str,
    quality: str,
    reference_image_data_urls: Optional[List[str]] = None,
) -> Optional[str]:
    """Call ``/responses`` and return the generated image b64 payload."""
    import httpx

    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    payload = _build_responses_payload(
        prompt=prompt,
        responses_model=responses_model,
        api_model=api_model,
        size=size,
        quality=quality,
        reference_image_data_urls=reference_image_data_urls,
    )
    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=30.0, pool=30.0)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=timeout, headers=headers) as http:
        response = http.post(f"{root}/responses", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise RuntimeError(
                f"Responses API returned HTTP {exc.response.status_code}: {body}"
            ) from exc
        data = response.json()

    return _extract_image_b64(data)


def _save_openai_image_response(
    response: Any,
    *,
    tier_id: str,
    prompt: str,
    aspect: str,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist an OpenAI images response and return the uniform provider payload."""
    data = getattr(response, "data", None) or []
    if not data:
        return error_response(
            error="OpenAI returned no image data",
            error_type="empty_response",
            provider="openai",
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    first = data[0]
    b64 = getattr(first, "b64_json", None)
    url = getattr(first, "url", None)
    revised_prompt = getattr(first, "revised_prompt", None)

    if b64:
        try:
            saved_path = save_b64_image(b64, prefix=f"openai_{tier_id}")
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        image_ref = str(saved_path)
    elif url:
        # Defensive — gpt-image-2 returns b64 today, but OpenAI's API has
        # previously returned URLs. Cache the bytes locally so gateways never
        # depend on an expiring signed URL.
        try:
            saved_path = save_url_image(url, prefix=f"openai_{tier_id}")
        except Exception as exc:
            logger.warning(
                "OpenAI image URL %s could not be cached (%s); falling back to bare URL.",
                url,
                exc,
            )
            image_ref = url
        else:
            image_ref = str(saved_path)
    else:
        return error_response(
            error="OpenAI response contained neither b64_json nor URL",
            error_type="empty_response",
            provider="openai",
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    response_extra = dict(extra)
    if revised_prompt:
        response_extra["revised_prompt"] = revised_prompt

    return success_response(
        image=image_ref,
        model=tier_id,
        prompt=prompt,
        aspect_ratio=aspect,
        provider="openai",
        extra=response_extra,
    )


def _call_images_edit(client: Any, payload: Dict[str, Any], image_paths: List[Path]) -> Any:
    """Call ``images.edit`` with real local image file handles."""
    handles = [path.open("rb") for path in image_paths]
    try:
        edit_payload = dict(payload)
        edit_payload["image"] = handles if len(handles) != 1 else handles[0]
        edit_payload["input_fidelity"] = "high"
        try:
            return client.images.edit(**edit_payload)
        except Exception as exc:
            if "input_fidelity" not in str(exc).lower():
                raise
            logger.debug("Retrying OpenAI image edit without input_fidelity", exc_info=True)
            for handle in handles:
                try:
                    handle.seek(0)
                except Exception:
                    pass
            edit_payload.pop("input_fidelity", None)
            return client.images.edit(**edit_payload)
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` backend — gpt-image-2 at low/medium/high."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI",
            "badge": "paid",
            "tag": "gpt-image-2 at low/medium/high quality tiers",
            "env_vars": [
                {
                    "key": "OPENAI_API_KEY",
                    "prompt": "OpenAI API key",
                    "url": "https://platform.openai.com/api-keys",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai",
                aspect_ratio=aspect,
            )

        if not os.environ.get("OPENAI_API_KEY"):
            return error_response(
                error=(
                    "OPENAI_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="openai",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        api_key = os.environ.get("OPENAI_API_KEY", "")
        api_mode = _resolve_api_mode()
        api_model = _resolve_api_model()
        base_url = _resolve_base_url()
        size = _SIZES.get(aspect, _SIZES["square"])
        responses_fallback_error: Optional[Exception] = None

        try:
            reference_image_paths = _normalize_reference_image_paths(
                kwargs.get("reference_image_paths")
                or kwargs.get("reference_image_path")
                or kwargs.get("image_paths")
            )
        except ValueError as exc:
            return error_response(
                error=str(exc),
                error_type="invalid_reference_image",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if api_mode == API_MODE_RESPONSES:
            responses_model = _resolve_responses_model()
            responses_error: Optional[Exception] = None
            try:
                reference_image_data_urls = [
                    _reference_image_to_data_url(path) for path in reference_image_paths
                ]
                b64 = _collect_responses_image_b64(
                    api_key=api_key,
                    base_url=base_url,
                    prompt=prompt,
                    responses_model=responses_model,
                    api_model=api_model,
                    size=size,
                    quality=meta["quality"],
                    reference_image_data_urls=reference_image_data_urls,
                )
            except Exception as exc:
                logger.debug("OpenAI Responses image generation failed", exc_info=True)
                if not reference_image_paths:
                    return error_response(
                        error=f"OpenAI Responses image generation failed: {exc}",
                        error_type="api_error",
                        provider="openai",
                        model=tier_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                responses_error = exc
                b64 = None

            if not b64:
                if not reference_image_paths:
                    return error_response(
                        error="OpenAI Responses returned no image_generation_call result",
                        error_type="empty_response",
                        provider="openai",
                        model=tier_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                if responses_error is None:
                    responses_error = RuntimeError(
                        "OpenAI Responses returned no image_generation_call result"
                    )
                responses_fallback_error = responses_error
                logger.info(
                    "OpenAI Responses reference-image path failed; falling back to images.edit: %s",
                    responses_error,
                )
            else:
                try:
                    saved_path = save_b64_image(b64, prefix=f"openai_{tier_id}")
                except Exception as exc:
                    return error_response(
                        error=f"Could not save image to cache: {exc}",
                        error_type="io_error",
                        provider="openai",
                        model=tier_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )

                return success_response(
                    image=str(saved_path),
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    provider="openai",
                    extra={
                        "api_mode": api_mode,
                        "api_model": api_model,
                        "responses_model": responses_model,
                        "size": size,
                        "quality": meta["quality"],
                        "reference_image_count": len(reference_image_paths),
                    },
                )

        # gpt-image-2 returns b64_json unconditionally and REJECTS
        # ``response_format`` as an unknown parameter. Don't send it.
        payload: Dict[str, Any] = {
            "model": api_model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "quality": meta["quality"],
        }

        try:
            client_kwargs: Dict[str, Any] = {}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)
            if reference_image_paths:
                response = _call_images_edit(client, payload, reference_image_paths)
            else:
                response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("OpenAI image generation/edit failed", exc_info=True)
            action = "edit" if reference_image_paths else "generation"
            error = f"OpenAI image {action} failed: {exc}"
            if responses_fallback_error is not None:
                error = (
                    "OpenAI Responses reference-image path failed "
                    f"({responses_fallback_error}); images.edit fallback failed: {exc}"
                )
            return error_response(
                error=error,
                error_type="api_error",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {
            "api_mode": api_mode,
            "api_model": api_model,
            "size": size,
            "quality": meta["quality"],
            "reference_image_count": len(reference_image_paths),
        }
        if reference_image_paths:
            extra["api_action"] = "edit"

        return _save_openai_image_response(
            response,
            tier_id=tier_id,
            prompt=prompt,
            aspect=aspect,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
