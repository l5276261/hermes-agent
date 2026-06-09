"""Tests for the bundled OpenAI image_gen plugin (gpt-image-2, three tiers)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.openai as openai_plugin


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _fake_response(*, b64=None, url=None, revised_prompt=None):
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return openai_plugin.OpenAIImageGenProvider()


def _patched_openai(fake_client: MagicMock):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return patch.dict("sys.modules", {"openai": fake_openai})


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "openai"

    def test_default_model(self, provider):
        assert provider.default_model() == "gpt-image-2-medium"

    def test_list_models_three_tiers(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert ids == ["gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high"]

    def test_catalog_entries_have_display_speed_strengths(self, provider):
        for entry in provider.list_models():
            assert entry["display"].startswith("GPT Image 2")
            assert entry["speed"]
            assert entry["strengths"]


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert openai_plugin.OpenAIImageGenProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        assert openai_plugin.OpenAIImageGenProvider().is_available() is True


# ── Model resolution ────────────────────────────────────────────────────────


class TestModelResolution:
    def test_default_is_medium(self):
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-medium"
        assert meta["quality"] == "medium"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2-high")
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"

    def test_env_var_unknown_falls_back(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "bogus-tier")
        model_id, _ = openai_plugin._resolve_model()
        assert model_id == openai_plugin.DEFAULT_MODEL

    def test_config_openai_model(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"model": "gpt-image-2-low"}}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-low"
        assert meta["quality"] == "low"

    def test_config_top_level_model(self, tmp_path):
        """``image_gen.model: gpt-image-2-high`` also works (top-level)."""
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"model": "gpt-image-2-high"}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"

    def test_api_model_defaults_to_gpt_image_2(self):
        assert openai_plugin._resolve_api_model() == "gpt-image-2"

    def test_api_model_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_API_MODEL", "image-2")
        assert openai_plugin._resolve_api_model() == "image-2"

    def test_api_model_config_override(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"api_model": "image-2"}}})
        )
        assert openai_plugin._resolve_api_model() == "image-2"

    def test_base_url_config_override(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {"image_gen": {"openai": {"base_url": "http://127.0.0.1:8317/v1/"}}}
            )
        )
        assert openai_plugin._resolve_base_url() == "http://127.0.0.1:8317/v1"

    def test_base_url_openai_env_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8317/v1/")
        assert openai_plugin._resolve_base_url() == "http://127.0.0.1:8317/v1"

    def test_api_mode_defaults_to_images(self):
        assert openai_plugin._resolve_api_mode() == "images"

    def test_api_mode_env_responses(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_API_MODE", "responses")
        assert openai_plugin._resolve_api_mode() == "responses"

    def test_api_mode_config_responses(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"api_mode": "responses"}}})
        )
        assert openai_plugin._resolve_api_mode() == "responses"

    def test_responses_model_config_override(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"responses_model": "gpt-5.5"}}})
        )
        assert openai_plugin._resolve_responses_model() == "gpt-5.5"


# ── Generate ────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = openai_plugin.OpenAIImageGenProvider().generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_b64_saves_to_cache(self, provider, tmp_path):
        png_bytes = bytes.fromhex(_PNG_HEX)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["aspect_ratio"] == "landscape"
        assert result["provider"] == "openai"
        assert result["quality"] == "medium"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        call_kwargs = fake_client.images.generate.call_args.kwargs
        # All tiers hit the single underlying API model.
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["quality"] == "medium"
        assert call_kwargs["size"] == "1536x1024"
        # gpt-image-2 rejects response_format — we must NOT send it.
        assert "response_format" not in call_kwargs
        assert result["api_model"] == "gpt-image-2"

    def test_generate_uses_configured_api_model_and_base_url(self, provider, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_API_MODEL", "image-2")
        monkeypatch.setenv("OPENAI_IMAGE_BASE_URL", "http://127.0.0.1:8317/v1/")
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client) as patched:
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["api_model"] == "image-2"
        assert fake_client.images.generate.call_args.kwargs["model"] == "image-2"
        patched["openai"].OpenAI.assert_called_once_with(
            base_url="http://127.0.0.1:8317/v1"
        )

    def test_generate_via_responses_saves_b64(self, provider, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_API_MODE", "responses")
        monkeypatch.setenv("OPENAI_IMAGE_API_MODEL", "image-2")
        monkeypatch.setenv("OPENAI_IMAGE_RESPONSES_MODEL", "gpt-5.5")
        monkeypatch.setenv("OPENAI_IMAGE_BASE_URL", "http://127.0.0.1:8317/v1/")
        captured = {}

        def _collect(**kwargs):
            captured.update(kwargs)
            return _b64_png()

        monkeypatch.setattr(openai_plugin, "_collect_responses_image_b64", _collect)

        result = provider.generate("a cat", aspect_ratio="square")

        assert result["success"] is True
        assert result["api_mode"] == "responses"
        assert result["api_model"] == "image-2"
        assert result["responses_model"] == "gpt-5.5"
        assert captured["base_url"] == "http://127.0.0.1:8317/v1"
        assert captured["api_model"] == "image-2"
        assert captured["responses_model"] == "gpt-5.5"
        assert captured["size"] == "1024x1024"
        assert captured["reference_image_data_urls"] == []

    def test_generate_via_responses_passes_reference_image_bytes(self, provider, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_IMAGE_API_MODE", "responses")
        captured = {}

        ref = tmp_path / "reference.png"
        ref.write_bytes(bytes.fromhex(_PNG_HEX))

        def _collect(**kwargs):
            captured.update(kwargs)
            return _b64_png()

        monkeypatch.setattr(openai_plugin, "_collect_responses_image_b64", _collect)

        result = provider.generate(
            "make the reference image cinematic",
            aspect_ratio="square",
            reference_image_paths=[str(ref)],
        )

        assert result["success"] is True
        assert result["reference_image_count"] == 1
        assert captured["reference_image_data_urls"][0].startswith("data:image/png;base64,")

    @pytest.mark.parametrize("tier,expected_quality", [
        ("gpt-image-2-low", "low"),
        ("gpt-image-2-medium", "medium"),
        ("gpt-image-2-high", "high"),
    ])
    def test_tier_maps_to_quality(self, provider, monkeypatch, tier, expected_quality):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", tier)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["model"] == tier
        assert result["quality"] == expected_quality
        assert fake_client.images.generate.call_args.kwargs["quality"] == expected_quality
        # Always the same underlying API model regardless of tier.
        assert fake_client.images.generate.call_args.kwargs["model"] == "gpt-image-2"

    @pytest.mark.parametrize("aspect,expected_size", [
        ("landscape", "1536x1024"),
        ("square", "1024x1024"),
        ("portrait", "1024x1536"),
    ])
    def test_aspect_ratio_mapping(self, provider, aspect, expected_size):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            provider.generate("a cat", aspect_ratio=aspect)

        assert fake_client.images.generate.call_args.kwargs["size"] == expected_size

    def test_reference_image_uses_images_edit(self, provider, tmp_path):
        ref = tmp_path / "reference.png"
        ref.write_bytes(bytes.fromhex(_PNG_HEX))
        fake_client = MagicMock()
        fake_client.images.edit.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate(
                "edit the reference image",
                aspect_ratio="portrait",
                reference_image_paths=[str(ref)],
            )

        assert result["success"] is True
        assert result["api_action"] == "edit"
        assert result["reference_image_count"] == 1
        fake_client.images.generate.assert_not_called()
        call_kwargs = fake_client.images.edit.call_args.kwargs
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["quality"] == "medium"
        assert call_kwargs["size"] == "1024x1536"
        assert call_kwargs["input_fidelity"] == "high"
        assert call_kwargs["image"].name == str(ref.resolve())

    def test_revised_prompt_passed_through(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=_b64_png(), revised_prompt="A photo of a cat",
        )

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["revised_prompt"] == "A photo of a cat"

    def test_api_error_returns_error_response(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.side_effect = RuntimeError("boom")

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]

    def test_empty_response_data(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = SimpleNamespace(data=[])

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_url_response_is_cached_locally(self, provider):
        """OpenAI URL response (if API ever returns one) is cached locally.

        Pre-fix this asserted the bare URL passed through; symmetric to the
        xAI #26942 fix.  Even though gpt-image-2 returns b64 today, every
        ``image_gen`` provider must guarantee the gateway gets a stable
        file path so ephemeral signed URLs can't expire mid-flight.
        """
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.png",
        )

        with _patched_openai(fake_client), patch(
            "plugins.image_gen.openai.save_url_image",
            return_value=Path("/tmp/openai_gpt-image-2_20260524_000000_deadbeef.png"),
        ) as mock_save_url:
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"].startswith("/")
        assert "example.com" not in result["image"]
        mock_save_url.assert_called_once()

    def test_url_response_falls_back_to_bare_url_when_download_fails(self, provider):
        """Cache failure must not turn into a tool error — symmetric with xAI."""
        import requests as req_lib

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.png",
        )

        with _patched_openai(fake_client), patch(
            "plugins.image_gen.openai.save_url_image",
            side_effect=req_lib.HTTPError("404 from CDN"),
        ):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"] == "https://example.com/img.png"

    def test_responses_payload_shape(self):
        payload = openai_plugin._build_responses_payload(
            prompt="a cat",
            responses_model="gpt-5.5",
            api_model="image-2",
            size="1024x1024",
            quality="low",
        )
        assert payload["model"] == "gpt-5.5"
        assert payload["stream"] is False
        tool = payload["tools"][0]
        assert tool["type"] == "image_generation"
        assert tool["model"] == "image-2"
        assert tool["quality"] == "low"
        assert payload["tool_choice"]["tools"] == [{"type": "image_generation"}]

    def test_responses_payload_includes_reference_images(self):
        data_url = "data:image/png;base64,QUFB"
        payload = openai_plugin._build_responses_payload(
            prompt="edit this",
            responses_model="gpt-5.5",
            api_model="image-2",
            size="1024x1024",
            quality="low",
            reference_image_data_urls=[data_url],
        )
        content = payload["input"][0]["content"]
        assert content == [
            {"type": "input_text", "text": "edit this"},
            {"type": "input_image", "image_url": data_url},
        ]

    def test_extract_image_b64_from_responses_payload(self):
        payload = {
            "output": [{
                "type": "image_generation_call",
                "result": "abc",
            }]
        }
        assert openai_plugin._extract_image_b64(payload) == "abc"
