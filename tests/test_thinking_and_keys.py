import json
import os
import subprocess
import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

import history
import local_storage
from llm_filter import (
    MODEL_THINKING_CONFIG,
    get_model_thinking_config,
    estimate_cost,
    _call_gemini,
)


def test_model_thinking_config():
    # Gemini 3.8 Flash must have thinking config and mandatory flag True
    cfg_38 = get_model_thinking_config("gemini-3.8-flash")
    assert cfg_38 is not None
    assert cfg_38["mandatory"] is True
    assert "LOW" in cfg_38["levels"]
    assert "MEDIUM" in cfg_38["levels"]
    assert "HIGH" in cfg_38["levels"]
    assert cfg_38["default"] == "MEDIUM"

    # Gemini 3.1 Flash-Lite must have thinking config and MINIMAL level
    cfg_31 = get_model_thinking_config("models/gemini-3.1-flash-lite-preview")
    assert cfg_31 is not None
    assert cfg_31["mandatory"] is False
    assert "MINIMAL" in cfg_31["levels"]

    # Models without thinking support return None
    assert get_model_thinking_config("claude-haiku-4-5") is None


def test_estimate_cost_thinking_levels():
    messages = ["test batch message"]
    n_items = 10

    # Test Gemini 3.8 Flash with LOW vs HIGH thinking level
    cost_low = estimate_cost("gemini-3.8-flash", messages, n_items, input_tokens=100, thinking_level="LOW")
    cost_high = estimate_cost("gemini-3.8-flash", messages, n_items, input_tokens=100, thinking_level="HIGH")

    # Output tokens for HIGH must be greater than LOW
    assert cost_high["output_tokens"][0] > cost_low["output_tokens"][0]
    assert cost_high["output_tokens"][1] > cost_low["output_tokens"][1]
    assert cost_high["cost"][0] > cost_low["cost"][0]
    assert cost_high["cost"][1] > cost_low["cost"][1]


def test_call_gemini_thinking_config():
    from google.genai import types

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        _call_gemini(
            model="gemini-3.8-flash",
            message="test message",
            api_key="fake-key",
            thinking_level="HIGH",
        )

        assert mock_client.models.generate_content.called
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.8-flash"
        config = call_kwargs["config"]
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == types.ThinkingLevel.HIGH


def test_api_keys_persistent_disk_storage(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "data" / "api_keys.json"
        monkeypatch.setenv("API_KEYS_FILE", str(test_file))

        # Initially empty
        data = local_storage.load_keys_from_disk()
        assert data["keys"] == []

        # Add an API key
        k_id = local_storage.add_api_key(
            name="Test Gemini Key",
            key="AIzaSyFakeKey1234567890",
            provider="google",
            default_for_models=["gemini-3.8-flash"],
        )
        assert k_id.startswith("key_")
        assert test_file.exists()

        # Check permissions (0o600 - owner read/write only)
        mode = test_file.stat().st_mode & 0o777
        assert mode == 0o600

        # Read back directly from disk
        on_disk = json.loads(test_file.read_text(encoding="utf-8"))
        assert len(on_disk["keys"]) == 1
        assert on_disk["keys"][0]["name"] == "Test Gemini Key"
        assert on_disk["keys"][0]["key"] == "AIzaSyFakeKey1234567890"
        assert on_disk["default_keys"]["gemini-3.8-flash"] == k_id

        # Update API key
        local_storage.update_api_key(k_id, name="Updated Gemini Key")
        updated_disk = json.loads(test_file.read_text(encoding="utf-8"))
        assert updated_disk["keys"][0]["name"] == "Updated Gemini Key"

        # Delete API key
        local_storage.delete_api_key(k_id)
        deleted_disk = json.loads(test_file.read_text(encoding="utf-8"))
        assert len(deleted_disk["keys"]) == 0
        assert "gemini-3.8-flash" not in deleted_disk.get("default_keys", {})


def test_api_keys_file_is_gitignored():
    res = subprocess.run(
        ["git", "check-ignore", "data/api_keys.json", ".api_keys.json", "api_keys.json"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert res.returncode == 0
    ignored = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    assert "data/api_keys.json" in ignored
    assert ".api_keys.json" in ignored
    assert "api_keys.json" in ignored


def test_history_save_and_load_thinking_level(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_runs_dir = Path(tmpdir) / "runs"
        monkeypatch.setattr(history, "RUNS_DIR", test_runs_dir)

        run_id = "test-thinking-run"
        history.save_run(
            run_id,
            source="YouTube",
            criteria="test criteria",
            model="gemini-3.8-flash",
            backend="api",
            fetched={"link": []},
            results=[],
            errors=[],
            thinking_level="HIGH",
        )

        meta, data = history.load_run(run_id)
        assert meta["thinking_level"] == "HIGH"


def test_app_thinking_ui():
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    at.query_params["admin"] = os.environ.get("ADMIN_PASSWORD", "admin")
    at.run()

    # Selecting Claude Opus 5 -> no thinking level selector
    at.sidebar.selectbox(key="model_label").select("Claude Opus 5")
    at.run()
    radios_claude = [r for r in at.sidebar.radio if "thinking_level" in getattr(r, "key", "")]
    assert len(radios_claude) == 0

    # Switching to Gemini 3.8 Flash reveals thinking level selector with default MEDIUM
    at.sidebar.selectbox(key="model_label").select("Gemini 3.8 Flash")
    at.run()
    assert len(at.error) == 0

    radios_gemini = [r for r in at.sidebar.radio if "thinking_level" in getattr(r, "key", "")]
    assert len(radios_gemini) == 1
    assert radios_gemini[0].value == "MEDIUM"
    assert "Düşük (Hızlı, daha az token)" in radios_gemini[0].options
    assert "Yüksek (Derin akıl yürütme)" in radios_gemini[0].options
