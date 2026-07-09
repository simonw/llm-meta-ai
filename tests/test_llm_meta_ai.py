import httpx
import json
import llm
import llm_meta_ai
import pytest
from click.testing import CliRunner
from llm.cli import cli

MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {"id": "blue_hornbill_vip", "object": "model", "created": 0, "owned_by": "meta"},
        {"id": "muse-spark", "object": "model", "created": 0, "owned_by": "meta"},
    ],
}


@pytest.fixture(autouse=True)
def env_and_cache(monkeypatch, tmpdir):
    monkeypatch.setenv("LLM_USER_PATH", str(tmpdir))
    monkeypatch.setenv("META_AI_TOKEN", "x-test-key")


@pytest.fixture
def mocked_models(httpx_mock):
    httpx_mock.add_response(
        url="https://api.ai.meta.com/v1/models",
        method="GET",
        json=MODELS_RESPONSE,
        is_reusable=True,
    )
    return httpx_mock


def test_registers_models(mocked_models):
    model_ids = [model.model_id for model in llm.get_models()]
    assert "meta-ai/blue_hornbill_vip" in model_ids
    assert "meta-ai/muse-spark" in model_ids
    model = llm.get_model("meta-ai/muse-spark")
    assert model.model_name == "muse-spark"
    assert model.api_base == "https://api.ai.meta.com/v1"
    assert model.supports_tools
    assert model.supports_schema
    assert model.vision
    assert model.attachment_types == {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/x-icon",
        "application/pdf",
    }
    assert str(model) == "Meta AI: meta-ai/muse-spark"


def test_max_tokens_becomes_max_completion_tokens(mocked_models):
    model = llm.get_model("meta-ai/muse-spark")
    prompt = llm.models.Prompt("hello", model=model, options=model.Options(max_tokens=100))
    kwargs = model.build_kwargs(prompt, stream=False)
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == 100


def test_reasoning_effort_option(mocked_models):
    model = llm.get_model("meta-ai/muse-spark")
    prompt = llm.models.Prompt(
        "hello", model=model, options=model.Options(reasoning_effort="low")
    )
    kwargs = model.build_kwargs(prompt, stream=False)
    assert kwargs["reasoning_effort"] == "low"


def test_registers_async_models(mocked_models):
    model = llm.get_async_model("meta-ai/muse-spark")
    assert model.model_id == "meta-ai/muse-spark"


def test_models_command(mocked_models):
    runner = CliRunner()
    result = runner.invoke(cli, ["meta-ai", "models"])
    assert result.exit_code == 0
    assert result.output == ("meta-ai/blue_hornbill_vip\nmeta-ai/muse-spark\n")


def test_models_command_json(mocked_models):
    runner = CliRunner()
    result = runner.invoke(cli, ["meta-ai", "models", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == MODELS_RESPONSE["data"]


def test_refresh_command(mocked_models):
    runner = CliRunner()
    result = runner.invoke(cli, ["meta-ai", "refresh"])
    assert result.exit_code == 0
    assert "No changes" in result.output


def test_models_are_cached(mocked_models, monkeypatch):
    llm_meta_ai.get_meta_ai_models("x-test-key")
    llm_meta_ai.get_meta_ai_models("x-test-key")
    assert len(mocked_models.get_requests()) == 1


def test_no_key_no_models(monkeypatch, httpx_mock):
    monkeypatch.delenv("META_AI_TOKEN")
    model_ids = [model.model_id for model in llm.get_models()]
    assert not any(model_id.startswith("meta-ai/") for model_id in model_ids)
    assert len(httpx_mock.get_requests()) == 0


def test_download_error_skips_registration(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("connection failed"))
    model_ids = [model.model_id for model in llm.get_models()]
    assert not any(model_id.startswith("meta-ai/") for model_id in model_ids)
