import click
import httpx
import json
import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
from pathlib import Path
import time

API_BASE = "https://api.ai.meta.com/v1"


def get_meta_ai_models(key, skip_cache=False):
    return fetch_cached_json(
        url=API_BASE + "/models",
        path=llm.user_dir() / "meta_ai_models.json",
        cache_timeout=0 if skip_cache else 3600,
        headers={"Authorization": "Bearer {}".format(key)},
    )["data"]


def get_model_ids(key, skip_cache=False):
    return [model["id"] for model in get_meta_ai_models(key, skip_cache=skip_cache)]


class _mixin:
    needs_key = "meta-ai"
    key_env_var = "META_AI_TOKEN"

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        # The Meta AI API uses max_completion_tokens, not max_tokens -
        # requests without it are counted against the output token rate
        # limit as if they used the model maximum
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        return kwargs

    def __str__(self):
        return "Meta AI: {}".format(self.model_id)


class MetaAIChat(_mixin, Chat):
    pass


class MetaAIAsyncChat(_mixin, AsyncChat):
    pass


@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "meta-ai", "META_AI_TOKEN")
    if not key:
        return
    try:
        models = get_meta_ai_models(key)
    except DownloadError:
        return
    for model_definition in models:
        model_name = model_definition["id"]
        kwargs = dict(
            model_id="meta-ai/{}".format(model_name),
            model_name=model_name,
            api_base=API_BASE,
            supports_tools=True,
            supports_schema=True,
        )
        register(
            MetaAIChat(**kwargs),
            MetaAIAsyncChat(**kwargs),
        )


class DownloadError(Exception):
    pass


def fetch_cached_json(url, path, cache_timeout, headers=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        mod_time = path.stat().st_mtime
        if time.time() - mod_time < cache_timeout:
            with open(path, "r") as file:
                return json.load(file)

    try:
        response = httpx.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx.HTTPError:
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        raise DownloadError(
            "Failed to download data and no cache is available at {}".format(path)
        )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def meta_ai():
        "Commands relating to the llm-meta-ai plugin"

    @meta_ai.command()
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    @click.option("--key", help="Meta AI API key")
    def models(json_, key):
        "List of Meta AI models"
        key = llm.get_key(key, "meta-ai", "META_AI_TOKEN")
        all_models = get_meta_ai_models(key)
        if json_:
            click.echo(json.dumps(all_models, indent=2))
        else:
            for model in all_models:
                click.echo("meta-ai/{}".format(model["id"]))

    @meta_ai.command()
    @click.option("--key", help="Meta AI API key")
    def refresh(key):
        "Refresh the cached list of available Meta AI models"
        key = llm.get_key(key, "meta-ai", "META_AI_TOKEN")
        before = set(get_model_ids(key))
        after = set(get_model_ids(key, skip_cache=True))
        added = after - before
        removed = before - after
        if added:
            click.echo(
                "Added models: {}".format(
                    ", ".join("meta-ai/" + m for m in added)
                ),
                err=True,
            )
        if removed:
            click.echo(
                "Removed models: {}".format(
                    ", ".join("meta-ai/" + m for m in removed)
                ),
                err=True,
            )
        if not added and not removed:
            click.echo("No changes", err=True)
