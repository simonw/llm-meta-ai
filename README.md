# llm-meta-ai

[![PyPI](https://img.shields.io/pypi/v/llm-meta-ai.svg)](https://pypi.org/project/llm-meta-ai/)
[![Tests](https://github.com/simonw/llm-meta-ai/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-meta-ai/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-meta-ai?include_prereleases&label=changelog)](https://github.com/simonw/llm-meta-ai/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-meta-ai/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for models hosted by the [Meta AI API](https://developer.meta.com/ai/)

## Installation

Install this plugin in the same environment as LLM:
```bash
llm install llm-meta-ai
```

## Usage

First, obtain an API key for the Meta AI API and set it as a key called `meta-ai`:

```bash
llm keys set meta-ai
# Paste key here
```

You can also set the key using the `META_AI_TOKEN` environment variable.

Run `llm models` to get a full list of available models. Models are prefixed with `meta-ai/`, for example:

```bash
llm -m meta-ai/muse-spark "What is the capital of France?"
```

You can also list just the Meta AI models like this:

```bash
llm meta-ai models
```
Add `--json` for the full JSON model definitions.

The list of models is fetched from the API and cached for an hour. Run this command to refresh it:

```bash
llm meta-ai refresh
```

### Tools

Meta AI models support [tools](https://llm.datasette.io/en/stable/tools.html):

```bash
llm -m meta-ai/muse-spark -T llm_time 'What time is it?' --td
```

### Schemas

They support [schemas](https://llm.datasette.io/en/stable/schemas.html) as well:

```bash
llm -m meta-ai/muse-spark 'Invent a dog' --schema 'name, age int, breed'
```

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:
```bash
cd llm-meta-ai
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
python -m pip install -e . --group dev
```
To run the tests:
```bash
python -m pytest
```
