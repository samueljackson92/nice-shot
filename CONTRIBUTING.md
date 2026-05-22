# Contributing

Contributions are welcome. Here's how to get started.

## Setup

```sh
git clone <repo>
cd nice_shot
uv sync --extra shap
```

## Running the app locally

```sh
uv run nice-shot --shot-data path/to/shot_stats.parquet
```

Debug mode is on by default — Dash will hot-reload on Python file changes.

## Serving the docs

```sh
uv run --dev zensical serve
```

## Code style

- Format with `ruff format` before committing.
- Lint with `ruff check`.
- Keep the config schema (`nice_shot/config_schema.py`) in sync with `nice_shot/config.yaml` when adding new options.

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes with a clear commit message.
3. Open a pull request — describe what changed and why.

## Reporting issues

Open a GitHub issue with:
- What you were doing
- What you expected to happen
- What actually happened (including any error output)
- Your Python version and OS
