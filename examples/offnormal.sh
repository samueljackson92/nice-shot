#!/bin/bash
uv run python nice_shot/app.py shot_stats.parquet \
    --config configs/offnormal.yml \
    --shap-data shap_values.nc \
    --data-dir ~/projects/offnormal/data/mastu
    --port 8050
