#!/bin/bash
uv run python nice_shot/app.py \
    --config configs/offnormal.yml \
    --shot-data shot_stats.parquet \
    --shap-data shap_values.nc \
    --data-dir ~/projects/offnormal/data/mastu
