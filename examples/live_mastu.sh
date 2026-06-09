#!/bin/bash
uv run python nice_shot/app.py \
    --config configs/config_uda.yaml \
    --shot-data mastu_metadata.parquet \
    --port 8052
