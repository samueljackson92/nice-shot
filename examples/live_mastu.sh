#!/bin/bash
uv run python nice_shot/app.py mastu_metadata.parquet \
    --config configs/config_uda.yaml \
    --port 8052
