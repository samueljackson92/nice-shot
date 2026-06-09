#!/bin/bash
uv run python nice_shot/app.py \
    --config configs/plasma_events.yml \
    --shot-data mast_embeddings.parquet \
    --data-dir ~/projects/plasma-events/data/mast
    --port 8051
