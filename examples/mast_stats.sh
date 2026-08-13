#!/bin/bash
uv run python nice_shot/app.py mast_embeddings.parquet \
    --config configs/plasma_events.yml \
    --data-dir ~/projects/plasma-events/data/mast
    --port 8051
