from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class TimeWindow(BaseModel):
    min_time: float = 0.0
    max_time: float = 1.0

    @model_validator(mode="after")
    def check_order(self) -> TimeWindow:
        if self.min_time >= self.max_time:
            raise ValueError(
                f"time_window.min_time ({self.min_time}) must be less than time_window.max_time ({self.max_time})"
            )
        return self


class UDAOptions(BaseModel):
    timebase_hz: float | None = None


class AppConfig(BaseModel):
    backend: Literal["parquet", "uda", "sal"] = "parquet"
    signals: list[str] = ["ip", "ne", "dalpha", "loopv", "plasma_energy"]
    time_window: TimeWindow = TimeWindow()
    uda: UDAOptions = UDAOptions()
    projection_method: Literal["umap", "pca"] = "umap"
    umap_features: list[str] | None = None
    reference_shot_col: str | None = None
