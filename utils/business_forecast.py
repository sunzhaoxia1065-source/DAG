# -*- coding: utf-8 -*-
from typing import Optional, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for type-only imports
    torch = None

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def slice_business_evaluation_window(
    data: ArrayLike,
    evaluation_horizon: int,
    *,
    time_axis: Optional[int] = None,
) -> ArrayLike:
    """
    Take the last ``evaluation_horizon`` steps from a full day-ahead forecast.

    Business models predict ``horizon`` points (e.g. 156) at issue time, but metrics
    are computed on the trailing ``evaluation_horizon`` points (e.g. next-day 96).
    """
    if evaluation_horizon <= 0:
        raise ValueError("evaluation_horizon must be positive")

    if time_axis is None:
        time_axis = 0 if data.ndim <= 2 else 1

    full_horizon = data.shape[time_axis]
    if full_horizon < evaluation_horizon:
        raise ValueError(
            f"Forecast length {full_horizon} is shorter than "
            f"evaluation_horizon {evaluation_horizon}"
        )

    indexer = [slice(None)] * data.ndim
    indexer[time_axis] = slice(full_horizon - evaluation_horizon, full_horizon)
    return data[tuple(indexer)]
