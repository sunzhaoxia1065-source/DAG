# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.strategy.fixed_forecast import FixedForecast
from ts_benchmark.evaluation.strategy.rolling_forecast import RollingForecast
from ts_benchmark.evaluation.strategy.business_day_ahead import BusinessDayAheadForecast

STRATEGY = {
    "fixed_forecast": FixedForecast,
    "rolling_forecast": RollingForecast,
    "business_day_ahead": BusinessDayAheadForecast,
}
