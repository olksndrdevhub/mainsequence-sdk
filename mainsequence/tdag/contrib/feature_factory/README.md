# Feature Factory

Helper module to create features for assets, portfolios and asset pairs.

## Features from Bar Time Series

Creates features from historical Bar time series

```python
class FeatureBase:

    def __init__(self, rolling_window: int, time_serie_frequency: datetime.timedelta, *args, **kwargs):
        self.rolling_window = rolling_window
        self.time_serie_frequency = time_serie_frequency

```

### Single Asset Features

#### Volume Features

* VolumeRatio: Calcualtes ratio between numerator_window/rolling_window on target_column

#### Location

* ReturnFromMovingAverage: Return from last_observation(numerator_column)/MA(rolling_window)(target_column)

#### Volatility Fetures

* RealizedVolatilityContinuous: Simple volatility assumes there is no gap between observations
* DayCloseToOpenReturn: calculates return from trading day close to open
* VolatilityRatio: ratio of realized volatilities of two different estimators. The ratio is
  adjusted `vol_ratio=vol_ratio*(denominator.rolling_window/numerator.rolling_window)`

```python
class VolatilityRatio(FeatureBase):

    def __init__(self, numerator_vol_kwargs: dict, numerator_vol_type: str,
                 denominator_vol_kwargs: dict, denominator_vol_type: str,
                 *args, **kwargs):

```

* RogersSatchelVol

#### Technical Features

*RSI

#### Bar Features

*Shadow
*ReturnFromHighLows Calcualtes the return from a metric function from highs/lows given a buffer
```python
class ReturnFromHighLows(FeatureBase):

    def __init__(self, buffer_window: str, numerator_column: str, denominator_column: str,
                 agg_fun: str, *args, **kwargs):
        """

        :param buffer_window: window to calculate the return from, sometimes this is needed to avoid capturing
        a reversal effect.
        :param args:
        :param kwargs:
        """
        assert numerator_column in ["close", "vwap"]
        assert denominator_column in ["high", "low"]
        assert agg_fun in ["max", "min", "mean"]

```

### Cross Asset Features

*RollingResidualAlphaBeta: calculates a regression between two assets or asset vs portfolio. features are alpha, beta
and intercept. alpha corresponds to $\alpha_i=y_i-\betax_i$
*AlphaRealizedVolatility: calculates the realized volatility of the observed  $\alpha_i$

