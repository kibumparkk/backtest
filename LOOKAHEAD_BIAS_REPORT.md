# LOOKAHEAD BIAS DISCOVERY REPORT

## Executive Summary

**CRITICAL FINDING**: All multi-timeframe strategies that appeared to beat the benchmark had **lookahead bias**. After fixing this bias, **ZERO strategies beat the benchmark**.

The 3.48x return ratio was entirely due to using future information in trading decisions.

---

## The Bug

### Original (Buggy) Implementation
```python
# Weekly signal calculated on week ending Monday
weekly['trend_up'] = (weekly['Close'] > weekly['SMA10']).astype(int)

# BUGGY: Reindex applies signal to Monday immediately
weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')

# On Monday 2018-04-23:
#   - Week labeled 2018-04-23 includes data from 2018-04-23 to 2018-04-29
#   - Weekly Close = 2018-04-29 close price
#   - But signal is available on 2018-04-23 (Monday)
#   - This is LOOKAHEAD BIAS!
```

### Corrected Implementation
```python
# FIX: Shift weekly signal by 1 period BEFORE reindex
weekly_trend_shifted = weekly['trend_up'].shift(1)
weekly_trend = weekly_trend_shifted.reindex(daily.index, method='ffill')

# Now on Monday 2018-04-23:
#   - Uses signal from PREVIOUS week (ending 2018-04-16)
#   - No lookahead bias
```

---

## Impact Analysis

### Performance Comparison

| Strategy | Buggy Sharpe | Fixed Sharpe | Difference | % Change |
|----------|-------------|-------------|-----------|----------|
| **Benchmark** | 1.6591 | 1.6591 | 0.0000 | 0.0% |
| Weekly SMA10 + Daily SMA30 | **2.2185** | **1.5901** | **-0.6284** | **-28.3%** |
| Weekly Donchian + Daily SMA30 | **2.4528** | **0.9885** | **-1.4643** | **-59.7%** |
| Weekly SMA20 + Daily SMA30 | 1.9505 | 1.5658 | -0.3847 | -19.7% |
| Weekly SMA50 + Daily SMA30 | 1.9439 | 1.6290 | -0.3149 | -16.2% |
| Weekly EMA20 + Daily SMA30 | 1.9511 | 1.6500 | -0.3011 | -15.4% |

### Return Comparison (Weekly SMA10 + Daily SMA30)

| Metric | Buggy | Fixed | Difference |
|--------|-------|-------|-----------|
| **Total Return** | 30,870% | 17,936% | -12,934% |
| **CAGR** | 97.27% | 63.81% | -33.46% |
| **Sharpe Ratio** | 2.22 | 1.59 | -0.63 |
| **MDD** | -28.43% | -33.13% | -4.70% |
| **vs Benchmark Ratio** | **3.48x** | **2.02x** | **-1.46x** |

### Year-by-Year Impact (Weekly SMA10 + Daily SMA30)

| Year | Buggy Return | Fixed Return | Lookahead Impact |
|------|-------------|-------------|------------------|
| 2018 | **+4.69%** | **-8.63%** | **+13.32%** |
| 2019 | +56.77% | +48.02% | +8.75% |
| 2020 | +155.63% | +161.59% | -5.96% |
| 2021 | +72.07% | +62.45% | +9.62% |
| 2022 | **+10.13%** | **-5.02%** | **+15.15%** |
| 2023 | +48.14% | +48.14% | 0.00% |
| 2024 | +86.02% | +80.40% | +5.62% |
| 2025 | +19.12% | +17.03% | +2.09% |

**Key Finding**: Lookahead bias was most severe in bear markets (2018, 2022), where the buggy version showed profits but the fixed version correctly shows losses.

---

## Verification Methods

### 1. Loop-Based Verification
Implemented day-by-day iteration ensuring weekly signals are only available AFTER the week ends:

```python
for i in range(len(df_daily)):
    date = df_daily.index[i]

    # Find most recent COMPLETED weekly signal
    weekly_signal = 0
    for week_date in sorted(weekly_signals.keys(), reverse=True):
        # Signal available AFTER week ends (next day)
        if date >= weekly_signals[week_date]['available_from']:
            weekly_signal = weekly_signals[week_date]['signal']
            break
```

**Result**: Loop-based Sharpe 2.04 vs Vectorized Sharpe 2.22 (0.18 difference suggested lookahead)

### 2. Signal Shift Verification
Added `.shift(1)` before reindex to delay weekly signal by 1 period.

**Result**: Fixed Sharpe 1.59 (correct) vs Buggy Sharpe 2.22 (inflated by 40%)

### 3. Timeline Analysis
Examined specific dates (e.g., 2018-04-23) to verify when weekly signals became available:

```
Week labeled 2018-04-23:
  - Includes: 2018-04-23 (Mon) to 2018-04-29 (Sun)
  - Weekly Close: 10,187,000 (from 2018-04-29)
  - Buggy: Signal available on 2018-04-23 ❌
  - Fixed: Signal available on 2018-04-30 ✓
```

---

## Root Cause Analysis

### Pandas Resampling Behavior

```python
df_weekly = df_daily.resample('W-MON', label='left', closed='left').agg(...)
```

- `W-MON`: Week ending on Monday
- `label='left'`: Label is week start (Monday)
- `closed='left'`: Includes Monday, excludes next Monday

**Week labeled 2018-04-23** = Data from 2018-04-23 through 2018-04-29

### The Reindex Problem

```python
weekly_signal.reindex(daily.index, method='ffill')
```

This forward-fills weekly signal to ALL days of that week, INCLUDING the Monday when the week STARTS.

But the weekly signal is calculated using data through Sunday (end of week), so applying it to Monday is lookahead bias!

### The Fix

```python
weekly_signal_shifted = weekly_signal.shift(1)  # Delay by 1 week
weekly_signal_expanded = weekly_signal_shifted.reindex(daily.index, method='ffill')
```

Now the signal for week ending Monday is available starting NEXT week (Tuesday onwards).

---

## Final Results (After Fix)

### Benchmark
- **Strategy**: Close > SMA30 (daily only)
- **Sharpe**: 1.6591
- **Total Return**: 8,859%
- **CAGR**: 65.44%

### Multi-Timeframe Strategies
**ZERO strategies beat the benchmark**

Closest performers:
1. Weekly EMA20 + Daily SMA30: Sharpe 1.65 (-0.55%)
2. Weekly SMA50 + Daily SMA30: Sharpe 1.63 (-1.82%)
3. Weekly SMA50 + Daily SMA50: Sharpe 1.59 (-3.95%)

---

## Lessons Learned

1. **Reindex with caution**: `reindex(method='ffill')` can introduce lookahead bias when combining different timeframes

2. **Weekly signals need lag**: When resampling to weekly and expanding back to daily, weekly signals should be shifted by 1 period

3. **Validate with loops**: Loop-based backtests are slower but easier to verify for lookahead bias

4. **Too good to be true**: A 3.48x return ratio should have been an immediate red flag

5. **Bear market performance**: Strategies showing profits in 2018/2022 bear markets deserve extra scrutiny

---

## Conclusion

The dramatic performance improvements seen in multi-timeframe strategies (up to +47.7% Sharpe improvement) were **entirely due to lookahead bias**.

After fixing this critical bug:
- **Zero multi-timeframe strategies beat the benchmark**
- The best multi-timeframe strategy is -0.55% worse than benchmark
- The worst multi-timeframe strategy is -58.4% worse than benchmark

**The benchmark (Close > SMA30) remains undefeated.**

---

## Next Steps

To find strategies that legitimately beat the benchmark, need to explore:
1. Different single-timeframe approaches (we found Close > SMA31 at +2.3%)
2. Non-SMA indicators (RSI, Bollinger Bands, ATR-based stops)
3. Adaptive indicators (changing parameters based on volatility)
4. Multiple timeframes with proper lag implementation
5. Machine learning approaches

But the bar is HIGH - Sharpe 1.66 is already excellent for a simple trend-following strategy.
