# 🚨 CRITICAL FINDING: Lookahead Bias in "Loop-based" Implementation

## 📊 Cross-Validation Results

### Comparison: Previous Loop-based vs Fully Loop-based

| Strategy | Previous Sharpe | Fully Loop Sharpe | Difference | Previous Return | Fully Loop Return | Difference |
|----------|----------------|-------------------|------------|-----------------|-------------------|------------|
| **Benchmark** | 1.6591 | 1.6591 | **0%** | 8,859% | 8,859% | **0%** |
| Weekly Donchian | 2.3170 | **0.9891** | **-57%** | 5,149% | **417%** | **-92%** |
| Weekly EMA20 | 2.0780 | **1.6515** | **-21%** | 15,961% | **4,927%** | **-69%** |
| Weekly SMA10 | 2.0425 | **1.5918** | **-22%** | 18,500% | **4,719%** | **-74%** |
| Weekly SMA20 | 1.9095 | **1.5673** | **-18%** | 10,487% | **3,966%** | **-62%** |
| Weekly SMA50 | 1.7954 | **1.6305** | **-9%** | 7,437% | **4,507%** | **-39%** |

**Benchmark remains identical = validation that benchmark is correct.**

**All MTF strategies show HUGE drops = they had lookahead bias!**

---

## 🐛 What Was the Bug?

### Previous "Loop-based" Implementation (STILL HAD BIAS)

```python
# ❌ STILL BUGGY - even though it used loops!

# 1. Pre-calculate weekly data on FULL dataset
weekly = daily_data.resample('W-MON', ...).agg(...)  # Uses ALL data!
weekly['SMA10'] = weekly['Close'].rolling(10).mean()  # Calculated on full weekly dataset

# 2. Store signals with availability
for i in range(len(weekly)):
    week_date = weekly.index[i]
    signal = weekly.iloc[i]['signal']  # ⚠️ This signal was calculated using future data!
    available_from = week_date + pd.Timedelta(days=1)
    weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

# 3. Daily loop checks availability
for date in daily_dates:
    for week_date in weekly_signals.keys():
        if date >= weekly_signals[week_date]['available_from']:
            use_signal = weekly_signals[week_date]['signal']  # ⚠️ But signal itself had lookahead!
```

**The problem:**
- We checked WHEN to use the signal (good!)
- But the SIGNAL ITSELF was calculated using future data (bad!)

**Example:**
- On 2020-01-15, we correctly wait until the signal is "available"
- BUT: Weekly SMA10 used in that signal was calculated from weekly bars through 2025-10-31!
- The SMA10 value on week 2020-01-13 includes information about weeks in 2020, 2021, 2022, etc.

---

### Fully Loop-based Implementation (CORRECT)

```python
# ✅ CORRECT - recalculates weekly signals each day

for i in range(len(daily)):
    date = daily.index[i]

    # ⭐ Key: Use only data UP TO today
    data_until_today = daily.iloc[:i+1]

    # Recalculate weekly data from data_until_today
    weekly = data_until_today.resample('W-MON', ...).agg(...)

    # Calculate indicator on this dynamically created weekly data
    weekly['SMA10'] = weekly['Close'].rolling(10).mean()

    # Find completed weeks (exclude current incomplete week)
    current_week_start = date - pd.Timedelta(days=date.dayofweek)
    completed_weeks = weekly[weekly.index < current_week_start]

    # Use most recent completed week
    latest_week = completed_weeks.iloc[-1]
    weekly_signal = 1 if latest_week['Close'] > completed_weeks['Close'].rolling(10).mean().iloc[-1] else 0
```

**Why this works:**
- Every day recalculates weekly data using ONLY data up to that day
- Weekly indicators (SMA, EMA, Donchian) are calculated on this limited dataset
- Mathematically impossible to use future information

---

## 🎯 Corrected Results

### Reality Check: MTF Strategies vs Benchmark

| Strategy | Sharpe | vs Benchmark | Total Return | MDD | Trades |
|----------|--------|--------------|--------------|-----|--------|
| **Benchmark (Daily SMA30)** | **1.6591** | **-** | **8,859%** | **-38.09%** | **1,593** |
| Weekly Donchian + Daily SMA30 | 0.9891 | **-40% worse** | 417% | -37.05% | 78 |
| Weekly EMA20 + Daily SMA30 | 1.6515 | -0.5% (tie) | 4,927% | -36.43% | 106 |
| Weekly SMA10 + Daily SMA30 | 1.5918 | -4% | 4,719% | -33.10% | 108 |
| Weekly SMA20 + Daily SMA30 | 1.5673 | -6% | 3,966% | -32.60% | 108 |
| Weekly SMA50 + Daily SMA30 | 1.6305 | -2% | 4,507% | -32.92% | 118 |

**Conclusion: NONE of the MTF strategies beat the benchmark!**

- Weekly EMA20 is essentially tied with benchmark (1.65 vs 1.66)
- Weekly SMA50 is close (-2%)
- All others underperform
- Weekly Donchian dramatically underperforms (-40%)

---

## 💡 Key Lessons Learned

### 1. Lookahead Bias is Subtle

Even with:
- ✅ Loop-based implementation
- ✅ Sequential processing
- ✅ `available_from` checks

We STILL had lookahead bias because:
- ❌ Indicators were pre-calculated on full dataset

### 2. The Correct Approach

**For multi-timeframe strategies, you must:**

1. **Daily loop**: Process each day sequentially
2. **Dynamic weekly calculation**: Recalculate weekly data using only data up to current date
3. **Completed weeks only**: Exclude current incomplete week
4. **Fresh indicators**: Calculate indicators on the dynamically created weekly data

**Pseudo-code:**
```python
for each day:
    data_until_today = all_data[:today]
    weekly = resample(data_until_today)
    completed_weekly = weekly[exclude_current_week]
    indicator = calculate(completed_weekly)
    use_indicator_value
```

### 3. Cross-Validation is Essential

- First implementation (vectorized): Had obvious lookahead bias
- Second implementation (loop-based with pre-calculated weekly): Had subtle lookahead bias
- Third implementation (fully loop-based): Finally correct!

**Each validation revealed a deeper level of bias.**

---

## 🔬 Why the Huge Difference?

### Weekly Donchian Example

**Previous (buggy):**
- On 2020-01-15, used Donchian(20) from pre-calculated weekly data
- That Donchian(20) value was influenced by future weekly highs through 2025
- Result: Sharpe 2.32, Return 5,149%

**Fully Loop-based (correct):**
- On 2020-01-15, recalculated weekly data using only daily data through 2020-01-15
- Donchian(20) calculated on ~2 years of weekly data (104 weeks)
- Result: Sharpe 0.99, Return 417%

**Difference: -92% return overestimation!**

---

## ✅ Final Answer to Original Question

**Original request:** Find 5 Bitcoin strategies that beat "Close > SMA30" benchmark (Sharpe 1.66)

**Answer after rigorous cross-validation:**

**NONE of the multi-timeframe strategies actually beat the benchmark.**

The apparent improvements were due to lookahead bias. When properly implemented:
- Weekly EMA20 + Daily SMA30: Sharpe 1.65 (essentially tied)
- All others: Sharpe < 1.66 (worse than benchmark)

**True top strategies:**
1. **Benchmark: Close > SMA30** - Sharpe 1.6591 (simplest is best!)
2. Weekly EMA20 + Daily SMA30 - Sharpe 1.6515 (tie, but more complex)
3. Weekly SMA50 + Daily SMA30 - Sharpe 1.6305 (-2%)

---

## 🎓 Implications for Real Trading

### What This Means

1. **Simple > Complex**: The simple single-timeframe "Close > SMA30" is as good as (or better than) complex MTF strategies

2. **Weekly filters don't help**: Adding weekly trend filters doesn't improve risk-adjusted returns
   - They reduce trades (78-118 vs 1,593)
   - But also reduce returns proportionally
   - Net effect: Same or worse Sharpe ratio

3. **Lookahead bias is pervasive**: Even experienced developers can introduce subtle lookahead bias
   - Pre-calculating indicators is dangerous
   - Always recalculate using only available data

### Recommended Strategy

**Stick with the benchmark: Close > SMA30**
- Sharpe: 1.6591
- CAGR: 77.37%
- Total Return: 8,859% over ~8 years
- Trades: 1,593 (frequent rebalancing)
- MDD: -38.09%

Simple, transparent, and as good as it gets for trend-following on Bitcoin.

---

## 📝 Technical Implementation Notes

### How to Avoid Lookahead Bias in MTF Strategies

```python
def backtest_mtf_correct(daily_data):
    capital = 1.0
    position = 0

    for i in range(len(daily_data)):
        date = daily_data.index[i]

        # 1. Get data up to today only
        data_until_today = daily_data.iloc[:i+1]

        # 2. Recalculate weekly from daily
        weekly = data_until_today.resample('W-MON', label='left', closed='left').agg({
            'Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Volume': 'sum'
        }).dropna()

        # 3. Exclude current incomplete week
        current_week_start = date - pd.Timedelta(days=date.dayofweek)
        completed_weeks = weekly[weekly.index < current_week_start]

        if len(completed_weeks) < 20:  # Need minimum data
            continue

        # 4. Calculate indicator on completed weeks
        weekly_sma = completed_weeks['Close'].rolling(20).mean()
        latest_week = completed_weeks.iloc[-1]
        weekly_signal = 1 if latest_week['Close'] > weekly_sma.iloc[-1] else 0

        # 5. Daily signal
        daily_sma = data_until_today['Close'].rolling(30).mean()
        daily_signal = 1 if data_until_today['Close'].iloc[-1] > daily_sma.iloc[-1] else 0

        # 6. Combine
        final_signal = 1 if (weekly_signal == 1 and daily_signal == 1) else 0

        # 7. Update capital
        # ... (position changes, returns, etc.)
```

**Key principles:**
- `data_until_today = daily_data.iloc[:i+1]` - ONLY past data
- `weekly = data_until_today.resample(...)` - Recalculate every day
- `completed_weeks = weekly[weekly.index < current_week_start]` - Exclude incomplete week
- Calculate indicators on `completed_weeks` - NO future info

---

## 🔍 Validation Proof

**Benchmark is identical across all three implementations:**
- Vectorized: Sharpe 1.6591
- Loop-based (buggy): Sharpe 1.6591
- Fully loop-based (correct): Sharpe 1.6591

This proves:
1. Our data and calculation methods are consistent
2. Single-timeframe strategies are immune to this type of bias
3. The differences in MTF strategies are due to how weekly signals are calculated

**Cross-validation succeeded**: We found the remaining lookahead bias!

---

## 🎯 Final Recommendation

**Use the simplest strategy that works:**

```python
signal = 1 if close > SMA(30) else 0
```

**Why:**
- Best Sharpe ratio: 1.6591
- Transparent and easy to understand
- No multi-timeframe complexity
- No lookahead bias risk
- Proven through rigorous validation

**"Simplicity is the ultimate sophistication." - Leonardo da Vinci**

In algorithmic trading, this couldn't be more true.
