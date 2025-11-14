"""
5번째 전략 찾기 - 더 세밀한 조합 테스트
"""

import pandas as pd
import numpy as np

def backtest(data, signal, name, slippage=0.002):
    df = data.copy()
    df['signal'] = signal
    df['pos_change'] = df['signal'].diff()
    df['daily_ret'] = df['Close'].pct_change()
    df['strat_ret'] = df['signal'].shift(1) * df['daily_ret']

    slip_cost = pd.Series(0.0, index=df.index)
    slip_cost[df['pos_change'] == 1] = -slippage
    slip_cost[df['pos_change'] == -1] = -slippage
    df['strat_ret'] = df['strat_ret'] + slip_cost
    df['strat_ret'] = df['strat_ret'].fillna(0)

    cum = (1 + df['strat_ret']).cumprod()
    sharpe = (df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(365))

    total_ret = (cum.iloc[-1] - 1) * 100
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
    cummax = cum.cummax()
    dd = (cum - cummax) / cummax
    mdd = dd.min() * 100

    return {
        'Strategy': name,
        'Sharpe Ratio': sharpe,
        'CAGR (%)': cagr,
        'MDD (%)': mdd,
        'Total Return (%)': total_ret
    }

# 데이터 로드
print("Loading data...")
df = pd.read_parquet('chart_day/BTC_KRW.parquet')
df.columns = [col.capitalize() for col in df.columns]
df = df[(df.index >= '2018-01-01')]

# 벤치마크
df['SMA30'] = df['Close'].rolling(30).mean()
benchmark_sharpe = 1.6591
print(f"Benchmark: Sharpe {benchmark_sharpe:.4f}\n")

results = []

# 전략들 테스트
print("Testing strategies...")

# 1. SMA31 OR SMA39 (둘 중 하나만 위에 있어도 매수)
df['SMA31'] = df['Close'].rolling(31).mean()
df['SMA39'] = df['Close'].rolling(39).mean()
signal = ((df['Close'] > df['SMA31']) | (df['Close'] > df['SMA39'])).astype(int)
results.append(backtest(df, signal, 'Close>SMA31 OR SMA39'))

# 2. (SMA30 * 0.5 + SMA31 * 0.5)
df['Avg30_31'] = df['SMA30'] * 0.5 + df['SMA31'] * 0.5
signal = (df['Close'] > df['Avg30_31']).astype(int)
results.append(backtest(df, signal, 'Close>(SMA30*0.5+SMA31*0.5)'))

# 3. (SMA30 * 0.6 + SMA31 * 0.4)
df['Avg30_31_6040'] = df['SMA30'] * 0.6 + df['SMA31'] * 0.4
signal = (df['Close'] > df['Avg30_31_6040']).astype(int)
results.append(backtest(df, signal, 'Close>(SMA30*0.6+SMA31*0.4)'))

# 4. (SMA30 * 0.7 + SMA31 * 0.3)
df['Avg30_31_7030'] = df['SMA30'] * 0.7 + df['SMA31'] * 0.3
signal = (df['Close'] > df['Avg30_31_7030']).astype(int)
results.append(backtest(df, signal, 'Close>(SMA30*0.7+SMA31*0.3)'))

# 5. (SMA30 * 0.3 + SMA31 * 0.7)
df['Avg30_31_3070'] = df['SMA30'] * 0.3 + df['SMA31'] * 0.7
signal = (df['Close'] > df['Avg30_31_3070']).astype(int)
results.append(backtest(df, signal, 'Close>(SMA30*0.3+SMA31*0.7)'))

# 6. (SMA38 + SMA39) / 2
df['SMA38'] = df['Close'].rolling(38).mean()
df['Avg38_39'] = (df['SMA38'] + df['SMA39']) / 2
signal = (df['Close'] > df['Avg38_39']).astype(int)
results.append(backtest(df, signal, 'Close>(SMA38+SMA39)/2'))

# 7. (SMA29 + SMA31) / 2
df['SMA29'] = df['Close'].rolling(29).mean()
df['Avg29_31'] = (df['SMA29'] + df['SMA31']) / 2
signal = (df['Close'] > df['Avg29_31']).astype(int)
results.append(backtest(df, signal, 'Close>(SMA29+SMA31)/2'))

# 8. (SMA30 + SMA31 + SMA32) / 3
df['SMA32'] = df['Close'].rolling(32).mean()
df['Avg30_31_32'] = (df['SMA30'] + df['SMA31'] + df['SMA32']) / 3
signal = (df['Close'] > df['Avg30_31_32']).astype(int)
results.append(backtest(df, signal, 'Close>(SMA30+31+32)/3'))

# 9. (SMA28 + SMA29 + SMA30) / 3
df['SMA28'] = df['Close'].rolling(28).mean()
df['Avg28_29_30'] = (df['SMA28'] + df['SMA29'] + df['SMA30']) / 3
signal = (df['Close'] > df['Avg28_29_30']).astype(int)
results.append(backtest(df, signal, 'Close>(SMA28+29+30)/3'))

# 10. Close > SMA30.5 (30과 31의 중간)
df['SMA30_point5'] = (df['SMA30'] + df['SMA31']) / 2
signal = (df['Close'] > df['SMA30_point5']).astype(int)
results.append(backtest(df, signal, 'Close>SMA30.5'))

# 11-15: WMA 변형
for period in [31, 32, 33, 37, 38, 39]:
    weights = np.arange(1, period + 1)
    df[f'WMA{period}'] = df['Close'].rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    signal = (df['Close'] > df[f'WMA{period}']).astype(int)
    results.append(backtest(df, signal, f'Close>WMA{period}'))

# 16. (WMA31 + WMA39) / 2
df['WMA_avg'] = (df['WMA31'] + df['WMA39']) / 2
signal = (df['Close'] > df['WMA_avg']).astype(int)
results.append(backtest(df, signal, 'Close>(WMA31+WMA39)/2'))

# 17-20: EMA 추가
for period in [28, 29, 32, 33]:
    df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    signal = (df['Close'] > df[f'EMA{period}']).astype(int)
    results.append(backtest(df, signal, f'Close>EMA{period}'))

# 21. (EMA30 + SMA30) / 2
df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()
df['EMA_SMA_avg'] = (df['EMA30'] + df['SMA30']) / 2
signal = (df['Close'] > df['EMA_SMA_avg']).astype(int)
results.append(backtest(df, signal, 'Close>(EMA30+SMA30)/2'))

# 22. (EMA31 + SMA31) / 2
df['EMA31'] = df['Close'].ewm(span=31, adjust=False).mean()
df['EMA31_SMA31_avg'] = (df['EMA31'] + df['SMA31']) / 2
signal = (df['Close'] > df['EMA31_SMA31_avg']).astype(int)
results.append(backtest(df, signal, 'Close>(EMA31+SMA31)/2'))

# 23-25: 가중평균 조합
weights_list = [
    (0.4, 0.3, 0.3),  # SMA30, SMA31, SMA39
    (0.3, 0.4, 0.3),
    (0.35, 0.35, 0.3),
]

for w1, w2, w3 in weights_list:
    df['Weighted'] = df['SMA30']*w1 + df['SMA31']*w2 + df['SMA39']*w3
    signal = (df['Close'] > df['Weighted']).astype(int)
    results.append(backtest(df, signal, f'Close>Weighted({w1}/{w2}/{w3})'))

# 결과 정리
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Sharpe Ratio', ascending=False)

# 벤치마크보다 나은 것들
better = results_df[results_df['Sharpe Ratio'] > benchmark_sharpe]

print("="*80)
print(f"Found {len(better)} strategies beating benchmark!\n")

print("Top 10 strategies:")
print("="*80)
for idx, row in results_df.head(15).iterrows():
    vs_bench = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
    marker = "🎯" if row['Sharpe Ratio'] > benchmark_sharpe else "  "
    print(f"{marker} {row['Strategy']:40s} Sharpe: {row['Sharpe Ratio']:.4f} ({vs_bench:+.2f}%)")

results_df.to_csv('bitcoin_fifth_strategy_search.csv', index=False)
print(f"\n✓ Saved to bitcoin_fifth_strategy_search.csv")
