"""
더 많은 전략 발굴 - SMA31, SMA39 기반 변형 및 하이브리드
"""

import pandas as pd
import numpy as np

class MoreStrategies:
    def __init__(self, data, slippage=0.002):
        self.data = data
        self.slippage = slippage
        self.results = []

    def backtest(self, signal, name):
        """백테스트"""
        df = self.data.copy()
        df['signal'] = signal
        df['pos_change'] = df['signal'].diff()
        df['daily_ret'] = df['Close'].pct_change()
        df['strat_ret'] = df['signal'].shift(1) * df['daily_ret']

        slip_cost = pd.Series(0.0, index=df.index)
        slip_cost[df['pos_change'] == 1] = -self.slippage
        slip_cost[df['pos_change'] == -1] = -self.slippage
        df['strat_ret'] = df['strat_ret'] + slip_cost
        df['strat_ret'] = df['strat_ret'].fillna(0)

        cum = (1 + df['strat_ret']).cumprod()
        total_ret = (cum.iloc[-1] - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (cum.iloc[-1] ** (1/years) - 1) * 100
        cummax = cum.cummax()
        dd = (cum - cummax) / cummax
        mdd = dd.min() * 100
        sharpe = (df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(365))

        return {
            'Strategy': name,
            'Sharpe Ratio': sharpe,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Total Return (%)': total_ret
        }

    def test_all(self):
        """모든 변형 테스트"""
        df = self.data.copy()

        # 전략 1: (SMA31 + SMA39) / 2
        df['SMA31'] = df['Close'].rolling(31).mean()
        df['SMA39'] = df['Close'].rolling(39).mean()
        df['Avg'] = (df['SMA31'] + df['SMA39']) / 2
        signal = (df['Close'] > df['Avg']).astype(int)
        self.results.append(self.backtest(signal, 'Close>(SMA31+SMA39)/2'))

        # 전략 2: SMA31과 SMA39 둘 다 위
        signal = ((df['Close'] > df['SMA31']) & (df['Close'] > df['SMA39'])).astype(int)
        self.results.append(self.backtest(signal, 'Close>SMA31 AND SMA39'))

        # 전략 3-7: SMA28, 29, 32, 33, 34
        for period in [28, 29, 32, 33, 34]:
            df[f'SMA{period}'] = df['Close'].rolling(period).mean()
            signal = (df['Close'] > df[f'SMA{period}']).astype(int)
            self.results.append(self.backtest(signal, f'Close>SMA{period}'))

        # 전략 8: (SMA28 + SMA30 + SMA32) / 3
        df['SMA28'] = df['Close'].rolling(28).mean()
        df['SMA30'] = df['Close'].rolling(30).mean()
        df['SMA32'] = df['Close'].rolling(32).mean()
        df['Avg3'] = (df['SMA28'] + df['SMA30'] + df['SMA32']) / 3
        signal = (df['Close'] > df['Avg3']).astype(int)
        self.results.append(self.backtest(signal, 'Close>(SMA28+30+32)/3'))

        # 전략 9: (SMA29 + SMA30 + SMA31) / 3
        df['SMA29'] = df['Close'].rolling(29).mean()
        df['SMA31'] = df['Close'].rolling(31).mean()
        df['Avg3_2'] = (df['SMA29'] + df['SMA30'] + df['SMA31']) / 3
        signal = (df['Close'] > df['Avg3_2']).astype(int)
        self.results.append(self.backtest(signal, 'Close>(SMA29+30+31)/3'))

        # 전략 10: Close > WMA31
        weights = np.arange(1, 32)
        df['WMA31'] = df['Close'].rolling(31).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
        signal = (df['Close'] > df['WMA31']).astype(int)
        self.results.append(self.backtest(signal, 'Close>WMA31'))

        # 전략 11: Close > WMA39
        weights = np.arange(1, 40)
        df['WMA39'] = df['Close'].rolling(39).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
        signal = (df['Close'] > df['WMA39']).astype(int)
        self.results.append(self.backtest(signal, 'Close>WMA39'))

        # 전략 12: Close > (WMA30 + SMA30) / 2
        weights30 = np.arange(1, 31)
        df['WMA30'] = df['Close'].rolling(30).apply(lambda x: np.dot(x, weights30)/weights30.sum(), raw=True)
        df['Avg_ws'] = (df['WMA30'] + df['SMA30']) / 2
        signal = (df['Close'] > df['Avg_ws']).astype(int)
        self.results.append(self.backtest(signal, 'Close>(WMA30+SMA30)/2'))

        # 전략 13: Close > EMA31
        df['EMA31'] = df['Close'].ewm(span=31, adjust=False).mean()
        signal = (df['Close'] > df['EMA31']).astype(int)
        self.results.append(self.backtest(signal, 'Close>EMA31'))

        # 전략 14: Close > EMA39
        df['EMA39'] = df['Close'].ewm(span=39, adjust=False).mean()
        signal = (df['Close'] > df['EMA39']).astype(int)
        self.results.append(self.backtest(signal, 'Close>EMA39'))

        # 전략 15: Close > (SMA30 + WMA30) / 2 + 상승기울기
        df['Combo'] = (df['SMA30'] + df['WMA30']) / 2
        df['Slope'] = df['Combo'].diff(5)
        signal = ((df['Close'] > df['Combo']) & (df['Slope'] > 0)).astype(int)
        self.results.append(self.backtest(signal, 'Close>(SMA30+WMA30)/2+Slope'))

        # 전략 16: SMA31 상승 필터
        df['SMA31'] = df['Close'].rolling(31).mean()
        df['SMA31_slope'] = df['SMA31'].diff(5)
        signal = ((df['Close'] > df['SMA31']) & (df['SMA31_slope'] > 0)).astype(int)
        self.results.append(self.backtest(signal, 'Close>SMA31+Slope'))

        # 전략 17: SMA39 상승 필터
        df['SMA39'] = df['Close'].rolling(39).mean()
        df['SMA39_slope'] = df['SMA39'].diff(5)
        signal = ((df['Close'] > df['SMA39']) & (df['SMA39_slope'] > 0)).astype(int)
        self.results.append(self.backtest(signal, 'Close>SMA39+Slope'))

        # 전략 18-22: SMA26, 27, 36, 37, 38
        for period in [26, 27, 36, 37, 38]:
            df[f'SMA{period}'] = df['Close'].rolling(period).mean()
            signal = (df['Close'] > df[f'SMA{period}']).astype(int)
            self.results.append(self.backtest(signal, f'Close>SMA{period}'))

        # 전략 23: 가중평균 (SMA30*0.4 + SMA31*0.3 + SMA39*0.3)
        df['SMA30'] = df['Close'].rolling(30).mean()
        df['SMA31'] = df['Close'].rolling(31).mean()
        df['SMA39'] = df['Close'].rolling(39).mean()
        df['WeightedMA'] = df['SMA30']*0.4 + df['SMA31']*0.3 + df['SMA39']*0.3
        signal = (df['Close'] > df['WeightedMA']).astype(int)
        self.results.append(self.backtest(signal, 'Close>WeightedMA(30/31/39)'))

        return pd.DataFrame(self.results).sort_values('Sharpe Ratio', ascending=False)


# 메인 실행
print("Loading data...")
df = pd.read_parquet('chart_day/BTC_KRW.parquet')
df.columns = [col.capitalize() for col in df.columns]
df = df[(df.index >= '2018-01-01')]

print(f"Loaded {len(df)} data points\n")

# 벤치마크
df_bench = df.copy()
df_bench['SMA30'] = df_bench['Close'].rolling(30).mean()
signal_bench = (df_bench['Close'] > df_bench['SMA30']).astype(int)

tester = MoreStrategies(df)
benchmark_result = tester.backtest(signal_bench, 'BENCHMARK_Close>SMA30')
benchmark_sharpe = benchmark_result['Sharpe Ratio']

print("="*80)
print(f"Benchmark (Close>SMA30): Sharpe {benchmark_sharpe:.4f}")
print("="*80)

# 모든 변형 테스트
results_df = tester.test_all()

# 벤치마크보다 나은 전략
better = results_df[results_df['Sharpe Ratio'] > benchmark_sharpe]

print(f"\nFound {len(better)} strategies beating benchmark!\n")
print("="*80)
print("Top strategies:")
print("="*80)

for idx, row in results_df.head(10).iterrows():
    vs_bench = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
    marker = "🎯" if row['Sharpe Ratio'] > benchmark_sharpe else "  "
    print(f"{marker} {row['Strategy']:35s} Sharpe: {row['Sharpe Ratio']:.4f} ({vs_bench:+.2f}%)")

results_df.to_csv('bitcoin_additional_strategies.csv', index=False)
print(f"\n✓ Saved to bitcoin_additional_strategies.csv")
