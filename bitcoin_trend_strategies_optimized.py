"""
비트코인 추세추종 전략 발굴 - 최적화 버전
SMA 기간을 정밀하게 스캔하여 벤치마크를 이기는 전략 발굴
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class BitcoinStrategyOptimizer:
    """비트코인 전략 최적화 클래스"""

    def __init__(self, start_date='2018-01-01', end_date=None, slippage=0.002):
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = None
        self.results = []

    def load_data(self):
        """비트코인 데이터 로드"""
        print("Loading Bitcoin data...")
        df = pd.read_parquet('chart_day/BTC_KRW.parquet')
        df.columns = [col.capitalize() for col in df.columns]
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
        self.data = df
        print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}\n")

    def backtest_strategy(self, signal, name):
        """백테스트 실행"""
        df = self.data.copy()
        df['signal'] = signal
        df['position_change'] = df['signal'].diff()
        df['daily_return'] = df['Close'].pct_change()
        df['strategy_return'] = df['signal'].shift(1) * df['daily_return']

        # 슬리피지
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['strategy_return'] = df['strategy_return'] + slippage_cost
        df['strategy_return'] = df['strategy_return'].fillna(0)

        # 성과 지표
        cumulative = (1 + df['strategy_return']).cumprod()
        total_return = (cumulative.iloc[-1] - 1) * 100

        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100

        sharpe = (df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(365))

        total_trades = (df['strategy_return'] != 0).sum()
        winning_trades = (df['strategy_return'] > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'Strategy': name,
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades)
        }

    def test_sma_range(self, start_period=15, end_period=60):
        """SMA 기간 범위 테스트"""
        print(f"\nTesting SMA periods from {start_period} to {end_period}...")
        print("="*80)

        for period in range(start_period, end_period + 1):
            df = self.data.copy()
            df['SMA'] = df['Close'].rolling(window=period).mean()
            signal = (df['Close'] > df['SMA']).astype(int)

            metrics = self.backtest_strategy(signal, f'Close>SMA{period}')
            self.results.append(metrics)

            if period % 5 == 0:
                print(f"SMA{period}: Sharpe {metrics['Sharpe Ratio']:.4f}")

    def test_ema_range(self, start_period=15, end_period=60):
        """EMA 기간 범위 테스트"""
        print(f"\nTesting EMA periods from {start_period} to {end_period}...")
        print("="*80)

        for period in range(start_period, end_period + 1):
            df = self.data.copy()
            df['EMA'] = df['Close'].ewm(span=period, adjust=False).mean()
            signal = (df['Close'] > df['EMA']).astype(int)

            metrics = self.backtest_strategy(signal, f'Close>EMA{period}')
            self.results.append(metrics)

            if period % 5 == 0:
                print(f"EMA{period}: Sharpe {metrics['Sharpe Ratio']:.4f}")

    def test_wma_range(self, start_period=15, end_period=60):
        """WMA 기간 범위 테스트"""
        print(f"\nTesting WMA periods from {start_period} to {end_period}...")
        print("="*80)

        for period in range(start_period, end_period + 1):
            df = self.data.copy()
            weights = np.arange(1, period + 1)
            df['WMA'] = df['Close'].rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
            signal = (df['Close'] > df['WMA']).astype(int)

            metrics = self.backtest_strategy(signal, f'Close>WMA{period}')
            self.results.append(metrics)

            if period % 5 == 0:
                print(f"WMA{period}: Sharpe {metrics['Sharpe Ratio']:.4f}")

    def test_hybrid_strategies(self):
        """하이브리드 전략 테스트"""
        print(f"\nTesting Hybrid Strategies...")
        print("="*80)

        # 전략 1: Close > (SMA30 + EMA30) / 2
        df = self.data.copy()
        df['SMA30'] = df['Close'].rolling(window=30).mean()
        df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()
        df['Avg'] = (df['SMA30'] + df['EMA30']) / 2
        signal = (df['Close'] > df['Avg']).astype(int)
        metrics = self.backtest_strategy(signal, 'Close>(SMA30+EMA30)/2')
        self.results.append(metrics)
        print(f"Close>(SMA30+EMA30)/2: Sharpe {metrics['Sharpe Ratio']:.4f}")

        # 전략 2: Close > (SMA25 + SMA30 + SMA35) / 3
        df = self.data.copy()
        df['SMA25'] = df['Close'].rolling(window=25).mean()
        df['SMA30'] = df['Close'].rolling(window=30).mean()
        df['SMA35'] = df['Close'].rolling(window=35).mean()
        df['Avg'] = (df['SMA25'] + df['SMA30'] + df['SMA35']) / 3
        signal = (df['Close'] > df['Avg']).astype(int)
        metrics = self.backtest_strategy(signal, 'Close>(SMA25+30+35)/3')
        self.results.append(metrics)
        print(f"Close>(SMA25+30+35)/3: Sharpe {metrics['Sharpe Ratio']:.4f}")

        # 전략 3: Close > TEMA30 (Triple EMA)
        df = self.data.copy()
        ema1 = df['Close'].ewm(span=30, adjust=False).mean()
        ema2 = ema1.ewm(span=30, adjust=False).mean()
        ema3 = ema2.ewm(span=30, adjust=False).mean()
        df['TEMA'] = 3 * ema1 - 3 * ema2 + ema3
        signal = (df['Close'] > df['TEMA']).astype(int)
        metrics = self.backtest_strategy(signal, 'Close>TEMA30')
        self.results.append(metrics)
        print(f"Close>TEMA30: Sharpe {metrics['Sharpe Ratio']:.4f}")

        # 전략 4: Close > DEMA30 (Double EMA)
        df = self.data.copy()
        ema1 = df['Close'].ewm(span=30, adjust=False).mean()
        ema2 = ema1.ewm(span=30, adjust=False).mean()
        df['DEMA'] = 2 * ema1 - ema2
        signal = (df['Close'] > df['DEMA']).astype(int)
        metrics = self.backtest_strategy(signal, 'Close>DEMA30')
        self.results.append(metrics)
        print(f"Close>DEMA30: Sharpe {metrics['Sharpe Ratio']:.4f}")

        # 전략 5: Close > HMA30 (Hull MA)
        df = self.data.copy()
        half_length = int(30 / 2)
        sqrt_length = int(np.sqrt(30))
        wma_half = df['Close'].rolling(window=half_length).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
        )
        wma_full = df['Close'].rolling(window=30).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
        )
        raw_hma = 2 * wma_half - wma_full
        df['HMA'] = raw_hma.rolling(window=sqrt_length).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
        )
        signal = (df['Close'] > df['HMA']).astype(int)
        metrics = self.backtest_strategy(signal, 'Close>HMA30')
        self.results.append(metrics)
        print(f"Close>HMA30: Sharpe {metrics['Sharpe Ratio']:.4f}")

    def run_optimization(self):
        """전체 최적화 실행"""
        print("="*80)
        print("Bitcoin Strategy Optimization - Finding strategies beating benchmark")
        print("="*80)

        self.load_data()

        # 벤치마크 먼저 계산
        df = self.data.copy()
        df['SMA30'] = df['Close'].rolling(window=30).mean()
        signal = (df['Close'] > df['SMA30']).astype(int)
        benchmark = self.backtest_strategy(signal, 'BENCHMARK_Close>SMA30')
        self.results.append(benchmark)

        print(f"\nBenchmark (Close>SMA30): Sharpe {benchmark['Sharpe Ratio']:.4f}")
        print("="*80)

        # 정밀 스캔
        self.test_sma_range(15, 60)
        self.test_ema_range(15, 60)
        self.test_wma_range(15, 60)
        self.test_hybrid_strategies()

        # 결과 정리
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('Sharpe Ratio', ascending=False)

        # 벤치마크보다 나은 전략 찾기
        benchmark_sharpe = benchmark['Sharpe Ratio']
        better_strategies = results_df[
            (~results_df['Strategy'].str.contains('BENCHMARK')) &
            (results_df['Sharpe Ratio'] > benchmark_sharpe)
        ]

        print("\n" + "="*80)
        print(f"Results: {len(better_strategies)} strategies beat the benchmark!")
        print("="*80)

        if len(better_strategies) > 0:
            print("\n🎉 Top 5 Strategies Beating Benchmark:")
            print("-"*80)
            top5 = better_strategies.head(5)
            for idx, row in top5.iterrows():
                improvement = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
                print(f"\n{row['Strategy']}:")
                print(f"  Sharpe: {row['Sharpe Ratio']:.4f} (+{improvement:.2f}% vs benchmark)")
                print(f"  CAGR: {row['CAGR (%)']:.2f}%")
                print(f"  MDD: {row['MDD (%)']:.2f}%")
                print(f"  Total Return: {row['Total Return (%)']:.2f}%")
        else:
            print("\n⚠ No strategies beat the benchmark.")
            print("\nClosest strategies:")
            print("-"*80)
            closest = results_df[~results_df['Strategy'].str.contains('BENCHMARK')].head(5)
            for idx, row in closest.iterrows():
                diff = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
                print(f"\n{row['Strategy']}:")
                print(f"  Sharpe: {row['Sharpe Ratio']:.4f} ({diff:+.2f}% vs benchmark)")
                print(f"  CAGR: {row['CAGR (%)']:.2f}%")

        # CSV 저장
        results_df.to_csv('bitcoin_optimized_strategies.csv', index=False)
        print(f"\n✓ Results saved to bitcoin_optimized_strategies.csv")
        print(f"✓ Total strategies tested: {len(results_df)}")

        return results_df, better_strategies


def main():
    optimizer = BitcoinStrategyOptimizer(
        start_date='2018-01-01',
        slippage=0.002
    )

    results_df, better_strategies = optimizer.run_optimization()

    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)

    return results_df, better_strategies


if __name__ == "__main__":
    main()
