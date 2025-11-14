"""
비트코인 멀티 타임프레임 전략
주봉 필터 + 일봉 타이밍으로 유의미한 성과 개선

벤치마크: Close > SMA30 (일봉 only, Sharpe 1.66)
목표: Sharpe 1.8+ 달성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class MultiTimeframeStrategy:
    """멀티 타임프레임 전략 백테스터"""

    def __init__(self, slippage=0.002):
        self.slippage = slippage
        self.daily_data = None
        self.weekly_data = None
        self.results = {}

    def load_data(self):
        """일봉 및 주봉 데이터 로드"""
        print("="*80)
        print("Loading Bitcoin data...")
        print("="*80)

        # 일봉 데이터
        df_daily = pd.read_parquet('chart_day/BTC_KRW.parquet')
        df_daily.columns = [col.capitalize() for col in df_daily.columns]
        df_daily = df_daily[df_daily.index >= '2018-01-01']
        self.daily_data = df_daily

        # 주봉 데이터 생성 (일봉에서 리샘플링)
        df_weekly = df_daily.resample('W-MON', label='left', closed='left').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        self.weekly_data = df_weekly

        print(f"\nDaily data: {len(df_daily)} bars from {df_daily.index[0]} to {df_daily.index[-1]}")
        print(f"Weekly data: {len(df_weekly)} bars from {df_weekly.index[0]} to {df_weekly.index[-1]}")
        print("="*80 + "\n")

    def backtest(self, signal, name):
        """백테스트 실행"""
        df = self.daily_data.copy()
        df['signal'] = signal
        df['pos_change'] = df['signal'].diff()
        df['daily_ret'] = df['Close'].pct_change()
        df['strat_ret'] = df['signal'].shift(1) * df['daily_ret']

        # 슬리피지
        slip_cost = pd.Series(0.0, index=df.index)
        slip_cost[df['pos_change'] == 1] = -self.slippage
        slip_cost[df['pos_change'] == -1] = -self.slippage
        df['strat_ret'] = df['strat_ret'] + slip_cost
        df['strat_ret'] = df['strat_ret'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['strat_ret']).cumprod()

        # 성과 지표
        total_return = (df['cumulative'].iloc[-1] - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['cumulative'].iloc[-1] ** (1/years) - 1) * 100

        cummax = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - cummax) / cummax
        mdd = drawdown.min() * 100

        sharpe = (df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(365)) if df['strat_ret'].std() > 0 else 0

        total_trades = (df['strat_ret'] != 0).sum()
        winning_trades = (df['strat_ret'] > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        self.results[name] = {
            'returns': df['strat_ret'],
            'cumulative': df['cumulative'],
            'metrics': {
                'Strategy': name,
                'Total Return (%)': total_return,
                'CAGR (%)': cagr,
                'MDD (%)': mdd,
                'Sharpe Ratio': sharpe,
                'Win Rate (%)': win_rate,
                'Total Trades': int(total_trades)
            }
        }

        return self.results[name]['metrics']

    # ==================== 벤치마크 ====================
    def benchmark_daily_sma30(self):
        """벤치마크: Close > SMA30 (일봉만)"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()
        signal = (df['Close'] > df['SMA30']).astype(int)
        return signal

    # ==================== 전략 1: Weekly SMA20 필터 ====================
    def strategy_weekly_sma20_daily_sma30(self):
        """
        주봉 SMA20 위 + 일봉 SMA30 위
        - Weekly Close > Weekly SMA20 → 상승 추세
        - 상승 추세일 때만 Daily Close > Daily SMA30 신호 활성화
        """
        # 주봉 신호
        weekly = self.weekly_data.copy()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['SMA20']).astype(int)

        # 일봉에 주봉 신호 병합
        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        # 주봉 신호를 일봉 인덱스에 맞게 확장 (forward fill)
        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')

        # 복합 신호: 주봉 상승 AND 일봉 SMA30 위
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_trend == 1)).astype(int)

        return signal

    # ==================== 전략 2: Weekly SMA50 필터 ====================
    def strategy_weekly_sma50_daily_sma30(self):
        """주봉 SMA50 위 + 일봉 SMA30 위"""
        weekly = self.weekly_data.copy()
        weekly['SMA50'] = weekly['Close'].rolling(50).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['SMA50']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_trend == 1)).astype(int)

        return signal

    # ==================== 전략 3: Weekly SMA10 필터 ====================
    def strategy_weekly_sma10_daily_sma30(self):
        """주봉 SMA10 위 + 일봉 SMA30 위 (더 민감)"""
        weekly = self.weekly_data.copy()
        weekly['SMA10'] = weekly['Close'].rolling(10).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['SMA10']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_trend == 1)).astype(int)

        return signal

    # ==================== 전략 4: Weekly EMA20 필터 ====================
    def strategy_weekly_ema20_daily_sma30(self):
        """주봉 EMA20 위 + 일봉 SMA30 위"""
        weekly = self.weekly_data.copy()
        weekly['EMA20'] = weekly['Close'].ewm(span=20, adjust=False).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['EMA20']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_trend == 1)).astype(int)

        return signal

    # ==================== 전략 5: Weekly 골든크로스 필터 ====================
    def strategy_weekly_golden_cross_daily_sma30(self):
        """주봉 SMA10 > SMA20 + 일봉 SMA30 위"""
        weekly = self.weekly_data.copy()
        weekly['SMA10'] = weekly['Close'].rolling(10).mean()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['golden_cross'] = (weekly['SMA10'] > weekly['SMA20']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_signal = weekly['golden_cross'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_signal == 1)).astype(int)

        return signal

    # ==================== 전략 6: Weekly MACD 필터 ====================
    def strategy_weekly_macd_daily_sma30(self):
        """주봉 MACD > Signal + 일봉 SMA30 위"""
        weekly = self.weekly_data.copy()

        # MACD 계산
        ema12 = weekly['Close'].ewm(span=12, adjust=False).mean()
        ema26 = weekly['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        weekly['macd_bullish'] = (macd > signal_line).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_signal = weekly['macd_bullish'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_signal == 1)).astype(int)

        return signal

    # ==================== 전략 7: Weekly 200 SMA 필터 ====================
    def strategy_weekly_sma200_daily_sma30(self):
        """주봉 SMA200 위 + 일봉 SMA30 위 (강력한 장기 추세)"""
        weekly = self.weekly_data.copy()
        weekly['SMA200'] = weekly['Close'].rolling(200).mean()
        weekly['bull_market'] = (weekly['Close'] > weekly['SMA200']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_signal = weekly['bull_market'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_signal == 1)).astype(int)

        return signal

    # ==================== 전략 8: Weekly 듀얼 SMA 필터 ====================
    def strategy_weekly_dual_sma_daily_sma30(self):
        """주봉 SMA20 & SMA50 위 + 일봉 SMA30 위"""
        weekly = self.weekly_data.copy()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['SMA50'] = weekly['Close'].rolling(50).mean()
        weekly['strong_trend'] = ((weekly['Close'] > weekly['SMA20']) &
                                  (weekly['Close'] > weekly['SMA50'])).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_signal = weekly['strong_trend'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_signal == 1)).astype(int)

        return signal

    # ==================== 전략 9: Weekly SMA 정렬 필터 ====================
    def strategy_weekly_alignment_daily_sma30(self):
        """주봉 SMA10 > SMA20 > SMA50 정렬 + 일봉 SMA30 위"""
        weekly = self.weekly_data.copy()
        weekly['SMA10'] = weekly['Close'].rolling(10).mean()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['SMA50'] = weekly['Close'].rolling(50).mean()
        weekly['aligned'] = ((weekly['SMA10'] > weekly['SMA20']) &
                            (weekly['SMA20'] > weekly['SMA50'])).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_signal = weekly['aligned'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_signal == 1)).astype(int)

        return signal

    # ==================== 전략 10: Weekly Donchian 필터 ====================
    def strategy_weekly_donchian_daily_sma30(self):
        """주봉 20주 신고가 근처 + 일봉 SMA30 위"""
        weekly = self.weekly_data.copy()
        weekly['high_20'] = weekly['High'].rolling(20).max()
        # 신고가의 95% 이상일 때 강세
        weekly['near_high'] = (weekly['Close'] > weekly['high_20'] * 0.95).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_signal = weekly['near_high'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA30']) & (weekly_signal == 1)).astype(int)

        return signal

    # ==================== 전략 11-15: 일봉 타이밍 변형 ====================
    def strategy_weekly_sma20_daily_sma20(self):
        """주봉 SMA20 위 + 일봉 SMA20 위"""
        weekly = self.weekly_data.copy()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['SMA20']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA20'] = daily['Close'].rolling(20).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA20']) & (weekly_trend == 1)).astype(int)

        return signal

    def strategy_weekly_sma20_daily_sma50(self):
        """주봉 SMA20 위 + 일봉 SMA50 위"""
        weekly = self.weekly_data.copy()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['SMA20']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA50'] = daily['Close'].rolling(50).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA50']) & (weekly_trend == 1)).astype(int)

        return signal

    def strategy_weekly_sma50_daily_sma20(self):
        """주봉 SMA50 위 + 일봉 SMA20 위"""
        weekly = self.weekly_data.copy()
        weekly['SMA50'] = weekly['Close'].rolling(50).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['SMA50']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA20'] = daily['Close'].rolling(20).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA20']) & (weekly_trend == 1)).astype(int)

        return signal

    def strategy_weekly_sma50_daily_sma50(self):
        """주봉 SMA50 위 + 일봉 SMA50 위"""
        weekly = self.weekly_data.copy()
        weekly['SMA50'] = weekly['Close'].rolling(50).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['SMA50']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA50'] = daily['Close'].rolling(50).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['SMA50']) & (weekly_trend == 1)).astype(int)

        return signal

    def strategy_weekly_ema20_daily_ema30(self):
        """주봉 EMA20 위 + 일봉 EMA30 위"""
        weekly = self.weekly_data.copy()
        weekly['EMA20'] = weekly['Close'].ewm(span=20, adjust=False).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['EMA20']).astype(int)

        daily = self.daily_data.copy()
        daily['EMA30'] = daily['Close'].ewm(span=30, adjust=False).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        signal = ((daily['Close'] > daily['EMA30']) & (weekly_trend == 1)).astype(int)

        return signal

    def run_all_strategies(self):
        """모든 전략 실행"""
        print("="*80)
        print("Running Multi-Timeframe Strategies")
        print("="*80 + "\n")

        strategies = {
            '0_BENCHMARK_Daily_SMA30': self.benchmark_daily_sma30,
            '1_Weekly_SMA20+Daily_SMA30': self.strategy_weekly_sma20_daily_sma30,
            '2_Weekly_SMA50+Daily_SMA30': self.strategy_weekly_sma50_daily_sma30,
            '3_Weekly_SMA10+Daily_SMA30': self.strategy_weekly_sma10_daily_sma30,
            '4_Weekly_EMA20+Daily_SMA30': self.strategy_weekly_ema20_daily_sma30,
            '5_Weekly_GoldenCross+Daily_SMA30': self.strategy_weekly_golden_cross_daily_sma30,
            '6_Weekly_MACD+Daily_SMA30': self.strategy_weekly_macd_daily_sma30,
            '7_Weekly_SMA200+Daily_SMA30': self.strategy_weekly_sma200_daily_sma30,
            '8_Weekly_DualSMA+Daily_SMA30': self.strategy_weekly_dual_sma_daily_sma30,
            '9_Weekly_Alignment+Daily_SMA30': self.strategy_weekly_alignment_daily_sma30,
            '10_Weekly_Donchian+Daily_SMA30': self.strategy_weekly_donchian_daily_sma30,
            '11_Weekly_SMA20+Daily_SMA20': self.strategy_weekly_sma20_daily_sma20,
            '12_Weekly_SMA20+Daily_SMA50': self.strategy_weekly_sma20_daily_sma50,
            '13_Weekly_SMA50+Daily_SMA20': self.strategy_weekly_sma50_daily_sma20,
            '14_Weekly_SMA50+Daily_SMA50': self.strategy_weekly_sma50_daily_sma50,
            '15_Weekly_EMA20+Daily_EMA30': self.strategy_weekly_ema20_daily_ema30,
        }

        metrics_list = []

        for name, func in strategies.items():
            try:
                print(f"Running {name}...")
                signal = func()
                metrics = self.backtest(signal, name)
                metrics_list.append(metrics)
                print(f"  Sharpe: {metrics['Sharpe Ratio']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80)

        return pd.DataFrame(metrics_list)

    def analyze_results(self, metrics_df):
        """결과 분석"""
        benchmark_sharpe = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0]

        # 벤치마크보다 나은 전략
        better = metrics_df[
            (~metrics_df['Strategy'].str.contains('BENCHMARK')) &
            (metrics_df['Sharpe Ratio'] > benchmark_sharpe)
        ].sort_values('Sharpe Ratio', ascending=False)

        print("\n" + "="*120)
        print(f"{'Multi-Timeframe Strategy Results':^120}")
        print("="*120)
        print(f"\nBenchmark (Daily SMA30 only): Sharpe {benchmark_sharpe:.4f}")
        print(f"Strategies beating benchmark: {len(better)}")

        if len(better) > 0:
            print("\n" + "-"*120)
            print("🎯 Top Strategies (Beating Benchmark):")
            print("-"*120)

            for idx, row in better.head(10).iterrows():
                improvement = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
                print(f"\n{row['Strategy']}")
                print(f"  Sharpe: {row['Sharpe Ratio']:.4f} (+{improvement:.2f}% vs benchmark)")
                print(f"  Total Return: {row['Total Return (%)']:.2f}%")
                print(f"  CAGR: {row['CAGR (%)']:.2f}%")
                print(f"  MDD: {row['MDD (%)']:.2f}%")
                print(f"  Win Rate: {row['Win Rate (%)']:.2f}%")
        else:
            print("\n⚠ No strategies beat the benchmark")
            print("\nClosest strategies:")
            print("-"*120)

            closest = metrics_df[~metrics_df['Strategy'].str.contains('BENCHMARK')].nlargest(5, 'Sharpe Ratio')
            for idx, row in closest.iterrows():
                diff = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
                print(f"\n{row['Strategy']}")
                print(f"  Sharpe: {row['Sharpe Ratio']:.4f} ({diff:+.2f}% vs benchmark)")
                print(f"  CAGR: {row['CAGR (%)']:.2f}%")
                print(f"  MDD: {row['MDD (%)']:.2f}%")

        print("\n" + "="*120)

        return better


def main():
    """메인 실행"""
    print("\n" + "="*80)
    print("Bitcoin Multi-Timeframe Strategy Analysis")
    print("Goal: Beat Daily SMA30 benchmark (Sharpe 1.66) with meaningful improvement")
    print("="*80)

    analyzer = MultiTimeframeStrategy(slippage=0.002)
    analyzer.load_data()

    # 모든 전략 실행
    metrics_df = analyzer.run_all_strategies()

    # 결과 분석
    better_strategies = analyzer.analyze_results(metrics_df)

    # CSV 저장
    metrics_df.to_csv('bitcoin_multi_timeframe_results.csv', index=False)
    print(f"\n✓ Results saved to bitcoin_multi_timeframe_results.csv")

    return metrics_df, better_strategies


if __name__ == "__main__":
    main()
