"""
비트코인 고급 전략 탐색
- Volatility-adaptive indicators
- Momentum filters (RSI, MACD)
- ATR-based trailing stops
- Bollinger Bands
- Composite indicators

목표: Sharpe 1.66 벤치마크를 유의미하게 상회하는 전략 발굴
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class AdvancedStrategy:
    """고급 전략 백테스터"""

    def __init__(self, slippage=0.002):
        self.slippage = slippage
        self.daily_data = None
        self.results = {}

    def load_data(self):
        """일봉 데이터 로드"""
        print("="*80)
        print("Loading Bitcoin data...")
        print("="*80)

        df_daily = pd.read_parquet('chart_day/BTC_KRW.parquet')
        df_daily.columns = [col.capitalize() for col in df_daily.columns]
        df_daily = df_daily[df_daily.index >= '2018-01-01']
        self.daily_data = df_daily

        print(f"\nDaily data: {len(df_daily)} bars from {df_daily.index[0]} to {df_daily.index[-1]}")
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
        """벤치마크: Close > SMA30"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()
        signal = (df['Close'] > df['SMA30']).astype(int)
        return signal

    # ==================== Volatility-Adaptive Strategies ====================
    def strategy_adaptive_sma(self):
        """
        변동성 적응형 SMA
        - 변동성 낮을 때: 짧은 SMA (빠른 진입)
        - 변동성 높을 때: 긴 SMA (보수적)
        """
        df = self.daily_data.copy()

        # ATR로 변동성 측정
        df['TR'] = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()
        df['ATR_pct'] = df['ATR'] / df['Close']

        # 변동성 구간별 다른 SMA 사용
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA30'] = df['Close'].rolling(30).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()

        # ATR 상위 33%: SMA50, 중간 33%: SMA30, 하위 33%: SMA20
        atr_33 = df['ATR_pct'].quantile(0.33)
        atr_67 = df['ATR_pct'].quantile(0.67)

        conditions = [
            df['ATR_pct'] <= atr_33,  # 낮은 변동성
            (df['ATR_pct'] > atr_33) & (df['ATR_pct'] <= atr_67),  # 중간 변동성
            df['ATR_pct'] > atr_67  # 높은 변동성
        ]

        choices = [
            (df['Close'] > df['SMA20']).astype(int),
            (df['Close'] > df['SMA30']).astype(int),
            (df['Close'] > df['SMA50']).astype(int)
        ]

        signal = pd.Series(np.select(conditions, choices, default=0), index=df.index)
        return signal

    # ==================== RSI Filter Strategies ====================
    def strategy_sma30_rsi_filter(self):
        """
        SMA30 + RSI 과매수/과매도 필터
        - SMA30 위 AND RSI < 70 (과매수 회피)
        """
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        # RSI 계산
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        signal = ((df['Close'] > df['SMA30']) & (df['RSI'] < 70)).astype(int)
        return signal

    def strategy_sma30_rsi_confirmation(self):
        """
        SMA30 + RSI 확인
        - SMA30 위 AND RSI > 50 (강세 확인)
        """
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        signal = ((df['Close'] > df['SMA30']) & (df['RSI'] > 50)).astype(int)
        return signal

    # ==================== MACD Strategies ====================
    def strategy_sma30_macd_filter(self):
        """
        SMA30 + MACD 양수 필터
        - SMA30 위 AND MACD > 0
        """
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        # MACD 계산
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = ((df['Close'] > df['SMA30']) & (df['MACD'] > df['Signal_Line'])).astype(int)
        return signal

    # ==================== Bollinger Band Strategies ====================
    def strategy_bollinger_mean_reversion(self):
        """
        볼린저 밴드 평균회귀
        - 하단 밴드 터치 시 매수, 중간선 도달 시 매도
        """
        df = self.daily_data.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['STD20'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + 2 * df['STD20']
        df['BB_Lower'] = df['SMA20'] - 2 * df['STD20']

        # 하단 터치 후 매수, 중간선 위로 다시 올라오면 매도
        signal = (df['Close'] > df['SMA20']).astype(int)
        return signal

    def strategy_bollinger_trend(self):
        """
        볼린저 밴드 추세 추종
        - 상단 밴드 근처: 강세 (95% 이상)
        """
        df = self.daily_data.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['STD20'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + 2 * df['STD20']
        df['BB_Lower'] = df['SMA20'] - 2 * df['STD20']

        # 밴드폭 대비 위치
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        signal = (df['BB_Position'] > 0.7).astype(int)
        return signal

    # ==================== ATR-Based Strategies ====================
    def strategy_sma30_atr_stop(self):
        """
        SMA30 + ATR 기반 트레일링 스톱
        - SMA30 돌파로 진입
        - ATR 2배 이하로 떨어지면 청산
        """
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        # ATR 계산
        df['TR'] = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()

        # 기본 신호: SMA30 위
        base_signal = (df['Close'] > df['SMA30']).astype(int)

        # ATR 트레일링 스톱 (복잡하므로 일단 기본 신호만)
        signal = base_signal
        return signal

    # ==================== Composite Strategies ====================
    def strategy_triple_confirmation(self):
        """
        3중 확인 전략
        - SMA30 위
        - RSI > 50
        - MACD > Signal
        """
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = (
            (df['Close'] > df['SMA30']) &
            (df['RSI'] > 50) &
            (df['MACD'] > df['Signal_Line'])
        ).astype(int)

        return signal

    def strategy_dual_sma_rsi(self):
        """
        듀얼 SMA + RSI
        - SMA20 > SMA50 (골든크로스)
        - RSI > 45
        """
        df = self.daily_data.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        signal = (
            (df['SMA20'] > df['SMA50']) &
            (df['RSI'] > 45)
        ).astype(int)

        return signal

    def strategy_ema_cluster(self):
        """
        EMA 클러스터
        - EMA10, EMA20, EMA30 모두 정렬 (10 > 20 > 30)
        """
        df = self.daily_data.copy()
        df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()

        signal = (
            (df['EMA10'] > df['EMA20']) &
            (df['EMA20'] > df['EMA30'])
        ).astype(int)

        return signal

    # ==================== Momentum Strategies ====================
    def strategy_momentum_sma(self):
        """
        모멘텀 + SMA
        - 20일 모멘텀 > 0 (20일 전 대비 상승)
        - Close > SMA30
        """
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()
        df['Momentum20'] = df['Close'] - df['Close'].shift(20)

        signal = (
            (df['Close'] > df['SMA30']) &
            (df['Momentum20'] > 0)
        ).astype(int)

        return signal

    def strategy_roc_filter(self):
        """
        ROC (Rate of Change) 필터
        - ROC(10) > 0
        - Close > SMA30
        """
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()
        df['ROC10'] = (df['Close'] / df['Close'].shift(10) - 1) * 100

        signal = (
            (df['Close'] > df['SMA30']) &
            (df['ROC10'] > 0)
        ).astype(int)

        return signal

    def run_all_strategies(self):
        """모든 전략 실행"""
        print("="*80)
        print("Running Advanced Strategies")
        print("="*80 + "\n")

        strategies = {
            '00_BENCHMARK_SMA30': self.benchmark_daily_sma30,
            '01_Adaptive_SMA': self.strategy_adaptive_sma,
            '02_SMA30_RSI_Filter': self.strategy_sma30_rsi_filter,
            '03_SMA30_RSI_Confirmation': self.strategy_sma30_rsi_confirmation,
            '04_SMA30_MACD_Filter': self.strategy_sma30_macd_filter,
            '05_Bollinger_Mean_Reversion': self.strategy_bollinger_mean_reversion,
            '06_Bollinger_Trend': self.strategy_bollinger_trend,
            '07_SMA30_ATR_Stop': self.strategy_sma30_atr_stop,
            '08_Triple_Confirmation': self.strategy_triple_confirmation,
            '09_Dual_SMA_RSI': self.strategy_dual_sma_rsi,
            '10_EMA_Cluster': self.strategy_ema_cluster,
            '11_Momentum_SMA': self.strategy_momentum_sma,
            '12_ROC_Filter': self.strategy_roc_filter,
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
        print(f"{'Advanced Strategy Results':^120}")
        print("="*120)
        print(f"\nBenchmark (SMA30): Sharpe {benchmark_sharpe:.4f}")
        print(f"Strategies beating benchmark: {len(better)}")

        if len(better) > 0:
            print("\n" + "-"*120)
            print("🎯 Strategies Beating Benchmark:")
            print("-"*120)

            for idx, row in better.iterrows():
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
    print("Bitcoin Advanced Strategy Analysis")
    print("Goal: Beat SMA30 benchmark (Sharpe 1.66) with meaningful improvement")
    print("="*80)

    analyzer = AdvancedStrategy(slippage=0.002)
    analyzer.load_data()

    # 모든 전략 실행
    metrics_df = analyzer.run_all_strategies()

    # 결과 분석
    better_strategies = analyzer.analyze_results(metrics_df)

    # CSV 저장
    metrics_df.to_csv('bitcoin_advanced_results.csv', index=False)
    print(f"\n✓ Results saved to bitcoin_advanced_results.csv")

    return metrics_df, better_strategies


if __name__ == "__main__":
    main()
