"""
비트코인 최종 전략 - 성공 패턴 확장
- MACD 필터 변형
- RSI 필터 변형
- 복합 지표 조합

목표: Sharpe 1.66 벤치마크 대비 5%+ 개선된 전략 5개 발굴
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class FinalStrategy:
    """최종 전략 백테스터"""

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

    # ==================== Top 3 이미 발견된 전략 ====================
    def strategy_sma30_macd(self):
        """#1: SMA30 + MACD (Sharpe 1.75)"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = ((df['Close'] > df['SMA30']) & (df['MACD'] > df['Signal_Line'])).astype(int)
        return signal

    def strategy_triple_confirmation(self):
        """#2: Triple Confirmation (Sharpe 1.72)"""
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

    def strategy_sma30_rsi(self):
        """#3: SMA30 + RSI (Sharpe 1.69)"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        signal = ((df['Close'] > df['SMA30']) & (df['RSI'] > 50)).astype(int)
        return signal

    # ==================== 새로운 변형 전략 ====================
    def strategy_sma25_macd(self):
        """SMA25 + MACD (더 민감한 진입)"""
        df = self.daily_data.copy()
        df['SMA25'] = df['Close'].rolling(25).mean()

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = ((df['Close'] > df['SMA25']) & (df['MACD'] > df['Signal_Line'])).astype(int)
        return signal

    def strategy_sma35_macd(self):
        """SMA35 + MACD (더 보수적 진입)"""
        df = self.daily_data.copy()
        df['SMA35'] = df['Close'].rolling(35).mean()

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = ((df['Close'] > df['SMA35']) & (df['MACD'] > df['Signal_Line'])).astype(int)
        return signal

    def strategy_ema30_macd(self):
        """EMA30 + MACD (SMA 대신 EMA)"""
        df = self.daily_data.copy()
        df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = ((df['Close'] > df['EMA30']) & (df['MACD'] > df['Signal_Line'])).astype(int)
        return signal

    def strategy_sma30_macd_hist_positive(self):
        """SMA30 + MACD 히스토그램 양수"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

        signal = ((df['Close'] > df['SMA30']) & (df['MACD_Hist'] > 0)).astype(int)
        return signal

    def strategy_sma30_rsi55(self):
        """SMA30 + RSI > 55 (더 강한 모멘텀)"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        signal = ((df['Close'] > df['SMA30']) & (df['RSI'] > 55)).astype(int)
        return signal

    def strategy_sma30_rsi45(self):
        """SMA30 + RSI > 45 (더 빠른 진입)"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        signal = ((df['Close'] > df['SMA30']) & (df['RSI'] > 45)).astype(int)
        return signal

    def strategy_sma30_volume_filter(self):
        """SMA30 + 거래량 필터 (평균 이상)"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()
        df['Vol_SMA20'] = df['Volume'].rolling(20).mean()

        signal = ((df['Close'] > df['SMA30']) & (df['Volume'] > df['Vol_SMA20'])).astype(int)
        return signal

    def strategy_dual_sma_macd(self):
        """Dual SMA (20 > 50) + MACD"""
        df = self.daily_data.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = (
            (df['SMA20'] > df['SMA50']) &
            (df['MACD'] > df['Signal_Line'])
        ).astype(int)

        return signal

    def strategy_close_above_dual_sma_macd(self):
        """Close > SMA20 AND SMA20 > SMA50 + MACD"""
        df = self.daily_data.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = (
            (df['Close'] > df['SMA20']) &
            (df['SMA20'] > df['SMA50']) &
            (df['MACD'] > df['Signal_Line'])
        ).astype(int)

        return signal

    def strategy_sma30_macd_rsi(self):
        """SMA30 + MACD + RSI (3중 필터)"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        signal = (
            (df['Close'] > df['SMA30']) &
            (df['MACD'] > df['Signal_Line']) &
            (df['RSI'] > 45)
        ).astype(int)

        return signal

    def strategy_price_above_ema_ribbon(self):
        """가격이 EMA 리본 위 + MACD"""
        df = self.daily_data.copy()
        df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = (
            (df['Close'] > df['EMA10']) &
            (df['Close'] > df['EMA20']) &
            (df['Close'] > df['EMA30']) &
            (df['MACD'] > df['Signal_Line'])
        ).astype(int)

        return signal

    def run_all_strategies(self):
        """모든 전략 실행"""
        print("="*80)
        print("Running Final Strategies")
        print("="*80 + "\n")

        strategies = {
            '00_BENCHMARK_SMA30': self.benchmark_daily_sma30,
            '01_SMA30_MACD': self.strategy_sma30_macd,
            '02_Triple_Confirmation': self.strategy_triple_confirmation,
            '03_SMA30_RSI': self.strategy_sma30_rsi,
            '04_SMA25_MACD': self.strategy_sma25_macd,
            '05_SMA35_MACD': self.strategy_sma35_macd,
            '06_EMA30_MACD': self.strategy_ema30_macd,
            '07_SMA30_MACD_Hist': self.strategy_sma30_macd_hist_positive,
            '08_SMA30_RSI55': self.strategy_sma30_rsi55,
            '09_SMA30_RSI45': self.strategy_sma30_rsi45,
            '10_SMA30_Volume': self.strategy_sma30_volume_filter,
            '11_DualSMA_MACD': self.strategy_dual_sma_macd,
            '12_Close_DualSMA_MACD': self.strategy_close_above_dual_sma_macd,
            '13_SMA30_MACD_RSI45': self.strategy_sma30_macd_rsi,
            '14_EMA_Ribbon_MACD': self.strategy_price_above_ema_ribbon,
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
        """결과 분석 - Top 5만 추출"""
        benchmark_sharpe = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0]

        # 벤치마크보다 나은 전략
        better = metrics_df[
            (~metrics_df['Strategy'].str.contains('BENCHMARK')) &
            (metrics_df['Sharpe Ratio'] > benchmark_sharpe)
        ].sort_values('Sharpe Ratio', ascending=False)

        print("\n" + "="*120)
        print(f"{'🏆 TOP 5 BITCOIN TREND STRATEGIES 🏆':^120}")
        print("="*120)
        print(f"\n벤치마크 (Close > SMA30): Sharpe {benchmark_sharpe:.4f}")
        print(f"벤치마크를 상회하는 전략: {len(better)}개\n")

        if len(better) >= 5:
            print("="*120)
            print("✓ SUCCESS: Found 5+ strategies beating benchmark!")
            print("="*120)

            for rank, (idx, row) in enumerate(better.head(5).iterrows(), 1):
                improvement = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
                print(f"\n{'='*120}")
                print(f"#{rank}: {row['Strategy']}")
                print(f"{'='*120}")
                print(f"  Sharpe Ratio: {row['Sharpe Ratio']:.4f} (+{improvement:.2f}% vs benchmark)")
                print(f"  Total Return: {row['Total Return (%)']:,.2f}%")
                print(f"  CAGR: {row['CAGR (%)']:.2f}%")
                print(f"  MDD: {row['MDD (%)']:.2f}%")
                print(f"  Win Rate: {row['Win Rate (%)']:.2f}%")
                print(f"  Total Trades: {row['Total Trades']}")

        elif len(better) > 0:
            print(f"⚠ Found {len(better)} strategies beating benchmark (need 5)")
            print("="*120)

            for rank, (idx, row) in enumerate(better.iterrows(), 1):
                improvement = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
                print(f"\n#{rank}: {row['Strategy']}")
                print(f"  Sharpe: {row['Sharpe Ratio']:.4f} (+{improvement:.2f}% vs benchmark)")
                print(f"  CAGR: {row['CAGR (%)']:.2f}%")
                print(f"  MDD: {row['MDD (%)']:.2f}%")
        else:
            print("⚠ No strategies beat the benchmark")

        print("\n" + "="*120)

        return better


def main():
    """메인 실행"""
    print("\n" + "="*80)
    print("Bitcoin Final Strategy Search")
    print("Goal: Find Top 5 strategies beating SMA30 benchmark (Sharpe 1.66)")
    print("="*80)

    analyzer = FinalStrategy(slippage=0.002)
    analyzer.load_data()

    # 모든 전략 실행
    metrics_df = analyzer.run_all_strategies()

    # 결과 분석
    top5 = analyzer.analyze_results(metrics_df)

    # CSV 저장
    metrics_df.to_csv('bitcoin_final_top5_results.csv', index=False)
    print(f"\n✓ Results saved to bitcoin_final_top5_results.csv")

    return metrics_df, top5


if __name__ == "__main__":
    main()
