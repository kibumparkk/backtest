"""
비트코인 추세추종 전략 발굴
목표: 전일종가 > SMA30 벤치마크보다 샤프지수가 높은 전략 5개 발굴
제약: 레버리지 없음, 숏 없음 (롱 온리)
평가: 샤프 지수
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class BitcoinTrendStrategyFinder:
    """비트코인 추세추종 전략 발굴 클래스"""

    def __init__(self, start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = None
        self.strategy_results = {}

    def load_data(self):
        """비트코인 데이터 로드"""
        print("="*80)
        print("Loading Bitcoin data...")
        print("="*80)

        file_path = 'chart_day/BTC_KRW.parquet'
        print(f"\nLoading BTC_KRW from {file_path}...")
        df = pd.read_parquet(file_path)

        # 컬럼명 변경 (소문자 -> 대문자)
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df
        print(f"  Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        print("\n" + "="*80)
        print("Data loading completed!")
        print("="*80 + "\n")

    def calculate_returns(self, signal_series):
        """
        신호에 기반한 수익률 계산 (롱 온리)

        Args:
            signal_series: 매수 신호 (1=매수, 0=현금)

        Returns:
            returns, cumulative
        """
        df = self.data.copy()
        df['signal'] = signal_series

        # 포지션 변화 감지
        df['position_change'] = df['signal'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 매수/매도 시 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage  # 매수
        slippage_cost[df['position_change'] == -1] = -self.slippage  # 매도

        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df['returns'], df['cumulative']

    def calculate_metrics(self, returns_series, name):
        """성과 지표 계산"""
        # 누적 수익률
        cumulative = (1 + returns_series).cumprod()

        # 총 수익률
        total_return = (cumulative.iloc[-1] - 1) * 100

        # 연간 수익률 (CAGR)
        years = (returns_series.index[-1] - returns_series.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # MDD
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100

        # 샤프 비율 (연율화)
        sharpe = (returns_series.mean() / returns_series.std() * np.sqrt(365)) if returns_series.std() > 0 else 0

        # 승률
        total_trades = (returns_series != 0).sum()
        winning_trades = (returns_series > 0).sum()
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

    # ==================== 벤치마크 전략: 전일종가 > SMA30 ====================
    def benchmark_close_above_sma30(self):
        """
        벤치마크: 전일종가 > SMA30
        - 전일 종가가 SMA30 이상일 때 매수
        - 전일 종가가 SMA30 미만일 때 매도
        """
        df = self.data.copy()

        # SMA30 계산
        df['SMA30'] = df['Close'].rolling(window=30).mean()

        # 전일 종가 사용
        df['prev_close'] = df['Close'].shift(1)

        # 신호: 전일종가 > SMA30
        signal = (df['prev_close'] > df['SMA30']).astype(int)

        return signal

    # ==================== 전략 1: SMA10/SMA30 크로스오버 ====================
    def strategy_sma10_sma30_cross(self):
        """SMA10이 SMA30 위에 있을 때 매수"""
        df = self.data.copy()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA30'] = df['Close'].rolling(window=30).mean()
        signal = (df['SMA10'] > df['SMA30']).astype(int)
        return signal

    # ==================== 전략 2: SMA20/SMA50 크로스오버 ====================
    def strategy_sma20_sma50_cross(self):
        """SMA20이 SMA50 위에 있을 때 매수"""
        df = self.data.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        signal = (df['SMA20'] > df['SMA50']).astype(int)
        return signal

    # ==================== 전략 3: EMA12/EMA26 크로스오버 ====================
    def strategy_ema12_ema26_cross(self):
        """EMA12가 EMA26 위에 있을 때 매수"""
        df = self.data.copy()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        signal = (df['EMA12'] > df['EMA26']).astype(int)
        return signal

    # ==================== 전략 4: Donchian Channel (20/10) ====================
    def strategy_donchian_20_10(self):
        """
        20일 최고가 돌파 시 매수, 10일 최저가 하향 돌파 시 매도
        """
        df = self.data.copy()
        df['high_20'] = df['High'].rolling(window=20).max().shift(1)
        df['low_10'] = df['Low'].rolling(window=10).min().shift(1)

        signal = pd.Series(0, index=df.index)
        position = 0

        for i in range(1, len(df)):
            if df.iloc[i]['High'] > df.iloc[i]['high_20'] and position == 0:
                position = 1
            elif df.iloc[i]['Low'] < df.iloc[i]['low_10'] and position == 1:
                position = 0
            signal.iloc[i] = position

        return signal

    # ==================== 전략 5: Donchian Channel (30/15) ====================
    def strategy_donchian_30_15(self):
        """
        30일 최고가 돌파 시 매수, 15일 최저가 하향 돌파 시 매도
        """
        df = self.data.copy()
        df['high_30'] = df['High'].rolling(window=30).max().shift(1)
        df['low_15'] = df['Low'].rolling(window=15).min().shift(1)

        signal = pd.Series(0, index=df.index)
        position = 0

        for i in range(1, len(df)):
            if df.iloc[i]['High'] > df.iloc[i]['high_30'] and position == 0:
                position = 1
            elif df.iloc[i]['Low'] < df.iloc[i]['low_15'] and position == 1:
                position = 0
            signal.iloc[i] = position

        return signal

    # ==================== 전략 6: 삼중 SMA 정렬 ====================
    def strategy_triple_sma_alignment(self):
        """SMA10 > SMA20 > SMA50일 때 매수"""
        df = self.data.copy()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        signal = ((df['SMA10'] > df['SMA20']) & (df['SMA20'] > df['SMA50'])).astype(int)
        return signal

    # ==================== 전략 7: Close > SMA20 ====================
    def strategy_close_above_sma20(self):
        """종가가 SMA20 위에 있을 때 매수"""
        df = self.data.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        signal = (df['Close'] > df['SMA20']).astype(int)
        return signal

    # ==================== 전략 8: Close > SMA50 ====================
    def strategy_close_above_sma50(self):
        """종가가 SMA50 위에 있을 때 매수"""
        df = self.data.copy()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        signal = (df['Close'] > df['SMA50']).astype(int)
        return signal

    # ==================== 전략 9: Close > SMA100 ====================
    def strategy_close_above_sma100(self):
        """종가가 SMA100 위에 있을 때 매수"""
        df = self.data.copy()
        df['SMA100'] = df['Close'].rolling(window=100).mean()
        signal = (df['Close'] > df['SMA100']).astype(int)
        return signal

    # ==================== 전략 10: MACD 크로스오버 ====================
    def strategy_macd_cross(self):
        """MACD > Signal일 때 매수"""
        df = self.data.copy()

        # MACD 계산
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        signal = (df['MACD'] > df['Signal']).astype(int)
        return signal

    # ==================== 전략 11: SMA 기울기 기반 ====================
    def strategy_sma50_slope(self):
        """SMA50 상승 중 + Close > SMA50"""
        df = self.data.copy()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA50_slope'] = df['SMA50'].diff(5)  # 5일간 변화

        signal = ((df['SMA50_slope'] > 0) & (df['Close'] > df['SMA50'])).astype(int)
        return signal

    # ==================== 전략 12: Moving Average Ribbon ====================
    def strategy_ma_ribbon(self):
        """SMA5 > SMA10 > SMA20일 때 매수"""
        df = self.data.copy()
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        signal = ((df['SMA5'] > df['SMA10']) & (df['SMA10'] > df['SMA20'])).astype(int)
        return signal

    # ==================== 전략 13: Donchian (40/20) ====================
    def strategy_donchian_40_20(self):
        """40일 최고가 돌파 시 매수, 20일 최저가 하향 돌파 시 매도"""
        df = self.data.copy()
        df['high_40'] = df['High'].rolling(window=40).max().shift(1)
        df['low_20'] = df['Low'].rolling(window=20).min().shift(1)

        signal = pd.Series(0, index=df.index)
        position = 0

        for i in range(1, len(df)):
            if df.iloc[i]['High'] > df.iloc[i]['high_40'] and position == 0:
                position = 1
            elif df.iloc[i]['Low'] < df.iloc[i]['low_20'] and position == 1:
                position = 0
            signal.iloc[i] = position

        return signal

    # ==================== 전략 14: EMA20/EMA50 크로스오버 ====================
    def strategy_ema20_ema50_cross(self):
        """EMA20이 EMA50 위에 있을 때 매수"""
        df = self.data.copy()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        signal = (df['EMA20'] > df['EMA50']).astype(int)
        return signal

    # ==================== 전략 15: SMA5/SMA20 크로스오버 ====================
    def strategy_sma5_sma20_cross(self):
        """SMA5가 SMA20 위에 있을 때 매수"""
        df = self.data.copy()
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        signal = (df['SMA5'] > df['SMA20']).astype(int)
        return signal

    # ==================== 전략 16: Close > EMA30 ====================
    def strategy_close_above_ema30(self):
        """종가가 EMA30 위에 있을 때 매수"""
        df = self.data.copy()
        df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()
        signal = (df['Close'] > df['EMA30']).astype(int)
        return signal

    # ==================== 전략 17: Close > EMA50 ====================
    def strategy_close_above_ema50(self):
        """종가가 EMA50 위에 있을 때 매수"""
        df = self.data.copy()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        signal = (df['Close'] > df['EMA50']).astype(int)
        return signal

    # ==================== 전략 18: Dual Donchian ====================
    def strategy_donchian_50_25(self):
        """50일 최고가 돌파 시 매수, 25일 최저가 하향 돌파 시 매도"""
        df = self.data.copy()
        df['high_50'] = df['High'].rolling(window=50).max().shift(1)
        df['low_25'] = df['Low'].rolling(window=25).min().shift(1)

        signal = pd.Series(0, index=df.index)
        position = 0

        for i in range(1, len(df)):
            if df.iloc[i]['High'] > df.iloc[i]['high_50'] and position == 0:
                position = 1
            elif df.iloc[i]['Low'] < df.iloc[i]['low_25'] and position == 1:
                position = 0
            signal.iloc[i] = position

        return signal

    # ==================== 전략 19: SMA15/SMA40 크로스오버 ====================
    def strategy_sma15_sma40_cross(self):
        """SMA15가 SMA40 위에 있을 때 매수"""
        df = self.data.copy()
        df['SMA15'] = df['Close'].rolling(window=15).mean()
        df['SMA40'] = df['Close'].rolling(window=40).mean()
        signal = (df['SMA15'] > df['SMA40']).astype(int)
        return signal

    # ==================== 전략 20: Close > SMA200 ====================
    def strategy_close_above_sma200(self):
        """종가가 SMA200 위에 있을 때 매수 (장기 추세)"""
        df = self.data.copy()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        signal = (df['Close'] > df['SMA200']).astype(int)
        return signal

    def run_all_strategies(self):
        """모든 전략 실행 및 성과 평가"""
        print("\n" + "="*80)
        print("Running all trend-following strategies...")
        print("="*80 + "\n")

        # 전략 정의
        strategies = {
            '0_BENCHMARK_Close>SMA30': self.benchmark_close_above_sma30,
            '1_SMA10/SMA30_Cross': self.strategy_sma10_sma30_cross,
            '2_SMA20/SMA50_Cross': self.strategy_sma20_sma50_cross,
            '3_EMA12/EMA26_Cross': self.strategy_ema12_ema26_cross,
            '4_Donchian_20_10': self.strategy_donchian_20_10,
            '5_Donchian_30_15': self.strategy_donchian_30_15,
            '6_Triple_SMA_Align': self.strategy_triple_sma_alignment,
            '7_Close>SMA20': self.strategy_close_above_sma20,
            '8_Close>SMA50': self.strategy_close_above_sma50,
            '9_Close>SMA100': self.strategy_close_above_sma100,
            '10_MACD_Cross': self.strategy_macd_cross,
            '11_SMA50_Slope': self.strategy_sma50_slope,
            '12_MA_Ribbon': self.strategy_ma_ribbon,
            '13_Donchian_40_20': self.strategy_donchian_40_20,
            '14_EMA20/EMA50_Cross': self.strategy_ema20_ema50_cross,
            '15_SMA5/SMA20_Cross': self.strategy_sma5_sma20_cross,
            '16_Close>EMA30': self.strategy_close_above_ema30,
            '17_Close>EMA50': self.strategy_close_above_ema50,
            '18_Donchian_50_25': self.strategy_donchian_50_25,
            '19_SMA15/SMA40_Cross': self.strategy_sma15_sma40_cross,
            '20_Close>SMA200': self.strategy_close_above_sma200,
        }

        metrics_list = []

        for strategy_name, strategy_func in strategies.items():
            try:
                print(f"Running {strategy_name}...")

                # 신호 생성
                signal = strategy_func()

                # 수익률 계산
                returns, cumulative = self.calculate_returns(signal)

                # 성과 지표 계산
                metrics = self.calculate_metrics(returns, strategy_name)
                metrics_list.append(metrics)

                # 결과 저장
                self.strategy_results[strategy_name] = {
                    'signal': signal,
                    'returns': returns,
                    'cumulative': cumulative
                }

                print(f"  ✓ Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                continue

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80 + "\n")

        return pd.DataFrame(metrics_list)

    def select_top_strategies(self, metrics_df, top_n=5):
        """벤치마크보다 샤프지수가 높은 상위 전략 선정"""
        print("\n" + "="*80)
        print(f"Selecting top {top_n} strategies...")
        print("="*80 + "\n")

        # 벤치마크 샤프 비율
        benchmark_sharpe = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0]
        print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.4f}\n")

        # 벤치마크 제외
        candidate_strategies = metrics_df[~metrics_df['Strategy'].str.contains('BENCHMARK')].copy()

        # 샤프 비율로 정렬
        candidate_strategies = candidate_strategies.sort_values('Sharpe Ratio', ascending=False)

        # 벤치마크보다 높은 전략만 선택
        better_than_benchmark = candidate_strategies[candidate_strategies['Sharpe Ratio'] > benchmark_sharpe]

        print(f"Strategies better than benchmark: {len(better_than_benchmark)}")

        if len(better_than_benchmark) >= top_n:
            top_strategies = better_than_benchmark.head(top_n)
            print(f"\n✓ Found {top_n} strategies better than benchmark!\n")
        else:
            print(f"\n⚠ Only {len(better_than_benchmark)} strategies beat the benchmark.")
            print(f"Selecting top {top_n} strategies overall...\n")
            top_strategies = candidate_strategies.head(top_n)

        # 벤치마크 포함하여 반환
        benchmark_row = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]
        final_df = pd.concat([benchmark_row, top_strategies]).reset_index(drop=True)

        return final_df

    def plot_comparison(self, metrics_df, save_path='bitcoin_trend_strategies_comparison.png'):
        """전략 비교 시각화"""
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 누적 수익률 비교 (상위 전략만)
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in metrics_df['Strategy']:
            if strategy_name in self.strategy_results:
                cumulative = self.strategy_results[strategy_name]['cumulative']

                # 벤치마크는 굵게
                if 'BENCHMARK' in strategy_name:
                    ax1.plot(cumulative.index, cumulative, label=strategy_name,
                            linewidth=3.5, alpha=0.9, color='black', linestyle='--')
                else:
                    ax1.plot(cumulative.index, cumulative, label=strategy_name,
                            linewidth=2.5, alpha=0.8)

        ax1.set_title('Cumulative Returns Comparison - Bitcoin Trend Following Strategies',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 샤프 비율 비교
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'green' if x > sorted_df[sorted_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0] else 'orange'
                  for s, x in zip(sorted_df['Strategy'], sorted_df['Sharpe Ratio'])]
        ax2.barh(range(len(sorted_df)), sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_df)))
        ax2.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax2.set_xlabel('Sharpe Ratio', fontsize=11)
        ax2.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=sorted_df[sorted_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0],
                   color='red', linestyle='--', linewidth=2, alpha=0.7, label='Benchmark')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'green' if x > 0 else 'orange'
                  for s, x in zip(sorted_df['Strategy'], sorted_df['CAGR (%)'])]
        ax3.barh(range(len(sorted_df)), sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(sorted_df)))
        ax3.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        colors = ['red' if 'BENCHMARK' in s else 'crimson'
                  for s in sorted_df['Strategy']]
        ax4.barh(range(len(sorted_df)), sorted_df['MDD (%)'], color=colors, alpha=0.7)
        ax4.set_yticks(range(len(sorted_df)))
        ax4.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. Return vs Risk 산점도
        ax5 = fig.add_subplot(gs[2, 0])
        for idx, row in metrics_df.iterrows():
            if 'BENCHMARK' in row['Strategy']:
                ax5.scatter(row['MDD (%)'], row['CAGR (%)'], s=400, alpha=0.8,
                           c='red', marker='s', edgecolors='black', linewidths=2)
            else:
                ax5.scatter(row['MDD (%)'], row['CAGR (%)'], s=300, alpha=0.6,
                           c=row['Sharpe Ratio'], cmap='RdYlGn', vmin=0, vmax=2)

        for idx, row in metrics_df.iterrows():
            label = row['Strategy'].replace('0_BENCHMARK_', 'BM: ')
            ax5.annotate(label, (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=8, ha='center', va='bottom')

        ax5.set_xlabel('MDD (%)', fontsize=11)
        ax5.set_ylabel('CAGR (%)', fontsize=11)
        ax5.set_title('Return vs Risk (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 6. Total Return 비교
        ax6 = fig.add_subplot(gs[2, 1])
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'green' if x > 0 else 'orange'
                  for s, x in zip(sorted_df['Strategy'], sorted_df['Total Return (%)'])]
        ax6.barh(range(len(sorted_df)), sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax6.set_yticks(range(len(sorted_df)))
        ax6.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax6.set_xlabel('Total Return (%)', fontsize=11)
        ax6.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        # 7. Win Rate 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'steelblue'
                  for s in sorted_df['Strategy']]
        ax7.barh(range(len(sorted_df)), sorted_df['Win Rate (%)'], color=colors, alpha=0.7)
        ax7.set_yticks(range(len(sorted_df)))
        ax7.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax7.set_xlabel('Win Rate (%)', fontsize=11)
        ax7.set_title('Win Rate Comparison', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')
        ax7.axvline(x=50, color='black', linestyle='--', linewidth=1, alpha=0.5)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_results(self, metrics_df):
        """결과 출력"""
        print("\n" + "="*120)
        print(f"{'Bitcoin Trend-Following Strategy Comparison':^120}")
        print("="*120)
        print(f"\nPeriod: {self.start_date} ~ {self.end_date}")
        print(f"Asset: BTC_KRW")
        print(f"Slippage: {self.slippage*100}%")
        print(f"Benchmark: Close > SMA30 (previous close)")

        print("\n" + "-"*120)
        print(f"{'Strategy Performance Metrics':^120}")
        print("-"*120)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}')

        print(metrics_df.to_string(index=False))
        print("\n" + "="*120 + "\n")


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("비트코인 추세추종 전략 발굴 시작")
    print("="*80)

    # 전략 발굴기 초기화
    finder = BitcoinTrendStrategyFinder(
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )

    # 데이터 로드
    finder.load_data()

    # 모든 전략 실행
    all_metrics = finder.run_all_strategies()

    # 상위 5개 전략 선정
    top_metrics = finder.select_top_strategies(all_metrics, top_n=5)

    # 결과 출력
    finder.print_results(top_metrics)

    # 시각화
    finder.plot_comparison(top_metrics)

    # 결과 저장
    print("\nSaving results...")
    all_metrics.to_csv('bitcoin_all_strategies_metrics.csv', index=False)
    top_metrics.to_csv('bitcoin_top5_strategies_metrics.csv', index=False)
    print("  ✓ bitcoin_all_strategies_metrics.csv")
    print("  ✓ bitcoin_top5_strategies_metrics.csv")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")

    return top_metrics


if __name__ == "__main__":
    main()
