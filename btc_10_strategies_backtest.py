"""
BTC 10가지 전략 백테스트 비교 분석

비트코인(BTC)에 대해 10가지 다양한 거래 전략을 적용하여 성과 비교:
1. Turtle Trading - 터틀 트레이딩 (돌파 전략)
2. RSI Oversold/Overbought - RSI 과매도/과매수 전략
3. SMA Crossover - 단순 이동평균 교차
4. EMA Crossover - 지수 이동평균 교차
5. MACD - MACD 교차 전략
6. Bollinger Bands - 볼린저 밴드
7. Mean Reversion - 평균 회귀
8. Momentum - 모멘텀 전략
9. Dual Momentum - 듀얼 모멘텀
10. Combined RSI+MACD - RSI와 MACD 결합
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


class BTCStrategyComparison:
    """BTC 10가지 전략 비교 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01',
                 end_date=None, slippage=0.002):
        """
        Args:
            symbol: 종목 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = None
        self.strategy_results = {}

    def load_data(self):
        """데이터 로드"""
        print("="*80)
        print(f"Loading {self.symbol} data...")
        print("="*80)

        file_path = f'chart_day/{self.symbol}.parquet'
        print(f"\nLoading from {file_path}...")
        df = pd.read_parquet(file_path)

        # 컬럼명 변경 (소문자 -> 대문자)
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df
        print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        print("="*80 + "\n")

    # ==================== 전략 1: Turtle Trading ====================
    def strategy_turtle_trading(self, df, entry_period=20, exit_period=10):
        """
        터틀 트레이딩 전략
        - N일 최고가 돌파 시 매수
        - M일 최저가 하향 돌파 시 매도
        """
        df = df.copy()

        # 터틀 채널
        df['entry_high'] = df['High'].rolling(window=entry_period).max().shift(1)
        df['exit_low'] = df['Low'].rolling(window=exit_period).min().shift(1)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 최고가 돌파 시 매수
            if df.iloc[i]['High'] > df.iloc[i]['entry_high'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 최저가 하향 돌파 시 매도
            elif df.iloc[i]['Low'] < df.iloc[i]['exit_low'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                # 당일 종가에 매수 (슬리피지 포함)
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['Close'] * (1 + self.slippage)
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                # 당일 종가에 매도 (슬리피지 포함)
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                sell_price = df.iloc[i]['Close'] * (1 - self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)
            elif df.iloc[i]['position'] == 1:
                # 포지션 유지
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 2: RSI Oversold/Overbought ====================
    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def strategy_rsi_oversold_overbought(self, df, rsi_period=14,
                                         oversold=30, overbought=70):
        """
        RSI 과매도/과매수 전략
        - RSI < 30: 매수 신호
        - RSI > 70: 매도 신호
        """
        df = df.copy()

        # RSI 계산
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 과매도 시 매수
            if df.iloc[i]['RSI'] < oversold and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 과매수 시 매도
            elif df.iloc[i]['RSI'] > overbought and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 3: SMA Crossover ====================
    def strategy_sma_crossover(self, df, short_period=20, long_period=50):
        """
        SMA 교차 전략
        - 단기 SMA > 장기 SMA: 매수
        - 단기 SMA < 장기 SMA: 매도
        """
        df = df.copy()

        # SMA 계산
        df['SMA_short'] = df['Close'].rolling(window=short_period).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_period).mean()

        # 포지션 계산
        df['position'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 4: EMA Crossover ====================
    def strategy_ema_crossover(self, df, short_period=12, long_period=26):
        """
        EMA 교차 전략
        - 단기 EMA > 장기 EMA: 매수
        - 단기 EMA < 장기 EMA: 매도
        """
        df = df.copy()

        # EMA 계산
        df['EMA_short'] = df['Close'].ewm(span=short_period, adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=long_period, adjust=False).mean()

        # 포지션 계산
        df['position'] = np.where(df['EMA_short'] > df['EMA_long'], 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 5: MACD ====================
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """MACD 계산"""
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def strategy_macd(self, df, fast=12, slow=26, signal=9):
        """
        MACD 전략
        - MACD > Signal: 매수
        - MACD < Signal: 매도
        """
        df = df.copy()

        # MACD 계산
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.calculate_macd(df, fast, slow, signal)

        # 포지션 계산
        df['position'] = np.where(df['MACD'] > df['MACD_signal'], 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 6: Bollinger Bands ====================
    def strategy_bollinger_bands(self, df, period=20, std_dev=2):
        """
        볼린저 밴드 전략
        - 가격이 하단 밴드 아래: 매수
        - 가격이 상단 밴드 위: 매도
        """
        df = df.copy()

        # 볼린저 밴드 계산
        df['BB_middle'] = df['Close'].rolling(window=period).mean()
        df['BB_std'] = df['Close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * std_dev)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * std_dev)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 하단 밴드 아래로 하락 시 매수
            if df.iloc[i]['Close'] < df.iloc[i]['BB_lower'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 상단 밴드 위로 상승 시 매도
            elif df.iloc[i]['Close'] > df.iloc[i]['BB_upper'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 7: Mean Reversion ====================
    def strategy_mean_reversion(self, df, period=20, threshold=1.5):
        """
        평균 회귀 전략
        - 가격이 평균에서 threshold * std 이상 벗어나면 매수
        - 평균으로 회귀하면 매도
        """
        df = df.copy()

        # 이동평균과 표준편차 계산
        df['MA'] = df['Close'].rolling(window=period).mean()
        df['STD'] = df['Close'].rolling(window=period).std()

        # Z-score 계산
        df['z_score'] = (df['Close'] - df['MA']) / df['STD']

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 과도하게 하락 시 매수
            if df.iloc[i]['z_score'] < -threshold and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 평균으로 회귀 시 매도
            elif df.iloc[i]['z_score'] > 0 and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 8: Momentum ====================
    def strategy_momentum(self, df, period=20, threshold=0.05):
        """
        모멘텀 전략
        - N일 수익률이 threshold 이상이면 매수
        - N일 수익률이 threshold 이하면 매도
        """
        df = df.copy()

        # 모멘텀 계산 (N일 수익률)
        df['momentum'] = df['Close'].pct_change(period)

        # 포지션 계산
        df['position'] = np.where(df['momentum'] > threshold, 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 9: Dual Momentum ====================
    def strategy_dual_momentum(self, df, short_period=10, long_period=30):
        """
        듀얼 모멘텀 전략
        - 단기 모멘텀 > 0 AND 장기 모멘텀 > 0: 매수
        - 그 외: 매도
        """
        df = df.copy()

        # 단기/장기 모멘텀 계산
        df['momentum_short'] = df['Close'].pct_change(short_period)
        df['momentum_long'] = df['Close'].pct_change(long_period)

        # 포지션 계산
        df['position'] = np.where(
            (df['momentum_short'] > 0) & (df['momentum_long'] > 0), 1, 0
        )

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 10: Combined RSI + MACD ====================
    def strategy_rsi_macd_combined(self, df, rsi_period=14, rsi_threshold=50,
                                   macd_fast=12, macd_slow=26, macd_signal=9):
        """
        RSI + MACD 결합 전략
        - RSI > 50 AND MACD > Signal: 매수
        - RSI < 50 OR MACD < Signal: 매도
        """
        df = df.copy()

        # RSI 계산
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)

        # MACD 계산
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.calculate_macd(
            df, macd_fast, macd_slow, macd_signal
        )

        # 포지션 계산
        df['position'] = np.where(
            (df['RSI'] > rsi_threshold) & (df['MACD'] > df['MACD_signal']), 1, 0
        )

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 실행 ====================
    def run_all_strategies(self):
        """모든 전략 실행"""
        strategies = {
            '1. Turtle Trading': lambda df: self.strategy_turtle_trading(df, entry_period=20, exit_period=10),
            '2. RSI Oversold/Overbought': lambda df: self.strategy_rsi_oversold_overbought(df, rsi_period=14, oversold=30, overbought=70),
            '3. SMA Crossover': lambda df: self.strategy_sma_crossover(df, short_period=20, long_period=50),
            '4. EMA Crossover': lambda df: self.strategy_ema_crossover(df, short_period=12, long_period=26),
            '5. MACD': lambda df: self.strategy_macd(df, fast=12, slow=26, signal=9),
            '6. Bollinger Bands': lambda df: self.strategy_bollinger_bands(df, period=20, std_dev=2),
            '7. Mean Reversion': lambda df: self.strategy_mean_reversion(df, period=20, threshold=1.5),
            '8. Momentum': lambda df: self.strategy_momentum(df, period=20, threshold=0.05),
            '9. Dual Momentum': lambda df: self.strategy_dual_momentum(df, short_period=10, long_period=30),
            '10. RSI+MACD Combined': lambda df: self.strategy_rsi_macd_combined(df, rsi_period=14, rsi_threshold=50)
        }

        print("\n" + "="*80)
        print("Running all 10 strategies...")
        print("="*80 + "\n")

        for strategy_name, strategy_func in strategies.items():
            print(f"Running {strategy_name}...")
            result = strategy_func(self.data.copy())
            self.strategy_results[strategy_name] = result

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80 + "\n")

    # ==================== 성과 지표 계산 ====================
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

        # 샤프 비율
        sharpe = (returns_series.mean() / returns_series.std() * np.sqrt(365)) if returns_series.std() > 0 else 0

        # 승률
        total_trades = (returns_series != 0).sum()
        winning_trades = (returns_series > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit Factor
        total_profit = returns_series[returns_series > 0].sum()
        total_loss = abs(returns_series[returns_series < 0].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        return {
            'Strategy': name,
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades),
            'Profit Factor': profit_factor
        }

    def calculate_all_metrics(self):
        """모든 전략의 성과 지표 계산"""
        metrics_list = []

        for strategy_name in self.strategy_results.keys():
            returns = self.strategy_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, strategy_name)
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================
    def plot_comparison(self, metrics_df, save_path='btc_10_strategies_comparison.png'):
        """전략 비교 시각화"""
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.strategy_results.keys():
            cumulative = self.strategy_results[strategy_name]['cumulative']
            ax1.plot(cumulative.index, cumulative, label=strategy_name,
                    linewidth=2, alpha=0.8)

        # Buy & Hold 추가
        buy_hold_cumulative = self.data['Close'] / self.data['Close'].iloc[0]
        ax1.plot(buy_hold_cumulative.index, buy_hold_cumulative,
                label='Buy & Hold', linewidth=2.5, alpha=0.8,
                linestyle='--', color='black')

        ax1.set_title(f'BTC Strategy Comparison - Cumulative Returns\n'
                     f'Period: {self.start_date} to {self.end_date}',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 총 수익률 비교
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
        bars = ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'],
                       color=colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 값 표시
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%',
                    ha='left' if width > 0 else 'right',
                    va='center', fontsize=8)

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['CAGR (%)']]
        bars = ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'],
                       color=colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        for bar in bars:
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%',
                    ha='left' if width > 0 else 'right',
                    va='center', fontsize=8)

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        bars = ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'],
                       color='crimson', alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        for bar in bars:
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%',
                    ha='right', va='center', fontsize=8)

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Sharpe Ratio']]
        bars = ax5.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'],
                       color=colors, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        for bar in bars:
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}',
                    ha='left' if width > 0 else 'right',
                    va='center', fontsize=8)

        # 6. Return vs Risk 산점도
        ax6 = fig.add_subplot(gs[2, 1])
        scatter = ax6.scatter(metrics_df['MDD (%)'], metrics_df['CAGR (%)'],
                   s=300, alpha=0.6, c=metrics_df['Sharpe Ratio'],
                   cmap='RdYlGn', edgecolors='black', linewidth=1.5)

        for idx, row in metrics_df.iterrows():
            # 전략 번호만 표시
            label = row['Strategy'].split('.')[0]
            ax6.annotate(label,
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=9, ha='center', va='center',
                        fontweight='bold')

        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk (colored by Sharpe)',
                     fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 컬러바
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Sharpe Ratio', fontsize=10)

        # 7. Profit Factor 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.copy()
        sorted_df = sorted_df[sorted_df['Profit Factor'] != np.inf]
        if len(sorted_df) > 0:
            sorted_df = sorted_df.sort_values('Profit Factor', ascending=True)
            colors = ['green' if x > 1 else 'red' for x in sorted_df['Profit Factor']]
            bars = ax7.barh(sorted_df['Strategy'], sorted_df['Profit Factor'],
                           color=colors, alpha=0.7)

            for bar in bars:
                width = bar.get_width()
                ax7.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.2f}',
                        ha='left', va='center', fontsize=8)

        ax7.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax7.set_xlabel('Profit Factor', fontsize=11)
        ax7.set_title('Profit Factor Comparison', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8. 승률 비교
        ax8 = fig.add_subplot(gs[3, 0])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=True)
        colors = ['green' if x >= 50 else 'orange' for x in sorted_df['Win Rate (%)']]
        bars = ax8.barh(sorted_df['Strategy'], sorted_df['Win Rate (%)'],
                       color=colors, alpha=0.7)
        ax8.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax8.set_xlabel('Win Rate (%)', fontsize=11)
        ax8.set_title('Win Rate Comparison', fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')

        for bar in bars:
            width = bar.get_width()
            ax8.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%',
                    ha='left', va='center', fontsize=8)

        # 9. 거래 횟수 비교
        ax9 = fig.add_subplot(gs[3, 1])
        sorted_df = metrics_df.sort_values('Total Trades', ascending=True)
        bars = ax9.barh(sorted_df['Strategy'], sorted_df['Total Trades'],
                       color='steelblue', alpha=0.7)
        ax9.set_xlabel('Total Trades', fontsize=11)
        ax9.set_title('Total Trades Comparison', fontsize=13, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='x')

        for bar in bars:
            width = bar.get_width()
            ax9.text(width, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}',
                    ha='left', va='center', fontsize=8)

        # 10. 성과 지표 히트맵
        ax10 = fig.add_subplot(gs[3, 2])

        # 정규화를 위한 데이터 준비
        heatmap_data = metrics_df[['Strategy', 'Total Return (%)', 'CAGR (%)',
                                   'MDD (%)', 'Sharpe Ratio', 'Win Rate (%)']].copy()
        heatmap_data = heatmap_data.set_index('Strategy')

        # MDD는 절대값으로 변환 (음수이므로)
        heatmap_data['MDD (%)'] = heatmap_data['MDD (%)'].abs()

        # 정규화 (0-1 범위)
        heatmap_normalized = heatmap_data.copy()
        for col in heatmap_normalized.columns:
            min_val = heatmap_normalized[col].min()
            max_val = heatmap_normalized[col].max()
            if max_val > min_val:
                heatmap_normalized[col] = (heatmap_normalized[col] - min_val) / (max_val - min_val)
            else:
                heatmap_normalized[col] = 0.5

        sns.heatmap(heatmap_normalized.T, annot=False, cmap='RdYlGn',
                   center=0.5, ax=ax10, cbar_kws={'label': 'Normalized Score'},
                   linewidths=0.5)
        ax10.set_title('Performance Metrics Heatmap (Normalized)',
                      fontsize=13, fontweight='bold')
        ax10.set_xlabel('')
        ax10.set_ylabel('Metrics', fontsize=11)

        # 11. 드로우다운 비교
        ax11 = fig.add_subplot(gs[4, :])
        for strategy_name in self.strategy_results.keys():
            cumulative = self.strategy_results[strategy_name]['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax11.plot(drawdown.index, drawdown, label=strategy_name,
                     linewidth=1.5, alpha=0.7)

        ax11.set_title('Drawdown Comparison Over Time',
                      fontsize=14, fontweight='bold')
        ax11.set_ylabel('Drawdown (%)', fontsize=12)
        ax11.set_xlabel('Date', fontsize=12)
        ax11.legend(loc='lower right', fontsize=9, ncol=3)
        ax11.grid(True, alpha=0.3)
        ax11.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'BTC 10가지 전략 백테스트 성과 비교':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {self.symbol}")
        print(f"슬리피지: {self.slippage*100}%")

        print("\n" + "-"*150)
        print(f"{'전략별 성과 지표':^150}")
        print("-"*150)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(metrics_df.to_string(index=False))

        print("\n" + "="*150 + "\n")

        # 최고 성과 전략 요약
        print("="*150)
        print(f"{'최고 성과 전략 요약':^150}")
        print("="*150)
        print(f"\n최고 총 수익률: {metrics_df.loc[metrics_df['Total Return (%)'].idxmax(), 'Strategy']}")
        print(f"  → {metrics_df['Total Return (%)'].max():.2f}%")

        print(f"\n최고 CAGR: {metrics_df.loc[metrics_df['CAGR (%)'].idxmax(), 'Strategy']}")
        print(f"  → {metrics_df['CAGR (%)'].max():.2f}%")

        print(f"\n최소 MDD: {metrics_df.loc[metrics_df['MDD (%)'].idxmax(), 'Strategy']}")
        print(f"  → {metrics_df['MDD (%)'].max():.2f}%")

        print(f"\n최고 샤프 비율: {metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax(), 'Strategy']}")
        print(f"  → {metrics_df['Sharpe Ratio'].max():.2f}")

        print("\n" + "="*150 + "\n")

    def run_analysis(self):
        """전체 분석 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 모든 전략 실행
        self.run_all_strategies()

        # 3. 성과 지표 계산
        metrics_df = self.calculate_all_metrics()

        # 4. 결과 출력
        self.print_metrics_table(metrics_df)

        # 5. 시각화
        self.plot_comparison(metrics_df)

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("BTC 10가지 전략 백테스트 비교 분석 시작")
    print("="*80)

    # 백테스트 실행
    comparison = BTCStrategyComparison(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = comparison.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('btc_10_strategies_metrics.csv', index=False)
    print("Metrics saved to btc_10_strategies_metrics.csv")

    # 각 전략별 상세 결과 저장
    for strategy_name in comparison.strategy_results.keys():
        filename = f"strategy_{strategy_name.replace(' ', '_').replace('.', '').replace('/', '_').lower()}.csv"
        comparison.strategy_results[strategy_name][['Close', 'position', 'returns', 'cumulative']].to_csv(filename)
        print(f"Strategy details saved to {filename}")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
