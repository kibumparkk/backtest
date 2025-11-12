"""
비트코인 추세추종 전략 10종
각 전략은 2개 지표를 OR 조건으로 사용하여 매수 시그널 생성
조건이 아닐 시 매도하는 전략

전략 목록:
1. SMA_20 OR RSI_60: Close > SMA20 또는 RSI > 60
2. EMA_12 OR MACD: Close > EMA12 또는 MACD > 0
3. SMA_50 OR ADX_25: Close > SMA50 또는 ADX > 25
4. RSI_55 OR BB_UPPER: RSI > 55 또는 Close가 볼린저밴드 상단 근처
5. EMA_20 OR RSI_50: Close > EMA20 또는 RSI > 50
6. SMA_30 OR MACD_SIGNAL: Close > SMA30 또는 MACD > Signal
7. DONCHIAN_20 OR RSI_65: Close > 돈치안채널20 또는 RSI > 65
8. EMA_50 OR MOMENTUM: Close > EMA50 또는 Momentum > 0
9. SMA_100 OR RSI_70: Close > SMA100 또는 RSI > 70
10. EMA_30 OR TURTLE: Close > EMA30 또는 Turtle 돌파
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


class BitcoinTrendFollowingStrategies:
    """비트코인 추세추종 전략 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbol: 종목 (default: BTC_KRW)
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
        df = pd.read_parquet(file_path)

        # 컬럼명 변경 (소문자 -> 대문자)
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df
        print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        print("="*80 + "\n")

    # ==================== 보조 지표 계산 함수 ====================
    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_ema(self, prices, period):
        """EMA 계산"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD 계산"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd, signal)
        return macd, macd_signal

    def calculate_adx(self, df, period=14):
        """ADX 계산"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        # +DM, -DM 계산
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # TR 계산
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR 계산
        atr = tr.rolling(window=period).mean()

        # +DI, -DI 계산
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # DX, ADX 계산
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def calculate_donchian(self, df, period=20):
        """돈치안 채널 계산"""
        upper = df['High'].rolling(window=period).max()
        lower = df['Low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        return upper, middle, lower

    def calculate_momentum(self, prices, period=10):
        """모멘텀 계산"""
        return prices.diff(period)

    # ==================== 전략 1: SMA_20 OR RSI_60 ====================
    def strategy_sma20_or_rsi60(self, df):
        """
        전략 1: SMA_20 OR RSI_60
        매수: Close > SMA20 또는 RSI > 60
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'], period=14)

        # 신호 생성 (OR 조건)
        condition1 = df['Close'] > df['SMA20']
        condition2 = df['RSI'] > 60
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산 (shift로 look-ahead bias 방지)
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 2: EMA_12 OR MACD ====================
    def strategy_ema12_or_macd(self, df):
        """
        전략 2: EMA_12 OR MACD
        매수: Close > EMA12 또는 MACD > 0
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['EMA12'] = self.calculate_ema(df['Close'], 12)
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])

        # 신호 생성 (OR 조건)
        condition1 = df['Close'] > df['EMA12']
        condition2 = df['MACD'] > 0
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 3: SMA_50 OR ADX_25 ====================
    def strategy_sma50_or_adx25(self, df):
        """
        전략 3: SMA_50 OR ADX_25
        매수: Close > SMA50 또는 ADX > 25
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['ADX'] = self.calculate_adx(df, period=14)

        # 신호 생성 (OR 조건)
        condition1 = df['Close'] > df['SMA50']
        condition2 = df['ADX'] > 25
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 4: RSI_55 OR BB_UPPER ====================
    def strategy_rsi55_or_bbupper(self, df):
        """
        전략 4: RSI_55 OR BB_UPPER
        매수: RSI > 55 또는 Close가 볼린저밴드 상단 근처 (상단의 90% 이상)
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['RSI'] = self.calculate_rsi(df['Close'], period=14)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])

        # 신호 생성 (OR 조건)
        condition1 = df['RSI'] > 55
        # Close가 볼린저밴드 상단의 90% 이상
        bb_threshold = df['BB_Middle'] + (df['BB_Upper'] - df['BB_Middle']) * 0.7
        condition2 = df['Close'] > bb_threshold
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 5: EMA_20 OR RSI_50 ====================
    def strategy_ema20_or_rsi50(self, df):
        """
        전략 5: EMA_20 OR RSI_50
        매수: Close > EMA20 또는 RSI > 50
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['EMA20'] = self.calculate_ema(df['Close'], 20)
        df['RSI'] = self.calculate_rsi(df['Close'], period=14)

        # 신호 생성 (OR 조건)
        condition1 = df['Close'] > df['EMA20']
        condition2 = df['RSI'] > 50
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 6: SMA_30 OR MACD_SIGNAL ====================
    def strategy_sma30_or_macdsignal(self, df):
        """
        전략 6: SMA_30 OR MACD_SIGNAL
        매수: Close > SMA30 또는 MACD > Signal
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['SMA30'] = df['Close'].rolling(window=30).mean()
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])

        # 신호 생성 (OR 조건)
        condition1 = df['Close'] > df['SMA30']
        condition2 = df['MACD'] > df['MACD_Signal']
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 7: DONCHIAN_20 OR RSI_65 ====================
    def strategy_donchian20_or_rsi65(self, df):
        """
        전략 7: DONCHIAN_20 OR RSI_65
        매수: Close > 돈치안채널20 중간선 또는 RSI > 65
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['Donchian_Upper'], df['Donchian_Middle'], df['Donchian_Lower'] = self.calculate_donchian(df, period=20)
        df['RSI'] = self.calculate_rsi(df['Close'], period=14)

        # 신호 생성 (OR 조건)
        condition1 = df['Close'] > df['Donchian_Middle']
        condition2 = df['RSI'] > 65
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 8: EMA_50 OR MOMENTUM ====================
    def strategy_ema50_or_momentum(self, df):
        """
        전략 8: EMA_50 OR MOMENTUM
        매수: Close > EMA50 또는 Momentum(10일) > 0
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['EMA50'] = self.calculate_ema(df['Close'], 50)
        df['Momentum'] = self.calculate_momentum(df['Close'], period=10)

        # 신호 생성 (OR 조건)
        condition1 = df['Close'] > df['EMA50']
        condition2 = df['Momentum'] > 0
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 9: SMA_100 OR RSI_70 ====================
    def strategy_sma100_or_rsi70(self, df):
        """
        전략 9: SMA_100 OR RSI_70
        매수: Close > SMA100 또는 RSI > 70
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['SMA100'] = df['Close'].rolling(window=100).mean()
        df['RSI'] = self.calculate_rsi(df['Close'], period=14)

        # 신호 생성 (OR 조건)
        condition1 = df['Close'] > df['SMA100']
        condition2 = df['RSI'] > 70
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 10: EMA_30 OR TURTLE ====================
    def strategy_ema30_or_turtle(self, df):
        """
        전략 10: EMA_30 OR TURTLE
        매수: Close > EMA30 또는 High가 20일 최고가 돌파
        매도: 위 조건이 아닐 때
        """
        df = df.copy()

        # 지표 계산
        df['EMA30'] = self.calculate_ema(df['Close'], 30)
        df['Turtle_High'] = df['High'].rolling(window=20).max().shift(1)

        # 신호 생성 (OR 조건)
        condition1 = df['Close'] > df['EMA30']
        condition2 = df['High'] > df['Turtle_High']
        df['signal'] = (condition1 | condition2).astype(int)

        # 포지션 계산
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 실행 ====================
    def run_all_strategies(self):
        """모든 전략 실행"""
        strategies = {
            '1. SMA20 OR RSI60': self.strategy_sma20_or_rsi60,
            '2. EMA12 OR MACD': self.strategy_ema12_or_macd,
            '3. SMA50 OR ADX25': self.strategy_sma50_or_adx25,
            '4. RSI55 OR BB_UPPER': self.strategy_rsi55_or_bbupper,
            '5. EMA20 OR RSI50': self.strategy_ema20_or_rsi50,
            '6. SMA30 OR MACD_SIGNAL': self.strategy_sma30_or_macdsignal,
            '7. DONCHIAN20 OR RSI65': self.strategy_donchian20_or_rsi65,
            '8. EMA50 OR MOMENTUM': self.strategy_ema50_or_momentum,
            '9. SMA100 OR RSI70': self.strategy_sma100_or_rsi70,
            '10. EMA30 OR TURTLE': self.strategy_ema30_or_turtle
        }

        print("\n" + "="*80)
        print("Running all Bitcoin trend-following strategies...")
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
        """모든 전략 성과 지표 계산"""
        metrics_list = []

        for strategy_name, result_df in self.strategy_results.items():
            returns = result_df['returns']
            metrics = self.calculate_metrics(returns, strategy_name)
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================
    def plot_comparison(self, metrics_df, save_path='bitcoin_trend_following_strategies.png'):
        """전략 비교 시각화"""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name, result_df in self.strategy_results.items():
            ax1.plot(result_df.index, result_df['cumulative'],
                    label=strategy_name, linewidth=2, alpha=0.7)

        ax1.set_title(f'Bitcoin Trend Following Strategies - Cumulative Returns Comparison\n{self.symbol}',
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
        ax2.barh(range(len(sorted_df)), sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_df)))
        ax2.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['CAGR (%)']]
        ax3.barh(range(len(sorted_df)), sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(sorted_df)))
        ax3.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        ax4.barh(range(len(sorted_df)), sorted_df['MDD (%)'], color='crimson', alpha=0.7)
        ax4.set_yticks(range(len(sorted_df)))
        ax4.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Sharpe Ratio']]
        ax5.barh(range(len(sorted_df)), sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax5.set_yticks(range(len(sorted_df)))
        ax5.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Return vs Risk 산점도
        ax6 = fig.add_subplot(gs[2, 1])
        scatter = ax6.scatter(metrics_df['MDD (%)'], metrics_df['CAGR (%)'],
                   s=300, alpha=0.6, c=metrics_df['Sharpe Ratio'], cmap='RdYlGn')
        for idx, row in metrics_df.iterrows():
            ax6.annotate(row['Strategy'].split('.')[0],
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=9, ha='center', va='bottom')
        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax6, label='Sharpe Ratio')

        # 7. Profit Factor 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.copy()
        sorted_df = sorted_df[sorted_df['Profit Factor'] != np.inf]
        if len(sorted_df) > 0:
            sorted_df = sorted_df.sort_values('Profit Factor', ascending=True)
            colors = ['green' if x > 1 else 'red' for x in sorted_df['Profit Factor']]
            ax7.barh(range(len(sorted_df)), sorted_df['Profit Factor'], color=colors, alpha=0.7)
            ax7.set_yticks(range(len(sorted_df)))
            ax7.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax7.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax7.set_xlabel('Profit Factor', fontsize=11)
        ax7.set_title('Profit Factor Comparison', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8. 승률 비교
        ax8 = fig.add_subplot(gs[3, 0])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=True)
        colors = ['green' if x > 50 else 'red' for x in sorted_df['Win Rate (%)']]
        ax8.barh(range(len(sorted_df)), sorted_df['Win Rate (%)'], color=colors, alpha=0.7)
        ax8.set_yticks(range(len(sorted_df)))
        ax8.set_yticklabels(sorted_df['Strategy'], fontsize=9)
        ax8.axvline(x=50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax8.set_xlabel('Win Rate (%)', fontsize=11)
        ax8.set_title('Win Rate Comparison', fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')

        # 9. 드로우다운 비교
        ax9 = fig.add_subplot(gs[3, 1:])
        for strategy_name, result_df in self.strategy_results.items():
            cumulative = result_df['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax9.plot(drawdown.index, drawdown, label=strategy_name, linewidth=1.5, alpha=0.7)

        ax9.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax9.set_ylabel('Drawdown (%)', fontsize=12)
        ax9.set_xlabel('Date', fontsize=12)
        ax9.legend(loc='lower right', fontsize=9, ncol=2)
        ax9.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*140)
        print(f"{'비트코인 추세추종 전략 성과 비교':^140}")
        print("="*140)
        print(f"\n종목: {self.symbol}")
        print(f"기간: {self.start_date} ~ {self.end_date}")
        print(f"슬리피지: {self.slippage*100}%")
        print(f"\n각 전략은 2개 지표를 OR 조건으로 사용:")
        print("  - 조건 충족 시: 매수/보유")
        print("  - 조건 불충족 시: 매도/현금 보유")

        print("\n" + "-"*140)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 140)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(metrics_df.to_string(index=False))
        print("\n" + "="*140 + "\n")

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
    print("비트코인 추세추종 전략 10종 백테스팅 시작")
    print("="*80)

    # 백테스트 실행
    btc_strategies = BitcoinTrendFollowingStrategies(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = btc_strategies.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('bitcoin_trend_following_metrics.csv', index=False)
    print("Metrics saved to bitcoin_trend_following_metrics.csv")

    # 각 전략 상세 결과 저장
    for strategy_name, result_df in btc_strategies.strategy_results.items():
        filename = f"strategy_{strategy_name.split('.')[0].strip()}.csv"
        result_df.to_csv(filename)
        print(f"Strategy details saved to {filename}")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
