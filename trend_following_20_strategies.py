"""
추세추종 전략 20개 백테스트

20가지 다양한 추세추종 전략을 BTC, ETH, ADA, XRP에 적용하여 성과 비교:

1. Turtle Trading (20/10) - 터틀 트레이딩
2. Turtle Trading (55/20) - 장기 터틀 트레이딩
3. SMA 30 Crossover - 단순이동평균 30일
4. SMA 50 Crossover - 단순이동평균 50일
5. SMA 200 Crossover - 장기 단순이동평균
6. Dual SMA (50/200) - 골든크로스/데드크로스
7. Triple SMA (20/50/200) - 삼중 이동평균
8. EMA 21 Crossover - 지수이동평균
9. Dual EMA (12/26) - 이중 지수이동평균
10. MACD Crossover - MACD 크로스오버
11. Bollinger Bands Breakout - 볼린저 밴드 돌파
12. Donchian Channel (20) - 돈치안 채널
13. Donchian Channel (55) - 장기 돈치안 채널
14. RSI 55 - RSI 55 전략
15. RSI 70/30 - RSI 과매수/과매도
16. ADX Trend Strength - ADX 추세 강도
17. Parabolic SAR - 파라볼릭 SAR
18. Keltner Channel - 켈트너 채널
19. Price Channel Breakout - 가격 채널 돌파
20. Ichimoku Cloud - 일목균형표

각 전략은 4개 종목에 25%씩 동일 비중 투자
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


class TrendFollowing20Strategies:
    """20가지 추세추종 전략 백테스트 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = {}
        self.strategy_results = {}
        self.portfolio_results = {}

    def load_data(self):
        """모든 종목 데이터 로드"""
        print("="*80)
        print("Loading data for all symbols...")
        print("="*80)

        for symbol in self.symbols:
            file_path = f'chart_day/{symbol}.parquet'
            print(f"\nLoading {symbol} from {file_path}...")
            df = pd.read_parquet(file_path)

            # 컬럼명 변경 (소문자 -> 대문자)
            df.columns = [col.capitalize() for col in df.columns]

            # 날짜 필터링
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            self.data[symbol] = df
            print(f"  Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

        print("\n" + "="*80)
        print("Data loading completed!")
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

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def calculate_adx(self, high, low, close, period=14):
        """ADX 계산"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=close.index).rolling(window=period).mean()
        minus_dm = pd.Series(minus_dm, index=close.index).rolling(window=period).mean()

        plus_di = 100 * (plus_dm / atr)
        minus_di = 100 * (minus_dm / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    def calculate_parabolic_sar(self, high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
        """Parabolic SAR 계산 (간소화 버전)"""
        sar = close.copy()
        trend = pd.Series(1, index=close.index)  # 1 = uptrend, -1 = downtrend

        # 초기값
        sar.iloc[0] = low.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > sar.iloc[i-1]:
                trend.iloc[i] = 1
                sar.iloc[i] = sar.iloc[i-1] + af_start * (high.iloc[:i].max() - sar.iloc[i-1])
            else:
                trend.iloc[i] = -1
                sar.iloc[i] = sar.iloc[i-1] + af_start * (low.iloc[:i].min() - sar.iloc[i-1])

        return sar, trend

    def calculate_keltner_channel(self, high, low, close, period=20, atr_multiplier=2):
        """켈트너 채널 계산"""
        # Middle Line (EMA)
        middle = close.ewm(span=period, adjust=False).mean()

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        # Upper and Lower bands
        upper = middle + (atr * atr_multiplier)
        lower = middle - (atr * atr_multiplier)

        return upper, middle, lower

    def calculate_ichimoku(self, high, low, close):
        """일목균형표 계산"""
        # Tenkan-sen (Conversion Line): 9-period
        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line): 26-period
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        # Senkou Span B (Leading Span B): 52-period
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

    # ==================== 전략 1: Turtle Trading (20/10) ====================

    def strategy_turtle_20_10(self, df):
        """터틀 트레이딩 (20일 돌파, 10일 청산)"""
        df = df.copy()

        df['entry_high'] = df['High'].rolling(window=20).max().shift(1)
        df['exit_low'] = df['Low'].rolling(window=10).min().shift(1)

        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            if df.iloc[i]['High'] > df.iloc[i]['entry_high'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1
            elif df.iloc[i]['Low'] < df.iloc[i]['exit_low'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        df = self._calculate_returns(df)
        return df

    # ==================== 전략 2: Turtle Trading (55/20) ====================

    def strategy_turtle_55_20(self, df):
        """장기 터틀 트레이딩 (55일 돌파, 20일 청산)"""
        df = df.copy()

        df['entry_high'] = df['High'].rolling(window=55).max().shift(1)
        df['exit_low'] = df['Low'].rolling(window=20).min().shift(1)

        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            if df.iloc[i]['High'] > df.iloc[i]['entry_high'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1
            elif df.iloc[i]['Low'] < df.iloc[i]['exit_low'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        df = self._calculate_returns(df)
        return df

    # ==================== 전략 3-5: SMA Crossover ====================

    def strategy_sma_crossover(self, df, period):
        """SMA 크로스오버 전략"""
        df = df.copy()

        df['SMA'] = df['Close'].rolling(window=period).mean()
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)
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

    def strategy_sma_30(self, df):
        """SMA 30 전략"""
        return self.strategy_sma_crossover(df, 30)

    def strategy_sma_50(self, df):
        """SMA 50 전략"""
        return self.strategy_sma_crossover(df, 50)

    def strategy_sma_200(self, df):
        """SMA 200 전략"""
        return self.strategy_sma_crossover(df, 200)

    # ==================== 전략 6: Dual SMA (50/200) ====================

    def strategy_dual_sma(self, df):
        """골든크로스/데드크로스 전략"""
        df = df.copy()

        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        df['position'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
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

    # ==================== 전략 7: Triple SMA ====================

    def strategy_triple_sma(self, df):
        """삼중 이동평균 전략 (20/50/200)"""
        df = df.copy()

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # 모든 조건이 상승 정렬일 때만 매수
        df['position'] = np.where(
            (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200']), 1, 0
        )
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

    # ==================== 전략 8-9: EMA Crossover ====================

    def strategy_ema_21(self, df):
        """EMA 21 크로스오버 전략"""
        df = df.copy()

        df['EMA'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['position'] = np.where(df['Close'] >= df['EMA'], 1, 0)
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

    def strategy_dual_ema(self, df):
        """Dual EMA (12/26) 전략"""
        df = df.copy()

        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        df['position'] = np.where(df['EMA_12'] > df['EMA_26'], 1, 0)
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

    # ==================== 전략 10: MACD Crossover ====================

    def strategy_macd(self, df):
        """MACD 크로스오버 전략"""
        df = df.copy()

        macd, macd_signal, macd_hist = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal

        df['position'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
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

    # ==================== 전략 11: Bollinger Bands Breakout ====================

    def strategy_bollinger_bands(self, df):
        """볼린저 밴드 돌파 전략"""
        df = df.copy()

        upper, middle, lower = self.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower

        # 상단 밴드 돌파 시 매수, 중간선 하향 돌파 시 매도
        df['position'] = 0
        for i in range(1, len(df)):
            if df.iloc[i-1]['position'] == 0 and df.iloc[i]['Close'] > df.iloc[i]['BB_Upper']:
                df.iloc[i, df.columns.get_loc('position')] = 1
            elif df.iloc[i-1]['position'] == 1 and df.iloc[i]['Close'] < df.iloc[i]['BB_Middle']:
                df.iloc[i, df.columns.get_loc('position')] = 0
            else:
                df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1]['position']

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

    # ==================== 전략 12-13: Donchian Channel ====================

    def strategy_donchian_20(self, df):
        """돈치안 채널 (20일) 전략"""
        df = df.copy()

        df['DC_Upper'] = df['High'].rolling(window=20).max()
        df['DC_Lower'] = df['Low'].rolling(window=20).min()

        df['position'] = np.where(df['Close'] >= df['DC_Upper'].shift(1), 1, 0)
        df['position'] = np.where(df['Close'] <= df['DC_Lower'].shift(1), 0, df['position'])

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

    def strategy_donchian_55(self, df):
        """돈치안 채널 (55일) 전략"""
        df = df.copy()

        df['DC_Upper'] = df['High'].rolling(window=55).max()
        df['DC_Lower'] = df['Low'].rolling(window=55).min()

        df['position'] = np.where(df['Close'] >= df['DC_Upper'].shift(1), 1, 0)
        df['position'] = np.where(df['Close'] <= df['DC_Lower'].shift(1), 0, df['position'])

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

    # ==================== 전략 14-15: RSI ====================

    def strategy_rsi_55(self, df):
        """RSI 55 전략"""
        df = df.copy()

        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['position'] = np.where(df['RSI'] >= 55, 1, 0)
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

    def strategy_rsi_70_30(self, df):
        """RSI 70/30 전략"""
        df = df.copy()

        df['RSI'] = self.calculate_rsi(df['Close'], 14)

        # RSI > 70: 매수, RSI < 30: 매도
        df['position'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['RSI'] > 70:
                df.iloc[i, df.columns.get_loc('position')] = 1
            elif df.iloc[i]['RSI'] < 30:
                df.iloc[i, df.columns.get_loc('position')] = 0
            else:
                df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1]['position']

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

    # ==================== 전략 16: ADX Trend Strength ====================

    def strategy_adx(self, df):
        """ADX 추세 강도 전략"""
        df = df.copy()

        adx, plus_di, minus_di = self.calculate_adx(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di

        # ADX > 25이고 +DI > -DI일 때 매수
        df['position'] = np.where((df['ADX'] > 25) & (df['Plus_DI'] > df['Minus_DI']), 1, 0)
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

    # ==================== 전략 17: Parabolic SAR ====================

    def strategy_parabolic_sar(self, df):
        """파라볼릭 SAR 전략"""
        df = df.copy()

        sar, trend = self.calculate_parabolic_sar(df['High'], df['Low'], df['Close'])
        df['SAR'] = sar

        df['position'] = np.where(df['Close'] > df['SAR'], 1, 0)
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

    # ==================== 전략 18: Keltner Channel ====================

    def strategy_keltner_channel(self, df):
        """켈트너 채널 전략"""
        df = df.copy()

        upper, middle, lower = self.calculate_keltner_channel(
            df['High'], df['Low'], df['Close']
        )
        df['KC_Upper'] = upper
        df['KC_Middle'] = middle
        df['KC_Lower'] = lower

        # 상단 채널 돌파 시 매수, 하단 채널 돌파 시 매도
        df['position'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['Close'] > df.iloc[i]['KC_Upper']:
                df.iloc[i, df.columns.get_loc('position')] = 1
            elif df.iloc[i]['Close'] < df.iloc[i]['KC_Lower']:
                df.iloc[i, df.columns.get_loc('position')] = 0
            else:
                df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1]['position']

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

    # ==================== 전략 19: Price Channel Breakout ====================

    def strategy_price_channel(self, df):
        """가격 채널 돌파 전략 (30일)"""
        df = df.copy()

        df['PC_Upper'] = df['High'].rolling(window=30).max()
        df['PC_Lower'] = df['Low'].rolling(window=30).min()
        df['PC_Middle'] = (df['PC_Upper'] + df['PC_Lower']) / 2

        df['position'] = np.where(df['Close'] > df['PC_Middle'].shift(1), 1, 0)
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

    # ==================== 전략 20: Ichimoku Cloud ====================

    def strategy_ichimoku(self, df):
        """일목균형표 전략"""
        df = df.copy()

        tenkan, kijun, senkou_a, senkou_b = self.calculate_ichimoku(
            df['High'], df['Low'], df['Close']
        )
        df['Tenkan'] = tenkan
        df['Kijun'] = kijun
        df['Senkou_A'] = senkou_a
        df['Senkou_B'] = senkou_b

        # 전환선 > 기준선이고 가격이 구름 위에 있을 때 매수
        df['position'] = np.where(
            (df['Tenkan'] > df['Kijun']) &
            (df['Close'] > df['Senkou_A']) &
            (df['Close'] > df['Senkou_B']),
            1, 0
        )
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

    # ==================== 보조 함수: 수익률 계산 ====================

    def _calculate_returns(self, df):
        """포지션 기반 수익률 계산 (터틀 전략용)"""
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                # 매수
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['Close'] * (1 + self.slippage)
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                # 매도
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                sell_price = df.iloc[i]['Close'] * (1 - self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)
            elif df.iloc[i]['position'] == 1:
                # 포지션 유지
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 실행 ====================

    def run_all_strategies(self):
        """모든 20개 전략을 모든 종목에 대해 실행"""
        strategies = {
            '01_Turtle_20_10': self.strategy_turtle_20_10,
            '02_Turtle_55_20': self.strategy_turtle_55_20,
            '03_SMA_30': self.strategy_sma_30,
            '04_SMA_50': self.strategy_sma_50,
            '05_SMA_200': self.strategy_sma_200,
            '06_Dual_SMA_50_200': self.strategy_dual_sma,
            '07_Triple_SMA': self.strategy_triple_sma,
            '08_EMA_21': self.strategy_ema_21,
            '09_Dual_EMA_12_26': self.strategy_dual_ema,
            '10_MACD': self.strategy_macd,
            '11_Bollinger_Bands': self.strategy_bollinger_bands,
            '12_Donchian_20': self.strategy_donchian_20,
            '13_Donchian_55': self.strategy_donchian_55,
            '14_RSI_55': self.strategy_rsi_55,
            '15_RSI_70_30': self.strategy_rsi_70_30,
            '16_ADX': self.strategy_adx,
            '17_Parabolic_SAR': self.strategy_parabolic_sar,
            '18_Keltner_Channel': self.strategy_keltner_channel,
            '19_Price_Channel': self.strategy_price_channel,
            '20_Ichimoku': self.strategy_ichimoku,
        }

        print("\n" + "="*80)
        print(f"Running all {len(strategies)} strategies for all symbols...")
        print("="*80 + "\n")

        for idx, (strategy_name, strategy_func) in enumerate(strategies.items(), 1):
            print(f"\n>>> [{idx}/{len(strategies)}] Running {strategy_name}...")
            self.strategy_results[strategy_name] = {}

            for symbol in self.symbols:
                try:
                    df = self.data[symbol].copy()
                    result = strategy_func(df)
                    self.strategy_results[strategy_name][symbol] = result
                    print(f"  ✓ {symbol}")
                except Exception as e:
                    print(f"  ✗ {symbol} - Error: {str(e)}")

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80 + "\n")

    # ==================== 포트폴리오 구성 ====================

    def create_portfolios(self):
        """각 전략별로 동일 비중 포트폴리오 생성"""
        print("\n" + "="*80)
        print("Creating equal-weight portfolios...")
        print("="*80 + "\n")

        weight = 1.0 / len(self.symbols)

        for idx, strategy_name in enumerate(self.strategy_results.keys(), 1):
            print(f"\n>>> [{idx}/{len(self.strategy_results)}] Creating portfolio for {strategy_name}...")

            # 모든 종목의 공통 날짜 인덱스 찾기
            all_indices = [self.strategy_results[strategy_name][symbol].index
                          for symbol in self.symbols]
            common_index = all_indices[0]
            for idx_temp in all_indices[1:]:
                common_index = common_index.intersection(idx_temp)

            # 포트폴리오 수익률 계산
            portfolio_returns = pd.Series(0.0, index=common_index)

            for symbol in self.symbols:
                symbol_returns = self.strategy_results[strategy_name][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight

            # 포트폴리오 누적 수익률
            portfolio_cumulative = (1 + portfolio_returns).cumprod()

            # 결과 저장
            self.portfolio_results[strategy_name] = pd.DataFrame({
                'returns': portfolio_returns,
                'cumulative': portfolio_cumulative
            }, index=common_index)

        print("\n" + "="*80)
        print("Portfolio creation completed!")
        print("="*80 + "\n")

    # ==================== 성과 지표 계산 ====================

    def calculate_metrics(self, returns_series, name):
        """성과 지표 계산"""
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
        """모든 전략의 포트폴리오 성과 지표 계산"""
        metrics_list = []

        for strategy_name in self.portfolio_results.keys():
            returns = self.portfolio_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, strategy_name)
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================

    def plot_comparison(self, metrics_df, save_path='trend_following_20_strategies.png'):
        """20개 전략 비교 시각화"""
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)

        # 1. 누적 수익률 비교 (상위 10개)
        ax1 = fig.add_subplot(gs[0:2, :])

        # CAGR 기준 상위 10개 전략 선택
        top_10_strategies = metrics_df.nlargest(10, 'CAGR (%)')['Strategy'].tolist()

        for strategy_name in self.portfolio_results.keys():
            if strategy_name in top_10_strategies:
                cumulative = self.portfolio_results[strategy_name]['cumulative']
                ax1.plot(cumulative.index, cumulative, label=strategy_name.replace('_', ' '),
                        linewidth=2, alpha=0.8)

        ax1.set_title('Top 10 Strategies: Cumulative Returns (by CAGR)',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. CAGR 비교 (전체)
        ax2 = fig.add_subplot(gs[2, :])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['CAGR (%)']]
        bars = ax2.barh(range(len(sorted_df)), sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_df)))
        ax2.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy']], fontsize=8)
        ax2.set_xlabel('CAGR (%)', fontsize=11)
        ax2.set_title('All 20 Strategies: CAGR Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # 3. MDD 비교
        ax3 = fig.add_subplot(gs[3, 0])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        ax3.barh(range(len(sorted_df[:10])), sorted_df['MDD (%)'][:10], color='crimson', alpha=0.7)
        ax3.set_yticks(range(10))
        ax3.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy'][:10]], fontsize=8)
        ax3.set_xlabel('MDD (%)', fontsize=10)
        ax3.set_title('Top 10 Worst MDD', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Sharpe Ratio 비교
        ax4 = fig.add_subplot(gs[3, 1])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=False)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Sharpe Ratio'][:10]]
        ax4.barh(range(10), sorted_df['Sharpe Ratio'][:10], color=colors, alpha=0.7)
        ax4.set_yticks(range(10))
        ax4.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy'][:10]], fontsize=8)
        ax4.set_xlabel('Sharpe Ratio', fontsize=10)
        ax4.set_title('Top 10 Sharpe Ratio', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. Return vs Risk
        ax5 = fig.add_subplot(gs[3, 2])
        scatter = ax5.scatter(metrics_df['MDD (%)'], metrics_df['CAGR (%)'],
                   s=200, alpha=0.6, c=metrics_df['Sharpe Ratio'], cmap='RdYlGn')
        plt.colorbar(scatter, ax=ax5, label='Sharpe Ratio')

        # 상위 5개만 라벨 표시
        top_5 = metrics_df.nlargest(5, 'Sharpe Ratio')
        for idx, row in top_5.iterrows():
            ax5.annotate(row['Strategy'].replace('_', ' '),
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=7, ha='center', va='bottom')

        ax5.set_xlabel('MDD (%)', fontsize=10)
        ax5.set_ylabel('CAGR (%)', fontsize=10)
        ax5.set_title('Return vs Risk (Top 5 labeled)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 6. Total Return 비교
        ax6 = fig.add_subplot(gs[4, 0])
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=False)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)'][:10]]
        ax6.barh(range(10), sorted_df['Total Return (%)'][:10], color=colors, alpha=0.7)
        ax6.set_yticks(range(10))
        ax6.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy'][:10]], fontsize=8)
        ax6.set_xlabel('Total Return (%)', fontsize=10)
        ax6.set_title('Top 10 Total Return', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        # 7. Win Rate 비교
        ax7 = fig.add_subplot(gs[4, 1])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=False)
        ax7.barh(range(10), sorted_df['Win Rate (%)'][:10], color='steelblue', alpha=0.7)
        ax7.set_yticks(range(10))
        ax7.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy'][:10]], fontsize=8)
        ax7.set_xlabel('Win Rate (%)', fontsize=10)
        ax7.set_title('Top 10 Win Rate', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')
        ax7.axvline(x=50, color='orange', linestyle='--', linewidth=1, alpha=0.5)

        # 8. Profit Factor 비교
        ax8 = fig.add_subplot(gs[4, 2])
        sorted_df_pf = metrics_df[metrics_df['Profit Factor'] != np.inf].copy()
        if len(sorted_df_pf) > 0:
            sorted_df_pf = sorted_df_pf.sort_values('Profit Factor', ascending=False)
            colors = ['green' if x > 1 else 'red' for x in sorted_df_pf['Profit Factor'][:10]]
            ax8.barh(range(len(sorted_df_pf[:10])), sorted_df_pf['Profit Factor'][:10],
                    color=colors, alpha=0.7)
            ax8.set_yticks(range(len(sorted_df_pf[:10])))
            ax8.set_yticklabels([s.replace('_', ' ') for s in sorted_df_pf['Strategy'][:10]], fontsize=8)
        ax8.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax8.set_xlabel('Profit Factor', fontsize=10)
        ax8.set_title('Top 10 Profit Factor', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')

        # 전체 제목
        fig.suptitle('20 Trend Following Strategies Performance Comparison\n'
                    f'Period: {self.start_date} to {self.end_date} | '
                    f'Assets: {", ".join([s.split("_")[0] for s in self.symbols])} | '
                    f'Equal-Weight Portfolio (25% each)',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'20 Trend Following Strategies Performance Comparison':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"포트폴리오 구성: 각 종목 동일 비중 (25%)")
        print(f"슬리피지: {self.slippage*100}%")

        print("\n" + "-"*150)
        print(f"{'전략 성과 요약 (CAGR 순위)':^150}")
        print("-"*150)

        sorted_metrics = metrics_df.sort_values('CAGR (%)', ascending=False)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(sorted_metrics.to_string(index=False))

        print("\n" + "="*150 + "\n")

        # 상위 5개 전략 강조
        print("\n" + "🏆 "*30)
        print("TOP 5 STRATEGIES (by CAGR):")
        print("🏆 "*30 + "\n")

        for idx, row in sorted_metrics.head(5).iterrows():
            print(f"{row.name + 1}. {row['Strategy'].replace('_', ' ')}")
            print(f"   CAGR: {row['CAGR (%)']:.2f}% | Total Return: {row['Total Return (%)']:.2f}% | "
                  f"MDD: {row['MDD (%)']:.2f}% | Sharpe: {row['Sharpe Ratio']:.2f}")
            print()

    def run_analysis(self):
        """전체 분석 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 모든 전략 실행
        self.run_all_strategies()

        # 3. 포트폴리오 생성
        self.create_portfolios()

        # 4. 성과 지표 계산
        metrics_df = self.calculate_all_metrics()

        # 5. 결과 출력
        self.print_metrics_table(metrics_df)

        # 6. 시각화
        self.plot_comparison(metrics_df)

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("20 Trend Following Strategies Backtest")
    print("="*80)

    # 백테스트 실행
    backtest = TrendFollowing20Strategies(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = backtest.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('trend_following_20_strategies_results.csv', index=False)
    print("✓ Metrics saved to trend_following_20_strategies_results.csv")

    # 각 전략별 포트폴리오 상세 결과 저장
    print("\nSaving individual portfolio results...")
    for strategy_name in backtest.portfolio_results.keys():
        filename = f"portfolio_{strategy_name.lower()}.csv"
        backtest.portfolio_results[strategy_name].to_csv(filename)
        print(f"✓ {filename}")

    print("\n" + "="*80)
    print("백테스트 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
