"""
BTC 새로운 10가지 전략 백테스트 비교 분석

비트코인(BTC)에 대해 새로운 10가지 거래 전략을 적용하여 성과 비교:
1. Keltner Channel - ATR 기반 채널 돌파
2. ADX Trend - ADX로 트렌드 강도 측정
3. Parabolic SAR - 추세 추종 및 역전 감지
4. Ichimoku Cloud - 일목균형표 전략
5. Williams %R - 모멘텀 오실레이터
6. Stochastic - 스토캐스틱 교차
7. Triple MA - 3중 이동평균
8. Donchian Channel - 채널 돌파
9. SuperTrend - ATR 기반 트렌드
10. Golden/Death Cross - 50/200일 MA
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


class BTCNewStrategiesComparison:
    """BTC 새로운 10가지 전략 비교 클래스"""

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

    # ==================== Helper Functions ====================
    def calculate_atr(self, df, period=14):
        """ATR (Average True Range) 계산"""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_adx(self, df, period=14):
        """ADX (Average Directional Index) 계산"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        # +DM, -DM 계산
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # ATR 계산
        atr = self.calculate_atr(df, period)

        # +DI, -DI 계산
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # DX, ADX 계산
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    # ==================== 전략 1: Keltner Channel ====================
    def strategy_keltner_channel(self, df, ema_period=20, atr_period=10, multiplier=2):
        """
        Keltner Channel 전략
        - 가격이 상단 채널 돌파: 매수
        - 가격이 하단 채널 하향 돌파: 매도
        """
        df = df.copy()

        # EMA 계산
        df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

        # ATR 계산
        df['ATR'] = self.calculate_atr(df, atr_period)

        # Keltner Channel
        df['KC_upper'] = df['EMA'] + (multiplier * df['ATR'])
        df['KC_lower'] = df['EMA'] - (multiplier * df['ATR'])

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 상단 채널 돌파 시 매수
            if df.iloc[i]['Close'] > df.iloc[i]['KC_upper'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 하단 채널 하향 돌파 시 매도
            elif df.iloc[i]['Close'] < df.iloc[i]['KC_lower'] and df.iloc[i-1]['position'] == 1:
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

    # ==================== 전략 2: ADX Trend ====================
    def strategy_adx_trend(self, df, adx_period=14, adx_threshold=25):
        """
        ADX Trend 전략
        - ADX > threshold AND +DI > -DI: 매수
        - ADX < threshold OR +DI < -DI: 매도
        """
        df = df.copy()

        # ADX, +DI, -DI 계산
        df['ADX'], df['plus_DI'], df['minus_DI'] = self.calculate_adx(df, adx_period)

        # 포지션 계산
        df['position'] = np.where(
            (df['ADX'] > adx_threshold) & (df['plus_DI'] > df['minus_DI']), 1, 0
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

    # ==================== 전략 3: Parabolic SAR ====================
    def calculate_parabolic_sar(self, df, af_start=0.02, af_increment=0.02, af_max=0.2):
        """Parabolic SAR 계산"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        sar = np.zeros(len(df))
        ep = np.zeros(len(df))
        af = np.zeros(len(df))
        trend = np.zeros(len(df))  # 1: uptrend, -1: downtrend

        # 초기값 설정
        sar[0] = low[0]
        ep[0] = high[0]
        af[0] = af_start
        trend[0] = 1

        for i in range(1, len(df)):
            # SAR 계산
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])

            # 상승 추세
            if trend[i-1] == 1:
                # 추세 전환 확인
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            # 하락 추세
            else:
                # 추세 전환 확인
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]

        return sar, trend

    def strategy_parabolic_sar(self, df):
        """
        Parabolic SAR 전략
        - 가격 > SAR (상승 추세): 매수
        - 가격 < SAR (하락 추세): 매도
        """
        df = df.copy()

        # Parabolic SAR 계산
        sar, trend = self.calculate_parabolic_sar(df)
        df['SAR'] = sar
        df['trend'] = trend

        # 포지션 계산 (상승 추세일 때 1)
        df['position'] = np.where(df['trend'] == 1, 1, 0)

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

    # ==================== 전략 4: Ichimoku Cloud ====================
    def strategy_ichimoku_cloud(self, df, tenkan=9, kijun=26, senkou_b=52):
        """
        Ichimoku Cloud 전략
        - 전환선 > 기준선 AND 가격 > 구름: 매수
        - 전환선 < 기준선 OR 가격 < 구름: 매도
        """
        df = df.copy()

        # 전환선 (Tenkan-sen): 9일 최고가+최저가 / 2
        high_9 = df['High'].rolling(window=tenkan).max()
        low_9 = df['Low'].rolling(window=tenkan).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2

        # 기준선 (Kijun-sen): 26일 최고가+최저가 / 2
        high_26 = df['High'].rolling(window=kijun).max()
        low_26 = df['Low'].rolling(window=kijun).min()
        df['kijun_sen'] = (high_26 + low_26) / 2

        # 선행스팬A (Senkou Span A): (전환선 + 기준선) / 2, 26일 선행
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun)

        # 선행스팬B (Senkou Span B): 52일 최고가+최저가 / 2, 26일 선행
        high_52 = df['High'].rolling(window=senkou_b).max()
        low_52 = df['Low'].rolling(window=senkou_b).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(kijun)

        # 구름 (Cloud)
        df['cloud_top'] = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        df['cloud_bottom'] = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)

        # 포지션 계산
        df['position'] = np.where(
            (df['tenkan_sen'] > df['kijun_sen']) & (df['Close'] > df['cloud_top']), 1, 0
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

    # ==================== 전략 5: Williams %R ====================
    def calculate_williams_r(self, df, period=14):
        """Williams %R 계산"""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()

        williams_r = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)

        return williams_r

    def strategy_williams_r(self, df, period=14, oversold=-20, overbought=-80):
        """
        Williams %R 전략
        - Williams %R > -20 (과매수 탈출): 매수
        - Williams %R < -80 (과매도 진입): 매도
        """
        df = df.copy()

        # Williams %R 계산
        df['williams_r'] = self.calculate_williams_r(df, period)

        # 포지션 계산
        df['position'] = np.where(df['williams_r'] > overbought, 1, 0)

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

    # ==================== 전략 6: Stochastic ====================
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """Stochastic Oscillator 계산"""
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()

        k = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()

        return k, d

    def strategy_stochastic(self, df, k_period=14, d_period=3):
        """
        Stochastic 교차 전략
        - %K > %D: 매수
        - %K < %D: 매도
        """
        df = df.copy()

        # Stochastic 계산
        df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df, k_period, d_period)

        # 포지션 계산
        df['position'] = np.where(df['stoch_k'] > df['stoch_d'], 1, 0)

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

    # ==================== 전략 7: Triple MA ====================
    def strategy_triple_ma(self, df, short=5, mid=20, long=60):
        """
        Triple MA 전략
        - 단기 > 중기 > 장기: 매수
        - 그 외: 매도
        """
        df = df.copy()

        # 이동평균 계산
        df['MA_short'] = df['Close'].rolling(window=short).mean()
        df['MA_mid'] = df['Close'].rolling(window=mid).mean()
        df['MA_long'] = df['Close'].rolling(window=long).mean()

        # 포지션 계산
        df['position'] = np.where(
            (df['MA_short'] > df['MA_mid']) & (df['MA_mid'] > df['MA_long']), 1, 0
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

    # ==================== 전략 8: Donchian Channel ====================
    def strategy_donchian_channel(self, df, entry_period=55, exit_period=20):
        """
        Donchian Channel 전략
        - N일 최고가 돌파: 매수
        - M일 최저가 하향 돌파: 매도
        """
        df = df.copy()

        # Donchian Channel
        df['upper_band'] = df['High'].rolling(window=entry_period).max()
        df['lower_band'] = df['Low'].rolling(window=exit_period).min()

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 상단 밴드 돌파 시 매수
            if df.iloc[i]['Close'] > df.iloc[i-1]['upper_band'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 하단 밴드 하향 돌파 시 매도
            elif df.iloc[i]['Close'] < df.iloc[i-1]['lower_band'] and df.iloc[i-1]['position'] == 1:
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

    # ==================== 전략 9: SuperTrend ====================
    def strategy_supertrend(self, df, atr_period=10, multiplier=3):
        """
        SuperTrend 전략
        - 가격 > SuperTrend: 매수
        - 가격 < SuperTrend: 매도
        """
        df = df.copy()

        # ATR 계산
        df['ATR'] = self.calculate_atr(df, atr_period)

        # Basic Bands
        hl_avg = (df['High'] + df['Low']) / 2
        df['basic_ub'] = hl_avg + (multiplier * df['ATR'])
        df['basic_lb'] = hl_avg - (multiplier * df['ATR'])

        # Final Bands
        df['final_ub'] = 0.0
        df['final_lb'] = 0.0

        for i in range(atr_period, len(df)):
            # Upper Band
            if i == atr_period:
                df.iloc[i, df.columns.get_loc('final_ub')] = df.iloc[i]['basic_ub']
            else:
                if df.iloc[i]['basic_ub'] < df.iloc[i-1]['final_ub'] or df.iloc[i-1]['Close'] > df.iloc[i-1]['final_ub']:
                    df.iloc[i, df.columns.get_loc('final_ub')] = df.iloc[i]['basic_ub']
                else:
                    df.iloc[i, df.columns.get_loc('final_ub')] = df.iloc[i-1]['final_ub']

            # Lower Band
            if i == atr_period:
                df.iloc[i, df.columns.get_loc('final_lb')] = df.iloc[i]['basic_lb']
            else:
                if df.iloc[i]['basic_lb'] > df.iloc[i-1]['final_lb'] or df.iloc[i-1]['Close'] < df.iloc[i-1]['final_lb']:
                    df.iloc[i, df.columns.get_loc('final_lb')] = df.iloc[i]['basic_lb']
                else:
                    df.iloc[i, df.columns.get_loc('final_lb')] = df.iloc[i-1]['final_lb']

        # SuperTrend
        df['supertrend'] = 0.0
        for i in range(atr_period, len(df)):
            if i == atr_period:
                df.iloc[i, df.columns.get_loc('supertrend')] = df.iloc[i]['final_ub']
            else:
                if df.iloc[i-1]['supertrend'] == df.iloc[i-1]['final_ub']:
                    if df.iloc[i]['Close'] <= df.iloc[i]['final_ub']:
                        df.iloc[i, df.columns.get_loc('supertrend')] = df.iloc[i]['final_ub']
                    else:
                        df.iloc[i, df.columns.get_loc('supertrend')] = df.iloc[i]['final_lb']
                else:
                    if df.iloc[i]['Close'] >= df.iloc[i]['final_lb']:
                        df.iloc[i, df.columns.get_loc('supertrend')] = df.iloc[i]['final_lb']
                    else:
                        df.iloc[i, df.columns.get_loc('supertrend')] = df.iloc[i]['final_ub']

        # 포지션 계산
        df['position'] = np.where(df['Close'] > df['supertrend'], 1, 0)

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

    # ==================== 전략 10: Golden/Death Cross ====================
    def strategy_golden_death_cross(self, df, short_period=50, long_period=200):
        """
        Golden/Death Cross 전략
        - 50일 MA > 200일 MA: 매수 (Golden Cross)
        - 50일 MA < 200일 MA: 매도 (Death Cross)
        """
        df = df.copy()

        # 이동평균 계산
        df['MA_50'] = df['Close'].rolling(window=short_period).mean()
        df['MA_200'] = df['Close'].rolling(window=long_period).mean()

        # 포지션 계산
        df['position'] = np.where(df['MA_50'] > df['MA_200'], 1, 0)

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
            '1. Keltner Channel': lambda df: self.strategy_keltner_channel(df, ema_period=20, atr_period=10, multiplier=2),
            '2. ADX Trend': lambda df: self.strategy_adx_trend(df, adx_period=14, adx_threshold=25),
            '3. Parabolic SAR': lambda df: self.strategy_parabolic_sar(df),
            '4. Ichimoku Cloud': lambda df: self.strategy_ichimoku_cloud(df, tenkan=9, kijun=26, senkou_b=52),
            '5. Williams %R': lambda df: self.strategy_williams_r(df, period=14),
            '6. Stochastic': lambda df: self.strategy_stochastic(df, k_period=14, d_period=3),
            '7. Triple MA': lambda df: self.strategy_triple_ma(df, short=5, mid=20, long=60),
            '8. Donchian Channel': lambda df: self.strategy_donchian_channel(df, entry_period=55, exit_period=20),
            '9. SuperTrend': lambda df: self.strategy_supertrend(df, atr_period=10, multiplier=3),
            '10. Golden/Death Cross': lambda df: self.strategy_golden_death_cross(df, short_period=50, long_period=200)
        }

        print("\n" + "="*80)
        print("Running all 10 NEW strategies...")
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
    def plot_comparison(self, metrics_df, save_path='btc_10_new_strategies_comparison.png'):
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

        ax1.set_title(f'BTC New 10 Strategies Comparison - Cumulative Returns\n'
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

        heatmap_data = metrics_df[['Strategy', 'Total Return (%)', 'CAGR (%)',
                                   'MDD (%)', 'Sharpe Ratio', 'Win Rate (%)']].copy()
        heatmap_data = heatmap_data.set_index('Strategy')

        # MDD는 절대값으로 변환
        heatmap_data['MDD (%)'] = heatmap_data['MDD (%)'].abs()

        # 정규화
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
        print(f"{'BTC 새로운 10가지 전략 백테스트 성과 비교':^150}")
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
    print("BTC 새로운 10가지 전략 백테스트 비교 분석 시작")
    print("="*80)

    # 백테스트 실행
    comparison = BTCNewStrategiesComparison(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = comparison.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('btc_10_new_strategies_metrics.csv', index=False)
    print("Metrics saved to btc_10_new_strategies_metrics.csv")

    # 각 전략별 상세 결과 저장
    for strategy_name in comparison.strategy_results.keys():
        filename = f"new_strategy_{strategy_name.replace(' ', '_').replace('.', '').replace('/', '_').lower()}.csv"
        comparison.strategy_results[strategy_name][['Close', 'position', 'returns', 'cumulative']].to_csv(filename)
        print(f"Strategy details saved to {filename}")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
