"""
비트코인 돌파신호 기반 백테스트 전략 30개
단일 지표 전략 15개 + 조합 전략 15개

백테스팅 체크리스트 준수:
- ✅ shift(1) 적용으로 Look-ahead Bias 방지
- ✅ 슬리피지 0.2% 적용
- ✅ 현실적인 체결 가격 (종가 기준)
- ✅ 수수료 반영
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


class BitcoinBreakoutStrategies:
    """비트코인 돌파신호 기반 전략 30개 백테스트"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01',
                 end_date=None, slippage=0.002, fee=0.001):
        """
        Args:
            symbol: 종목 (default: BTC_KRW)
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
            fee: 거래 수수료 (default: 0.1%)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.fee = fee
        self.data = None
        self.strategy_results = {}

    def load_data(self):
        """비트코인 데이터 로드"""
        print("=" * 80)
        print(f"Loading {self.symbol} data...")
        print("=" * 80)

        file_path = f'chart_day/{self.symbol}.parquet'
        df = pd.read_parquet(file_path)

        # 컬럼명 변경
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df
        print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        print("=" * 80 + "\n")

    # ==================== 보조 지표 계산 ====================

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

    def calculate_atr(self, df, period=14):
        """ATR (Average True Range) 계산"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_donchian_channel(self, df, period=20):
        """도니안 채널 계산"""
        upper = df['High'].rolling(window=period).max()
        lower = df['Low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        return upper, middle, lower

    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """스토캐스틱 계산"""
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()

        k = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d

    def calculate_cci(self, df, period=20):
        """CCI (Commodity Channel Index) 계산"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci

    # ==================== 수익률 계산 헬퍼 ====================

    def calculate_returns(self, df, signal_col='signal'):
        """
        신호에 따른 수익률 계산
        - shift(1) 적용으로 Look-ahead Bias 방지
        - 슬리피지 및 수수료 적용
        """
        df = df.copy()

        # 포지션 변화 감지
        df['position_change'] = df[signal_col].diff()

        # 일일 가격 수익률
        df['daily_price_return'] = df['Close'].pct_change()

        # 전략 수익률 (신호를 shift(1)하여 다음날 체결)
        df['returns'] = df[signal_col].shift(1) * df['daily_price_return']

        # 거래 비용 (슬리피지 + 수수료)
        transaction_cost = self.slippage + self.fee
        cost = pd.Series(0.0, index=df.index)
        cost[df['position_change'].abs() > 0] = -transaction_cost

        df['returns'] = df['returns'] + cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ==================== 단일 지표 전략 (15개) ====================

    def strategy_01_sma_cross_10_30(self, df):
        """전략 1: SMA(10) > SMA(30) 골든크로스"""
        df = df.copy()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA30'] = df['Close'].rolling(window=30).mean()
        df['signal'] = (df['SMA10'] > df['SMA30']).astype(int)
        return self.calculate_returns(df)

    def strategy_02_sma_cross_20_60(self, df):
        """전략 2: SMA(20) > SMA(60) 골든크로스"""
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA60'] = df['Close'].rolling(window=60).mean()
        df['signal'] = (df['SMA20'] > df['SMA60']).astype(int)
        return self.calculate_returns(df)

    def strategy_03_sma_cross_50_200(self, df):
        """전략 3: SMA(50) > SMA(200) 골든크로스 (장기)"""
        df = df.copy()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['signal'] = (df['SMA50'] > df['SMA200']).astype(int)
        return self.calculate_returns(df)

    def strategy_04_price_above_sma20(self, df):
        """전략 4: 가격 > SMA(20) 돌파"""
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['signal'] = (df['Close'] > df['SMA20']).astype(int)
        return self.calculate_returns(df)

    def strategy_05_price_above_sma50(self, df):
        """전략 5: 가격 > SMA(50) 돌파"""
        df = df.copy()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['signal'] = (df['Close'] > df['SMA50']).astype(int)
        return self.calculate_returns(df)

    def strategy_06_bollinger_breakout_upper(self, df):
        """전략 6: 볼린저 밴드 상단 돌파"""
        df = df.copy()
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'], 20, 2)
        df['BB_upper'] = upper
        df['signal'] = (df['Close'] > df['BB_upper']).astype(int)
        return self.calculate_returns(df)

    def strategy_07_bollinger_middle_cross(self, df):
        """전략 7: 볼린저 밴드 중심선 돌파"""
        df = df.copy()
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'], 20, 2)
        df['BB_middle'] = middle
        df['signal'] = (df['Close'] > df['BB_middle']).astype(int)
        return self.calculate_returns(df)

    def strategy_08_donchian_breakout_20(self, df):
        """전략 8: 도니안 채널(20) 상단 돌파"""
        df = df.copy()
        upper, middle, lower = self.calculate_donchian_channel(df, 20)
        df['DC_upper'] = upper.shift(1)

        # 돌파 신호
        df['signal'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['High'] > df.iloc[i]['DC_upper'] and df.iloc[i-1]['signal'] == 0:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif df.iloc[i-1]['signal'] == 1:
                df.iloc[i, df.columns.get_loc('signal')] = 1

        return self.calculate_returns(df)

    def strategy_09_donchian_breakout_55(self, df):
        """전략 9: 도니안 채널(55) 상단 돌파 (터틀 방식)"""
        df = df.copy()
        upper, middle, lower = self.calculate_donchian_channel(df, 55)
        df['DC_upper'] = upper.shift(1)

        # 돌파 신호
        df['signal'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['High'] > df.iloc[i]['DC_upper'] and df.iloc[i-1]['signal'] == 0:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif df.iloc[i-1]['signal'] == 1:
                df.iloc[i, df.columns.get_loc('signal')] = 1

        return self.calculate_returns(df)

    def strategy_10_rsi_above_50(self, df):
        """전략 10: RSI(14) > 50 돌파"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['signal'] = (df['RSI'] > 50).astype(int)
        return self.calculate_returns(df)

    def strategy_11_rsi_above_55(self, df):
        """전략 11: RSI(14) > 55 돌파"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['signal'] = (df['RSI'] > 55).astype(int)
        return self.calculate_returns(df)

    def strategy_12_macd_cross_signal(self, df):
        """전략 12: MACD 신호선 골든크로스"""
        df = df.copy()
        macd, macd_signal, macd_hist = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['signal'] = (df['MACD'] > df['MACD_signal']).astype(int)
        return self.calculate_returns(df)

    def strategy_13_macd_zero_cross(self, df):
        """전략 13: MACD > 0 돌파"""
        df = df.copy()
        macd, macd_signal, macd_hist = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['signal'] = (df['MACD'] > 0).astype(int)
        return self.calculate_returns(df)

    def strategy_14_high_breakout_20d(self, df):
        """전략 14: 20일 최고가 돌파"""
        df = df.copy()
        df['high_20'] = df['High'].rolling(window=20).max().shift(1)
        df['signal'] = (df['High'] > df['high_20']).astype(int)
        return self.calculate_returns(df)

    def strategy_15_high_breakout_60d(self, df):
        """전략 15: 60일 최고가 돌파"""
        df = df.copy()
        df['high_60'] = df['High'].rolling(window=60).max().shift(1)
        df['signal'] = (df['High'] > df['high_60']).astype(int)
        return self.calculate_returns(df)

    # ==================== 조합 전략 (15개) ====================

    def strategy_16_sma_rsi_combo(self, df):
        """전략 16: SMA(20) 상향 + RSI > 50"""
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['signal'] = ((df['Close'] > df['SMA20']) & (df['RSI'] > 50)).astype(int)
        return self.calculate_returns(df)

    def strategy_17_sma_macd_combo(self, df):
        """전략 17: SMA(50) 상향 + MACD 골든크로스"""
        df = df.copy()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        macd, macd_signal, _ = self.calculate_macd(df['Close'])
        df['signal'] = ((df['Close'] > df['SMA50']) & (macd > macd_signal)).astype(int)
        return self.calculate_returns(df)

    def strategy_18_bollinger_rsi_combo(self, df):
        """전략 18: 볼린저 중심선 상향 + RSI > 50"""
        df = df.copy()
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'], 20, 2)
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['signal'] = ((df['Close'] > middle) & (df['RSI'] > 50)).astype(int)
        return self.calculate_returns(df)

    def strategy_19_triple_sma(self, df):
        """전략 19: 트리플 SMA (10 > 20 > 50)"""
        df = df.copy()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['signal'] = ((df['SMA10'] > df['SMA20']) & (df['SMA20'] > df['SMA50'])).astype(int)
        return self.calculate_returns(df)

    def strategy_20_donchian_atr_combo(self, df):
        """전략 20: 도니안(20) 돌파 + ATR 필터"""
        df = df.copy()
        upper, middle, lower = self.calculate_donchian_channel(df, 20)
        df['ATR'] = self.calculate_atr(df, 14)
        df['ATR_avg'] = df['ATR'].rolling(window=20).mean()

        # ATR이 평균보다 클 때만 돌파 신호 유효
        df['DC_upper'] = upper.shift(1)
        df['high_breakout'] = (df['High'] > df['DC_upper']).astype(int)
        df['atr_filter'] = (df['ATR'] > df['ATR_avg']).astype(int)
        df['signal'] = (df['high_breakout'] & df['atr_filter']).astype(int)

        return self.calculate_returns(df)

    def strategy_21_rsi_macd_combo(self, df):
        """전략 21: RSI > 50 + MACD 골든크로스"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        macd, macd_signal, _ = self.calculate_macd(df['Close'])
        df['signal'] = ((df['RSI'] > 50) & (macd > macd_signal)).astype(int)
        return self.calculate_returns(df)

    def strategy_22_stochastic_cross(self, df):
        """전략 22: 스토캐스틱 %K > %D 골든크로스"""
        df = df.copy()
        k, d = self.calculate_stochastic(df, 14, 3)
        df['signal'] = (k > d).astype(int)
        return self.calculate_returns(df)

    def strategy_23_cci_breakout(self, df):
        """전략 23: CCI > 100 돌파"""
        df = df.copy()
        df['CCI'] = self.calculate_cci(df, 20)
        df['signal'] = (df['CCI'] > 100).astype(int)
        return self.calculate_returns(df)

    def strategy_24_sma_volume_combo(self, df):
        """전략 24: SMA(20) 상향 + 거래량 증가"""
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['Vol_avg'] = df['Volume'].rolling(window=20).mean()
        df['signal'] = ((df['Close'] > df['SMA20']) & (df['Volume'] > df['Vol_avg'])).astype(int)
        return self.calculate_returns(df)

    def strategy_25_ema_cross_12_26(self, df):
        """전략 25: EMA(12) > EMA(26) 골든크로스"""
        df = df.copy()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['signal'] = (df['EMA12'] > df['EMA26']).astype(int)
        return self.calculate_returns(df)

    def strategy_26_price_rsi_momentum(self, df):
        """전략 26: 가격 > SMA(20) + RSI(14) > 60 (강한 모멘텀)"""
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['signal'] = ((df['Close'] > df['SMA20']) & (df['RSI'] > 60)).astype(int)
        return self.calculate_returns(df)

    def strategy_27_bollinger_squeeze(self, df):
        """전략 27: 볼린저 밴드 스퀴즈 이후 상단 돌파"""
        df = df.copy()
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'], 20, 2)
        df['BB_width'] = (upper - lower) / middle
        df['BB_width_avg'] = df['BB_width'].rolling(window=20).mean()

        # 스퀴즈: 밴드폭이 평균보다 좁을 때
        df['squeeze'] = (df['BB_width'] < df['BB_width_avg'] * 0.8).fillna(0).astype(int)
        df['breakout'] = (df['Close'] > upper).fillna(0).astype(int)
        df['signal'] = ((df['squeeze'].shift(1).fillna(0) == 1) & (df['breakout'] == 1)).astype(int)

        return self.calculate_returns(df)

    def strategy_28_multiple_timeframe(self, df):
        """전략 28: 멀티 타임프레임 (SMA 10, 20, 50 모두 상승)"""
        df = df.copy()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()

        df['signal'] = (
            (df['Close'] > df['SMA10']) &
            (df['SMA10'] > df['SMA20']) &
            (df['SMA20'] > df['SMA50'])
        ).astype(int)
        return self.calculate_returns(df)

    def strategy_29_rsi_oversold_bounce(self, df):
        """전략 29: RSI 과매도(30) 이후 반등 (50 돌파)"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)

        # 과매도 경험 후 50 돌파
        df['oversold'] = (df['RSI'] < 30).astype(int)
        df['oversold_recent'] = df['oversold'].rolling(window=5).max()
        df['signal'] = ((df['oversold_recent'] > 0) & (df['RSI'] > 50)).astype(int)

        return self.calculate_returns(df)

    def strategy_30_atr_volatility_breakout(self, df):
        """전략 30: ATR 기반 변동성 돌파"""
        df = df.copy()
        df['ATR'] = self.calculate_atr(df, 14)
        df['SMA20'] = df['Close'].rolling(window=20).mean()

        # ATR의 2배만큼 상승하면 매수
        df['breakout_level'] = df['SMA20'] + (df['ATR'] * 2)
        df['signal'] = (df['Close'] > df['breakout_level'].shift(1)).astype(int)

        return self.calculate_returns(df)

    # ==================== 전략 실행 ====================

    def get_all_strategies(self):
        """모든 전략 함수 리스트 반환"""
        return {
            '01_SMA_Cross_10_30': self.strategy_01_sma_cross_10_30,
            '02_SMA_Cross_20_60': self.strategy_02_sma_cross_20_60,
            '03_SMA_Cross_50_200': self.strategy_03_sma_cross_50_200,
            '04_Price_Above_SMA20': self.strategy_04_price_above_sma20,
            '05_Price_Above_SMA50': self.strategy_05_price_above_sma50,
            '06_Bollinger_Upper_Breakout': self.strategy_06_bollinger_breakout_upper,
            '07_Bollinger_Middle_Cross': self.strategy_07_bollinger_middle_cross,
            '08_Donchian_20_Breakout': self.strategy_08_donchian_breakout_20,
            '09_Donchian_55_Breakout': self.strategy_09_donchian_breakout_55,
            '10_RSI_Above_50': self.strategy_10_rsi_above_50,
            '11_RSI_Above_55': self.strategy_11_rsi_above_55,
            '12_MACD_Signal_Cross': self.strategy_12_macd_cross_signal,
            '13_MACD_Zero_Cross': self.strategy_13_macd_zero_cross,
            '14_High_20D_Breakout': self.strategy_14_high_breakout_20d,
            '15_High_60D_Breakout': self.strategy_15_high_breakout_60d,
            '16_SMA_RSI_Combo': self.strategy_16_sma_rsi_combo,
            '17_SMA_MACD_Combo': self.strategy_17_sma_macd_combo,
            '18_Bollinger_RSI_Combo': self.strategy_18_bollinger_rsi_combo,
            '19_Triple_SMA': self.strategy_19_triple_sma,
            '20_Donchian_ATR_Combo': self.strategy_20_donchian_atr_combo,
            '21_RSI_MACD_Combo': self.strategy_21_rsi_macd_combo,
            '22_Stochastic_Cross': self.strategy_22_stochastic_cross,
            '23_CCI_Breakout': self.strategy_23_cci_breakout,
            '24_SMA_Volume_Combo': self.strategy_24_sma_volume_combo,
            '25_EMA_Cross_12_26': self.strategy_25_ema_cross_12_26,
            '26_Price_RSI_Momentum': self.strategy_26_price_rsi_momentum,
            '27_Bollinger_Squeeze': self.strategy_27_bollinger_squeeze,
            '28_Multiple_Timeframe': self.strategy_28_multiple_timeframe,
            '29_RSI_Oversold_Bounce': self.strategy_29_rsi_oversold_bounce,
            '30_ATR_Volatility_Breakout': self.strategy_30_atr_volatility_breakout,
        }

    def run_all_strategies(self):
        """모든 전략 실행"""
        print("\n" + "=" * 80)
        print(f"Running 30 breakout strategies on {self.symbol}...")
        print("=" * 80 + "\n")

        strategies = self.get_all_strategies()

        for i, (strategy_name, strategy_func) in enumerate(strategies.items(), 1):
            print(f"[{i}/30] Running {strategy_name}...")
            try:
                result = strategy_func(self.data.copy())
                self.strategy_results[strategy_name] = result
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        print("\n" + "=" * 80)
        print(f"Completed! {len(self.strategy_results)}/30 strategies executed successfully.")
        print("=" * 80 + "\n")

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

        # 최대 연속 손실
        losses = (returns_series < 0).astype(int)
        loss_groups = (losses != losses.shift()).cumsum()
        max_consecutive_losses = losses.groupby(loss_groups).sum().max() if len(losses) > 0 else 0

        return {
            'Strategy': name,
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades),
            'Profit Factor': profit_factor,
            'Max Consecutive Losses': int(max_consecutive_losses)
        }

    def calculate_all_metrics(self):
        """모든 전략의 성과 지표 계산"""
        metrics_list = []

        for strategy_name, result_df in self.strategy_results.items():
            returns = result_df['returns']
            metrics = self.calculate_metrics(returns, strategy_name)
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================

    def plot_performance_comparison(self, metrics_df, save_path='bitcoin_breakout_strategies_comparison.png'):
        """성과 비교 시각화"""
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)

        # 1. 누적 수익률 비교 (상위 10개)
        ax1 = fig.add_subplot(gs[0:2, :])
        top_10 = metrics_df.nlargest(10, 'Total Return (%)')['Strategy'].tolist()

        for strategy_name in top_10:
            if strategy_name in self.strategy_results:
                cumulative = self.strategy_results[strategy_name]['cumulative']
                ax1.plot(cumulative.index, cumulative, label=strategy_name, linewidth=2, alpha=0.7)

        ax1.set_title(f'Top 10 Strategies - Cumulative Returns (BTC)\nPeriod: {self.start_date} to {self.end_date}',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 총 수익률 비교 (상위 15개)
        ax2 = fig.add_subplot(gs[2, :])
        top_15 = metrics_df.nlargest(15, 'Total Return (%)').sort_values('Total Return (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in top_15['Total Return (%)']]
        ax2.barh(range(len(top_15)), top_15['Total Return (%)'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_15)))
        ax2.set_yticklabels(top_15['Strategy'], fontsize=9)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Top 15 Strategies - Total Return', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교 (상위 15개)
        ax3 = fig.add_subplot(gs[3, 0])
        top_15_cagr = metrics_df.nlargest(15, 'CAGR (%)').sort_values('CAGR (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in top_15_cagr['CAGR (%)']]
        ax3.barh(range(len(top_15_cagr)), top_15_cagr['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(top_15_cagr)))
        ax3.set_yticklabels(top_15_cagr['Strategy'], fontsize=8)
        ax3.set_xlabel('CAGR (%)', fontsize=10)
        ax3.set_title('Top 15 - CAGR', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. 샤프 비율 비교 (상위 15개)
        ax4 = fig.add_subplot(gs[3, 1])
        top_15_sharpe = metrics_df.nlargest(15, 'Sharpe Ratio').sort_values('Sharpe Ratio', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in top_15_sharpe['Sharpe Ratio']]
        ax4.barh(range(len(top_15_sharpe)), top_15_sharpe['Sharpe Ratio'], color=colors, alpha=0.7)
        ax4.set_yticks(range(len(top_15_sharpe)))
        ax4.set_yticklabels(top_15_sharpe['Strategy'], fontsize=8)
        ax4.set_xlabel('Sharpe Ratio', fontsize=10)
        ax4.set_title('Top 15 - Sharpe Ratio', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. MDD 비교 (상위 15개 - MDD가 작을수록 좋음)
        ax5 = fig.add_subplot(gs[3, 2])
        top_15_mdd = metrics_df.nlargest(15, 'MDD (%)').sort_values('MDD (%)', ascending=False)
        ax5.barh(range(len(top_15_mdd)), top_15_mdd['MDD (%)'], color='crimson', alpha=0.7)
        ax5.set_yticks(range(len(top_15_mdd)))
        ax5.set_yticklabels(top_15_mdd['Strategy'], fontsize=8)
        ax5.set_xlabel('MDD (%)', fontsize=10)
        ax5.set_title('Top 15 - Maximum Drawdown', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Return vs Risk (Sharpe 기준 색상)
        ax6 = fig.add_subplot(gs[4, 0])
        scatter = ax6.scatter(metrics_df['MDD (%)'], metrics_df['CAGR (%)'],
                   s=200, alpha=0.6, c=metrics_df['Sharpe Ratio'], cmap='RdYlGn')
        plt.colorbar(scatter, ax=ax6, label='Sharpe Ratio')
        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 7. 승률 분포
        ax7 = fig.add_subplot(gs[4, 1])
        ax7.hist(metrics_df['Win Rate (%)'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax7.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50%')
        ax7.set_xlabel('Win Rate (%)', fontsize=11)
        ax7.set_ylabel('Frequency', fontsize=11)
        ax7.set_title('Win Rate Distribution', fontsize=13, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. Profit Factor 분포
        ax8 = fig.add_subplot(gs[4, 2])
        pf_data = metrics_df[metrics_df['Profit Factor'] != np.inf]['Profit Factor']
        ax8.hist(pf_data, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax8.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax8.set_xlabel('Profit Factor', fontsize=11)
        ax8.set_ylabel('Frequency', fontsize=11)
        ax8.set_title('Profit Factor Distribution', fontsize=13, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. 전략별 총 수익률 히트맵
        ax9 = fig.add_subplot(gs[5, :])
        sorted_metrics = metrics_df.sort_values('Total Return (%)', ascending=False)
        colors_bar = ['green' if x > 0 else 'red' for x in sorted_metrics['Total Return (%)']]
        bars = ax9.bar(range(len(sorted_metrics)), sorted_metrics['Total Return (%)'],
                       color=colors_bar, alpha=0.7, edgecolor='black')
        ax9.set_xticks(range(len(sorted_metrics)))
        ax9.set_xticklabels(sorted_metrics['Strategy'], rotation=90, fontsize=8)
        ax9.set_ylabel('Total Return (%)', fontsize=11)
        ax9.set_title('All 30 Strategies - Total Return Ranking', fontsize=13, fontweight='bold')
        ax9.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax9.grid(True, alpha=0.3, axis='y')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPerformance comparison chart saved: {save_path}")
        plt.close()

    def plot_top_strategies_detail(self, metrics_df, top_n=5, save_path='bitcoin_top_strategies_detail.png'):
        """상위 전략 상세 분석"""
        top_strategies = metrics_df.nlargest(top_n, 'Total Return (%)')['Strategy'].tolist()

        fig = plt.figure(figsize=(24, 6 * top_n))
        gs = fig.add_gridspec(top_n, 4, hspace=0.4, wspace=0.3)

        for idx, strategy_name in enumerate(top_strategies):
            result_df = self.strategy_results[strategy_name]

            # 1. 누적 수익률
            ax1 = fig.add_subplot(gs[idx, 0])
            ax1.plot(result_df.index, result_df['cumulative'], color='blue', linewidth=2)
            ax1.set_title(f'{strategy_name}\nCumulative Returns', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Cumulative Return', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')

            # 2. Drawdown
            ax2 = fig.add_subplot(gs[idx, 1])
            cummax = result_df['cumulative'].cummax()
            drawdown = (result_df['cumulative'] - cummax) / cummax * 100
            ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
            ax2.plot(drawdown.index, drawdown, color='darkred', linewidth=1.5)
            ax2.set_title('Drawdown', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)', fontsize=10)
            ax2.grid(True, alpha=0.3)

            # 3. 월별 수익률
            ax3 = fig.add_subplot(gs[idx, 2])
            monthly_returns = result_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            colors_monthly = ['green' if x > 0 else 'red' for x in monthly_returns]
            ax3.bar(monthly_returns.index, monthly_returns, color=colors_monthly, alpha=0.7, width=20)
            ax3.set_title('Monthly Returns', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Return (%)', fontsize=10)
            ax3.axhline(y=0, color='black', linewidth=1)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(axis='x', rotation=45, labelsize=8)

            # 4. 성과 지표
            ax4 = fig.add_subplot(gs[idx, 3])
            ax4.axis('off')

            metrics = self.calculate_metrics(result_df['returns'], strategy_name)
            metrics_text = f"Performance Metrics\n{'='*25}\n\n"
            metrics_text += f"Total Return: {metrics['Total Return (%)']:.2f}%\n"
            metrics_text += f"CAGR: {metrics['CAGR (%)']:.2f}%\n"
            metrics_text += f"MDD: {metrics['MDD (%)']:.2f}%\n"
            metrics_text += f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}\n"
            metrics_text += f"Win Rate: {metrics['Win Rate (%)']:.2f}%\n"
            metrics_text += f"Total Trades: {metrics['Total Trades']}\n"
            if metrics['Profit Factor'] != np.inf:
                metrics_text += f"Profit Factor: {metrics['Profit Factor']:.2f}\n"
            metrics_text += f"Max Loss Streak: {metrics['Max Consecutive Losses']}\n"

            ax4.text(0.1, 0.95, metrics_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        fig.suptitle(f'Top {top_n} Strategies - Detailed Analysis\n{self.symbol}: {self.start_date} to {self.end_date}',
                    fontsize=18, fontweight='bold')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top strategies detail chart saved: {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "=" * 160)
        print(f"{'비트코인 돌파신호 기반 전략 30개 성과 분석':^160}")
        print("=" * 160)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {self.symbol}")
        print(f"슬리피지: {self.slippage*100}% | 수수료: {self.fee*100}%")

        print("\n" + "-" * 160)
        print(f"{'전체 전략 성과 (Total Return 기준 정렬)':^160}")
        print("-" * 160)

        sorted_metrics = metrics_df.sort_values('Total Return (%)', ascending=False)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 160)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(sorted_metrics.to_string(index=False))

        # 상위 5개 요약
        print("\n" + "=" * 160)
        print(f"{'TOP 5 전략 요약':^160}")
        print("=" * 160)
        top_5 = sorted_metrics.head(5)
        for idx, row in top_5.iterrows():
            print(f"\n🏆 {row['Strategy']}")
            print(f"   Total Return: {row['Total Return (%)']:.2f}% | CAGR: {row['CAGR (%)']:.2f}% | "
                  f"MDD: {row['MDD (%)']:.2f}% | Sharpe: {row['Sharpe Ratio']:.2f}")

        print("\n" + "=" * 160 + "\n")

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
        self.plot_performance_comparison(metrics_df)
        self.plot_top_strategies_detail(metrics_df, top_n=5)

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "=" * 80)
    print("비트코인 돌파신호 기반 백테스트 전략 30개 분석 시작")
    print("=" * 80)

    # 백테스트 실행
    backtest = BitcoinBreakoutStrategies(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002,  # 0.2%
        fee=0.001        # 0.1%
    )

    # 분석 실행
    metrics_df = backtest.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('bitcoin_breakout_strategies_metrics.csv', index=False)
    print("✅ Metrics saved to bitcoin_breakout_strategies_metrics.csv")

    # 각 전략별 상세 결과 저장
    print("\nSaving individual strategy results...")
    for strategy_name, result_df in backtest.strategy_results.items():
        filename = f"strategy_results/strategy_{strategy_name}.csv"
        import os
        os.makedirs('strategy_results', exist_ok=True)
        result_df[['Close', 'signal', 'returns', 'cumulative']].to_csv(filename)
    print(f"✅ {len(backtest.strategy_results)} strategy results saved to strategy_results/")

    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
