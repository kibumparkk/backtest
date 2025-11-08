"""
암호화폐 포트폴리오 조합 전략 분석

20가지 다양한 기술적 지표 조합 전략:
1. RSI + MACD
2. RSI + Bollinger Bands
3. MACD + Bollinger Bands
4. RSI + Stochastic
5. MACD + Stochastic
6. SMA + RSI
7. EMA Crossover + RSI
8. SMA + Bollinger Bands
9. Triple EMA
10. Ichimoku Cloud
11. RSI + ADX
12. MACD + ADX
13. Bollinger Bands + Stochastic
14. Williams %R + RSI
15. CCI (Commodity Channel Index)
16. MACD + SMA Crossover
17. RSI Divergence + MACD
18. Volume Weighted RSI
19. Multiple Timeframe RSI
20. Composite Momentum (RSI + MACD + Stochastic)

각 전략은 4개 종목(BTC, ETH, ADA, XRP)에 25%씩 동일 비중 투자
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


class CombinedStrategies:
    """여러 지표를 조합한 전략 클래스"""

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

    # ==================== 보조 지표 계산 함수들 ====================
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

    def calculate_stochastic(self, high, low, close, period=14, smooth_k=3):
        """스토캐스틱 계산"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k.rolling(window=smooth_k).mean()
        d = k.rolling(window=3).mean()
        return k, d

    def calculate_adx(self, high, low, close, period=14):
        """ADX (Average Directional Index) 계산"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di

    def calculate_williams_r(self, high, low, close, period=14):
        """Williams %R 계산"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r

    def calculate_cci(self, high, low, close, period=20):
        """CCI (Commodity Channel Index) 계산"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (typical_price - sma) / (0.015 * mad)
        return cci

    def calculate_ichimoku(self, high, low, close):
        """일목균형표 계산"""
        # 전환선 (Tenkan-sen): 9일 중간값
        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        # 기준선 (Kijun-sen): 26일 중간값
        kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        # 선행스팬A (Senkou Span A): (전환선 + 기준선) / 2, 26일 선행
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        # 선행스팬B (Senkou Span B): 52일 중간값, 26일 선행
        senkou_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        # 후행스팬 (Chikou Span): 종가를 26일 후행
        chikou = close.shift(-26)
        return tenkan, kijun, senkou_a, senkou_b, chikou

    # ==================== 전략 1: RSI + MACD ====================
    def strategy_rsi_macd(self, df):
        """RSI >= 50 AND MACD > Signal: 매수"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        macd, macd_signal, macd_hist = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal

        # 매수 조건: RSI >= 50 AND MACD > Signal
        df['signal'] = ((df['RSI'] >= 50) & (df['MACD'] > df['MACD_Signal'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 2: RSI + Bollinger Bands ====================
    def strategy_rsi_bb(self, df):
        """RSI >= 50 AND 가격 > BB 중심선: 매수"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower

        # 매수 조건: RSI >= 50 AND 가격 > BB 중심선
        df['signal'] = ((df['RSI'] >= 50) & (df['Close'] > df['BB_Middle'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 3: MACD + Bollinger Bands ====================
    def strategy_macd_bb(self, df):
        """MACD > Signal AND 가격 > BB 중심선: 매수"""
        df = df.copy()
        macd, macd_signal, _ = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'])
        df['BB_Middle'] = middle

        # 매수 조건
        df['signal'] = ((df['MACD'] > df['MACD_Signal']) & (df['Close'] > df['BB_Middle'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 4: RSI + Stochastic ====================
    def strategy_rsi_stoch(self, df):
        """RSI >= 50 AND Stochastic %K > %D: 매수"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        k, d = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = k
        df['Stoch_D'] = d

        # 매수 조건
        df['signal'] = ((df['RSI'] >= 50) & (df['Stoch_K'] > df['Stoch_D'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 5: MACD + Stochastic ====================
    def strategy_macd_stoch(self, df):
        """MACD > Signal AND Stochastic %K > %D: 매수"""
        df = df.copy()
        macd, macd_signal, _ = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        k, d = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = k
        df['Stoch_D'] = d

        # 매수 조건
        df['signal'] = ((df['MACD'] > df['MACD_Signal']) & (df['Stoch_K'] > df['Stoch_D'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 6: SMA + RSI ====================
    def strategy_sma_rsi(self, df):
        """가격 > SMA(30) AND RSI >= 50: 매수"""
        df = df.copy()
        df['SMA'] = df['Close'].rolling(window=30).mean()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)

        # 매수 조건
        df['signal'] = ((df['Close'] > df['SMA']) & (df['RSI'] >= 50)).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 7: EMA Crossover + RSI ====================
    def strategy_ema_cross_rsi(self, df):
        """EMA(12) > EMA(26) AND RSI >= 50: 매수"""
        df = df.copy()
        df['EMA_Fast'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)

        # 매수 조건
        df['signal'] = ((df['EMA_Fast'] > df['EMA_Slow']) & (df['RSI'] >= 50)).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 8: SMA + Bollinger Bands ====================
    def strategy_sma_bb(self, df):
        """가격 > SMA(30) AND 가격 > BB 하단: 매수"""
        df = df.copy()
        df['SMA'] = df['Close'].rolling(window=30).mean()
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'])
        df['BB_Lower'] = lower

        # 매수 조건
        df['signal'] = ((df['Close'] > df['SMA']) & (df['Close'] > df['BB_Lower'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 9: Triple EMA ====================
    def strategy_triple_ema(self, df):
        """EMA(5) > EMA(20) > EMA(50): 매수"""
        df = df.copy()
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # 매수 조건
        df['signal'] = ((df['EMA_5'] > df['EMA_20']) & (df['EMA_20'] > df['EMA_50'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 10: Ichimoku Cloud ====================
    def strategy_ichimoku(self, df):
        """가격 > 선행스팬 AND 전환선 > 기준선: 매수"""
        df = df.copy()
        tenkan, kijun, senkou_a, senkou_b, chikou = self.calculate_ichimoku(df['High'], df['Low'], df['Close'])
        df['Tenkan'] = tenkan
        df['Kijun'] = kijun
        df['Senkou_A'] = senkou_a
        df['Senkou_B'] = senkou_b

        # 매수 조건: 가격이 구름 위에 있고 전환선이 기준선 위
        cloud_top = df[['Senkou_A', 'Senkou_B']].max(axis=1)
        df['signal'] = ((df['Close'] > cloud_top) & (df['Tenkan'] > df['Kijun'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 11: RSI + ADX ====================
    def strategy_rsi_adx(self, df):
        """RSI >= 50 AND ADX > 25 (강한 추세): 매수"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        adx, plus_di, minus_di = self.calculate_adx(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx
        df['Plus_DI'] = plus_di

        # 매수 조건
        df['signal'] = ((df['RSI'] >= 50) & (df['ADX'] > 25) & (df['Plus_DI'] > 20)).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 12: MACD + ADX ====================
    def strategy_macd_adx(self, df):
        """MACD > Signal AND ADX > 25: 매수"""
        df = df.copy()
        macd, macd_signal, _ = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        adx, plus_di, minus_di = self.calculate_adx(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx

        # 매수 조건
        df['signal'] = ((df['MACD'] > df['MACD_Signal']) & (df['ADX'] > 25)).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 13: Bollinger Bands + Stochastic ====================
    def strategy_bb_stoch(self, df):
        """가격 > BB 중심선 AND Stochastic %K > %D: 매수"""
        df = df.copy()
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'])
        df['BB_Middle'] = middle
        k, d = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = k
        df['Stoch_D'] = d

        # 매수 조건
        df['signal'] = ((df['Close'] > df['BB_Middle']) & (df['Stoch_K'] > df['Stoch_D'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 14: Williams %R + RSI ====================
    def strategy_williams_rsi(self, df):
        """Williams %R > -50 AND RSI >= 50: 매수"""
        df = df.copy()
        df['Williams_R'] = self.calculate_williams_r(df['High'], df['Low'], df['Close'])
        df['RSI'] = self.calculate_rsi(df['Close'], 14)

        # 매수 조건
        df['signal'] = ((df['Williams_R'] > -50) & (df['RSI'] >= 50)).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 15: CCI ====================
    def strategy_cci(self, df):
        """CCI > 0: 매수"""
        df = df.copy()
        df['CCI'] = self.calculate_cci(df['High'], df['Low'], df['Close'])

        # 매수 조건: CCI > 0
        df['signal'] = (df['CCI'] > 0).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 16: MACD + SMA Crossover ====================
    def strategy_macd_sma_cross(self, df):
        """MACD > Signal AND 가격 > SMA(50): 매수"""
        df = df.copy()
        macd, macd_signal, _ = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['SMA'] = df['Close'].rolling(window=50).mean()

        # 매수 조건
        df['signal'] = ((df['MACD'] > df['MACD_Signal']) & (df['Close'] > df['SMA'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 17: RSI Mean Reversion ====================
    def strategy_rsi_mean_reversion(self, df):
        """RSI < 30: 매수 (과매도), RSI > 70: 매도 (과매수)"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)

        # 매수 조건: RSI < 30 (역추세 전략)
        df['signal'] = (df['RSI'] < 30).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 18: Volume Weighted RSI ====================
    def strategy_volume_rsi(self, df):
        """RSI >= 50 AND 거래량 > 20일 평균: 매수"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

        # 매수 조건
        df['signal'] = ((df['RSI'] >= 50) & (df['Volume'] > df['Volume_MA'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 19: Multiple Timeframe RSI ====================
    def strategy_multi_rsi(self, df):
        """RSI(14) >= 50 AND RSI(28) >= 50: 매수 (다중 시간프레임)"""
        df = df.copy()
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        df['RSI_28'] = self.calculate_rsi(df['Close'], 28)

        # 매수 조건
        df['signal'] = ((df['RSI_14'] >= 50) & (df['RSI_28'] >= 50)).astype(int)
        return self._calculate_returns(df)

    # ==================== 전략 20: Composite Momentum ====================
    def strategy_composite_momentum(self, df):
        """RSI >= 50 AND MACD > Signal AND Stochastic %K > %D: 매수 (복합 모멘텀)"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        macd, macd_signal, _ = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        k, d = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = k
        df['Stoch_D'] = d

        # 매수 조건: 모든 모멘텀 지표가 긍정적일 때
        df['signal'] = ((df['RSI'] >= 50) &
                       (df['MACD'] > df['MACD_Signal']) &
                       (df['Stoch_K'] > df['Stoch_D'])).astype(int)
        return self._calculate_returns(df)

    # ==================== 수익률 계산 공통 함수 ====================
    def _calculate_returns(self, df):
        """수익률 계산 (공통)"""
        # 포지션 변화 감지
        df['position_change'] = df['signal'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage  # 매수
        slippage_cost[df['position_change'] == -1] = -self.slippage  # 매도

        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 실행 ====================
    def run_all_strategies(self):
        """모든 전략을 모든 종목에 대해 실행"""
        strategies = {
            '1. RSI + MACD': self.strategy_rsi_macd,
            '2. RSI + Bollinger Bands': self.strategy_rsi_bb,
            '3. MACD + Bollinger Bands': self.strategy_macd_bb,
            '4. RSI + Stochastic': self.strategy_rsi_stoch,
            '5. MACD + Stochastic': self.strategy_macd_stoch,
            '6. SMA + RSI': self.strategy_sma_rsi,
            '7. EMA Crossover + RSI': self.strategy_ema_cross_rsi,
            '8. SMA + Bollinger Bands': self.strategy_sma_bb,
            '9. Triple EMA': self.strategy_triple_ema,
            '10. Ichimoku Cloud': self.strategy_ichimoku,
            '11. RSI + ADX': self.strategy_rsi_adx,
            '12. MACD + ADX': self.strategy_macd_adx,
            '13. Bollinger Bands + Stochastic': self.strategy_bb_stoch,
            '14. Williams %R + RSI': self.strategy_williams_rsi,
            '15. CCI': self.strategy_cci,
            '16. MACD + SMA Crossover': self.strategy_macd_sma_cross,
            '17. RSI Mean Reversion': self.strategy_rsi_mean_reversion,
            '18. Volume Weighted RSI': self.strategy_volume_rsi,
            '19. Multiple Timeframe RSI': self.strategy_multi_rsi,
            '20. Composite Momentum': self.strategy_composite_momentum
        }

        print("\n" + "="*80)
        print("Running all 20 combined strategies for all symbols...")
        print("="*80 + "\n")

        for strategy_name, strategy_func in strategies.items():
            print(f"\n>>> Running {strategy_name} strategy...")
            self.strategy_results[strategy_name] = {}

            for symbol in self.symbols:
                print(f"  - {symbol}...")
                df = self.data[symbol].copy()
                result = strategy_func(df)
                self.strategy_results[strategy_name][symbol] = result

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80 + "\n")

    # ==================== 포트폴리오 구성 ====================
    def create_portfolios(self):
        """각 전략별로 동일 비중 포트폴리오 생성"""
        print("\n" + "="*80)
        print("Creating equal-weight portfolios...")
        print("="*80 + "\n")

        weight = 1.0 / len(self.symbols)  # 동일 비중 (25% each)

        for strategy_name in self.strategy_results.keys():
            print(f"\n>>> Creating portfolio for {strategy_name}...")

            # 모든 종목의 공통 날짜 인덱스 찾기
            all_indices = [self.strategy_results[strategy_name][symbol].index
                          for symbol in self.symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

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
        """모든 전략의 포트폴리오 성과 지표 계산"""
        metrics_list = []

        # 각 전략의 포트폴리오 성과
        for strategy_name in self.portfolio_results.keys():
            returns = self.portfolio_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, f"{strategy_name}")
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================
    def plot_comparison(self, metrics_df, save_path='combined_strategies_comparison.png'):
        """20개 전략 비교 시각화"""
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)

        # 1. 포트폴리오 누적 수익률 비교 (상위 10개)
        ax1 = fig.add_subplot(gs[0:2, :])

        # 성과 상위 10개 전략만 표시
        top_strategies = metrics_df.nlargest(10, 'CAGR (%)')['Strategy'].tolist()

        for strategy_name in self.portfolio_results.keys():
            if strategy_name in top_strategies:
                cumulative = self.portfolio_results[strategy_name]['cumulative']
                ax1.plot(cumulative.index, cumulative, label=strategy_name,
                        linewidth=2, alpha=0.7)

        ax1.set_title('Top 10 Strategies: Portfolio Cumulative Returns - Equal-Weight: BTC, ETH, ADA, XRP',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. CAGR 비교 (전체 20개)
        ax2 = fig.add_subplot(gs[2, :])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['CAGR (%)']]
        ax2.barh(range(len(sorted_df)), sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_df)))
        ax2.set_yticklabels(sorted_df['Strategy'], fontsize=8)
        ax2.set_xlabel('CAGR (%)', fontsize=11)
        ax2.set_title('CAGR Comparison - All 20 Strategies', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. MDD 비교
        ax3 = fig.add_subplot(gs[3, 0])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False).head(15)
        ax3.barh(range(len(sorted_df)), sorted_df['MDD (%)'], color='crimson', alpha=0.7)
        ax3.set_yticks(range(len(sorted_df)))
        ax3.set_yticklabels([s[:25] for s in sorted_df['Strategy']], fontsize=7)
        ax3.set_xlabel('MDD (%)', fontsize=10)
        ax3.set_title('Maximum Drawdown (Top 15)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. 샤프 비율 비교
        ax4 = fig.add_subplot(gs[3, 1])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True).tail(15)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Sharpe Ratio']]
        ax4.barh(range(len(sorted_df)), sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax4.set_yticks(range(len(sorted_df)))
        ax4.set_yticklabels([s[:25] for s in sorted_df['Strategy']], fontsize=7)
        ax4.set_xlabel('Sharpe Ratio', fontsize=10)
        ax4.set_title('Sharpe Ratio (Top 15)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. Return vs Risk 산점도
        ax5 = fig.add_subplot(gs[3, 2])
        scatter = ax5.scatter(metrics_df['MDD (%)'], metrics_df['CAGR (%)'],
                   s=200, alpha=0.6, c=metrics_df['Sharpe Ratio'], cmap='RdYlGn')
        plt.colorbar(scatter, ax=ax5, label='Sharpe Ratio')

        # 상위 5개만 라벨 표시
        top5 = metrics_df.nlargest(5, 'Sharpe Ratio')
        for idx, row in top5.iterrows():
            ax5.annotate(row['Strategy'][:15],
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=7, ha='center', va='bottom')

        ax5.set_xlabel('MDD (%)', fontsize=10)
        ax5.set_ylabel('CAGR (%)', fontsize=10)
        ax5.set_title('Return vs Risk (colored by Sharpe)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 6. 승률 비교
        ax6 = fig.add_subplot(gs[4, 0])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=True).tail(15)
        ax6.barh(range(len(sorted_df)), sorted_df['Win Rate (%)'], color='steelblue', alpha=0.7)
        ax6.set_yticks(range(len(sorted_df)))
        ax6.set_yticklabels([s[:25] for s in sorted_df['Strategy']], fontsize=7)
        ax6.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax6.set_xlabel('Win Rate (%)', fontsize=10)
        ax6.set_title('Win Rate (Top 15)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        # 7. Profit Factor 비교
        ax7 = fig.add_subplot(gs[4, 1])
        sorted_df = metrics_df.copy()
        sorted_df = sorted_df[sorted_df['Profit Factor'] != np.inf].sort_values('Profit Factor', ascending=True).tail(15)
        if len(sorted_df) > 0:
            colors = ['green' if x > 1 else 'red' for x in sorted_df['Profit Factor']]
            ax7.barh(range(len(sorted_df)), sorted_df['Profit Factor'], color=colors, alpha=0.7)
            ax7.set_yticks(range(len(sorted_df)))
            ax7.set_yticklabels([s[:25] for s in sorted_df['Strategy']], fontsize=7)
        ax7.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax7.set_xlabel('Profit Factor', fontsize=10)
        ax7.set_title('Profit Factor (Top 15)', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8. 총 수익률 비교
        ax8 = fig.add_subplot(gs[4, 2])
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True).tail(15)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
        ax8.barh(range(len(sorted_df)), sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax8.set_yticks(range(len(sorted_df)))
        ax8.set_yticklabels([s[:25] for s in sorted_df['Strategy']], fontsize=7)
        ax8.set_xlabel('Total Return (%)', fontsize=10)
        ax8.set_title('Total Return (Top 15)', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')

        # 9. 드로우다운 비교 (상위 5개 전략)
        ax9 = fig.add_subplot(gs[5, :])
        top5_strategies = metrics_df.nlargest(5, 'Sharpe Ratio')['Strategy'].tolist()

        for strategy_name in top5_strategies:
            if strategy_name in self.portfolio_results:
                cumulative = self.portfolio_results[strategy_name]['cumulative']
                cummax = cumulative.cummax()
                drawdown = (cumulative - cummax) / cummax * 100
                ax9.plot(drawdown.index, drawdown, label=strategy_name, linewidth=2, alpha=0.7)

        ax9.set_title('Drawdown Over Time (Top 5 by Sharpe Ratio)', fontsize=14, fontweight='bold')
        ax9.set_ylabel('Drawdown (%)', fontsize=12)
        ax9.set_xlabel('Date', fontsize=12)
        ax9.legend(loc='lower right', fontsize=10)
        ax9.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'20개 조합 전략 성과 비교':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"포트폴리오 구성: 각 종목 동일 비중 (25%)")
        print(f"슬리피지: {self.slippage*100}%")

        # 성과 지표 출력
        print("\n" + "-"*150)
        print(f"{'포트폴리오 성과 (CAGR 기준 정렬)':^150}")
        print("-"*150)

        sorted_metrics = metrics_df.sort_values('CAGR (%)', ascending=False)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(sorted_metrics.to_string(index=False))

        print("\n" + "="*150 + "\n")

        # 상위 5개 전략 하이라이트
        print("\n" + "="*80)
        print("TOP 5 STRATEGIES (by Sharpe Ratio):")
        print("="*80)
        top5 = metrics_df.nlargest(5, 'Sharpe Ratio')
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"{i}. {row['Strategy']}")
            print(f"   CAGR: {row['CAGR (%)']:.2f}% | MDD: {row['MDD (%)']:.2f}% | Sharpe: {row['Sharpe Ratio']:.2f}")
        print("="*80 + "\n")

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
    print("20개 조합 전략 분석 시작")
    print("="*80)

    # 백테스트 실행
    strategy_analyzer = CombinedStrategies(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = strategy_analyzer.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('combined_strategies_metrics.csv', index=False)
    print("Metrics saved to combined_strategies_metrics.csv")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
