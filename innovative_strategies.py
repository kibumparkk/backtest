"""
혁신적인 새로운 지표 창조 및 백테스트

기존 유명 지표가 아닌 완전히 새로운 개념의 지표들:
1. Volatility-Adjusted Momentum (VAM) - 변동성 조정 모멘텀
2. Volume-Weighted Strength (VWS) - 거래량 가중 강도
3. Price Acceleration Index (PAI) - 가격 가속도 지표
4. Trend Consistency Score (TCS) - 추세 일관성 점수
5. Adaptive Volatility Channel (AVC) - 적응형 변동성 채널
6. Multi-Timeframe Momentum Alignment (MTMA) - 멀티타임프레임 모멘텀 정렬
7. Momentum Quality Index (MQI) - 모멘텀 품질 지표
8. Dynamic Support/Resistance Breakout (DSRB) - 동적 지지/저항 돌파
9. Market Regime Adaptive Strategy (MRAS) - 시장 체제 적응 전략
10. Composite Momentum Score (CMS) - 복합 모멘텀 점수

레버리지 없음
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class InnovativeStrategies:
    """혁신적인 새로운 전략 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = {}
        self.strategy_results = {}
        self.portfolio_results = {}

    def load_data(self):
        """데이터 로드"""
        print("="*80)
        print("Loading data...")
        print("="*80)

        for symbol in self.symbols:
            file_path = f'chart_day/{symbol}.parquet'
            df = pd.read_parquet(file_path)
            df.columns = [col.capitalize() for col in df.columns]
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            self.data[symbol] = df
            print(f"Loaded {symbol}: {len(df)} days")

        print("\n" + "="*80 + "\n")

    # ========== 보조 함수 ==========
    def calculate_atr(self, df, period=14):
        """Average True Range 계산"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    # ========== 1. Volatility-Adjusted Momentum (VAM) ==========
    def strategy_volatility_adjusted_momentum(self, df, momentum_period=20, vol_period=20, threshold=1.0):
        """
        변동성 조정 모멘텀 전략

        아이디어: 변동성이 낮을 때의 모멘텀이 더 신뢰성 있다
        - 높은 변동성 = 노이즈 많음
        - 낮은 변동성 = 진짜 추세

        VAM = Momentum / Volatility
        VAM > threshold일 때 매수
        """
        df = df.copy()

        # 모멘텀 계산 (ROC - Rate of Change)
        df['momentum'] = df['Close'].pct_change(momentum_period)

        # 변동성 계산 (표준편차)
        df['volatility'] = df['Close'].pct_change().rolling(window=vol_period).std()

        # VAM = Momentum / Volatility (변동성으로 정규화)
        df['VAM'] = df['momentum'] / (df['volatility'] + 1e-10)  # 0으로 나누기 방지

        # 매매 신호: VAM이 임계값보다 높을 때
        df['position'] = np.where(df['VAM'] > threshold, 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 2. Volume-Weighted Strength (VWS) ==========
    def strategy_volume_weighted_strength(self, df, ma_period=30, volume_ma_period=30):
        """
        거래량 가중 강도 전략

        아이디어: 거래량이 많을 때의 가격 상승이 더 신뢰성 있다
        - 거래량 없는 상승 = 가짜 상승
        - 거래량 있는 상승 = 진짜 상승

        VWS = (Price - MA) * (Volume / Volume_MA)
        VWS > 0일 때 매수
        """
        df = df.copy()

        # 이동평균
        df['MA'] = df['Close'].rolling(window=ma_period).mean()

        # 거래량 이동평균
        df['Volume_MA'] = df['Volume'].rolling(window=volume_ma_period).mean()

        # 가격 강도
        df['price_strength'] = (df['Close'] - df['MA']) / df['MA']

        # 거래량 비율
        df['volume_ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)

        # VWS = Price Strength * Volume Ratio
        df['VWS'] = df['price_strength'] * df['volume_ratio']

        # 매매 신호: VWS > 0이고 가격이 MA 위
        df['position'] = np.where((df['VWS'] > 0) & (df['Close'] > df['MA']), 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 3. Price Acceleration Index (PAI) ==========
    def strategy_price_acceleration(self, df, period=10):
        """
        가격 가속도 지표

        아이디어: 가격 변화의 가속 (2차 미분)
        - 속도가 증가 = 가속 (추세 강화)
        - 속도가 감소 = 감속 (추세 약화)

        PAI = d²(price)/dt² = 가격 변화율의 변화율
        PAI > 0일 때 매수 (가속 중)
        """
        df = df.copy()

        # 1차 미분: 속도 (가격 변화율)
        df['velocity'] = df['Close'].pct_change(period)

        # 2차 미분: 가속도 (속도의 변화율)
        df['acceleration'] = df['velocity'].diff(period)

        # 가속도를 평활화
        df['PAI'] = df['acceleration'].rolling(window=5).mean()

        # 가격도 상승 추세여야 함
        df['trend'] = df['Close'] > df['Close'].rolling(window=30).mean()

        # 매매 신호: 가속도 > 0이고 상승 추세
        df['position'] = np.where((df['PAI'] > 0) & df['trend'], 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 4. Trend Consistency Score (TCS) ==========
    def strategy_trend_consistency(self, df):
        """
        추세 일관성 점수

        아이디어: 여러 기간에서 추세가 일치할수록 신뢰성 높음
        - 5일, 10일, 20일, 30일, 50일 모두 상승 추세
        - 점수가 높을수록 강한 추세

        TCS = 상승 추세인 기간의 개수 / 전체 기간 개수
        """
        df = df.copy()

        periods = [5, 10, 20, 30, 50]
        df['TCS'] = 0

        for period in periods:
            # 현재 가격이 해당 기간 평균보다 높으면 +1
            df[f'trend_{period}'] = (df['Close'] > df['Close'].rolling(window=period).mean()).astype(int)
            df['TCS'] += df[f'trend_{period}']

        # 정규화: 0-1 범위로 (0-5를 0-1로)
        df['TCS'] = df['TCS'] / len(periods)

        # 매매 신호: TCS >= 0.8 (5개 중 4개 이상 상승 추세)
        df['position'] = np.where(df['TCS'] >= 0.8, 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 5. Adaptive Volatility Channel (AVC) ==========
    def strategy_adaptive_volatility_channel(self, df, ma_period=20, atr_period=14):
        """
        적응형 변동성 채널

        아이디어: 변동성에 따라 자동으로 채널 폭 조정
        - 변동성 높을 때 = 넓은 채널
        - 변동성 낮을 때 = 좁은 채널

        Upper = MA + ATR * multiplier
        Lower = MA - ATR * multiplier
        가격이 Upper 돌파 시 매수
        """
        df = df.copy()

        # 중심선
        df['MA'] = df['Close'].rolling(window=ma_period).mean()

        # ATR (변동성)
        df['ATR'] = self.calculate_atr(df, atr_period)

        # 변동성 기반 배수 (ATR이 클수록 배수 작게)
        df['ATR_pct'] = df['ATR'] / df['Close']
        df['multiplier'] = np.where(df['ATR_pct'] > 0.05, 1.5, 2.5)  # 적응형

        # 채널
        df['upper'] = df['MA'] + df['ATR'] * df['multiplier']
        df['lower'] = df['MA'] - df['ATR'] * df['multiplier']

        # 매매 신호: 가격이 중심선 위
        df['position'] = np.where(df['Close'] > df['MA'], 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 6. Multi-Timeframe Momentum Alignment (MTMA) ==========
    def strategy_multi_timeframe_momentum(self, df):
        """
        멀티타임프레임 모멘텀 정렬

        아이디어: 단기/중기/장기 모멘텀이 모두 양수일 때만 매수
        - 5일, 20일, 50일 모멘텀이 모두 양수
        - 모든 시간대가 일치 = 강한 추세
        """
        df = df.copy()

        # 여러 기간의 모멘텀
        df['momentum_5'] = df['Close'].pct_change(5)
        df['momentum_20'] = df['Close'].pct_change(20)
        df['momentum_50'] = df['Close'].pct_change(50)

        # 모든 모멘텀이 양수일 때
        df['all_positive'] = ((df['momentum_5'] > 0) &
                             (df['momentum_20'] > 0) &
                             (df['momentum_50'] > 0))

        df['position'] = df['all_positive'].astype(int)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 7. Momentum Quality Index (MQI) ==========
    def strategy_momentum_quality(self, df, period=20):
        """
        모멘텀 품질 지표

        아이디어: 모멘텀의 품질 평가
        - 강도: 모멘텀 크기
        - 일관성: 양수 일수 / 전체 일수
        - 거래량 확인: 거래량 증가 비율

        MQI = strength * consistency * volume_confirmation
        """
        df = df.copy()

        # 1. 모멘텀 강도
        df['momentum'] = df['Close'].pct_change(period)
        df['strength'] = df['momentum'].rolling(window=10).mean()

        # 2. 일관성 (최근 N일 중 상승한 날 비율)
        df['daily_change'] = df['Close'].pct_change()
        df['consistency'] = df['daily_change'].rolling(window=period).apply(
            lambda x: (x > 0).sum() / len(x)
        )

        # 3. 거래량 확인
        df['volume_change'] = df['Volume'].pct_change(period)
        df['volume_confirmation'] = np.where(df['volume_change'] > 0, 1.2, 0.8)

        # MQI = strength * consistency * volume_confirmation
        df['MQI'] = df['strength'] * df['consistency'] * df['volume_confirmation']

        # 매매 신호: MQI > 0.01 (임계값)
        df['position'] = np.where(df['MQI'] > 0.01, 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 8. Dynamic Support/Resistance Breakout (DSRB) ==========
    def strategy_dynamic_sr_breakout(self, df, lookback=20):
        """
        동적 지지/저항 돌파

        아이디어: 최근 고점/저점을 동적으로 계산
        - 변동성 기반 임계값
        - 진짜 돌파만 매수 (노이즈 필터)
        """
        df = df.copy()

        # 동적 저항선 (최근 N일 최고가)
        df['resistance'] = df['High'].rolling(window=lookback).max().shift(1)

        # ATR 기반 임계값
        df['ATR'] = self.calculate_atr(df, 14)
        df['threshold'] = df['resistance'] + df['ATR'] * 0.5

        # 돌파 신호: Close가 저항선 + 임계값 돌파
        df['breakout'] = df['Close'] > df['threshold']

        # 추세 필터: 가격이 30일 평균 위
        df['trend_filter'] = df['Close'] > df['Close'].rolling(window=30).mean()

        df['position'] = (df['breakout'] & df['trend_filter']).astype(int)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 9. Market Regime Adaptive Strategy (MRAS) ==========
    def strategy_market_regime_adaptive(self, df):
        """
        시장 체제 적응 전략

        아이디어: 시장 상태에 따라 전략 변경
        - 고변동성: 보수적 (장기 추세만)
        - 저변동성: 공격적 (단기 추세)
        """
        df = df.copy()

        # 변동성 계산
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
        df['vol_ma'] = df['volatility'].rolling(window=50).mean()

        # 체제 분류
        df['high_vol_regime'] = df['volatility'] > df['vol_ma']

        # 단기/장기 신호
        df['short_term_signal'] = df['Close'] > df['Close'].rolling(window=10).mean()
        df['long_term_signal'] = df['Close'] > df['Close'].rolling(window=50).mean()

        # 체제별 전략 선택
        df['position'] = np.where(
            df['high_vol_regime'],
            df['long_term_signal'].astype(int),  # 고변동성: 장기만
            df['short_term_signal'].astype(int)   # 저변동성: 단기
        )

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 10. Composite Momentum Score (CMS) ==========
    def strategy_composite_momentum_score(self, df):
        """
        복합 모멘텀 점수

        아이디어: 여러 모멘텀 지표를 Z-score로 정규화하여 합산
        - ROC, RSI, MACD 등을 표준화
        - 종합 점수가 높을 때만 매수
        """
        df = df.copy()

        # 1. ROC (Rate of Change)
        df['roc'] = df['Close'].pct_change(20)

        # 2. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # -1 to 1

        # 3. MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 4. 거리 (Price vs MA)
        df['ma30'] = df['Close'].rolling(window=30).mean()
        df['distance'] = (df['Close'] - df['ma30']) / df['ma30']

        # Z-score 정규화
        df['roc_z'] = (df['roc'] - df['roc'].rolling(50).mean()) / (df['roc'].rolling(50).std() + 1e-10)
        df['macd_z'] = (df['macd_hist'] - df['macd_hist'].rolling(50).mean()) / (df['macd_hist'].rolling(50).std() + 1e-10)
        df['distance_z'] = (df['distance'] - df['distance'].rolling(50).mean()) / (df['distance'].rolling(50).std() + 1e-10)

        # Composite Score (가중 평균)
        df['CMS'] = (df['roc_z'] * 0.3 +
                    df['rsi_normalized'] * 0.2 +
                    df['macd_z'] * 0.3 +
                    df['distance_z'] * 0.2)

        # 매매 신호: CMS > 0.5
        df['position'] = np.where(df['CMS'] > 0.5, 1, 0)

        # 수익률 계산
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ========== 기준선: SMA 30 ==========
    def strategy_sma_30_baseline(self, df):
        """SMA 30 기준선"""
        df = df.copy()
        df['SMA'] = df['Close'].rolling(window=30).mean()
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 실행 ====================
    def run_all_strategies(self):
        """모든 전략 실행"""
        strategies = {
            'SMA 30 (Baseline)': self.strategy_sma_30_baseline,
            'VAM - Volatility Adjusted Momentum': self.strategy_volatility_adjusted_momentum,
            'VWS - Volume Weighted Strength': self.strategy_volume_weighted_strength,
            'PAI - Price Acceleration': self.strategy_price_acceleration,
            'TCS - Trend Consistency': self.strategy_trend_consistency,
            'AVC - Adaptive Vol Channel': self.strategy_adaptive_volatility_channel,
            'MTMA - Multi-Timeframe Momentum': self.strategy_multi_timeframe_momentum,
            'MQI - Momentum Quality': self.strategy_momentum_quality,
            'DSRB - Dynamic SR Breakout': self.strategy_dynamic_sr_breakout,
            'MRAS - Market Regime Adaptive': self.strategy_market_regime_adaptive,
            'CMS - Composite Momentum': self.strategy_composite_momentum_score,
        }

        print("="*80)
        print("Running innovative strategies...")
        print("="*80 + "\n")

        for strategy_name, strategy_func in strategies.items():
            print(f">>> {strategy_name}")
            self.strategy_results[strategy_name] = {}
            for symbol in self.symbols:
                df = self.data[symbol].copy()
                result = strategy_func(df)
                self.strategy_results[strategy_name][symbol] = result

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80 + "\n")

    def create_portfolios(self):
        """포트폴리오 생성"""
        weight = 1.0 / len(self.symbols)

        for strategy_name in self.strategy_results.keys():
            all_indices = [self.strategy_results[strategy_name][symbol].index
                          for symbol in self.symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            portfolio_returns = pd.Series(0.0, index=common_index)
            for symbol in self.symbols:
                symbol_returns = self.strategy_results[strategy_name][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight

            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            self.portfolio_results[strategy_name] = pd.DataFrame({
                'returns': portfolio_returns,
                'cumulative': portfolio_cumulative
            }, index=common_index)

    def calculate_metrics(self, returns_series, name):
        """성과 지표 계산"""
        cumulative = (1 + returns_series).cumprod()
        total_return = (cumulative.iloc[-1] - 1) * 100
        years = (returns_series.index[-1] - returns_series.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100
        sharpe = (returns_series.mean() / returns_series.std() * np.sqrt(365)) if returns_series.std() > 0 else 0
        total_trades = (returns_series != 0).sum()
        winning_trades = (returns_series > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
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
        """모든 전략 성과 계산"""
        metrics_list = []
        for strategy_name in self.portfolio_results.keys():
            returns = self.portfolio_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, strategy_name)
            metrics_list.append(metrics)
        return pd.DataFrame(metrics_list)

    def plot_comparison(self, metrics_df, save_path='innovative_strategies_results.png'):
        """시각화"""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            linewidth = 3.5 if 'Baseline' in strategy_name else 2
            alpha = 1.0 if 'Baseline' in strategy_name else 0.7
            ax1.plot(cumulative.index, cumulative, label=strategy_name, linewidth=linewidth, alpha=alpha)
        ax1.set_title('Innovative Strategies: Cumulative Returns (Log Scale)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. CAGR 랭킹
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['gold' if 'Baseline' in x else 'green' for x in sorted_df['Strategy']]
        ax2.barh(range(len(sorted_df)), sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_df)))
        ax2.set_yticklabels([s.replace(' - ', '\n') for s in sorted_df['Strategy']], fontsize=8)
        ax2.set_xlabel('CAGR (%)', fontsize=11)
        ax2.set_title('CAGR Ranking', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. Total Return 랭킹
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True)
        colors = ['gold' if 'Baseline' in x else 'green' for x in sorted_df['Strategy']]
        ax3.barh(range(len(sorted_df)), sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(sorted_df)))
        ax3.set_yticklabels([s.replace(' - ', '\n') for s in sorted_df['Strategy']], fontsize=8)
        ax3.set_xlabel('Total Return (%)', fontsize=11)
        ax3.set_title('Total Return Ranking', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        colors = ['gold' if 'Baseline' in x else 'crimson' for x in sorted_df['Strategy']]
        ax4.barh(range(len(sorted_df)), sorted_df['MDD (%)'], color=colors, alpha=0.7)
        ax4.set_yticks(range(len(sorted_df)))
        ax4.set_yticklabels([s.replace(' - ', '\n') for s in sorted_df['Strategy']], fontsize=8)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. Sharpe Ratio
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['gold' if 'Baseline' in x else 'teal' for x in sorted_df['Strategy']]
        ax5.barh(range(len(sorted_df)), sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax5.set_yticks(range(len(sorted_df)))
        ax5.set_yticklabels([s.replace(' - ', '\n') for s in sorted_df['Strategy']], fontsize=8)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Ranking', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Return vs Risk
        ax6 = fig.add_subplot(gs[2, 1])
        colors_scatter = ['gold' if 'Baseline' in x else 'steelblue' for x in metrics_df['Strategy']]
        sizes = [500 if 'Baseline' in x else 250 for x in metrics_df['Strategy']]
        ax6.scatter(metrics_df['MDD (%)'], metrics_df['CAGR (%)'], s=sizes, alpha=0.6,
                   c=colors_scatter, edgecolors='black', linewidths=1.5)
        for idx, row in metrics_df.iterrows():
            label = row['Strategy'].replace(' - ', '\n')
            ax6.annotate(label, (row['MDD (%)'], row['CAGR (%)']), fontsize=7, ha='left', va='bottom')
        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # 7. Win Rate
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=True)
        colors = ['gold' if 'Baseline' in x else 'purple' for x in sorted_df['Strategy']]
        ax7.barh(range(len(sorted_df)), sorted_df['Win Rate (%)'], color=colors, alpha=0.7)
        ax7.set_yticks(range(len(sorted_df)))
        ax7.set_yticklabels([s.replace(' - ', '\n') for s in sorted_df['Strategy']], fontsize=8)
        ax7.set_xlabel('Win Rate (%)', fontsize=11)
        ax7.set_title('Win Rate', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')
        ax7.axvline(x=50, color='red', linestyle='--', alpha=0.5)

        # 8-10. Top 3 전략 드로우다운
        top3 = metrics_df.nlargest(3, 'CAGR (%)')
        baseline = metrics_df[metrics_df['Strategy'].str.contains('Baseline')]
        top3_with_baseline = pd.concat([top3, baseline]).drop_duplicates()

        for idx, (_, row) in enumerate(top3_with_baseline.iterrows()):
            if idx >= 3:
                break
            ax = fig.add_subplot(gs[3, idx])
            strategy_name = row['Strategy']
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
            ax.plot(drawdown.index, drawdown, color='darkred', linewidth=2)
            ax.set_title(f'{strategy_name}\nDrawdown', fontsize=10, fontweight='bold')
            ax.set_ylabel('DD (%)', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Innovative Trading Strategies: New Indicators Created from Scratch',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved: {save_path}")
        plt.close()

    def print_results(self, metrics_df):
        """결과 출력"""
        print("\n" + "="*120)
        print(f"{'혁신적인 새로운 지표 전략 성과':^120}")
        print("="*120)

        sorted_metrics = metrics_df.sort_values('CAGR (%)', ascending=False)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}')
        print(sorted_metrics.to_string(index=False))

        best = sorted_metrics.iloc[0]
        baseline = metrics_df[metrics_df['Strategy'].str.contains('Baseline')].iloc[0]

        print("\n" + "="*120)
        print(f"{'🏆 최고 성과 전략':^120}")
        print("="*120)
        print(f"\n{best['Strategy']}")
        print(f"  CAGR: {best['CAGR (%)']:.2f}%")
        print(f"  Total Return: {best['Total Return (%)']:.2f}%")
        print(f"  MDD: {best['MDD (%)']:.2f}%")
        print(f"  Sharpe: {best['Sharpe Ratio']:.2f}")

        print(f"\nBaseline (SMA 30) 대비:")
        print(f"  CAGR: {best['CAGR (%)'] - baseline['CAGR (%)']:+.2f}%p")
        print(f"  Total Return: {best['Total Return (%)'] - baseline['Total Return (%)']:+.2f}%p")
        print(f"  Sharpe: {best['Sharpe Ratio'] - baseline['Sharpe Ratio']:+.2f}")
        print("\n" + "="*120 + "\n")

    def run_analysis(self):
        """전체 분석 실행"""
        self.load_data()
        self.run_all_strategies()
        self.create_portfolios()
        metrics_df = self.calculate_all_metrics()
        self.print_results(metrics_df)
        self.plot_comparison(metrics_df)
        metrics_df.to_csv('innovative_strategies_results.csv', index=False)
        print("Results saved: innovative_strategies_results.csv\n")
        return metrics_df


def main():
    print("\n" + "="*80)
    print("혁신적인 새로운 지표 창조 및 백테스트")
    print("="*80)

    comparison = InnovativeStrategies(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        slippage=0.002
    )

    metrics_df = comparison.run_analysis()

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
