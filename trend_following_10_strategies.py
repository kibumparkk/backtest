"""
10개 추세추종전략 백테스트

전체 데이터 구간 사용:
- BTC 단일 전략
- BTC + ETH 포트폴리오 전략

전략 목록:
1. SMA Crossover (20/50)
2. EMA Crossover (12/26)
3. MACD
4. ADX Trend
5. Bollinger Bands Breakout
6. Donchian Channel
7. MA Slope (이동평균 기울기)
8. Parabolic SAR
9. SMA30 Above (전일종가 > SMA30)
10. Triple EMA (Fast/Medium/Slow)
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


class TrendFollowingStrategies:
    """10개 추세추종전략 백테스트 클래스"""

    def __init__(self, symbols=['BTC_KRW'], start_date='2018-01-01',
                 end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
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

    # ==================== Strategy 1: SMA Crossover ====================
    def strategy_sma_crossover(self, df, fast=20, slow=50):
        """
        SMA 교차 전략
        - Fast SMA가 Slow SMA를 상향 돌파 시 매수
        - Fast SMA가 Slow SMA를 하향 돌파 시 매도
        """
        df = df.copy()

        # SMA 계산
        df['SMA_fast'] = df['Close'].rolling(window=fast).mean()
        df['SMA_slow'] = df['Close'].rolling(window=slow).mean()

        # 신호 생성
        df['signal'] = 0
        df.loc[df['SMA_fast'] > df['SMA_slow'], 'signal'] = 1

        # 포지션 변화
        df['position_change'] = df['signal'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = df['signal']

        return df

    # ==================== Strategy 2: EMA Crossover ====================
    def strategy_ema_crossover(self, df, fast=12, slow=26):
        """
        EMA 교차 전략
        - Fast EMA가 Slow EMA를 상향 돌파 시 매수
        - Fast EMA가 Slow EMA를 하향 돌파 시 매도
        """
        df = df.copy()

        # EMA 계산
        df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()

        # 신호 생성
        df['signal'] = 0
        df.loc[df['EMA_fast'] > df['EMA_slow'], 'signal'] = 1

        # 포지션 변화
        df['position_change'] = df['signal'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = df['signal']

        return df

    # ==================== Strategy 3: MACD ====================
    def strategy_macd(self, df, fast=12, slow=26, signal=9):
        """
        MACD 전략
        - MACD가 시그널선을 상향 돌파 시 매수
        - MACD가 시그널선을 하향 돌파 시 매도
        """
        df = df.copy()

        # MACD 계산
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

        # 신호 생성
        df['signal'] = 0
        df.loc[df['MACD'] > df['MACD_signal'], 'signal'] = 1

        # 포지션 변화
        df['position_change'] = df['signal'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = df['signal']

        return df

    # ==================== Strategy 4: ADX Trend ====================
    def calculate_adx(self, df, period=14):
        """ADX 계산"""
        df = df.copy()

        # True Range
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        # Directional Movement
        df['H_diff'] = df['High'] - df['High'].shift(1)
        df['L_diff'] = df['Low'].shift(1) - df['Low']

        df['DM+'] = np.where((df['H_diff'] > df['L_diff']) & (df['H_diff'] > 0), df['H_diff'], 0)
        df['DM-'] = np.where((df['L_diff'] > df['H_diff']) & (df['L_diff'] > 0), df['L_diff'], 0)

        # Smoothed values
        df['TR_smooth'] = df['TR'].rolling(window=period).sum()
        df['DM+_smooth'] = df['DM+'].rolling(window=period).sum()
        df['DM-_smooth'] = df['DM-'].rolling(window=period).sum()

        # Directional Indicators
        df['DI+'] = 100 * df['DM+_smooth'] / df['TR_smooth']
        df['DI-'] = 100 * df['DM-_smooth'] / df['TR_smooth']

        # ADX
        df['DX'] = 100 * abs(df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-'])
        df['ADX'] = df['DX'].rolling(window=period).mean()

        return df

    def strategy_adx_trend(self, df, period=14, threshold=25):
        """
        ADX 추세 전략
        - ADX > threshold이고 DI+ > DI- 일 때 매수
        - ADX > threshold이고 DI+ < DI- 일 때 매도
        - ADX < threshold일 때 현금 보유
        """
        df = self.calculate_adx(df, period)

        # 신호 생성
        df['signal'] = 0
        df.loc[(df['ADX'] > threshold) & (df['DI+'] > df['DI-']), 'signal'] = 1

        # 포지션 변화
        df['position_change'] = df['signal'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = df['signal']

        return df

    # ==================== Strategy 5: Bollinger Bands Breakout ====================
    def strategy_bollinger_bands(self, df, period=20, std_dev=2):
        """
        볼린저 밴드 돌파 전략
        - 가격이 상단 밴드를 돌파하면 매수
        - 가격이 중간선 이하로 내려오면 매도
        """
        df = df.copy()

        # 볼린저 밴드 계산
        df['BB_middle'] = df['Close'].rolling(window=period).mean()
        df['BB_std'] = df['Close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (std_dev * df['BB_std'])
        df['BB_lower'] = df['BB_middle'] - (std_dev * df['BB_std'])

        # 신호 생성
        df['signal'] = 0
        df.loc[df['Close'] > df['BB_upper'], 'signal'] = 1
        df.loc[df['Close'] < df['BB_middle'], 'signal'] = 0

        # Forward fill to maintain position
        df['signal'] = df['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)

        # 포지션 변화
        df['position_change'] = df['signal'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = df['signal']

        return df

    # ==================== Strategy 6: Donchian Channel ====================
    def strategy_donchian_channel(self, df, entry_period=20, exit_period=10):
        """
        돈치안 채널 전략
        - N일 최고가 돌파 시 매수
        - M일 최저가 하향 돌파 시 매도
        """
        df = df.copy()

        # 돈치안 채널
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
                # 당일 종가에 매수
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['Close'] * (1 + self.slippage)
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                # 당일 종가에 매도
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                sell_price = df.iloc[i]['Close'] * (1 - self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)
            elif df.iloc[i]['position'] == 1:
                # 포지션 유지
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== Strategy 7: MA Slope ====================
    def strategy_ma_slope(self, df, period=30, slope_threshold=0):
        """
        이동평균 기울기 전략
        - MA의 기울기가 양수이고 가격이 MA 위에 있으면 매수
        - MA의 기울기가 음수이거나 가격이 MA 아래면 매도
        """
        df = df.copy()

        # 이동평균 계산
        df['MA'] = df['Close'].rolling(window=period).mean()

        # 기울기 계산 (일간 변화율)
        df['MA_slope'] = df['MA'].pct_change()

        # 신호 생성
        df['signal'] = 0
        df.loc[(df['MA_slope'] > slope_threshold) & (df['Close'] > df['MA']), 'signal'] = 1

        # 포지션 변화
        df['position_change'] = df['signal'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = df['signal']

        return df

    # ==================== Strategy 8: Parabolic SAR ====================
    def calculate_psar(self, df, af_start=0.02, af_increment=0.02, af_max=0.2):
        """Parabolic SAR 계산"""
        df = df.copy()

        # 초기값 설정
        psar = df['Close'].iloc[0]
        psars = [psar]
        trend = 1  # 1: 상승, -1: 하락
        ep = df['High'].iloc[0]  # Extreme Point
        af = af_start

        for i in range(1, len(df)):
            # PSAR 업데이트
            psar = psar + af * (ep - psar)

            # 추세 변경 확인
            if trend == 1:  # 상승 추세
                if df['Low'].iloc[i] < psar:
                    # 추세 전환: 상승 -> 하락
                    trend = -1
                    psar = ep
                    ep = df['Low'].iloc[i]
                    af = af_start
                else:
                    # 추세 유지
                    if df['High'].iloc[i] > ep:
                        ep = df['High'].iloc[i]
                        af = min(af + af_increment, af_max)
            else:  # 하락 추세
                if df['High'].iloc[i] > psar:
                    # 추세 전환: 하락 -> 상승
                    trend = 1
                    psar = ep
                    ep = df['High'].iloc[i]
                    af = af_start
                else:
                    # 추세 유지
                    if df['Low'].iloc[i] < ep:
                        ep = df['Low'].iloc[i]
                        af = min(af + af_increment, af_max)

            psars.append(psar)

        df['PSAR'] = psars
        return df

    def strategy_parabolic_sar(self, df):
        """
        Parabolic SAR 전략
        - 가격이 PSAR 위에 있으면 매수
        - 가격이 PSAR 아래에 있으면 매도
        """
        df = self.calculate_psar(df)

        # 신호 생성
        df['signal'] = 0
        df.loc[df['Close'] > df['PSAR'], 'signal'] = 1

        # 포지션 변화
        df['position_change'] = df['signal'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = df['signal']

        return df

    # ==================== Strategy 9: SMA30 Above ====================
    def strategy_sma30_above(self, df, period=30):
        """
        SMA30 Above 전략
        - 전일 종가가 SMA30보다 크면 매수
        - 전일 종가가 SMA30보다 작거나 같으면 매도
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=period).mean()

        # 신호 생성 (전일 종가 > 전일 SMA)
        df['prev_close'] = df['Close'].shift(1)
        df['prev_sma'] = df['SMA'].shift(1)
        df['signal'] = 0
        df.loc[df['prev_close'] > df['prev_sma'], 'signal'] = 1

        # 포지션 변화
        df['position_change'] = df['signal'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = df['signal']

        return df

    # ==================== Strategy 10: Triple EMA ====================
    def strategy_triple_ema(self, df, fast=8, medium=21, slow=55):
        """
        Triple EMA 전략
        - Fast > Medium > Slow 일 때 매수
        - 그 외에는 매도
        """
        df = df.copy()

        # EMA 계산
        df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_medium'] = df['Close'].ewm(span=medium, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()

        # 신호 생성
        df['signal'] = 0
        df.loc[(df['EMA_fast'] > df['EMA_medium']) & (df['EMA_medium'] > df['EMA_slow']), 'signal'] = 1

        # 포지션 변화
        df['position_change'] = df['signal'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost

        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = df['signal']

        return df

    # ==================== Run All Strategies ====================
    def run_all_strategies(self):
        """모든 전략을 모든 종목에 대해 실행"""
        strategies = {
            '1. SMA Crossover (20/50)': lambda df: self.strategy_sma_crossover(df, 20, 50),
            '2. EMA Crossover (12/26)': lambda df: self.strategy_ema_crossover(df, 12, 26),
            '3. MACD': lambda df: self.strategy_macd(df, 12, 26, 9),
            '4. ADX Trend': lambda df: self.strategy_adx_trend(df, 14, 25),
            '5. Bollinger Bands': lambda df: self.strategy_bollinger_bands(df, 20, 2),
            '6. Donchian Channel': lambda df: self.strategy_donchian_channel(df, 20, 10),
            '7. MA Slope': lambda df: self.strategy_ma_slope(df, 30, 0),
            '8. Parabolic SAR': lambda df: self.strategy_parabolic_sar(df),
            '9. SMA30 Above': lambda df: self.strategy_sma30_above(df, 30),
            '10. Triple EMA': lambda df: self.strategy_triple_ema(df, 8, 21, 55)
        }

        print("\n" + "="*80)
        print("Running all strategies for all symbols...")
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

    # ==================== Portfolio Creation ====================
    def create_portfolios(self):
        """각 전략별로 동일 비중 포트폴리오 생성"""
        print("\n" + "="*80)
        print("Creating equal-weight portfolios...")
        print("="*80 + "\n")

        weight = 1.0 / len(self.symbols)  # 동일 비중

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
                print(f"  - Added {symbol} with weight {weight:.2%}")

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

    # ==================== Metrics Calculation ====================
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
        """모든 전략 및 종목별 성과 지표 계산"""
        metrics_list = []

        # 각 전략의 포트폴리오 성과
        for strategy_name in self.portfolio_results.keys():
            returns = self.portfolio_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, f"{strategy_name}")
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== Visualization ====================
    def plot_comparison(self, metrics_df, save_path='trend_following_10_strategies.png'):
        """전략 비교 시각화"""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            ax1.plot(cumulative.index, cumulative, label=strategy_name,
                    linewidth=2, alpha=0.8)

        symbol_str = ' + '.join([s.split('_')[0] for s in self.symbols])
        ax1.set_title(f'10 Trend Following Strategies - Cumulative Returns ({symbol_str})',
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
        bars = ax2.barh(range(len(sorted_df)), sorted_df['Total Return (%)'], color=colors, alpha=0.7)
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
        ax6.scatter(metrics_df['MDD (%)'], metrics_df['CAGR (%)'],
                   s=300, alpha=0.6, c=metrics_df['Sharpe Ratio'], cmap='RdYlGn')
        for idx, row in metrics_df.iterrows():
            ax6.annotate(row['Strategy'].split('.')[0],
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=9, ha='center', va='bottom')
        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

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

        # 8. 드로우다운 비교
        ax8 = fig.add_subplot(gs[3, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax8.plot(drawdown.index, drawdown, label=strategy_name, linewidth=1.5, alpha=0.7)

        ax8.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Drawdown (%)', fontsize=12)
        ax8.set_xlabel('Date', fontsize=12)
        ax8.legend(loc='lower right', fontsize=9, ncol=2)
        ax8.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'10개 추세추종전략 백테스트 결과':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        if len(self.symbols) > 1:
            print(f"포트폴리오 구성: 각 종목 동일 비중 ({100/len(self.symbols):.1f}%)")
        print(f"슬리피지: {self.slippage*100}%")

        print("\n" + "-"*150)
        print(f"{'전략별 성과 비교':^150}")
        print("-"*150)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')

        # 순위 추가
        metrics_sorted = metrics_df.sort_values('CAGR (%)', ascending=False).reset_index(drop=True)
        metrics_sorted.insert(0, 'Rank', range(1, len(metrics_sorted) + 1))

        print(metrics_sorted.to_string(index=False))
        print("\n" + "="*150 + "\n")

        # Top 3 강조
        print("🏆 TOP 3 전략 (CAGR 기준):")
        for i in range(min(3, len(metrics_sorted))):
            row = metrics_sorted.iloc[i]
            emoji = ['🥇', '🥈', '🥉'][i]
            print(f"{emoji} {row['Rank']}. {row['Strategy']}")
            print(f"   CAGR: {row['CAGR (%)']:.2f}% | Sharpe: {row['Sharpe Ratio']:.2f} | MDD: {row['MDD (%)']:.2f}%")
        print()

    def run_analysis(self, chart_filename='trend_following_10_strategies.png'):
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
        self.plot_comparison(metrics_df, save_path=chart_filename)

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("10개 추세추종전략 백테스트 시작")
    print("="*80)

    # 1. BTC 단일 전략 백테스트
    print("\n" + "="*80)
    print("1. BTC 단일 전략 백테스트")
    print("="*80)

    btc_backtest = TrendFollowingStrategies(
        symbols=['BTC_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )
    btc_metrics = btc_backtest.run_analysis(chart_filename='trend_following_btc.png')
    btc_metrics.to_csv('trend_following_btc_metrics.csv', index=False)
    print("BTC metrics saved to trend_following_btc_metrics.csv")

    # 2. BTC + ETH 포트폴리오 백테스트
    print("\n" + "="*80)
    print("2. BTC + ETH 포트폴리오 백테스트")
    print("="*80)

    btc_eth_backtest = TrendFollowingStrategies(
        symbols=['BTC_KRW', 'ETH_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )
    btc_eth_metrics = btc_eth_backtest.run_analysis(chart_filename='trend_following_btc_eth.png')

    # 결과 저장
    btc_eth_metrics.to_csv('trend_following_btc_eth_metrics.csv', index=False)
    print("BTC+ETH metrics saved to trend_following_btc_eth_metrics.csv")

    # 각 포트폴리오 상세 결과 저장
    for strategy_name in btc_eth_backtest.portfolio_results.keys():
        filename = f"btc_eth_{strategy_name.split('.')[0].strip()}.csv"
        filename = filename.replace(' ', '_').lower()
        btc_eth_backtest.portfolio_results[strategy_name].to_csv(filename)
        print(f"Portfolio details saved to {filename}")

    print("\n" + "="*80)
    print("전체 분석 완료!")
    print("="*80)
    print("\n생성된 파일:")
    print("  - trend_following_btc_metrics.csv (BTC 단일)")
    print("  - trend_following_btc.png (BTC 단일 차트)")
    print("  - trend_following_btc_eth_metrics.csv (BTC+ETH)")
    print("  - trend_following_btc_eth.png (BTC+ETH 차트)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
