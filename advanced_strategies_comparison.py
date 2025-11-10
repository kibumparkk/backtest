"""
고급 암호화폐 트레이딩 전략 비교 분석

SMA 30 전략(수익률 5,942%, 샤프 1.60)보다 더 나은 성과를 목표로
5가지 새로운 고급 전략 구현:

1. Triple EMA Momentum - 3중 지수이동평균 트렌드 추종
2. RSI-SMA Hybrid - RSI 모멘텀 + SMA 트렌드 이중 필터
3. Adaptive ATR Channel - 변동성 기반 동적 채널 브레이크아웃
4. Bollinger RSI Strategy - 볼린저 밴드 + RSI 조합
5. MACD SMA Filter - MACD 크로스오버 + SMA 트렌드 필터

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


class AdvancedTradingStrategies:
    """고급 트레이딩 전략 비교 클래스"""

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
    def calculate_ema(self, prices, period):
        """EMA 계산"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df, period=14):
        """ATR 계산"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD 계산"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        return macd_line, signal_line

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    # ==================== 전략 1: Triple EMA Momentum ====================
    def strategy_triple_ema(self, df, fast_period=8, mid_period=21, slow_period=55):
        """
        Triple EMA Momentum 전략
        - 단기 EMA가 중기, 장기 EMA를 모두 상회할 때 매수
        - 단기 EMA가 중기 EMA 아래로 하락 시 매도
        - 강한 트렌드만 포착하여 잘못된 신호 최소화

        로직:
        - 매수: EMA(8) > EMA(21) AND EMA(21) > EMA(55) (완벽한 정렬)
        - 매도: EMA(8) < EMA(21) (트렌드 약화)
        """
        df = df.copy()

        # EMA 계산
        df['EMA_fast'] = self.calculate_ema(df['Close'], fast_period)
        df['EMA_mid'] = self.calculate_ema(df['Close'], mid_period)
        df['EMA_slow'] = self.calculate_ema(df['Close'], slow_period)

        # 매수 신호: 3개 EMA가 완벽하게 정렬
        df['signal'] = ((df['EMA_fast'] > df['EMA_mid']) &
                       (df['EMA_mid'] > df['EMA_slow'])).astype(int)

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

    # ==================== 전략 2: RSI-SMA Hybrid ====================
    def strategy_rsi_sma_hybrid(self, df, rsi_period=14, rsi_threshold=50, sma_period=30):
        """
        RSI-SMA Hybrid 전략
        - RSI > 50 (모멘텀 존재) AND 가격 > SMA(30) (상승 트렌드)
        - 두 조건 모두 충족 시에만 매수
        - 잘못된 브레이크아웃 필터링으로 승률 향상

        로직:
        - 매수: RSI >= 50 AND Close > SMA(30)
        - 매도: 둘 중 하나라도 조건 불충족
        """
        df = df.copy()

        # 지표 계산
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 이중 필터: RSI 모멘텀 + SMA 트렌드
        df['signal'] = ((df['RSI'] >= rsi_threshold) &
                       (df['Close'] > df['SMA'])).astype(int)

        # 포지션 변화 감지
        df['position_change'] = df['signal'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 3: Adaptive ATR Channel ====================
    def strategy_adaptive_atr_channel(self, df, atr_period=14, atr_multiplier=2.5):
        """
        Adaptive ATR Channel 전략
        - ATR 기반 동적 채널 브레이크아웃
        - 변동성이 낮을 때는 좁은 채널, 높을 때는 넓은 채널
        - 시장 상황에 맞게 적응

        로직:
        - Upper Channel = SMA(20) + ATR(14) * 2.5
        - Lower Channel = SMA(20) - ATR(14) * 2.5
        - 매수: Close > Upper Channel
        - 매도: Close < Lower Channel
        """
        df = df.copy()

        # 중간선 (SMA 20)
        df['SMA'] = df['Close'].rolling(window=20).mean()

        # ATR 계산
        df['ATR'] = self.calculate_atr(df, atr_period)

        # 동적 채널
        df['upper_channel'] = df['SMA'] + (df['ATR'] * atr_multiplier)
        df['lower_channel'] = df['SMA'] - (df['ATR'] * atr_multiplier)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 상단 채널 돌파 시 매수
            if df.iloc[i]['Close'] > df.iloc[i]['upper_channel'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 하단 채널 하향 돌파 시 매도
            elif df.iloc[i]['Close'] < df.iloc[i]['lower_channel'] and df.iloc[i-1]['position'] == 1:
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

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 4: Bollinger RSI Strategy ====================
    def strategy_bollinger_rsi(self, df, bb_period=20, bb_std=2, rsi_period=14,
                               rsi_oversold=30, rsi_overbought=70):
        """
        Bollinger RSI Strategy
        - 볼린저 밴드 하단 + RSI 과매도: 매수
        - 볼린저 밴드 중간선 돌파 OR RSI 과매수: 매도
        - 평균 회귀 + 모멘텀 조합

        로직:
        - 매수: Close < Lower BB AND RSI < 30 (과매도 상태)
        - 매도: Close > Middle BB OR RSI > 70 (목표 도달 or 과매수)
        """
        df = df.copy()

        # 볼린저 밴드 계산
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(
            df['Close'], bb_period, bb_std)

        # RSI 계산
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)

        # 매수 신호: 볼린저 밴드 하단 + RSI 과매도
        df['buy_signal'] = ((df['Close'] < df['BB_lower']) &
                           (df['RSI'] < rsi_oversold)).astype(int)

        # 매도 신호: 볼린저 중간선 복귀 OR RSI 과매수
        df['sell_signal'] = ((df['Close'] > df['BB_middle']) |
                            (df['RSI'] > rsi_overbought)).astype(int)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['buy_signal'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1
            elif df.iloc[i]['sell_signal'] == 1 and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0
            else:
                df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1]['position']

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

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 5: MACD SMA Filter ====================
    def strategy_macd_sma_filter(self, df, macd_fast=12, macd_slow=26, macd_signal=9,
                                 sma_period=50):
        """
        MACD SMA Filter 전략
        - MACD 크로스오버를 SMA로 필터링
        - 상승 트렌드에서만 MACD 매수 신호 활용
        - 잘못된 신호 필터링

        로직:
        - 매수: MACD > Signal AND Close > SMA(50) (상승 트렌드 내 MACD 골든크로스)
        - 매도: MACD < Signal (MACD 데드크로스)
        """
        df = df.copy()

        # MACD 계산
        df['MACD'], df['Signal'] = self.calculate_macd(
            df['Close'], macd_fast, macd_slow, macd_signal)

        # SMA 필터
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 매수 신호: MACD 골든크로스 + 상승 트렌드
        df['signal'] = ((df['MACD'] > df['Signal']) &
                       (df['Close'] > df['SMA'])).astype(int)

        # 포지션 변화 감지
        df['position_change'] = df['signal'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 기존 SMA 30 전략 (비교용) ====================
    def strategy_sma_30(self, df, sma_period=30):
        """
        SMA 30 교차 전략 (비교 기준)
        - 가격이 SMA 30 이상일 때 매수
        - 가격이 SMA 30 미만일 때 매도
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 포지션 계산
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

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
        """모든 전략을 모든 종목에 대해 실행"""
        strategies = {
            'SMA 30 (Baseline)': lambda df: self.strategy_sma_30(df, sma_period=30),
            'Triple EMA Momentum': lambda df: self.strategy_triple_ema(df, fast_period=8, mid_period=21, slow_period=55),
            'RSI-SMA Hybrid': lambda df: self.strategy_rsi_sma_hybrid(df, rsi_period=14, rsi_threshold=50, sma_period=30),
            'Adaptive ATR Channel': lambda df: self.strategy_adaptive_atr_channel(df, atr_period=14, atr_multiplier=2.5),
            'Bollinger RSI': lambda df: self.strategy_bollinger_rsi(df, bb_period=20, bb_std=2, rsi_period=14),
            'MACD SMA Filter': lambda df: self.strategy_macd_sma_filter(df, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=50)
        }

        print("\n" + "="*80)
        print("Running all advanced strategies for all symbols...")
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
        """모든 전략 및 종목별 성과 지표 계산"""
        metrics_list = []

        # 각 전략의 포트폴리오 성과
        for strategy_name in self.portfolio_results.keys():
            returns = self.portfolio_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, f"{strategy_name} Portfolio")
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================
    def plot_comparison(self, metrics_df, save_path='advanced_strategies_comparison.png'):
        """포트폴리오 비교 시각화"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 포트폴리오 누적 수익률 비교 (Log Scale)
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            linestyle = '--' if 'Baseline' in strategy_name else '-'
            linewidth = 2 if 'Baseline' in strategy_name else 2.5
            ax1.plot(cumulative.index, cumulative, label=strategy_name,
                    linewidth=linewidth, linestyle=linestyle, alpha=0.8)

        ax1.set_title('Advanced Strategies vs SMA 30 Baseline - Cumulative Returns (Log Scale)',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 포트폴리오만 필터링
        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()

        # 2. 총 수익률 비교
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = portfolio_metrics.sort_values('Total Return (%)', ascending=True)
        colors = ['red' if 'Baseline' in s else 'green' for s in sorted_df['Strategy']]
        ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = portfolio_metrics.sort_values('CAGR (%)', ascending=True)
        colors = ['red' if 'Baseline' in s else 'green' for s in sorted_df['Strategy']]
        ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Sharpe Ratio 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = portfolio_metrics.sort_values('Sharpe Ratio', ascending=True)
        colors = ['red' if 'Baseline' in s else 'green' for s in sorted_df['Strategy']]
        ax4.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax4.set_xlabel('Sharpe Ratio', fontsize=11)
        ax4.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.axvline(x=1.60, color='red', linestyle='--', linewidth=2, alpha=0.5, label='SMA 30 Sharpe')
        ax4.legend(fontsize=9)

        # 5. MDD 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = portfolio_metrics.sort_values('MDD (%)', ascending=False)
        colors = ['red' if 'Baseline' in s else 'crimson' for s in sorted_df['Strategy']]
        ax5.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color=colors, alpha=0.7)
        ax5.set_xlabel('MDD (%)', fontsize=11)
        ax5.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Return vs Risk 산점도
        ax6 = fig.add_subplot(gs[2, 1])
        baseline_mask = portfolio_metrics['Strategy'].str.contains('Baseline')

        # 새로운 전략들
        new_strategies = portfolio_metrics[~baseline_mask]
        ax6.scatter(new_strategies['MDD (%)'], new_strategies['CAGR (%)'],
                   s=400, alpha=0.6, c=new_strategies['Sharpe Ratio'],
                   cmap='RdYlGn', vmin=0, vmax=2.5, edgecolors='black', linewidth=2)

        # 베이스라인
        baseline = portfolio_metrics[baseline_mask]
        ax6.scatter(baseline['MDD (%)'], baseline['CAGR (%)'],
                   s=400, alpha=0.8, c='red', marker='*',
                   edgecolors='black', linewidth=2, label='SMA 30 Baseline')

        for idx, row in portfolio_metrics.iterrows():
            strategy_label = row['Strategy'].replace(' Portfolio', '').replace(' (Baseline)', '')
            ax6.annotate(strategy_label,
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=9, ha='center', va='bottom')

        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=10)

        # 컬러바 추가
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=2.5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax6)
        cbar.set_label('Sharpe Ratio', fontsize=10)

        # 7. Profit Factor 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = portfolio_metrics.copy()
        sorted_df = sorted_df[sorted_df['Profit Factor'] != np.inf]
        if len(sorted_df) > 0:
            sorted_df = sorted_df.sort_values('Profit Factor', ascending=True)
            colors = ['red' if 'Baseline' in s else 'green' for s in sorted_df['Strategy']]
            ax7.barh(sorted_df['Strategy'], sorted_df['Profit Factor'], color=colors, alpha=0.7)
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
            linestyle = '--' if 'Baseline' in strategy_name else '-'
            linewidth = 2.5 if 'Baseline' in strategy_name else 2
            alpha = 0.9 if 'Baseline' in strategy_name else 0.7
            ax8.plot(drawdown.index, drawdown, label=strategy_name,
                    linewidth=linewidth, linestyle=linestyle, alpha=alpha)

        ax8.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Drawdown (%)', fontsize=12)
        ax8.set_xlabel('Date', fontsize=12)
        ax8.legend(loc='lower right', fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 전체 타이틀
        fig.suptitle(f'Advanced Trading Strategies Comparison\n'
                    f'Target: Beat SMA 30 (5,942% return, 1.60 Sharpe)\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'고급 트레이딩 전략 성과 비교':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"포트폴리오 구성: 각 종목 동일 비중 (25%)")
        print(f"슬리피지: {self.slippage*100}%")
        print(f"\n🎯 목표: SMA 30 전략(수익률 5,942%, 샤프 1.60) 초과 달성")

        # 포트폴리오 성과
        print("\n" + "-"*150)
        print(f"{'포트폴리오 성과 비교':^150}")
        print("-"*150)
        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()

        # SMA 30 베이스라인 값
        baseline = portfolio_metrics[portfolio_metrics['Strategy'].str.contains('Baseline')].iloc[0]
        baseline_return = baseline['Total Return (%)']
        baseline_sharpe = baseline['Sharpe Ratio']

        # 성과 표시 추가
        portfolio_metrics['vs Baseline'] = portfolio_metrics.apply(
            lambda row: '✅ BETTER' if row['Total Return (%)'] > baseline_return and row['Sharpe Ratio'] > baseline_sharpe
            else '⚠️ PARTIAL' if row['Total Return (%)'] > baseline_return or row['Sharpe Ratio'] > baseline_sharpe
            else '❌ WORSE' if 'Baseline' not in row['Strategy'] else '-', axis=1
        )

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(portfolio_metrics.to_string(index=False))

        # 최고 성과 전략 찾기
        best_strategy = portfolio_metrics.loc[portfolio_metrics['Sharpe Ratio'].idxmax()]
        if 'Baseline' not in best_strategy['Strategy']:
            print("\n" + "="*150)
            print(f"🏆 최고 성과 전략: {best_strategy['Strategy']}")
            print(f"   총 수익률: {best_strategy['Total Return (%)']:.2f}% (vs SMA 30: {baseline_return:.2f}%)")
            print(f"   샤프 비율: {best_strategy['Sharpe Ratio']:.2f} (vs SMA 30: {baseline_sharpe:.2f})")
            print("="*150 + "\n")

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

        # 6. 포트폴리오 비교 시각화
        self.plot_comparison(metrics_df)

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("고급 트레이딩 전략 비교 분석 시작")
    print("목표: SMA 30 전략(5,942% 수익률, 1.60 샤프) 초과 달성")
    print("="*80)

    # 백테스트 실행
    comparison = AdvancedTradingStrategies(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = comparison.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('advanced_strategies_metrics.csv', index=False)
    print("Metrics saved to advanced_strategies_metrics.csv")

    # 각 포트폴리오 상세 결과 저장
    for strategy_name in comparison.portfolio_results.keys():
        filename = f"portfolio_{strategy_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.csv"
        comparison.portfolio_results[strategy_name].to_csv(filename)
        print(f"Portfolio details saved to {filename}")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
