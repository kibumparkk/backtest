"""
유명한 돌파 전략 10개 구현 및 성과 비교

구현된 전략:
1. Donchian Channel Breakout (돈치안 채널 돌파)
2. Volatility Breakout (변동성 돌파)
3. Range Breakout (레인지 돌파)
4. Opening Range Breakout (시가 레인지 돌파)
5. ATR Breakout (ATR 돌파)
6. Turtle Trading (터틀 트레이딩)
7. Bollinger Band Breakout (볼린저 밴드 돌파)
8. High/Low Breakout (고/저가 돌파)
9. Momentum Breakout (모멘텀 돌파)
10. Keltner Channel Breakout (켈트너 채널 돌파)
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


class BreakthroughStrategiesBacktest:
    """돌파 전략 통합 백테스트 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbol: 티커 심볼 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = None
        self.results = {}

    def load_data(self):
        """데이터 로드"""
        file_path = f'chart_day/{self.symbol}.parquet'
        print(f"Loading data from {file_path}...")
        df = pd.read_parquet(file_path)

        # 컬럼명 변경 (소문자 -> 대문자)
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df
        print(f"Loaded {len(self.data)} data points from {df.index[0]} to {df.index[-1]}")
        return self.data

    # ==================== 전략 1: Donchian Channel Breakout ====================
    def strategy_donchian_channel(self, period=20):
        """
        돈치안 채널 돌파 전략
        - N일 최고가 돌파 시 매수, N일 최저가 하향 돌파 시 매도
        """
        df = self.data.copy()

        # 돈치안 채널 계산
        df['upper_band'] = df['High'].rolling(window=period).max().shift(1)
        df['lower_band'] = df['Low'].rolling(window=period).min().shift(1)

        # 매수/매도 신호
        df['position'] = 0
        for i in range(1, len(df)):
            # 전일 포지션 유지
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 최고가 돌파 시 매수 (포지션이 없을 때)
            if df.iloc[i]['High'] > df.iloc[i]['upper_band'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 최저가 하향 돌파 시 매도
            elif df.iloc[i]['Low'] < df.iloc[i]['lower_band'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan
        df['sell_price'] = np.nan

        for i in range(1, len(df)):
            # 매수 진입
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['upper_band']

            # 매도 청산
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                df.iloc[i, df.columns.get_loc('sell_price')] = df.iloc[i]['lower_band']
                df.iloc[i, df.columns.get_loc('returns')] = (df.iloc[i]['sell_price'] / buy_price - 1) - self.slippage

            # 포지션 유지
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 2: Volatility Breakout ====================
    def strategy_volatility_breakout(self, k=0.5):
        """
        변동성 돌파 전략 (래리 윌리엄스)
        - 목표가 = 시가 + (전일 고가 - 전일 저가) * k
        - 목표가 돌파 시 매수, 당일 종가에 매도
        """
        df = self.data.copy()

        # 전일 변동성 계산
        df['prev_high'] = df['High'].shift(1)
        df['prev_low'] = df['Low'].shift(1)
        df['volatility'] = df['prev_high'] - df['prev_low']

        # 목표가 계산
        df['target_price'] = df['Open'] + df['volatility'] * k

        # 매수 신호: 고가가 목표가를 돌파
        df['buy_signal'] = df['High'] >= df['target_price']
        df['buy_price'] = np.where(df['buy_signal'], df['target_price'], np.nan)
        df['sell_price'] = np.where(df['buy_signal'], df['Close'], np.nan)

        # 수익률 계산
        df['returns'] = np.where(
            df['buy_signal'],
            (df['sell_price'] / df['buy_price'] - 1) - self.slippage,
            0
        )

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 3: Range Breakout ====================
    def strategy_range_breakout(self, period=10, threshold=0.02):
        """
        레인지 돌파 전략
        - N일간 레인지 폭이 작을 때 (횡보장) 상단 돌파 시 매수
        - 하단 돌파 시 매도
        """
        df = self.data.copy()

        # N일 고가/저가
        df['range_high'] = df['High'].rolling(window=period).max().shift(1)
        df['range_low'] = df['Low'].rolling(window=period).min().shift(1)
        df['range_width'] = (df['range_high'] - df['range_low']) / df['range_low']

        # 레인지가 좁은 구간 (횡보장)
        df['is_consolidation'] = df['range_width'] < threshold

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 횡보장에서 상단 돌파 시 매수
            if (df.iloc[i-1]['is_consolidation'] and
                df.iloc[i]['High'] > df.iloc[i]['range_high'] and
                df.iloc[i-1]['position'] == 0):
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 하단 돌파 시 매도
            elif df.iloc[i]['Low'] < df.iloc[i]['range_low'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['range_high']
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                df.iloc[i, df.columns.get_loc('returns')] = (df.iloc[i]['range_low'] / buy_price - 1) - self.slippage
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 4: Opening Range Breakout ====================
    def strategy_opening_range_breakout(self, lookback=5):
        """
        시가 레인지 돌파 전략
        - N일 평균 시가 대비 현재 시가가 높으면 매수 신호
        - 당일 종가에 청산
        """
        df = self.data.copy()

        # 평균 시가 계산
        df['avg_open'] = df['Open'].rolling(window=lookback).mean().shift(1)
        df['open_diff'] = (df['Open'] - df['avg_open']) / df['avg_open']

        # 시가가 평균보다 높으면 매수
        df['buy_signal'] = df['open_diff'] > 0
        df['buy_price'] = np.where(df['buy_signal'], df['Open'], np.nan)
        df['sell_price'] = np.where(df['buy_signal'], df['Close'], np.nan)

        # 수익률 계산
        df['returns'] = np.where(
            df['buy_signal'],
            (df['sell_price'] / df['buy_price'] - 1) - self.slippage,
            0
        )

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 5: ATR Breakout ====================
    def strategy_atr_breakout(self, period=14, multiplier=2):
        """
        ATR 돌파 전략
        - ATR(Average True Range) 기반 변동성 돌파
        - 시가 + ATR * multiplier 돌파 시 매수
        """
        df = self.data.copy()

        # True Range 계산
        df['prev_close'] = df['Close'].shift(1)
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['prev_close'])
        df['tr3'] = abs(df['Low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # ATR 계산
        df['atr'] = df['tr'].rolling(window=period).mean()

        # 목표가 = 시가 + ATR * multiplier
        df['target_price'] = df['Open'] + df['atr'].shift(1) * multiplier

        # 매수 신호
        df['buy_signal'] = df['High'] >= df['target_price']
        df['buy_price'] = np.where(df['buy_signal'], df['target_price'], np.nan)
        df['sell_price'] = np.where(df['buy_signal'], df['Close'], np.nan)

        # 수익률 계산
        df['returns'] = np.where(
            df['buy_signal'],
            (df['sell_price'] / df['buy_price'] - 1) - self.slippage,
            0
        )

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 6: Turtle Trading ====================
    def strategy_turtle_trading(self, entry_period=20, exit_period=10):
        """
        터틀 트레이딩 전략
        - N일 최고가 돌파 시 매수
        - M일 최저가 하향 돌파 시 매도
        """
        df = self.data.copy()

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
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['entry_high']
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                df.iloc[i, df.columns.get_loc('returns')] = (df.iloc[i]['exit_low'] / buy_price - 1) - self.slippage
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 7: Bollinger Band Breakout ====================
    def strategy_bollinger_breakout(self, period=20, std_dev=2):
        """
        볼린저 밴드 돌파 전략
        - 상단 밴드 돌파 시 매수
        - 하단 밴드 하향 돌파 시 매도
        """
        df = self.data.copy()

        # 볼린저 밴드 계산
        df['ma'] = df['Close'].rolling(window=period).mean()
        df['std'] = df['Close'].rolling(window=period).std()
        df['upper_band'] = df['ma'] + (df['std'] * std_dev)
        df['lower_band'] = df['ma'] - (df['std'] * std_dev)

        df['upper_band'] = df['upper_band'].shift(1)
        df['lower_band'] = df['lower_band'].shift(1)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 상단 밴드 돌파 시 매수
            if df.iloc[i]['High'] > df.iloc[i]['upper_band'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 하단 밴드 하향 돌파 시 매도
            elif df.iloc[i]['Low'] < df.iloc[i]['lower_band'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['upper_band']
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                df.iloc[i, df.columns.get_loc('returns')] = (df.iloc[i]['lower_band'] / buy_price - 1) - self.slippage
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 8: High/Low Breakout ====================
    def strategy_high_low_breakout(self, period=5):
        """
        고/저가 돌파 전략
        - N일 최고가 갱신 시 매수
        - N일 최저가 갱신 시 매도
        """
        df = self.data.copy()

        # N일 최고가/최저가
        df['highest'] = df['High'].rolling(window=period).max().shift(1)
        df['lowest'] = df['Low'].rolling(window=period).min().shift(1)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 최고가 돌파 시 매수
            if df.iloc[i]['High'] > df.iloc[i]['highest'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 최저가 하향 돌파 시 매도
            elif df.iloc[i]['Low'] < df.iloc[i]['lowest'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['highest']
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                df.iloc[i, df.columns.get_loc('returns')] = (df.iloc[i]['lowest'] / buy_price - 1) - self.slippage
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 9: Momentum Breakout ====================
    def strategy_momentum_breakout(self, period=10, threshold=0.05):
        """
        모멘텀 돌파 전략
        - N일 수익률이 임계값 이상이면 매수 (강한 상승 모멘텀)
        - 당일 종가에 청산
        """
        df = self.data.copy()

        # 모멘텀 계산 (N일 수익률)
        df['momentum'] = (df['Close'] / df['Close'].shift(period) - 1)

        # 모멘텀이 임계값 이상이면 매수
        df['buy_signal'] = df['momentum'].shift(1) > threshold
        df['buy_price'] = np.where(df['buy_signal'], df['Open'], np.nan)
        df['sell_price'] = np.where(df['buy_signal'], df['Close'], np.nan)

        # 수익률 계산
        df['returns'] = np.where(
            df['buy_signal'],
            (df['sell_price'] / df['buy_price'] - 1) - self.slippage,
            0
        )

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 10: Keltner Channel Breakout ====================
    def strategy_keltner_channel(self, period=20, atr_period=10, multiplier=2):
        """
        켈트너 채널 돌파 전략
        - EMA ± (ATR * multiplier) 채널
        - 상단 채널 돌파 시 매수
        - 하단 채널 하향 돌파 시 매도
        """
        df = self.data.copy()

        # EMA 계산
        df['ema'] = df['Close'].ewm(span=period, adjust=False).mean()

        # ATR 계산
        df['prev_close'] = df['Close'].shift(1)
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['prev_close'])
        df['tr3'] = abs(df['Low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=atr_period).mean()

        # 켈트너 채널
        df['upper_channel'] = df['ema'] + (df['atr'] * multiplier)
        df['lower_channel'] = df['ema'] - (df['atr'] * multiplier)

        df['upper_channel'] = df['upper_channel'].shift(1)
        df['lower_channel'] = df['lower_channel'].shift(1)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 상단 채널 돌파 시 매수
            if df.iloc[i]['High'] > df.iloc[i]['upper_channel'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 하단 채널 하향 돌파 시 매도
            elif df.iloc[i]['Low'] < df.iloc[i]['lower_channel'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['upper_channel']
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                df.iloc[i, df.columns.get_loc('returns')] = (df.iloc[i]['lower_channel'] / buy_price - 1) - self.slippage
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 성과 지표 계산 ====================
    def calculate_metrics(self, df, strategy_name):
        """전략 성과 지표 계산"""
        # 거래 횟수
        total_trades = (df['returns'] != 0).sum()
        winning_trades = (df['returns'] > 0).sum()
        losing_trades = (df['returns'] < 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # 수익률
        total_return = (df['cumulative'].iloc[-1] - 1) * 100

        # 연간 수익률 (CAGR)
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['cumulative'].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # MDD
        cummax = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - cummax) / cummax
        mdd = drawdown.min() * 100

        # 샤프 비율
        returns = df['returns']
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0

        # Profit Factor
        total_profit = df[df['returns'] > 0]['returns'].sum()
        total_loss = abs(df[df['returns'] < 0]['returns'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        # 평균 수익/손실
        avg_win = df[df['returns'] > 0]['returns'].mean() * 100 if winning_trades > 0 else 0
        avg_loss = df[df['returns'] < 0]['returns'].mean() * 100 if losing_trades > 0 else 0

        return {
            'Strategy': strategy_name,
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades,
            'Avg Win (%)': avg_win,
            'Avg Loss (%)': avg_loss,
            'Profit Factor': profit_factor
        }

    # ==================== 모든 전략 실행 ====================
    def run_all_strategies(self):
        """모든 전략 실행 및 결과 수집"""
        strategies = [
            ('Donchian Channel', lambda: self.strategy_donchian_channel(period=20)),
            ('Volatility Breakout', lambda: self.strategy_volatility_breakout(k=0.5)),
            ('Range Breakout', lambda: self.strategy_range_breakout(period=10)),
            ('Opening Range', lambda: self.strategy_opening_range_breakout(lookback=5)),
            ('ATR Breakout', lambda: self.strategy_atr_breakout(period=14, multiplier=2)),
            ('Turtle Trading', lambda: self.strategy_turtle_trading(entry_period=20, exit_period=10)),
            ('Bollinger Band', lambda: self.strategy_bollinger_breakout(period=20, std_dev=2)),
            ('High/Low Breakout', lambda: self.strategy_high_low_breakout(period=5)),
            ('Momentum Breakout', lambda: self.strategy_momentum_breakout(period=10, threshold=0.05)),
            ('Keltner Channel', lambda: self.strategy_keltner_channel(period=20, atr_period=10, multiplier=2))
        ]

        metrics_list = []

        for strategy_name, strategy_func in strategies:
            print(f"\nRunning {strategy_name}...")
            df_result = strategy_func()
            self.results[strategy_name] = df_result

            # 성과 지표 계산
            metrics = self.calculate_metrics(df_result, strategy_name)
            metrics_list.append(metrics)

        # Buy & Hold 추가
        df_bh = self.data.copy()
        df_bh['cumulative'] = df_bh['Close'] / df_bh['Close'].iloc[0]
        self.results['Buy & Hold'] = df_bh

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================
    def plot_comparison(self, metrics_df, save_path='breakthrough_strategies_comparison.png'):
        """전략 비교 시각화"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name, df in self.results.items():
            if strategy_name != 'Buy & Hold':
                ax1.plot(df.index, df['cumulative'], label=strategy_name, linewidth=1.5, alpha=0.8)

        # Buy & Hold 추가
        ax1.plot(self.results['Buy & Hold'].index,
                self.results['Buy & Hold']['cumulative'],
                label='Buy & Hold', linewidth=2, linestyle='--', color='black', alpha=0.7)

        ax1.set_title('Cumulative Returns Comparison - All Strategies', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', ncol=3, fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 총 수익률 비교
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
        ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['CAGR (%)']]
        ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color='crimson', alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Sharpe Ratio']]
        ax5.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. 승률 비교
        ax6 = fig.add_subplot(gs[2, 1])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=True)
        ax6.barh(sorted_df['Strategy'], sorted_df['Win Rate (%)'], color='skyblue', alpha=0.7)
        ax6.set_xlabel('Win Rate (%)', fontsize=11)
        ax6.set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        # 7. 거래 횟수 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.sort_values('Total Trades', ascending=True)
        ax7.barh(sorted_df['Strategy'], sorted_df['Total Trades'], color='orange', alpha=0.7)
        ax7.set_xlabel('Total Trades', fontsize=11)
        ax7.set_title('Total Trades Comparison', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8. Profit Factor 비교
        ax8 = fig.add_subplot(gs[3, 0])
        sorted_df = metrics_df.sort_values('Profit Factor', ascending=True)
        # Profit Factor가 무한대인 경우 처리
        sorted_df_clean = sorted_df[sorted_df['Profit Factor'] != np.inf].copy()
        if len(sorted_df_clean) > 0:
            colors = ['green' if x > 1 else 'red' for x in sorted_df_clean['Profit Factor']]
            ax8.barh(sorted_df_clean['Strategy'], sorted_df_clean['Profit Factor'], color=colors, alpha=0.7)
        ax8.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax8.set_xlabel('Profit Factor', fontsize=11)
        ax8.set_title('Profit Factor Comparison', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')

        # 9. Return vs Risk (MDD) 산점도
        ax9 = fig.add_subplot(gs[3, 1])
        ax9.scatter(metrics_df['MDD (%)'], metrics_df['CAGR (%)'],
                   s=200, alpha=0.6, c=metrics_df['Sharpe Ratio'], cmap='RdYlGn')
        for idx, row in metrics_df.iterrows():
            ax9.annotate(row['Strategy'],
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=8, ha='center', va='bottom')
        ax9.set_xlabel('MDD (%)', fontsize=11)
        ax9.set_ylabel('CAGR (%)', fontsize=11)
        ax9.set_title('Return vs Risk (colored by Sharpe)', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 10. 승률 vs Profit Factor
        ax10 = fig.add_subplot(gs[3, 2])
        # Profit Factor가 무한대가 아닌 경우만
        metrics_df_clean = metrics_df[metrics_df['Profit Factor'] != np.inf].copy()
        if len(metrics_df_clean) > 0:
            ax10.scatter(metrics_df_clean['Win Rate (%)'], metrics_df_clean['Profit Factor'],
                        s=200, alpha=0.6, c=metrics_df_clean['Total Return (%)'], cmap='RdYlGn')
            for idx, row in metrics_df_clean.iterrows():
                ax10.annotate(row['Strategy'],
                            (row['Win Rate (%)'], row['Profit Factor']),
                            fontsize=8, ha='center', va='bottom')
        ax10.set_xlabel('Win Rate (%)', fontsize=11)
        ax10.set_ylabel('Profit Factor', fontsize=11)
        ax10.set_title('Win Rate vs Profit Factor', fontsize=12, fontweight='bold')
        ax10.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax10.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.show()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'돌파 전략 성과 비교 (슬리피지 0.2% 반영)':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {self.symbol}")
        print(f"슬리피지: {self.slippage*100}%")
        print("\n" + "-"*150)

        # 데이터프레임을 보기 좋게 포맷팅
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')

        print(metrics_df.to_string(index=False))
        print("\n" + "="*150 + "\n")


def main():
    """메인 함수"""
    # 백테스트 실행
    backtest = BreakthroughStrategiesBacktest(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 데이터 로드
    backtest.load_data()

    # 모든 전략 실행
    print("\n" + "="*80)
    print("10가지 돌파 전략 백테스트 시작...")
    print("="*80)

    metrics_df = backtest.run_all_strategies()

    # 결과 출력
    backtest.print_metrics_table(metrics_df)

    # 시각화
    backtest.plot_comparison(metrics_df, save_path='breakthrough_strategies_comparison.png')

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('breakthrough_strategies_metrics.csv', index=False)
    print("Metrics saved to breakthrough_strategies_metrics.csv")

    # 각 전략의 상세 결과 저장
    for strategy_name, df in backtest.results.items():
        if strategy_name != 'Buy & Hold':
            filename = f"strategy_{strategy_name.replace('/', '_').replace(' ', '_').lower()}.csv"
            df.to_csv(filename)
            print(f"Strategy details saved to {filename}")

    print("\n" + "="*80)
    print("백테스트 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
