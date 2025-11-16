"""
암호화폐 포트폴리오 전략 비교 분석 (수정 버전)

세 가지 전략을 BTC, ETH, ADA, XRP에 적용하여 동일 비중 포트폴리오 성과 비교:
1. Turtle Trading (터틀트레이딩) - ✅ 현실적인 체결 가격으로 수정
2. RSI 55 전략
3. SMA 30 전략

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


class CryptoPortfolioComparisonFixed:
    """암호화폐 포트폴리오 전략 비교 클래스 (수정 버전)"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트 (default: ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'])
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

    # ==================== 전략 1: Turtle Trading (수정 버전) ====================
    def strategy_turtle_trading(self, df, entry_period=20, exit_period=10):
        """
        터틀 트레이딩 전략 (수정 버전)
        - N일 최고가 돌파 시 매수 → 당일 종가에 체결 (현실적)
        - M일 최저가 하향 돌파 시 매도 → 당일 종가에 체결 (현실적)

        수정 사항:
        - 기존: entry_high/exit_low 가격에 체결 (불가능)
        - 수정: 돌파 시그널 발생 시 당일 종가에 체결 + 슬리피지
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

        # 수익률 계산 (수정 버전)
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                # ✅ 수정: 당일 종가에 매수 (슬리피지 포함)
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['Close'] * (1 + self.slippage)
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                # ✅ 수정: 당일 종가에 매도 (슬리피지 포함)
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                sell_price = df.iloc[i]['Close'] * (1 - self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)
            elif df.iloc[i]['position'] == 1:
                # 포지션 유지
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 2: RSI 55 ====================
    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def strategy_rsi_55(self, df, rsi_period=14, rsi_threshold=55):
        """
        RSI 55 전략
        - RSI >= 55: 매수/보유
        - RSI < 55: 매도 후 현금 보유
        """
        df = df.copy()

        # RSI 계산
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)

        # 매수/매도 신호 생성
        df['signal'] = (df['RSI'] >= rsi_threshold).astype(int)

        # 포지션 변화 감지
        df['position_change'] = df['signal'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

        # 매수/매도 시 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage  # 매수
        slippage_cost[df['position_change'] == -1] = -self.slippage  # 매도

        df['returns'] = df['returns'] + slippage_cost

        # NaN 값 처리
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 3: SMA 30 ====================
    def strategy_sma_30(self, df, sma_period=30):
        """
        SMA 30 교차 전략
        - 가격이 SMA 30 이상일 때 매수 (보유)
        - 가격이 SMA 30 미만일 때 매도 후 현금 보유
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

        # 매수/매도 시 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 4: SMA 30 with Cooldown ====================
    def strategy_sma_30_with_cooldown(self, df, sma_period=30, cooldown_days=3):
        """
        SMA 30 교차 전략 + 매도 후 재매수 금지 기간
        - 가격이 SMA 30 이상일 때 매수 (보유)
        - 가격이 SMA 30 미만일 때 매도 후 현금 보유
        - 매도 후 cooldown_days 동안 재매수 금지

        Args:
            sma_period: SMA 기간 (default: 30)
            cooldown_days: 매도 후 재매수 금지 기간 (일)
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 기본 신호 계산
        df['raw_signal'] = np.where(df['Close'] >= df['SMA'], 1, 0)

        # 매도 후 재매수 금지 로직 적용
        df['position'] = 0
        df['days_since_sell'] = 0

        for i in range(1, len(df)):
            prev_position = df.iloc[i-1]['position']
            current_signal = df.iloc[i]['raw_signal']
            days_since_sell = df.iloc[i-1]['days_since_sell']

            # 매도 발생: 포지션 있었는데 신호가 0으로 전환
            if prev_position == 1 and current_signal == 0:
                df.iloc[i, df.columns.get_loc('position')] = 0
                df.iloc[i, df.columns.get_loc('days_since_sell')] = 1

            # 쿨다운 기간 중: 신호와 관계없이 매수 금지
            elif days_since_sell > 0 and days_since_sell < cooldown_days:
                df.iloc[i, df.columns.get_loc('position')] = 0
                df.iloc[i, df.columns.get_loc('days_since_sell')] = days_since_sell + 1

            # 쿨다운 종료 후 또는 쿨다운 없는 상태에서 신호가 1이면 매수
            elif current_signal == 1 and (days_since_sell == 0 or days_since_sell >= cooldown_days):
                df.iloc[i, df.columns.get_loc('position')] = 1
                df.iloc[i, df.columns.get_loc('days_since_sell')] = 0

            # 그 외: 포지션 없음, 쿨다운 리셋
            else:
                df.iloc[i, df.columns.get_loc('position')] = 0
                if days_since_sell >= cooldown_days:
                    df.iloc[i, df.columns.get_loc('days_since_sell')] = 0
                elif days_since_sell > 0:
                    # 쿨다운 기간이 아직 안 끝났는데 신호가 0인 경우는 위에서 처리됨
                    df.iloc[i, df.columns.get_loc('days_since_sell')] = 0
                else:
                    df.iloc[i, df.columns.get_loc('days_since_sell')] = 0

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 매수/매도 시 슬리피지 적용
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
            'Turtle Trading (Fixed)': lambda df: self.strategy_turtle_trading(df, entry_period=20, exit_period=10),
            'RSI 55': lambda df: self.strategy_rsi_55(df, rsi_period=14, rsi_threshold=55),
            'SMA 30': lambda df: self.strategy_sma_30(df, sma_period=30),
            'SMA 30 + 3D Cooldown': lambda df: self.strategy_sma_30_with_cooldown(df, sma_period=30, cooldown_days=3),
            'SMA 30 + 4D Cooldown': lambda df: self.strategy_sma_30_with_cooldown(df, sma_period=30, cooldown_days=4),
            'SMA 30 + 5D Cooldown': lambda df: self.strategy_sma_30_with_cooldown(df, sma_period=30, cooldown_days=5),
            'SMA 30 + 6D Cooldown': lambda df: self.strategy_sma_30_with_cooldown(df, sma_period=30, cooldown_days=6),
            'SMA 30 + 7D Cooldown': lambda df: self.strategy_sma_30_with_cooldown(df, sma_period=30, cooldown_days=7)
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

        # 개별 종목별 성과 (참고용)
        for strategy_name in self.strategy_results.keys():
            for symbol in self.symbols:
                returns = self.strategy_results[strategy_name][symbol]['returns']
                metrics = self.calculate_metrics(returns, f"{strategy_name} - {symbol.split('_')[0]}")
                metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================
    def plot_comparison(self, metrics_df, save_path='crypto_portfolio_comparison_fixed.png'):
        """포트폴리오 비교 시각화"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # 1. 포트폴리오 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            ax1.plot(cumulative.index, cumulative, label=f'{strategy_name} Portfolio',
                    linewidth=2.5, alpha=0.8)

        ax1.set_title('Portfolio Cumulative Returns Comparison (FIXED) - Equal-Weight: BTC, ETH, ADA, XRP',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 포트폴리오만 필터링
        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()

        # 2. 총 수익률 비교
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = portfolio_metrics.sort_values('Total Return (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
        ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = portfolio_metrics.sort_values('CAGR (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['CAGR (%)']]
        ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = portfolio_metrics.sort_values('MDD (%)', ascending=False)
        ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color='crimson', alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = portfolio_metrics.sort_values('Sharpe Ratio', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Sharpe Ratio']]
        ax5.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Return vs Risk 산점도
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.scatter(portfolio_metrics['MDD (%)'], portfolio_metrics['CAGR (%)'],
                   s=300, alpha=0.6, c=portfolio_metrics['Sharpe Ratio'], cmap='RdYlGn')
        for idx, row in portfolio_metrics.iterrows():
            ax6.annotate(row['Strategy'].replace(' Portfolio', '').replace(' (Fixed)', ''),
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=10, ha='center', va='bottom')
        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 7. Profit Factor 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = portfolio_metrics.copy()
        sorted_df = sorted_df[sorted_df['Profit Factor'] != np.inf]
        if len(sorted_df) > 0:
            sorted_df = sorted_df.sort_values('Profit Factor', ascending=True)
            colors = ['green' if x > 1 else 'red' for x in sorted_df['Profit Factor']]
            ax7.barh(sorted_df['Strategy'], sorted_df['Profit Factor'], color=colors, alpha=0.7)
        ax7.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax7.set_xlabel('Profit Factor', fontsize=11)
        ax7.set_title('Profit Factor Comparison', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8-10. 각 전략별 종목 비교
        turtle_metrics = metrics_df[metrics_df['Strategy'].str.contains('Turtle Trading')].copy()
        if len(turtle_metrics) > 0:
            ax8 = fig.add_subplot(gs[3, 0])
            sorted_df = turtle_metrics.sort_values('Total Return (%)', ascending=True)
            colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
            strategy_labels = [s.replace('Turtle Trading (Fixed) - ', '') for s in sorted_df['Strategy']]
            ax8.barh(strategy_labels, sorted_df['Total Return (%)'], color=colors, alpha=0.7)
            ax8.set_xlabel('Total Return (%)', fontsize=10)
            ax8.set_title('Turtle Trading (Fixed) - By Asset', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3, axis='x')

        rsi_metrics = metrics_df[metrics_df['Strategy'].str.contains('RSI 55')].copy()
        if len(rsi_metrics) > 0:
            ax9 = fig.add_subplot(gs[3, 1])
            sorted_df = rsi_metrics.sort_values('Total Return (%)', ascending=True)
            colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
            strategy_labels = [s.replace('RSI 55 - ', '') for s in sorted_df['Strategy']]
            ax9.barh(strategy_labels, sorted_df['Total Return (%)'], color=colors, alpha=0.7)
            ax9.set_xlabel('Total Return (%)', fontsize=10)
            ax9.set_title('RSI 55 - By Asset', fontsize=12, fontweight='bold')
            ax9.grid(True, alpha=0.3, axis='x')

        sma_metrics = metrics_df[metrics_df['Strategy'].str.contains('SMA 30')].copy()
        if len(sma_metrics) > 0:
            ax10 = fig.add_subplot(gs[3, 2])
            sorted_df = sma_metrics.sort_values('Total Return (%)', ascending=True)
            colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
            strategy_labels = [s.replace('SMA 30 - ', '') for s in sorted_df['Strategy']]
            ax10.barh(strategy_labels, sorted_df['Total Return (%)'], color=colors, alpha=0.7)
            ax10.set_xlabel('Total Return (%)', fontsize=10)
            ax10.set_title('SMA 30 - By Asset', fontsize=12, fontweight='bold')
            ax10.grid(True, alpha=0.3, axis='x')

        # 11. 드로우다운 비교
        ax11 = fig.add_subplot(gs[4, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax11.plot(drawdown.index, drawdown, label=f'{strategy_name}', linewidth=2, alpha=0.7)

        ax11.fill_between(drawdown.index, drawdown, 0, alpha=0.2)
        ax11.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax11.set_ylabel('Drawdown (%)', fontsize=12)
        ax11.set_xlabel('Date', fontsize=12)
        ax11.legend(loc='lower right', fontsize=11)
        ax11.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def plot_individual_ticker_analysis(self, strategy_name, symbol, save_dir='individual_results'):
        """개별 종목별 전략 상세 분석 시각화

        Args:
            strategy_name: 전략 이름
            symbol: 종목 심볼
            save_dir: 저장 디렉토리
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 데이터 가져오기
        result_df = self.strategy_results[strategy_name][symbol].copy()
        original_data = self.data[symbol].copy()

        # 파일명 생성
        strategy_clean = strategy_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        symbol_clean = symbol.split('_')[0]
        save_path = f"{save_dir}/{strategy_clean}_{symbol_clean}_analysis.png"

        # 시각화
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 가격 차트 + 누적 수익률
        ax1 = fig.add_subplot(gs[0, :2])
        ax1_twin = ax1.twinx()

        # 가격 (왼쪽 축)
        ax1.plot(original_data.index, original_data['Close'],
                label='Price', color='gray', linewidth=1.5, alpha=0.6)
        ax1.set_ylabel('Price (KRW)', fontsize=11, color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        # 누적 수익률 (오른쪽 축)
        ax1_twin.plot(result_df.index, result_df['cumulative'],
                     label='Cumulative Return', color='blue', linewidth=2.5)
        ax1_twin.set_ylabel('Cumulative Return', fontsize=11, color='blue')
        ax1_twin.tick_params(axis='y', labelcolor='blue')
        ax1_twin.set_yscale('log')

        ax1.set_title(f'{strategy_name} - {symbol_clean}: Price & Cumulative Returns',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)
        ax1_twin.legend(loc='upper right', fontsize=10)

        # 2. 매수/매도 신호 (포지션이 있는 경우)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'position' in result_df.columns:
            position_changes = result_df['position'].diff()
            buy_signals = result_df[position_changes == 1].index
            sell_signals = result_df[position_changes == -1].index

            ax2.plot(original_data.index, original_data['Close'],
                    color='gray', linewidth=1, alpha=0.5)
            ax2.scatter(buy_signals, original_data.loc[buy_signals, 'Close'],
                       color='green', marker='^', s=100, label='Buy', zorder=5, alpha=0.7)
            ax2.scatter(sell_signals, original_data.loc[sell_signals, 'Close'],
                       color='red', marker='v', s=100, label='Sell', zorder=5, alpha=0.7)
            ax2.set_title('Entry/Exit Signals', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Price (KRW)', fontsize=10)
            ax2.legend(fontsize=9)
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No position data\navailable',
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.set_title('Entry/Exit Signals', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Drawdown 차트
        ax3 = fig.add_subplot(gs[1, :])
        cummax = result_df['cumulative'].cummax()
        drawdown = (result_df['cumulative'] - cummax) / cummax * 100
        ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax3.plot(drawdown.index, drawdown, color='darkred', linewidth=2)
        ax3.set_title('Drawdown Over Time', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # MDD 표시
        mdd_value = drawdown.min()
        mdd_date = drawdown.idxmin()
        ax3.scatter([mdd_date], [mdd_value], color='red', s=200, zorder=5, marker='X')
        ax3.annotate(f'MDD: {mdd_value:.2f}%',
                    xy=(mdd_date, mdd_value),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        # 4. 월별 수익률 히트맵
        ax4 = fig.add_subplot(gs[2, :2])
        monthly_returns = result_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_returns_pivot = monthly_returns.to_frame('returns')
        monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
        heatmap_data = monthly_returns_pivot.pivot(index='year', columns='month', values='returns')

        # 히트맵 그리기
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax4, linewidths=0.5)
        ax4.set_title('Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Month', fontsize=11)
        ax4.set_ylabel('Year', fontsize=11)

        # 5. 수익률 분포
        ax5 = fig.add_subplot(gs[2, 2])
        returns_pct = result_df['returns'][result_df['returns'] != 0] * 100
        if len(returns_pct) > 0:
            ax5.hist(returns_pct, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax5.axvline(x=returns_pct.mean(), color='green', linestyle='--', linewidth=2,
                       label=f'Mean: {returns_pct.mean():.2f}%')
            ax5.set_title('Return Distribution', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Return (%)', fontsize=10)
            ax5.set_ylabel('Frequency', fontsize=10)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3, axis='y')

        # 6. 롤링 샤프 비율 (90일)
        ax6 = fig.add_subplot(gs[3, 0])
        rolling_window = 90
        rolling_sharpe = (result_df['returns'].rolling(rolling_window).mean() /
                         result_df['returns'].rolling(rolling_window).std() * np.sqrt(365))
        ax6.plot(rolling_sharpe.index, rolling_sharpe, color='purple', linewidth=2)
        ax6.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax6.axhline(y=1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1')
        ax6.axhline(y=2, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=2')
        ax6.set_title(f'Rolling Sharpe Ratio ({rolling_window}d)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Sharpe Ratio', fontsize=10)
        ax6.set_xlabel('Date', fontsize=10)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        # 7. 승률 분석
        ax7 = fig.add_subplot(gs[3, 1])
        winning_returns = result_df['returns'][result_df['returns'] > 0]
        losing_returns = result_df['returns'][result_df['returns'] < 0]

        win_count = len(winning_returns)
        loss_count = len(losing_returns)
        total_trades = win_count + loss_count

        if total_trades > 0:
            win_rate = win_count / total_trades * 100
            avg_win = winning_returns.mean() * 100 if win_count > 0 else 0
            avg_loss = losing_returns.mean() * 100 if loss_count > 0 else 0

            bars = ax7.bar(['Wins', 'Losses'], [win_count, loss_count],
                          color=['green', 'red'], alpha=0.7)
            ax7.set_title('Win/Loss Analysis', fontsize=12, fontweight='bold')
            ax7.set_ylabel('Number of Trades', fontsize=10)

            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

            # 텍스트 정보
            info_text = f'Win Rate: {win_rate:.1f}%\n'
            info_text += f'Avg Win: {avg_win:.2f}%\n'
            info_text += f'Avg Loss: {avg_loss:.2f}%\n'
            info_text += f'Total Trades: {total_trades}'

            ax7.text(0.98, 0.98, info_text, transform=ax7.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. 성과 지표 요약
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')

        # 성과 지표 계산
        metrics = self.calculate_metrics(result_df['returns'], f"{strategy_name} - {symbol_clean}")

        metrics_text = f"Performance Metrics\n{'='*30}\n\n"
        metrics_text += f"Total Return: {metrics['Total Return (%)']:.2f}%\n"
        metrics_text += f"CAGR: {metrics['CAGR (%)']:.2f}%\n"
        metrics_text += f"MDD: {metrics['MDD (%)']:.2f}%\n"
        metrics_text += f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}\n"
        metrics_text += f"Win Rate: {metrics['Win Rate (%)']:.2f}%\n"
        metrics_text += f"Total Trades: {metrics['Total Trades']}\n"

        if metrics['Profit Factor'] != np.inf:
            metrics_text += f"Profit Factor: {metrics['Profit Factor']:.2f}\n"
        else:
            metrics_text += f"Profit Factor: N/A\n"

        ax8.text(0.1, 0.95, metrics_text, transform=ax8.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 전체 제목
        fig.suptitle(f'{strategy_name} - {symbol_clean} Detailed Analysis\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=16, fontweight='bold', y=0.995)

        # 저장
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Chart saved: {save_path}")
        plt.close()

        return save_path

    def plot_all_individual_analyses(self, save_dir='individual_results'):
        """모든 전략-종목 조합에 대해 개별 분석 시각화 생성"""
        print("\n" + "="*80)
        print("Creating individual ticker analysis charts...")
        print("="*80 + "\n")

        saved_files = []

        for strategy_name in self.strategy_results.keys():
            print(f"\n>>> Processing {strategy_name}...")
            for symbol in self.symbols:
                print(f"  - {symbol}...")
                save_path = self.plot_individual_ticker_analysis(strategy_name, symbol, save_dir)
                saved_files.append(save_path)

        print("\n" + "="*80)
        print(f"Individual analysis completed! {len(saved_files)} charts saved.")
        print(f"Location: {save_dir}/")
        print("="*80 + "\n")

        return saved_files

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'암호화폐 포트폴리오 전략 성과 비교 (수정 버전)':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"포트폴리오 구성: 각 종목 동일 비중 (25%)")
        print(f"슬리피지: {self.slippage*100}%")
        print(f"\n✅ 터틀트레이딩 수정 사항:")
        print(f"  - 기존: entry_high/exit_low 가격에 체결 (비현실적)")
        print(f"  - 수정: 돌파 신호 발생 시 당일 종가에 체결 + 슬리피지 (현실적)")

        # 포트폴리오 성과
        print("\n" + "-"*150)
        print(f"{'포트폴리오 성과 비교 (동일 비중)':^150}")
        print("-"*150)
        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(portfolio_metrics.to_string(index=False))

        # 종목별 성과
        print("\n" + "-"*150)
        print(f"{'종목별 전략 성과':^150}")
        print("-"*150)
        asset_metrics = metrics_df[~metrics_df['Strategy'].str.contains('Portfolio')].copy()
        print(asset_metrics.to_string(index=False))

        print("\n" + "="*150 + "\n")

    def run_analysis(self, create_individual_charts=True):
        """전체 분석 실행

        Args:
            create_individual_charts: 개별 종목 차트 생성 여부 (default: True)
        """
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

        # 7. 개별 종목 시각화 (옵션)
        if create_individual_charts:
            self.plot_all_individual_analyses()

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("암호화폐 포트폴리오 전략 비교 분석 시작 (수정 버전)")
    print("="*80)

    # 백테스트 실행
    comparison = CryptoPortfolioComparisonFixed(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행 (개별 차트 생성 비활성화)
    metrics_df = comparison.run_analysis(create_individual_charts=False)

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('crypto_portfolio_metrics_fixed.csv', index=False)
    print("Metrics saved to crypto_portfolio_metrics_fixed.csv")

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
