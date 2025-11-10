"""
암호화폐 전략 확장 비교 분석

SMA30 전략보다 좋은 전략 찾기
여러 전략들을 테스트하고 비교:
1. SMA 20 (빠른 이동평균)
2. SMA 30 (기준선)
3. SMA 50 (느린 이동평균)
4. EMA 30 (지수 이동평균)
5. SMA 10/30 Crossover
6. SMA 30 + RSI 50 Combined
7. RSI 60
8. RSI 65
9. Price > SMA20 AND SMA20 > SMA50 (트렌드 필터)
10. Bollinger Bands Breakout

레버리지는 사용하지 않음
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


class ExtendedStrategyComparison:
    """확장된 전략 비교 클래스"""

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

    # ==================== 보조 함수 ====================
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

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return sma, upper_band, lower_band

    # ==================== 전략 1: SMA 20 ====================
    def strategy_sma_20(self, df):
        """SMA 20 전략 (빠른 반응)"""
        df = df.copy()
        df['SMA'] = df['Close'].rolling(window=20).mean()
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

    # ==================== 전략 2: SMA 30 (기준선) ====================
    def strategy_sma_30(self, df):
        """SMA 30 전략 (기준선)"""
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

    # ==================== 전략 3: SMA 50 ====================
    def strategy_sma_50(self, df):
        """SMA 50 전략 (느린 반응, 안정적)"""
        df = df.copy()
        df['SMA'] = df['Close'].rolling(window=50).mean()
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

    # ==================== 전략 4: EMA 30 ====================
    def strategy_ema_30(self, df):
        """EMA 30 전략 (최근 가격에 더 많은 가중치)"""
        df = df.copy()
        df['EMA'] = self.calculate_ema(df['Close'], 30)
        df['position'] = np.where(df['Close'] >= df['EMA'], 1, 0)
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 5: SMA 10/30 Crossover ====================
    def strategy_sma_crossover(self, df):
        """SMA 10/30 크로스오버 전략"""
        df = df.copy()
        df['SMA_fast'] = df['Close'].rolling(window=10).mean()
        df['SMA_slow'] = df['Close'].rolling(window=30).mean()
        df['position'] = np.where(df['SMA_fast'] > df['SMA_slow'], 1, 0)
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 6: SMA 30 + RSI 50 Combined ====================
    def strategy_sma_rsi_combined(self, df):
        """SMA 30 + RSI 50 결합 전략 (두 조건 모두 충족 시 매수)"""
        df = df.copy()
        df['SMA'] = df['Close'].rolling(window=30).mean()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)

        # 두 조건 모두 충족 시 매수
        df['position'] = np.where((df['Close'] >= df['SMA']) & (df['RSI'] >= 50), 1, 0)
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 7: RSI 60 ====================
    def strategy_rsi_60(self, df):
        """RSI 60 전략 (더 강한 모멘텀 요구)"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['position'] = np.where(df['RSI'] >= 60, 1, 0)
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 8: RSI 65 ====================
    def strategy_rsi_65(self, df):
        """RSI 65 전략 (매우 강한 모멘텀만 선택)"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['position'] = np.where(df['RSI'] >= 65, 1, 0)
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 9: Triple SMA Trend Filter ====================
    def strategy_triple_sma(self, df):
        """삼중 SMA 트렌드 필터 (Price > SMA20 AND SMA20 > SMA50)"""
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()

        # 가격이 SMA20 위이고, SMA20이 SMA50 위일 때 매수 (강한 상승 트렌드)
        df['position'] = np.where((df['Close'] >= df['SMA20']) & (df['SMA20'] > df['SMA50']), 1, 0)
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 10: Bollinger Bands ====================
    def strategy_bollinger_bands(self, df):
        """볼린저 밴드 브레이크아웃 전략 (가격 > 중간선)"""
        df = df.copy()
        middle_band, upper_band, lower_band = self.calculate_bollinger_bands(df['Close'], 20, 2)
        df['BB_middle'] = middle_band
        df['BB_upper'] = upper_band
        df['BB_lower'] = lower_band

        # 가격이 중간선 위에 있을 때 매수
        df['position'] = np.where(df['Close'] >= df['BB_middle'], 1, 0)
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
        """모든 전략을 모든 종목에 대해 실행"""
        strategies = {
            'SMA 20': self.strategy_sma_20,
            'SMA 30 (Baseline)': self.strategy_sma_30,
            'SMA 50': self.strategy_sma_50,
            'EMA 30': self.strategy_ema_30,
            'SMA 10/30 Crossover': self.strategy_sma_crossover,
            'SMA 30 + RSI 50': self.strategy_sma_rsi_combined,
            'RSI 60': self.strategy_rsi_60,
            'RSI 65': self.strategy_rsi_65,
            'Triple SMA Trend': self.strategy_triple_sma,
            'Bollinger Bands': self.strategy_bollinger_bands,
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
        """모든 전략의 포트폴리오 성과 지표 계산"""
        metrics_list = []

        for strategy_name in self.portfolio_results.keys():
            returns = self.portfolio_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, strategy_name)
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================
    def plot_comparison(self, metrics_df, save_path='strategy_comparison_extended.png'):
        """전략 비교 시각화"""
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 포트폴리오 누적 수익률 비교 (로그 스케일)
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            linewidth = 3 if 'Baseline' in strategy_name else 2
            alpha = 1.0 if 'Baseline' in strategy_name else 0.7
            ax1.plot(cumulative.index, cumulative, label=strategy_name,
                    linewidth=linewidth, alpha=alpha)

        ax1.set_title('Cumulative Returns Comparison (Log Scale) - All Strategies',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 총 수익률 비교 (정렬)
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True)
        colors = ['gold' if 'Baseline' in x else ('green' if y > 0 else 'red')
                  for x, y in zip(sorted_df['Strategy'], sorted_df['Total Return (%)'])]
        ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Ranking', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['gold' if 'Baseline' in x else ('green' if y > 0 else 'red')
                  for x, y in zip(sorted_df['Strategy'], sorted_df['CAGR (%)'])]
        ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Ranking', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        colors = ['gold' if 'Baseline' in x else 'crimson' for x in sorted_df['Strategy']]
        ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color=colors, alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['gold' if 'Baseline' in x else ('green' if y > 0 else 'red')
                  for x, y in zip(sorted_df['Strategy'], sorted_df['Sharpe Ratio'])]
        ax5.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Ranking', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Return vs Risk 산점도
        ax6 = fig.add_subplot(gs[2, 1])
        colors_scatter = ['gold' if 'Baseline' in x else 'steelblue' for x in metrics_df['Strategy']]
        sizes = [400 if 'Baseline' in x else 200 for x in metrics_df['Strategy']]
        ax6.scatter(metrics_df['MDD (%)'], metrics_df['CAGR (%)'],
                   s=sizes, alpha=0.6, c=colors_scatter, edgecolors='black', linewidths=1.5)
        for idx, row in metrics_df.iterrows():
            label = row['Strategy']
            if 'Baseline' in label:
                label = label + ' ⭐'
            ax6.annotate(label,
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=8, ha='left', va='bottom')
        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk (Gold = Baseline)', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # 7. 승률 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=True)
        colors = ['gold' if 'Baseline' in x else 'teal' for x in sorted_df['Strategy']]
        ax7.barh(sorted_df['Strategy'], sorted_df['Win Rate (%)'], color=colors, alpha=0.7)
        ax7.set_xlabel('Win Rate (%)', fontsize=11)
        ax7.set_title('Win Rate Ranking', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')
        ax7.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5)

        plt.suptitle('Extended Strategy Comparison: Finding Better Strategies than SMA 30',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_results(self, metrics_df):
        """결과 출력"""
        print("\n" + "="*120)
        print(f"{'전략 성과 비교 결과':^120}")
        print("="*120)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"포트폴리오: 동일 비중 (각 25%)")
        print(f"슬리피지: {self.slippage*100}%")
        print(f"레버리지: 사용 안 함 (1x)")

        # CAGR 기준으로 정렬
        sorted_metrics = metrics_df.sort_values('CAGR (%)', ascending=False)

        print("\n" + "-"*120)
        print(f"{'전략별 성과 (CAGR 순위)':^120}")
        print("-"*120)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}')
        print(sorted_metrics.to_string(index=False))

        # 베스트 전략 하이라이트
        best_strategy = sorted_metrics.iloc[0]
        baseline = metrics_df[metrics_df['Strategy'].str.contains('Baseline')].iloc[0]

        print("\n" + "="*120)
        print(f"{'🏆 최고 성과 전략':^120}")
        print("="*120)
        print(f"\n전략명: {best_strategy['Strategy']}")
        print(f"  - Total Return: {best_strategy['Total Return (%)']:.2f}%")
        print(f"  - CAGR: {best_strategy['CAGR (%)']:.2f}%")
        print(f"  - MDD: {best_strategy['MDD (%)']:.2f}%")
        print(f"  - Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")
        print(f"  - Win Rate: {best_strategy['Win Rate (%)']:.2f}%")

        print(f"\n기준선 (SMA 30) 대비:")
        print(f"  - CAGR 차이: {best_strategy['CAGR (%)'] - baseline['CAGR (%)']:+.2f}%p")
        print(f"  - Total Return 차이: {best_strategy['Total Return (%)'] - baseline['Total Return (%)']:+.2f}%p")
        print(f"  - Sharpe Ratio 차이: {best_strategy['Sharpe Ratio'] - baseline['Sharpe Ratio']:+.2f}")

        print("\n" + "="*120 + "\n")

    def run_analysis(self):
        """전체 분석 실행"""
        self.load_data()
        self.run_all_strategies()
        self.create_portfolios()
        metrics_df = self.calculate_all_metrics()
        self.print_results(metrics_df)
        self.plot_comparison(metrics_df)

        # 결과 저장
        metrics_df.to_csv('strategy_comparison_extended_results.csv', index=False)
        print("Results saved to: strategy_comparison_extended_results.csv\n")

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("확장 전략 비교 분석 시작: SMA 30보다 좋은 전략 찾기")
    print("="*80)

    comparison = ExtendedStrategyComparison(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )

    metrics_df = comparison.run_analysis()

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
