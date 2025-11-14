"""
SMA30 전략 with ATR 기반 변동성 조절 백테스트

전략 비교:
1. Buy & Hold (벤치마크)
2. SMA 30 전략 (기본)
3. SMA 30 + ATR 변동성 조절 전략

ATR 변동성 조절 로직:
- ATR(Average True Range)을 사용하여 최근 변동성 측정
- 변동성이 높을 때 포지션 크기 축소 (리스크 관리)
- 변동성이 낮을 때 포지션 크기 확대 (수익 극대화)
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


class SMA30ATRBacktest:
    """SMA30 + ATR 변동성 조절 백테스트 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
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

    # ==================== 벤치마크: Buy & Hold ====================
    def strategy_buy_and_hold(self, df):
        """
        Buy & Hold 전략 (벤치마크)
        - 시작 시점에 매수하여 끝까지 보유
        """
        df = df.copy()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['daily_price_return']

        # 첫 거래일에 슬리피지 적용
        if len(df) > 0:
            df.iloc[0, df.columns.get_loc('returns')] = -self.slippage

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        df['position'] = 1  # 항상 보유

        return df

    # ==================== 전략 1: SMA 30 ====================
    def strategy_sma_30(self, df, sma_period=30):
        """
        SMA 30 교차 전략
        - 전일종가 > SMA 30: 매수 (보유)
        - 전일종가 < SMA 30: 매도 후 현금 보유
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 전일종가 기준으로 포지션 결정
        df['prev_close'] = df['Close'].shift(1)
        df['position'] = np.where(df['prev_close'] >= df['SMA'], 1, 0)

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

    # ==================== ATR 계산 ====================
    def calculate_atr(self, df, period=14):
        """
        ATR (Average True Range) 계산

        Args:
            df: OHLC 데이터프레임
            period: ATR 계산 기간 (default: 14)

        Returns:
            ATR 시리즈
        """
        df = df.copy()

        # True Range 계산
        df['h-l'] = df['High'] - df['Low']
        df['h-pc'] = abs(df['High'] - df['Close'].shift(1))
        df['l-pc'] = abs(df['Low'] - df['Close'].shift(1))

        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

        # ATR = TR의 이동평균
        atr = df['tr'].rolling(window=period).mean()

        return atr

    # ==================== 전략 2: SMA 30 + ATR 변동성 조절 ====================
    def strategy_sma_30_atr(self, df, sma_period=30, atr_period=14, lookback_period=100):
        """
        SMA 30 + ATR 변동성 조절 전략

        - 기본: 전일종가 > SMA 30일 때 매수
        - 추가: ATR 기반 변동성에 따라 포지션 크기 조절

        포지션 크기 조절 (ATR percentile 기반):
        - ATR percentile 0-25% (낮은 변동성): 100% 투자
        - ATR percentile 25-50%: 75% 투자
        - ATR percentile 50-75%: 50% 투자
        - ATR percentile 75-100% (높은 변동성): 25% 투자

        Args:
            df: OHLC 데이터프레임
            sma_period: SMA 기간
            atr_period: ATR 계산 기간
            lookback_period: ATR percentile 계산을 위한 과거 기간
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # ATR 계산
        df['ATR'] = self.calculate_atr(df, period=atr_period)

        # ATR의 백분위수 계산 (과거 lookback_period일 기준)
        df['ATR_percentile'] = df['ATR'].rolling(window=lookback_period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # ATR percentile에 따른 포지션 크기 결정
        def get_position_size(percentile):
            if pd.isna(percentile):
                return 1.0
            elif percentile < 0.25:
                return 1.00  # 낮은 변동성 = 100% 투자
            elif percentile < 0.50:
                return 0.75  # 중저 변동성 = 75% 투자
            elif percentile < 0.75:
                return 0.50  # 중고 변동성 = 50% 투자
            else:
                return 0.25  # 높은 변동성 = 25% 투자

        df['position_size'] = df['ATR_percentile'].apply(get_position_size)

        # 전일종가 기준으로 매수/매도 신호
        df['prev_close'] = df['Close'].shift(1)
        df['signal'] = np.where(df['prev_close'] >= df['SMA'], 1, 0)

        # 최종 포지션 = 신호 * 포지션 크기
        df['position'] = df['signal'] * df['position_size']

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 매수/매도 시 슬리피지 적용
        # 포지션이 변경될 때마다 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        position_changed = df['position_change'] != 0
        slippage_cost[position_changed] = -self.slippage * abs(df.loc[position_changed, 'position_change'])

        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 실행 ====================
    def run_all_strategies(self):
        """모든 전략을 모든 종목에 대해 실행"""
        strategies = {
            'Buy & Hold': lambda df: self.strategy_buy_and_hold(df),
            'SMA 30': lambda df: self.strategy_sma_30(df, sma_period=30),
            'SMA 30 + ATR': lambda df: self.strategy_sma_30_atr(df, sma_period=30, atr_period=14, lookback_period=100)
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
    def plot_comparison(self, metrics_df, save_path='sma30_atr_volatility_comparison.png'):
        """포트폴리오 비교 시각화"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # 1. 포트폴리오 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            linewidth = 3.5 if 'ATR' in strategy_name else 2.5
            linestyle = '-' if 'ATR' in strategy_name else '--' if 'SMA' in strategy_name else ':'
            ax1.plot(cumulative.index, cumulative, label=f'{strategy_name} Portfolio',
                    linewidth=linewidth, alpha=0.8, linestyle=linestyle)

        ax1.set_title('Portfolio Cumulative Returns: SMA30 vs SMA30+ATR vs Buy&Hold\nEqual-Weight: BTC, ETH, ADA, XRP',
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
        colors = ['darkgreen' if 'ATR' in x else 'green' if 'SMA' in x else 'gray'
                 for x in sorted_df['Strategy']]
        ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = portfolio_metrics.sort_values('CAGR (%)', ascending=True)
        colors = ['darkgreen' if 'ATR' in x else 'green' if 'SMA' in x else 'gray'
                 for x in sorted_df['Strategy']]
        ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = portfolio_metrics.sort_values('MDD (%)', ascending=False)
        colors = ['darkred' if 'ATR' in x else 'red' if 'SMA' in x else 'gray'
                 for x in sorted_df['Strategy']]
        ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color=colors, alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = portfolio_metrics.sort_values('Sharpe Ratio', ascending=True)
        colors = ['darkgreen' if 'ATR' in x else 'green' if 'SMA' in x else 'gray'
                 for x in sorted_df['Strategy']]
        ax5.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Return vs Risk 산점도
        ax6 = fig.add_subplot(gs[2, 1])
        colors_scatter = ['darkgreen' if 'ATR' in x else 'green' if 'SMA' in x else 'gray'
                         for x in portfolio_metrics['Strategy']]
        ax6.scatter(portfolio_metrics['MDD (%)'], portfolio_metrics['CAGR (%)'],
                   s=400, alpha=0.6, c=colors_scatter)
        for idx, row in portfolio_metrics.iterrows():
            label = row['Strategy'].replace(' Portfolio', '')
            ax6.annotate(label, (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=10, ha='center', va='bottom', fontweight='bold')
        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 7. Profit Factor 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = portfolio_metrics.copy()
        sorted_df = sorted_df[sorted_df['Profit Factor'] != np.inf]
        if len(sorted_df) > 0:
            sorted_df = sorted_df.sort_values('Profit Factor', ascending=True)
            colors = ['darkgreen' if 'ATR' in x else 'green' if 'SMA' in x else 'gray'
                     for x in sorted_df['Strategy']]
            ax7.barh(sorted_df['Strategy'], sorted_df['Profit Factor'], color=colors, alpha=0.7)
        ax7.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax7.set_xlabel('Profit Factor', fontsize=11)
        ax7.set_title('Profit Factor Comparison', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8-10. ATR 전략 분석
        # 8. ATR 변동성별 포지션 크기 분포
        ax8 = fig.add_subplot(gs[3, 0])
        if 'SMA 30 + ATR' in self.strategy_results:
            # BTC 종목의 ATR 전략 데이터 사용
            atr_data = self.strategy_results['SMA 30 + ATR']['BTC_KRW']
            if 'position_size' in atr_data.columns:
                position_sizes = atr_data['position_size'].value_counts().sort_index()
                colors_bar = ['green' if x == 1.0 else 'yellow' if x >= 0.5 else 'orange'
                             for x in position_sizes.index]
                ax8.bar([f'{x*100:.0f}%' for x in position_sizes.index],
                       position_sizes.values, color=colors_bar, alpha=0.7)
                ax8.set_ylabel('Frequency (days)', fontsize=10)
                ax8.set_xlabel('Position Size', fontsize=10)
                ax8.set_title('Position Size Distribution (BTC)', fontsize=12, fontweight='bold')
                ax8.grid(True, alpha=0.3, axis='y')

        # 9. ATR Percentile 시계열
        ax9 = fig.add_subplot(gs[3, 1])
        if 'SMA 30 + ATR' in self.strategy_results:
            atr_data = self.strategy_results['SMA 30 + ATR']['BTC_KRW']
            if 'ATR_percentile' in atr_data.columns:
                ax9.plot(atr_data.index, atr_data['ATR_percentile'], color='purple', linewidth=1.5, alpha=0.7)
                ax9.axhline(y=0.25, color='green', linestyle='--', linewidth=1, alpha=0.5, label='25%')
                ax9.axhline(y=0.50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='50%')
                ax9.axhline(y=0.75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='75%')
                ax9.fill_between(atr_data.index, 0, atr_data['ATR_percentile'],
                                where=atr_data['ATR_percentile']<=0.25, alpha=0.2, color='green', label='Low Vol')
                ax9.fill_between(atr_data.index, 0.75, atr_data['ATR_percentile'],
                                where=atr_data['ATR_percentile']>=0.75, alpha=0.2, color='red', label='High Vol')
                ax9.set_ylabel('ATR Percentile', fontsize=10)
                ax9.set_xlabel('Date', fontsize=10)
                ax9.set_title('ATR Percentile Over Time (BTC)', fontsize=12, fontweight='bold')
                ax9.legend(fontsize=8, loc='upper right')
                ax9.grid(True, alpha=0.3)

        # 10. 종목별 성과 비교 (SMA 30 + ATR)
        ax10 = fig.add_subplot(gs[3, 2])
        atr_metrics = metrics_df[metrics_df['Strategy'].str.contains('SMA 30 \\+ ATR')].copy()
        if len(atr_metrics) > 0:
            sorted_df = atr_metrics.sort_values('Total Return (%)', ascending=True)
            colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
            strategy_labels = [s.replace('SMA 30 + ATR - ', '') for s in sorted_df['Strategy']]
            ax10.barh(strategy_labels, sorted_df['Total Return (%)'], color=colors, alpha=0.7)
            ax10.set_xlabel('Total Return (%)', fontsize=10)
            ax10.set_title('SMA 30 + ATR - By Asset', fontsize=12, fontweight='bold')
            ax10.grid(True, alpha=0.3, axis='x')

        # 11. 드로우다운 비교
        ax11 = fig.add_subplot(gs[4, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            linewidth = 3 if 'ATR' in strategy_name else 2 if 'SMA' in strategy_name else 1.5
            linestyle = '-' if 'ATR' in strategy_name else '--' if 'SMA' in strategy_name else ':'
            ax11.plot(drawdown.index, drawdown, label=f'{strategy_name}',
                     linewidth=linewidth, alpha=0.7, linestyle=linestyle)

        ax11.fill_between(drawdown.index, drawdown, 0, alpha=0.2)
        ax11.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax11.set_ylabel('Drawdown (%)', fontsize=12)
        ax11.set_xlabel('Date', fontsize=12)
        ax11.legend(loc='lower right', fontsize=11)
        ax11.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'SMA30 전략 with ATR 변동성 조절 백테스트 결과':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"포트폴리오 구성: 각 종목 동일 비중 ({100/len(self.symbols):.1f}%)")
        print(f"슬리피지: {self.slippage*100}%")
        print(f"\n전략 설명:")
        print(f"  1. Buy & Hold: 시작 시점 매수 후 끝까지 보유 (벤치마크)")
        print(f"  2. SMA 30: 전일종가가 SMA 30 이상일 때 매수")
        print(f"  3. SMA 30 + ATR: SMA 30 전략 + ATR 기반 변동성에 따른 포지션 크기 조절")
        print(f"\nATR 포지션 크기 조절:")
        print(f"  - ATR percentile 0-25% (낮은 변동성): 100% 투자")
        print(f"  - ATR percentile 25-50%: 75% 투자")
        print(f"  - ATR percentile 50-75%: 50% 투자")
        print(f"  - ATR percentile 75-100% (높은 변동성): 25% 투자")

        # 포트폴리오 성과
        print("\n" + "-"*150)
        print(f"{'포트폴리오 성과 비교 (동일 비중)':^150}")
        print("-"*150)
        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()

        # 전략 순서 정렬 (Buy & Hold -> SMA 30 -> SMA 30 + ATR)
        order = ['Buy & Hold Portfolio', 'SMA 30 Portfolio', 'SMA 30 + ATR Portfolio']
        portfolio_metrics['Strategy'] = pd.Categorical(portfolio_metrics['Strategy'], categories=order, ordered=True)
        portfolio_metrics = portfolio_metrics.sort_values('Strategy')

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
    print("SMA30 + ATR 변동성 조절 백테스트 시작")
    print("="*80)

    # 백테스트 실행
    backtest = SMA30ATRBacktest(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = backtest.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('sma30_atr_volatility_metrics.csv', index=False)
    print("Metrics saved to sma30_atr_volatility_metrics.csv")

    # 각 포트폴리오 상세 결과 저장
    for strategy_name in backtest.portfolio_results.keys():
        filename = f"portfolio_{strategy_name.replace(' ', '_').replace('+', 'plus').lower()}.csv"
        backtest.portfolio_results[strategy_name].to_csv(filename)
        print(f"Portfolio details saved to {filename}")

    print("\n" + "="*80)
    print("백테스트 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
