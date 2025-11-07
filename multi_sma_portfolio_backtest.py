"""
다중 SMA 윈도우 포트폴리오 백테스트

SMA 윈도우를 20, 25, 30, 35, 40으로 나누고 각각 개별 전략으로 수행한 후,
매일 리밸린싱하여 동일 비중 포트폴리오 성과를 측정합니다.

전략 구성:
- 5개의 SMA 전략: SMA20, SMA25, SMA30, SMA35, SMA40
- 각 전략은 BTC, ETH, ADA, XRP에 적용
- 매일 5개 전략을 동일 비중(각 20%)으로 리밸린싱
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


class MultiSMAPortfolioBacktest:
    """다중 SMA 윈도우 포트폴리오 백테스트 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 sma_windows=[20, 25, 30, 35, 40],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트
            sma_windows: SMA 윈도우 리스트 (default: [20, 25, 30, 35, 40])
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbols = symbols
        self.sma_windows = sma_windows
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = {}
        self.strategy_results = {}  # strategy_results[sma_window][symbol] = DataFrame
        self.rebalanced_portfolio = None

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
            if self.start_date is not None:
                df = df[df.index >= self.start_date]
            if self.end_date is not None:
                df = df[df.index <= self.end_date]

            self.data[symbol] = df
            print(f"  Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

        print("\n" + "="*80)
        print("Data loading completed!")
        print("="*80 + "\n")

    def strategy_sma(self, df, sma_period):
        """
        SMA 교차 전략
        - 가격이 SMA 이상일 때 매수 (보유)
        - 가격이 SMA 미만일 때 매도 후 현금 보유

        Args:
            df: 가격 데이터프레임
            sma_period: SMA 윈도우

        Returns:
            결과 데이터프레임 (position, returns, cumulative 등 포함)
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

    def run_all_sma_strategies(self):
        """모든 SMA 윈도우에 대해 모든 종목에 전략 실행"""
        print("\n" + "="*80)
        print("Running all SMA strategies for all symbols...")
        print("="*80 + "\n")

        for sma_window in self.sma_windows:
            print(f"\n>>> Running SMA {sma_window} strategy...")
            self.strategy_results[sma_window] = {}

            for symbol in self.symbols:
                print(f"  - {symbol}...")
                df = self.data[symbol].copy()
                result = self.strategy_sma(df, sma_window)
                self.strategy_results[sma_window][symbol] = result

        print("\n" + "="*80)
        print("All SMA strategies completed!")
        print("="*80 + "\n")

    def create_daily_rebalanced_portfolio(self):
        """
        매일 리밸런싱하는 포트폴리오 생성

        각 SMA 전략을 독립적인 전략으로 취급하고,
        5개 전략(SMA20, SMA25, SMA30, SMA35, SMA40)에 동일 비중(각 20%)으로 투자
        매일 리밸런싱하여 비중을 유지
        """
        print("\n" + "="*80)
        print("Creating daily rebalanced portfolio...")
        print("="*80 + "\n")

        # 각 SMA 전략의 포트폴리오 수익률 계산 (4개 종목 동일 비중)
        strategy_portfolio_returns = {}
        weight_per_symbol = 1.0 / len(self.symbols)

        print(f"Step 1: Creating individual SMA strategy portfolios (equal-weight across {len(self.symbols)} symbols)")
        print(f"  Weight per symbol: {weight_per_symbol:.2%}\n")

        for sma_window in self.sma_windows:
            print(f">>> Creating portfolio for SMA {sma_window}...")

            # 모든 종목의 공통 날짜 인덱스 찾기
            all_indices = [self.strategy_results[sma_window][symbol].index
                          for symbol in self.symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            # 포트폴리오 수익률 계산 (4개 종목 동일 비중)
            portfolio_returns = pd.Series(0.0, index=common_index)

            for symbol in self.symbols:
                symbol_returns = self.strategy_results[sma_window][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight_per_symbol
                print(f"  - Added {symbol} with weight {weight_per_symbol:.2%}")

            strategy_portfolio_returns[sma_window] = portfolio_returns
            print(f"  Portfolio returns shape: {portfolio_returns.shape}")

        # 모든 전략의 공통 날짜 인덱스 찾기
        print(f"\nStep 2: Finding common dates across all SMA strategies...")
        all_strategy_indices = [returns.index for returns in strategy_portfolio_returns.values()]
        common_index = all_strategy_indices[0]
        for idx in all_strategy_indices[1:]:
            common_index = common_index.intersection(idx)
        print(f"  Common dates: {len(common_index)} days")

        # 매일 리밸런싱: 5개 전략에 동일 비중 (각 20%)
        print(f"\nStep 3: Creating daily rebalanced portfolio (equal-weight across {len(self.sma_windows)} SMA strategies)")
        weight_per_strategy = 1.0 / len(self.sma_windows)
        print(f"  Weight per SMA strategy: {weight_per_strategy:.2%}\n")

        rebalanced_returns = pd.Series(0.0, index=common_index)

        for sma_window in self.sma_windows:
            strategy_returns = strategy_portfolio_returns[sma_window].loc[common_index]
            rebalanced_returns += strategy_returns * weight_per_strategy
            print(f"  - Added SMA {sma_window} strategy with weight {weight_per_strategy:.2%}")

        # 누적 수익률 계산
        rebalanced_cumulative = (1 + rebalanced_returns).cumprod()

        # 결과 저장
        self.rebalanced_portfolio = pd.DataFrame({
            'returns': rebalanced_returns,
            'cumulative': rebalanced_cumulative
        }, index=common_index)

        print(f"\n  Final portfolio shape: {self.rebalanced_portfolio.shape}")
        print("\n" + "="*80)
        print("Daily rebalanced portfolio creation completed!")
        print("="*80 + "\n")

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
        """모든 전략 및 리밸런싱 포트폴리오 성과 지표 계산"""
        metrics_list = []

        # 리밸런싱 포트폴리오 성과
        if self.rebalanced_portfolio is not None:
            returns = self.rebalanced_portfolio['returns']
            metrics = self.calculate_metrics(returns, "Daily Rebalanced Portfolio (SMA 20-40)")
            metrics_list.append(metrics)

        # 각 SMA 전략별 포트폴리오 성과
        for sma_window in self.sma_windows:
            # 각 SMA 전략의 포트폴리오 수익률 계산
            weight_per_symbol = 1.0 / len(self.symbols)

            # 공통 날짜 인덱스
            all_indices = [self.strategy_results[sma_window][symbol].index
                          for symbol in self.symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            # 포트폴리오 수익률
            portfolio_returns = pd.Series(0.0, index=common_index)
            for symbol in self.symbols:
                symbol_returns = self.strategy_results[sma_window][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight_per_symbol

            metrics = self.calculate_metrics(portfolio_returns, f"SMA {sma_window} Portfolio")
            metrics_list.append(metrics)

        # 개별 종목별 성과 (참고용)
        for sma_window in self.sma_windows:
            for symbol in self.symbols:
                returns = self.strategy_results[sma_window][symbol]['returns']
                metrics = self.calculate_metrics(returns, f"SMA {sma_window} - {symbol.split('_')[0]}")
                metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    def plot_analysis(self, metrics_df, save_path='multi_sma_portfolio_backtest.png'):
        """전체 분석 시각화"""
        fig = plt.figure(figsize=(22, 18))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률 비교 (리밸런싱 포트폴리오 vs 개별 SMA 전략)
        ax1 = fig.add_subplot(gs[0, :])

        # 리밸런싱 포트폴리오
        if self.rebalanced_portfolio is not None:
            ax1.plot(self.rebalanced_portfolio.index, self.rebalanced_portfolio['cumulative'],
                    label='Daily Rebalanced Portfolio (SMA 20-40)', linewidth=3.5, alpha=0.9, color='blue')

        # 개별 SMA 전략 포트폴리오
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.sma_windows)))
        for i, sma_window in enumerate(self.sma_windows):
            # 각 SMA 전략의 포트폴리오 수익률 재계산
            weight_per_symbol = 1.0 / len(self.symbols)
            all_indices = [self.strategy_results[sma_window][symbol].index for symbol in self.symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            portfolio_returns = pd.Series(0.0, index=common_index)
            for symbol in self.symbols:
                symbol_returns = self.strategy_results[sma_window][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight_per_symbol

            cumulative = (1 + portfolio_returns).cumprod()
            ax1.plot(cumulative.index, cumulative, label=f'SMA {sma_window}',
                    linewidth=2, alpha=0.7, color=colors[i])

        ax1.set_title('Multi-SMA Portfolio Strategy: Daily Rebalanced vs Individual SMA Strategies',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=11, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 포트폴리오만 필터링 (리밸런싱 + 개별 SMA 포트폴리오)
        portfolio_metrics = metrics_df[
            (metrics_df['Strategy'].str.contains('Portfolio')) |
            (metrics_df['Strategy'].str.contains('Rebalanced'))
        ].copy()

        # 2. 총 수익률 비교
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = portfolio_metrics.sort_values('Total Return (%)', ascending=True)
        colors_bar = ['darkblue' if 'Rebalanced' in x else ('green' if y > 0 else 'red')
                      for x, y in zip(sorted_df['Strategy'], sorted_df['Total Return (%)'])]
        ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'], color=colors_bar, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = portfolio_metrics.sort_values('CAGR (%)', ascending=True)
        colors_bar = ['darkblue' if 'Rebalanced' in x else ('green' if y > 0 else 'red')
                      for x, y in zip(sorted_df['Strategy'], sorted_df['CAGR (%)'])]
        ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors_bar, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = portfolio_metrics.sort_values('MDD (%)', ascending=False)
        colors_bar = ['darkblue' if 'Rebalanced' in x else 'crimson' for x in sorted_df['Strategy']]
        ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color=colors_bar, alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = portfolio_metrics.sort_values('Sharpe Ratio', ascending=True)
        colors_bar = ['darkblue' if 'Rebalanced' in x else ('green' if y > 0 else 'red')
                      for x, y in zip(sorted_df['Strategy'], sorted_df['Sharpe Ratio'])]
        ax5.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors_bar, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Return vs Risk 산점도
        ax6 = fig.add_subplot(gs[2, 1])
        colors_scatter = ['darkblue' if 'Rebalanced' in x else 'orange'
                         for x in portfolio_metrics['Strategy']]
        sizes = [500 if 'Rebalanced' in x else 300 for x in portfolio_metrics['Strategy']]

        for idx, row in portfolio_metrics.iterrows():
            color = 'darkblue' if 'Rebalanced' in row['Strategy'] else 'orange'
            size = 500 if 'Rebalanced' in row['Strategy'] else 300
            ax6.scatter(row['MDD (%)'], row['CAGR (%)'], s=size, alpha=0.7, color=color)

            label = row['Strategy'].replace(' Portfolio', '').replace('Daily Rebalanced Portfolio (SMA 20-40)', 'Rebalanced')
            ax6.annotate(label, (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=9, ha='center', va='bottom')

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
            colors_bar = ['darkblue' if 'Rebalanced' in x else ('green' if y > 1 else 'red')
                          for x, y in zip(sorted_df['Strategy'], sorted_df['Profit Factor'])]
            ax7.barh(sorted_df['Strategy'], sorted_df['Profit Factor'], color=colors_bar, alpha=0.7)
        ax7.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax7.set_xlabel('Profit Factor', fontsize=11)
        ax7.set_title('Profit Factor Comparison', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8. 드로우다운 비교
        ax8 = fig.add_subplot(gs[3, :])

        # 리밸런싱 포트폴리오 드로우다운
        if self.rebalanced_portfolio is not None:
            cumulative = self.rebalanced_portfolio['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax8.plot(drawdown.index, drawdown, label='Daily Rebalanced Portfolio',
                    linewidth=3, alpha=0.9, color='blue')
            ax8.fill_between(drawdown.index, drawdown, 0, alpha=0.2, color='blue')

        # 개별 SMA 전략 드로우다운
        for i, sma_window in enumerate(self.sma_windows):
            # 포트폴리오 수익률 재계산
            weight_per_symbol = 1.0 / len(self.symbols)
            all_indices = [self.strategy_results[sma_window][symbol].index for symbol in self.symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            portfolio_returns = pd.Series(0.0, index=common_index)
            for symbol in self.symbols:
                symbol_returns = self.strategy_results[sma_window][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight_per_symbol

            cumulative = (1 + portfolio_returns).cumprod()
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax8.plot(drawdown.index, drawdown, label=f'SMA {sma_window}',
                    linewidth=1.5, alpha=0.6, color=colors[i])

        ax8.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Drawdown (%)', fontsize=12)
        ax8.set_xlabel('Date', fontsize=12)
        ax8.legend(loc='lower right', fontsize=10, ncol=2)
        ax8.grid(True, alpha=0.3)

        # 9-10. 각 SMA 윈도우별 종목 비교
        for idx, sma_window in enumerate([20, 30, 40]):
            if idx >= 2:
                break

            ax = fig.add_subplot(gs[4, idx])
            sma_metrics = metrics_df[metrics_df['Strategy'].str.contains(f'SMA {sma_window} -')].copy()

            if len(sma_metrics) > 0:
                sorted_df = sma_metrics.sort_values('Total Return (%)', ascending=True)
                colors_bar = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
                strategy_labels = [s.replace(f'SMA {sma_window} - ', '') for s in sorted_df['Strategy']]
                ax.barh(strategy_labels, sorted_df['Total Return (%)'], color=colors_bar, alpha=0.7)
                ax.set_xlabel('Total Return (%)', fontsize=10)
                ax.set_title(f'SMA {sma_window} - By Asset', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')

        # 11. 성과 지표 요약 테이블
        ax11 = fig.add_subplot(gs[4, 2])
        ax11.axis('off')

        if self.rebalanced_portfolio is not None:
            rebalanced_metrics = metrics_df[metrics_df['Strategy'].str.contains('Rebalanced')].iloc[0]

            summary_text = "Daily Rebalanced Portfolio\n"
            summary_text += "="*35 + "\n\n"
            summary_text += f"Strategy: Equal-weight SMA 20-40\n"
            summary_text += f"Rebalancing: Daily\n"
            summary_text += f"Assets: {len(self.symbols)} cryptos\n\n"
            summary_text += "-"*35 + "\n"
            summary_text += f"Total Return: {rebalanced_metrics['Total Return (%)']:.2f}%\n"
            summary_text += f"CAGR: {rebalanced_metrics['CAGR (%)']:.2f}%\n"
            summary_text += f"MDD: {rebalanced_metrics['MDD (%)']:.2f}%\n"
            summary_text += f"Sharpe: {rebalanced_metrics['Sharpe Ratio']:.2f}\n"
            summary_text += f"Win Rate: {rebalanced_metrics['Win Rate (%)']:.2f}%\n"

            if rebalanced_metrics['Profit Factor'] != np.inf:
                summary_text += f"Profit Factor: {rebalanced_metrics['Profit Factor']:.2f}\n"

            ax11.text(0.1, 0.95, summary_text, transform=ax11.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'Multi-SMA Portfolio Backtest Results':^150}")
        print("="*150)

        # 실제 데이터 기간 확인
        if self.rebalanced_portfolio is not None and len(self.rebalanced_portfolio) > 0:
            actual_start = self.rebalanced_portfolio.index[0].strftime('%Y-%m-%d')
            actual_end = self.rebalanced_portfolio.index[-1].strftime('%Y-%m-%d')
            print(f"\n기간: {actual_start} ~ {actual_end} (전체 데이터)")
        else:
            start_str = self.start_date if self.start_date else "전체"
            end_str = self.end_date if self.end_date else "현재"
            print(f"\n기간: {start_str} ~ {end_str}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"SMA 윈도우: {', '.join([str(w) for w in self.sma_windows])}")
        print(f"포트폴리오 구성: 5개 SMA 전략에 동일 비중 (각 20%), 매일 리밸런싱")
        print(f"슬리피지: {self.slippage*100}%")

        # 리밸런싱 포트폴리오 성과
        print("\n" + "-"*150)
        print(f"{'Daily Rebalanced Portfolio Performance':^150}")
        print("-"*150)
        rebalanced_metrics = metrics_df[
            (metrics_df['Strategy'].str.contains('Rebalanced'))
        ].copy()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(rebalanced_metrics.to_string(index=False))

        # 개별 SMA 포트폴리오 성과
        print("\n" + "-"*150)
        print(f"{'Individual SMA Portfolio Performance':^150}")
        print("-"*150)
        sma_portfolio_metrics = metrics_df[
            (metrics_df['Strategy'].str.contains('SMA')) &
            (metrics_df['Strategy'].str.contains('Portfolio'))
        ].copy()
        print(sma_portfolio_metrics.to_string(index=False))

        # 종목별 성과
        print("\n" + "-"*150)
        print(f"{'Individual Asset Performance by SMA':^150}")
        print("-"*150)
        asset_metrics = metrics_df[
            (metrics_df['Strategy'].str.contains('SMA')) &
            (~metrics_df['Strategy'].str.contains('Portfolio')) &
            (~metrics_df['Strategy'].str.contains('Rebalanced'))
        ].copy()
        print(asset_metrics.to_string(index=False))

        print("\n" + "="*150 + "\n")

    def run_analysis(self):
        """전체 분석 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 모든 SMA 전략 실행
        self.run_all_sma_strategies()

        # 3. 매일 리밸런싱 포트폴리오 생성
        self.create_daily_rebalanced_portfolio()

        # 4. 성과 지표 계산
        metrics_df = self.calculate_all_metrics()

        # 5. 결과 출력
        self.print_metrics_table(metrics_df)

        # 6. 시각화
        self.plot_analysis(metrics_df)

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("Multi-SMA Portfolio Backtest 시작")
    print("="*80)

    # 백테스트 실행
    backtest = MultiSMAPortfolioBacktest(
        symbols=['BTC_KRW', 'ETH_KRW'],
        sma_windows=[20, 25, 30, 35, 40],
        start_date=None,  # 전체 데이터 사용
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = backtest.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('multi_sma_portfolio_metrics.csv', index=False)
    print("Metrics saved to multi_sma_portfolio_metrics.csv")

    # 리밸런싱 포트폴리오 상세 결과 저장
    if backtest.rebalanced_portfolio is not None:
        backtest.rebalanced_portfolio.to_csv('multi_sma_rebalanced_portfolio.csv')
        print("Rebalanced portfolio details saved to multi_sma_rebalanced_portfolio.csv")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
