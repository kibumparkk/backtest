#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
터틀 트레이딩 다중 전략 포트폴리오 분석
========================================

목적:
- 성과가 우수한 여러 파라미터 조합 선별
- 선별된 조합들의 균등투자 포트폴리오 구성
- 개별 전략 vs 포트폴리오 성과 비교
- 누적 PnL 및 drawdown 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class MultiStrategyPortfolioAnalyzer:
    """다중 전략 포트폴리오 분석 클래스"""

    def __init__(self, data_path='chart_day', slippage=0.002):
        """초기화"""
        self.data_path = Path(data_path)
        self.slippage = slippage
        self.data = {}
        self.strategy_results = {}
        self.portfolio_results = {}

    def load_data(self, symbols, start_date='2018-01-01'):
        """데이터 로드"""
        print("\n" + "="*80)
        print("Loading data...")
        print("="*80)

        for symbol in symbols:
            file_path = self.data_path / f"{symbol}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df.index = pd.to_datetime(df.index)
                df = df[df.index >= start_date]
                self.data[symbol] = df
                print(f"  - Loaded {symbol}: {len(df)} rows")
            else:
                print(f"  - WARNING: {symbol} file not found!")

        print(f"\nTotal symbols loaded: {len(self.data)}")

    def strategy_turtle_trading(self, df, entry_period=20, exit_period=10):
        """터틀 트레이딩 전략"""
        df = df.copy()

        # 터틀 채널
        df['entry_high'] = df['high'].rolling(window=entry_period).max().shift(1)
        df['exit_low'] = df['low'].rolling(window=exit_period).min().shift(1)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 최고가 돌파 시 매수
            if df.iloc[i]['high'] > df.iloc[i]['entry_high'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 최저가 하향 돌파 시 매도
            elif df.iloc[i]['low'] < df.iloc[i]['exit_low'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['close'] * (1 + self.slippage)
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['close']
                sell_price = df.iloc[i]['close'] * (1 - self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    def run_strategies(self, symbols, strategy_configs):
        """
        여러 전략 실행

        Parameters:
        -----------
        symbols : list
            종목 리스트
        strategy_configs : dict
            전략 설정 딕셔너리 {strategy_name: (entry_period, exit_period)}
        """
        print("\n" + "="*80)
        print("Running Multiple Strategies...")
        print("="*80)

        for strategy_name, (entry_period, exit_period) in strategy_configs.items():
            print(f"\n>>> Running {strategy_name} (Entry={entry_period}, Exit={exit_period})...")

            # 각 종목별로 전략 실행
            symbol_results = {}
            for symbol in symbols:
                df = self.data[symbol].copy()
                result = self.strategy_turtle_trading(df, entry_period, exit_period)
                symbol_results[symbol] = result

            # 포트폴리오 수익률 계산 (동일 비중)
            weight = 1.0 / len(symbols)

            # 공통 날짜 인덱스
            all_indices = [symbol_results[symbol].index for symbol in symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            # 포트폴리오 수익률
            portfolio_returns = pd.Series(0.0, index=common_index)
            for symbol in symbols:
                symbol_returns = symbol_results[symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight

            # 누적 수익률
            cumulative = (1 + portfolio_returns).cumprod()

            # 결과 저장
            self.strategy_results[strategy_name] = {
                'returns': portfolio_returns,
                'cumulative': cumulative,
                'entry_period': entry_period,
                'exit_period': exit_period
            }

            print(f"  - Strategy completed")

    def create_equal_weight_portfolio(self):
        """선별된 전략들의 균등투자 포트폴리오 생성"""
        print("\n" + "="*80)
        print("Creating Equal-Weight Multi-Strategy Portfolio...")
        print("="*80)

        strategy_names = list(self.strategy_results.keys())
        num_strategies = len(strategy_names)

        print(f"\nCombining {num_strategies} strategies:")
        for name in strategy_names:
            entry = self.strategy_results[name]['entry_period']
            exit = self.strategy_results[name]['exit_period']
            print(f"  - {name}: Entry={entry}, Exit={exit}")

        # 공통 날짜 인덱스
        all_indices = [self.strategy_results[name]['returns'].index for name in strategy_names]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)

        # 균등 비중 포트폴리오
        weight = 1.0 / num_strategies
        portfolio_returns = pd.Series(0.0, index=common_index)

        for strategy_name in strategy_names:
            strategy_returns = self.strategy_results[strategy_name]['returns'].loc[common_index]
            portfolio_returns += strategy_returns * weight
            print(f"  - Added {strategy_name} with weight {weight:.2%}")

        # 누적 수익률
        cumulative = (1 + portfolio_returns).cumprod()

        self.portfolio_results = {
            'returns': portfolio_returns,
            'cumulative': cumulative
        }

        print("\nMulti-Strategy Portfolio created successfully!")

    def calculate_metrics(self, returns_series, name):
        """성과 지표 계산"""
        cumulative = (1 + returns_series).cumprod()

        # 총 수익률
        total_return = (cumulative.iloc[-1] - 1) * 100

        # CAGR
        years = (returns_series.index[-1] - returns_series.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # MDD
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100

        # Sharpe
        sharpe = (returns_series.mean() / returns_series.std() * np.sqrt(365)) if returns_series.std() > 0 else 0

        # Calmar
        calmar = cagr / abs(mdd) if mdd != 0 else 0

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
            'Calmar Ratio': calmar,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades,
            'Profit Factor': profit_factor
        }

    def compare_performance(self):
        """성과 비교"""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)

        # 각 전략의 성과 지표
        all_metrics = []

        for strategy_name in self.strategy_results.keys():
            returns = self.strategy_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, strategy_name)
            all_metrics.append(metrics)

        # 포트폴리오 성과
        portfolio_metrics = self.calculate_metrics(
            self.portfolio_results['returns'],
            'Multi-Strategy Portfolio'
        )
        all_metrics.append(portfolio_metrics)

        # DataFrame으로 변환
        self.metrics_df = pd.DataFrame(all_metrics)

        # 출력
        print("\n" + self.metrics_df.to_string(index=False))

        return self.metrics_df

    def plot_comprehensive_analysis(self, output_file='turtle_multi_strategy_analysis.png'):
        """종합 분석 차트"""
        print("\n" + "="*80)
        print("Creating comprehensive analysis charts...")
        print("="*80)

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        strategy_names = list(self.strategy_results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))
        portfolio_color = 'red'

        # 1. 누적 수익률 (큰 차트)
        ax1 = fig.add_subplot(gs[0:2, :])
        for idx, strategy_name in enumerate(strategy_names):
            cumulative = self.strategy_results[strategy_name]['cumulative']
            ax1.plot(cumulative.index, (cumulative - 1) * 100,
                    label=strategy_name, linewidth=1.5, alpha=0.7, color=colors[idx])

        # 포트폴리오
        portfolio_cumulative = self.portfolio_results['cumulative']
        ax1.plot(portfolio_cumulative.index, (portfolio_cumulative - 1) * 100,
                label='Multi-Strategy Portfolio', linewidth=3, color=portfolio_color)

        ax1.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 2. Drawdown (큰 차트)
        ax2 = fig.add_subplot(gs[2, :])
        for idx, strategy_name in enumerate(strategy_names):
            cumulative = self.strategy_results[strategy_name]['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax2.plot(drawdown.index, drawdown, label=strategy_name,
                    linewidth=1.5, alpha=0.7, color=colors[idx])

        # 포트폴리오 drawdown
        cummax = portfolio_cumulative.cummax()
        drawdown = (portfolio_cumulative - cummax) / cummax * 100
        ax2.plot(drawdown.index, drawdown, label='Multi-Strategy Portfolio',
                linewidth=3, color=portfolio_color)

        ax2.set_title('Drawdown Comparison', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend(loc='lower left', fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[3, 0])
        cagr_data = self.metrics_df.sort_values('CAGR (%)', ascending=True)
        colors_bar = ['red' if x == 'Multi-Strategy Portfolio' else 'skyblue' for x in cagr_data['Strategy']]
        ax3.barh(range(len(cagr_data)), cagr_data['CAGR (%)'], color=colors_bar)
        ax3.set_yticks(range(len(cagr_data)))
        ax3.set_yticklabels(cagr_data['Strategy'], fontsize=9)
        ax3.set_xlabel('CAGR (%)', fontsize=10)
        ax3.set_title('CAGR Comparison', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # 4. MDD 비교 (절댓값)
        ax4 = fig.add_subplot(gs[3, 1])
        mdd_data = self.metrics_df.copy()
        mdd_data['MDD_abs'] = abs(mdd_data['MDD (%)'])
        mdd_data = mdd_data.sort_values('MDD_abs', ascending=True)
        colors_bar = ['red' if x == 'Multi-Strategy Portfolio' else 'lightcoral' for x in mdd_data['Strategy']]
        ax4.barh(range(len(mdd_data)), mdd_data['MDD_abs'], color=colors_bar)
        ax4.set_yticks(range(len(mdd_data)))
        ax4.set_yticklabels(mdd_data['Strategy'], fontsize=9)
        ax4.set_xlabel('MDD (abs %)', fontsize=10)
        ax4.set_title('Maximum Drawdown (lower is better)', fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)

        # 5. Sharpe Ratio 비교
        ax5 = fig.add_subplot(gs[3, 2])
        sharpe_data = self.metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors_bar = ['red' if x == 'Multi-Strategy Portfolio' else 'lightgreen' for x in sharpe_data['Strategy']]
        ax5.barh(range(len(sharpe_data)), sharpe_data['Sharpe Ratio'], color=colors_bar)
        ax5.set_yticks(range(len(sharpe_data)))
        ax5.set_yticklabels(sharpe_data['Strategy'], fontsize=9)
        ax5.set_xlabel('Sharpe Ratio', fontsize=10)
        ax5.set_title('Risk-Adjusted Return', fontsize=12, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comprehensive analysis saved to: {output_file}")
        plt.close()

    def plot_correlation_analysis(self, output_file='turtle_strategy_correlation.png'):
        """전략 간 상관관계 분석"""
        print("\n" + "="*80)
        print("Creating correlation analysis...")
        print("="*80)

        # 모든 전략의 수익률을 DataFrame으로 결합
        strategy_names = list(self.strategy_results.keys())

        # 공통 인덱스
        all_indices = [self.strategy_results[name]['returns'].index for name in strategy_names]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)

        # 수익률 DataFrame
        returns_df = pd.DataFrame()
        for strategy_name in strategy_names:
            returns_df[strategy_name] = self.strategy_results[strategy_name]['returns'].loc[common_index]

        # 상관관계 계산
        correlation_matrix = returns_df.corr()

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # 1. 상관관계 히트맵
        ax = axes[0]
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title('Strategy Returns Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

        # 2. 누적 수익률 분산도
        ax = axes[1]
        for idx, strategy_name in enumerate(strategy_names):
            cumulative = self.strategy_results[strategy_name]['cumulative']
            final_return = (cumulative.iloc[-1] - 1) * 100

            # MDD
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax
            mdd = drawdown.min() * 100

            ax.scatter(abs(mdd), final_return, s=200, alpha=0.6,
                      label=strategy_name, edgecolors='black', linewidth=1.5)

        # 포트폴리오
        portfolio_cumulative = self.portfolio_results['cumulative']
        final_return = (portfolio_cumulative.iloc[-1] - 1) * 100
        cummax = portfolio_cumulative.cummax()
        drawdown = (portfolio_cumulative - cummax) / cummax
        mdd = drawdown.min() * 100
        ax.scatter(abs(mdd), final_return, s=300, alpha=0.8, color='red',
                  label='Multi-Strategy Portfolio', edgecolors='black', linewidth=2, marker='*')

        ax.set_xlabel('Maximum Drawdown (%)', fontsize=12)
        ax.set_ylabel('Total Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Correlation analysis saved to: {output_file}")
        plt.close()

    def save_results(self, output_file='turtle_multi_strategy_results.csv'):
        """결과 저장"""
        self.metrics_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_file}")

    def print_summary(self):
        """요약 출력"""
        print("\n" + "="*80)
        print("MULTI-STRATEGY PORTFOLIO SUMMARY")
        print("="*80)

        # 포트폴리오 성과
        portfolio_row = self.metrics_df[self.metrics_df['Strategy'] == 'Multi-Strategy Portfolio'].iloc[0]

        print(f"\n[Multi-Strategy Portfolio Performance]")
        print(f"  Total Return: {portfolio_row['Total Return (%)']:.2f}%")
        print(f"  CAGR: {portfolio_row['CAGR (%)']:.2f}%")
        print(f"  MDD: {portfolio_row['MDD (%)']:.2f}%")
        print(f"  Sharpe Ratio: {portfolio_row['Sharpe Ratio']:.2f}")
        print(f"  Calmar Ratio: {portfolio_row['Calmar Ratio']:.2f}")
        print(f"  Win Rate: {portfolio_row['Win Rate (%)']:.2f}%")

        # 개별 전략 대비 비교
        individual_strategies = self.metrics_df[self.metrics_df['Strategy'] != 'Multi-Strategy Portfolio']

        print(f"\n[Comparison with Individual Strategies]")
        print(f"  Average CAGR of individuals: {individual_strategies['CAGR (%)'].mean():.2f}%")
        print(f"  Portfolio CAGR: {portfolio_row['CAGR (%)']:.2f}%")
        print(f"  CAGR Difference: {portfolio_row['CAGR (%)'] - individual_strategies['CAGR (%)'].mean():.2f}%")
        print()
        print(f"  Average MDD of individuals: {individual_strategies['MDD (%)'].mean():.2f}%")
        print(f"  Portfolio MDD: {portfolio_row['MDD (%)']:.2f}%")
        print(f"  MDD Improvement: {individual_strategies['MDD (%)'].mean() - portfolio_row['MDD (%)']:.2f}%")
        print()
        print(f"  Average Sharpe of individuals: {individual_strategies['Sharpe Ratio'].mean():.2f}")
        print(f"  Portfolio Sharpe: {portfolio_row['Sharpe Ratio']:.2f}")
        print(f"  Sharpe Difference: {portfolio_row['Sharpe Ratio'] - individual_strategies['Sharpe Ratio'].mean():.2f}")

        print("\n" + "="*80)


def main():
    """메인 실행 함수"""

    # 기본 설정
    SYMBOLS = ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW']
    START_DATE = '2018-01-01'
    SLIPPAGE = 0.002

    # 선별된 전략 (최적 파라미터 조합)
    STRATEGY_CONFIGS = {
        'Aggressive (E10/X20)': (10, 20),      # 최고 CAGR
        'Conservative (E55/X5)': (55, 5),      # 최고 Sharpe/Calmar, 최소 MDD
        'Balanced (E25/X10)': (25, 10),        # 균형잡힌 전략
        'High Sharpe (E15/X10)': (15, 10),     # 높은 Sharpe, 좋은 CAGR
        'Low Risk (E30/X10)': (30, 10),        # 낮은 MDD, 좋은 Calmar
    }

    print("\n" + "="*80)
    print("TURTLE TRADING MULTI-STRATEGY PORTFOLIO ANALYSIS")
    print("="*80)
    print(f"\nAssets: {SYMBOLS}")
    print(f"Start Date: {START_DATE}")
    print(f"Slippage: {SLIPPAGE*100:.1f}%")
    print(f"\nSelected Strategies: {len(STRATEGY_CONFIGS)}")
    for name, (entry, exit) in STRATEGY_CONFIGS.items():
        print(f"  - {name}: Entry={entry}, Exit={exit}")

    # 분석 객체 생성
    analyzer = MultiStrategyPortfolioAnalyzer(slippage=SLIPPAGE)

    # 데이터 로드
    analyzer.load_data(SYMBOLS, START_DATE)

    # 전략 실행
    analyzer.run_strategies(SYMBOLS, STRATEGY_CONFIGS)

    # 균등투자 포트폴리오 생성
    analyzer.create_equal_weight_portfolio()

    # 성과 비교
    analyzer.compare_performance()

    # 요약 출력
    analyzer.print_summary()

    # 결과 저장
    analyzer.save_results('turtle_multi_strategy_results.csv')

    # 시각화
    analyzer.plot_comprehensive_analysis('turtle_multi_strategy_analysis.png')
    analyzer.plot_correlation_analysis('turtle_strategy_correlation.png')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. turtle_multi_strategy_results.csv - Performance metrics comparison")
    print("  2. turtle_multi_strategy_analysis.png - Comprehensive analysis charts")
    print("  3. turtle_strategy_correlation.png - Correlation and risk-return analysis")
    print("\n")


if __name__ == '__main__':
    main()
