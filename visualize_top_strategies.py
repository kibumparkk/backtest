"""
상위 5개 전략 시각화 스크립트

loop_strategy_results_top5.csv 파일을 읽어서
상위 전략들을 다시 계산하고 벤치마크와 비교하는 시각화를 생성합니다.
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


class StrategyVisualizer:
    """전략 시각화 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = {}
        self.portfolio_returns_dict = {}

    def load_data(self):
        """데이터 로드"""
        print("Loading data...")
        for symbol in self.symbols:
            file_path = f'chart_day/{symbol}.parquet'
            df = pd.read_parquet(file_path)
            df.columns = [col.capitalize() for col in df.columns]
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            self.data[symbol] = df
        print("Data loaded successfully!\n")

    def strategy_sma(self, df, period):
        """SMA 전략"""
        df = df.copy()
        df['SMA'] = df['Close'].rolling(window=period).mean()
        df['prev_close'] = df['Close'].shift(1)
        df['position'] = np.where(df['prev_close'] >= df['SMA'], 1, 0)
        df['position_change'] = df['position'].diff()
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage
        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    def strategy_bollinger(self, df, period, std_dev):
        """Bollinger Band 전략"""
        df = df.copy()
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        df['BB_upper'] = sma + (std * std_dev)
        df['BB_middle'] = sma
        df['BB_lower'] = sma - (std * std_dev)
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

    def create_portfolio(self, strategy_results):
        """포트폴리오 생성"""
        weight = 1.0 / len(self.symbols)
        all_indices = [strategy_results[symbol].index for symbol in self.symbols]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)

        portfolio_returns = pd.Series(0.0, index=common_index)
        for symbol in self.symbols:
            symbol_returns = strategy_results[symbol].loc[common_index, 'returns']
            portfolio_returns += symbol_returns * weight

        return portfolio_returns

    def run_strategy(self, strategy_name, strategy_type, params):
        """전략 실행"""
        strategy_results = {}

        if strategy_type == 'SMA':
            period = int(params.split('=')[1])
            for symbol in self.symbols:
                df = self.data[symbol].copy()
                result = self.strategy_sma(df, period)
                strategy_results[symbol] = result

        elif strategy_type == 'Bollinger_Band':
            # "period=20, std_dev=1.5" 파싱
            parts = params.split(', ')
            period = int(parts[0].split('=')[1])
            std_dev = float(parts[1].split('=')[1])
            for symbol in self.symbols:
                df = self.data[symbol].copy()
                result = self.strategy_bollinger(df, period, std_dev)
                strategy_results[symbol] = result

        portfolio_returns = self.create_portfolio(strategy_results)
        return portfolio_returns

    def plot_comparison(self, top5_df, benchmark_name='SMA_30', save_path='top_strategies_visualization.png'):
        """시각화"""
        print("Creating visualization...")

        # 벤치마크 계산
        benchmark_row = top5_df[top5_df['Strategy'] == benchmark_name]
        if benchmark_row.empty:
            # 벤치마크가 top5에 없으면 직접 계산
            benchmark_returns = self.run_strategy('SMA_30', 'SMA', 'period=30')
            self.portfolio_returns_dict[benchmark_name] = benchmark_returns

        # 상위 5개 전략 계산
        strategies_to_plot = []
        for idx, row in top5_df.iterrows():
            strategy_name = row['Strategy']
            if strategy_name == benchmark_name:
                continue
            strategy_type = row['Type']
            params = row['Parameters']

            print(f"  Calculating {strategy_name}...")
            portfolio_returns = self.run_strategy(strategy_name, strategy_type, params)
            self.portfolio_returns_dict[strategy_name] = portfolio_returns
            strategies_to_plot.append(strategy_name)

        # 벤치마크가 없으면 추가
        if benchmark_name not in self.portfolio_returns_dict:
            print(f"  Calculating benchmark {benchmark_name}...")
            benchmark_returns = self.run_strategy(benchmark_name, 'SMA', 'period=30')
            self.portfolio_returns_dict[benchmark_name] = benchmark_returns

        # 시각화
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률 비교 (로그 스케일)
        ax1 = fig.add_subplot(gs[0, :])

        # 벤치마크 먼저 그리기
        returns = self.portfolio_returns_dict[benchmark_name]
        cumulative = (1 + returns).cumprod()
        ax1.plot(cumulative.index, cumulative, label=f'{benchmark_name} (Benchmark)',
                linewidth=3, color='black', linestyle='--', alpha=0.7)

        # 상위 전략들 그리기
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        for i, strategy_name in enumerate(strategies_to_plot[:5]):
            returns = self.portfolio_returns_dict[strategy_name]
            cumulative = (1 + returns).cumprod()
            ax1.plot(cumulative.index, cumulative, label=strategy_name,
                    linewidth=2.5, alpha=0.8, color=colors[i])

        ax1.set_title('Top 5 Strategies vs Benchmark: Cumulative Returns (Log Scale)',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Cumulative Return', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. CAGR 비교
        ax2 = fig.add_subplot(gs[1, 0])

        # 벤치마크 포함한 데이터 준비
        plot_df = top5_df.head(5).copy()

        # 벤치마크 데이터 추가
        if benchmark_name not in plot_df['Strategy'].values:
            # CSV에서 벤치마크 찾기
            all_results_df = pd.read_csv('loop_strategy_results_all.csv')
            benchmark_data = all_results_df[all_results_df['Strategy'] == benchmark_name]
            if not benchmark_data.empty:
                plot_df = pd.concat([plot_df, benchmark_data], ignore_index=True)

        sorted_df = plot_df.sort_values('CAGR (%)', ascending=True)
        colors_bar = ['green' if x > sorted_df[sorted_df['Strategy'] == benchmark_name]['CAGR (%)'].values[0]
                      else 'gray' if strategy == benchmark_name else 'red'
                      for strategy, x in zip(sorted_df['Strategy'], sorted_df['CAGR (%)'])]

        ax2.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors_bar, alpha=0.7)
        ax2.set_xlabel('CAGR (%)', fontsize=11, fontweight='bold')
        ax2.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=sorted_df[sorted_df['Strategy'] == benchmark_name]['CAGR (%)'].values[0],
                   color='red', linestyle='--', linewidth=2, alpha=0.5, label='Benchmark')

        # 3. Sharpe Ratio 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = plot_df.sort_values('Sharpe Ratio', ascending=True)
        colors_bar = ['green' if x > sorted_df[sorted_df['Strategy'] == benchmark_name]['Sharpe Ratio'].values[0]
                      else 'gray' if strategy == benchmark_name else 'red'
                      for strategy, x in zip(sorted_df['Strategy'], sorted_df['Sharpe Ratio'])]

        ax3.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors_bar, alpha=0.7)
        ax3.set_xlabel('Sharpe Ratio', fontsize=11, fontweight='bold')
        ax3.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=sorted_df[sorted_df['Strategy'] == benchmark_name]['Sharpe Ratio'].values[0],
                   color='red', linestyle='--', linewidth=2, alpha=0.5)

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = plot_df.sort_values('MDD (%)', ascending=False)
        colors_bar = ['green' if x > sorted_df[sorted_df['Strategy'] == benchmark_name]['MDD (%)'].values[0]
                      else 'gray' if strategy == benchmark_name else 'red'
                      for strategy, x in zip(sorted_df['Strategy'], sorted_df['MDD (%)'])]

        ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color=colors_bar, alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.axvline(x=sorted_df[sorted_df['Strategy'] == benchmark_name]['MDD (%)'].values[0],
                   color='red', linestyle='--', linewidth=2, alpha=0.5)

        # 5. Return vs Risk 산점도
        ax5 = fig.add_subplot(gs[2, 0])

        # 벤치마크를 별도로 표시
        benchmark_data = plot_df[plot_df['Strategy'] == benchmark_name]
        other_data = plot_df[plot_df['Strategy'] != benchmark_name]

        scatter = ax5.scatter(other_data['MDD (%)'], other_data['CAGR (%)'],
                   s=other_data['Sharpe Ratio']*200, alpha=0.6,
                   c=other_data['Sharpe Ratio'], cmap='RdYlGn',
                   edgecolors='black', linewidth=2)

        # 벤치마크 표시
        ax5.scatter(benchmark_data['MDD (%)'], benchmark_data['CAGR (%)'],
                   s=400, alpha=0.8, c='red', marker='D',
                   edgecolors='black', linewidth=2, label='Benchmark')

        for idx, row in plot_df.iterrows():
            label = row['Strategy']
            ax5.annotate(label, (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=9, ha='center', va='bottom')

        ax5.set_xlabel('MDD (%) - Lower is Better', fontsize=11, fontweight='bold')
        ax5.set_ylabel('CAGR (%) - Higher is Better', fontsize=11, fontweight='bold')
        ax5.set_title('Risk-Return Profile (Size = Sharpe Ratio)', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        plt.colorbar(scatter, ax=ax5, label='Sharpe Ratio')

        # 6. Drawdown 비교
        ax6 = fig.add_subplot(gs[2, 1:])

        # 벤치마크 드로우다운
        returns = self.portfolio_returns_dict[benchmark_name]
        cumulative = (1 + returns).cumprod()
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax * 100
        ax6.plot(drawdown.index, drawdown, label=f'{benchmark_name} (Benchmark)',
                linewidth=3, color='black', linestyle='--', alpha=0.7)

        # 상위 전략 드로우다운
        for i, strategy_name in enumerate(strategies_to_plot[:5]):
            returns = self.portfolio_returns_dict[strategy_name]
            cumulative = (1 + returns).cumprod()
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax6.plot(drawdown.index, drawdown, label=strategy_name,
                    linewidth=2, alpha=0.7, color=colors[i])

        ax6.fill_between(drawdown.index, drawdown, 0, alpha=0.2)
        ax6.set_title('Drawdown Comparison Over Time', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax6.legend(loc='lower right', fontsize=10)
        ax6.grid(True, alpha=0.3)

        # 7. 누적 수익률 비교 (선형 스케일)
        ax7 = fig.add_subplot(gs[3, :])

        # 벤치마크
        returns = self.portfolio_returns_dict[benchmark_name]
        cumulative = (1 + returns).cumprod()
        ax7.plot(cumulative.index, cumulative, label=f'{benchmark_name} (Benchmark)',
                linewidth=3, color='black', linestyle='--', alpha=0.7)

        # 상위 전략들
        for i, strategy_name in enumerate(strategies_to_plot[:5]):
            returns = self.portfolio_returns_dict[strategy_name]
            cumulative = (1 + returns).cumprod()
            ax7.plot(cumulative.index, cumulative, label=strategy_name,
                    linewidth=2.5, alpha=0.8, color=colors[i])

        ax7.set_title('Cumulative Returns (Linear Scale)', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax7.legend(loc='upper left', fontsize=11)
        ax7.grid(True, alpha=0.3)

        # 전체 제목
        fig.suptitle('Top 5 Strategies vs SMA_30 Benchmark - Performance Comparison\n'
                    f'Period: {self.start_date} to {self.end_date} | Portfolio: Equal-Weight (BTC, ETH, ADA, XRP)',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}\n")
        plt.close()


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("상위 5개 전략 시각화 시작")
    print("="*80 + "\n")

    # CSV 파일 읽기
    top5_df = pd.read_csv('loop_strategy_results_top5.csv')
    print("Top 5 strategies loaded from CSV:")
    print(top5_df[['Strategy', 'Type', 'Parameters', 'CAGR (%)', 'Sharpe Ratio', 'MDD (%)']].to_string(index=False))
    print("\n")

    # 시각화
    visualizer = StrategyVisualizer(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )

    visualizer.load_data()
    visualizer.plot_comparison(top5_df, benchmark_name='SMA_30')

    print("="*80)
    print("시각화 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
