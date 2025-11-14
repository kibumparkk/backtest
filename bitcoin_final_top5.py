"""
비트코인 추세추종 전략 - 최종 상위 5개 전략 백테스트 및 보고서
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class FinalTop5Analysis:
    def __init__(self):
        self.data = None
        self.results = {}

    def load_data(self):
        """데이터 로드"""
        df = pd.read_parquet('chart_day/BTC_KRW.parquet')
        df.columns = [col.capitalize() for col in df.columns]
        df = df[(df.index >= '2018-01-01')]
        self.data = df
        print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

    def backtest(self, signal, name):
        """백테스트 실행"""
        df = self.data.copy()
        df['signal'] = signal
        df['pos_change'] = df['signal'].diff()
        df['daily_ret'] = df['Close'].pct_change()
        df['strat_ret'] = df['signal'].shift(1) * df['daily_ret']

        # 슬리피지 0.2%
        slip_cost = pd.Series(0.0, index=df.index)
        slip_cost[df['pos_change'] == 1] = -0.002
        slip_cost[df['pos_change'] == -1] = -0.002
        df['strat_ret'] = df['strat_ret'] + slip_cost
        df['strat_ret'] = df['strat_ret'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['strat_ret']).cumprod()

        # 성과 지표
        total_return = (df['cumulative'].iloc[-1] - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['cumulative'].iloc[-1] ** (1/years) - 1) * 100

        cummax = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - cummax) / cummax
        mdd = drawdown.min() * 100

        sharpe = (df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(365))

        total_trades = (df['strat_ret'] != 0).sum()
        winning_trades = (df['strat_ret'] > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        self.results[name] = {
            'returns': df['strat_ret'],
            'cumulative': df['cumulative'],
            'metrics': {
                'Strategy': name,
                'Total Return (%)': total_return,
                'CAGR (%)': cagr,
                'MDD (%)': mdd,
                'Sharpe Ratio': sharpe,
                'Win Rate (%)': win_rate,
                'Total Trades': int(total_trades)
            }
        }

        return self.results[name]['metrics']

    def run_all(self):
        """5개 전략 + 벤치마크 실행"""
        print("\n" + "="*80)
        print("Running Top 5 Strategies + Benchmark")
        print("="*80 + "\n")

        df = self.data.copy()

        # 벤치마크: Close > SMA30
        df['SMA30'] = df['Close'].rolling(30).mean()
        signal = (df['Close'] > df['SMA30']).astype(int)
        self.backtest(signal, 'BENCHMARK_Close>SMA30')
        print("BENCHMARK_Close>SMA30: Done")

        # 1위: Close>SMA31 AND SMA39
        df['SMA31'] = df['Close'].rolling(31).mean()
        df['SMA39'] = df['Close'].rolling(39).mean()
        signal = ((df['Close'] > df['SMA31']) & (df['Close'] > df['SMA39'])).astype(int)
        self.backtest(signal, '1_Close>SMA31_AND_SMA39')
        print("1_Close>SMA31_AND_SMA39: Done")

        # 2위: Close>(SMA30+31+32)/3
        df['SMA32'] = df['Close'].rolling(32).mean()
        df['Avg'] = (df['SMA30'] + df['SMA31'] + df['SMA32']) / 3
        signal = (df['Close'] > df['Avg']).astype(int)
        self.backtest(signal, '2_Close>(SMA30+31+32)/3')
        print("2_Close>(SMA30+31+32)/3: Done")

        # 3위: Close>SMA31
        signal = (df['Close'] > df['SMA31']).astype(int)
        self.backtest(signal, '3_Close>SMA31')
        print("3_Close>SMA31: Done")

        # 4위: Close>SMA39
        signal = (df['Close'] > df['SMA39']).astype(int)
        self.backtest(signal, '4_Close>SMA39')
        print("4_Close>SMA39: Done")

        # 5위: Close>SMA31 OR SMA39
        signal = ((df['Close'] > df['SMA31']) | (df['Close'] > df['SMA39'])).astype(int)
        self.backtest(signal, '5_Close>SMA31_OR_SMA39')
        print("5_Close>SMA31_OR_SMA39: Done")

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80)

    def create_metrics_df(self):
        """성과 지표 DataFrame 생성"""
        metrics_list = [v['metrics'] for v in self.results.values()]
        return pd.DataFrame(metrics_list)

    def plot_comparison(self):
        """시각화"""
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 누적 수익률
        ax1 = fig.add_subplot(gs[0, :])
        for name in self.results.keys():
            cum = self.results[name]['cumulative']
            linewidth = 4 if 'BENCHMARK' in name else 2.5
            linestyle = '--' if 'BENCHMARK' in name else '-'
            alpha = 0.9 if 'BENCHMARK' in name else 0.8

            ax1.plot(cum.index, cum, label=name.replace('_', ' '),
                    linewidth=linewidth, linestyle=linestyle, alpha=alpha)

        ax1.set_title('Bitcoin Trend-Following Strategies: Top 5 vs Benchmark',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        metrics_df = self.create_metrics_df()

        # 2. 샤프 비율
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'green' for s in sorted_df['Strategy']]
        y_pos = range(len(sorted_df))
        ax2.barh(y_pos, sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy']], fontsize=9)
        ax2.set_xlabel('Sharpe Ratio', fontsize=11)
        ax2.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        benchmark_sharpe = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0]
        ax2.axvline(x=benchmark_sharpe, color='red', linestyle='--', linewidth=2, alpha=0.5)

        # 3. CAGR
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'green' for s in sorted_df['Strategy']]
        y_pos = range(len(sorted_df))
        ax3.barh(y_pos, sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy']], fontsize=9)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        colors = ['red' if 'BENCHMARK' in s else 'crimson' for s in sorted_df['Strategy']]
        y_pos = range(len(sorted_df))
        ax4.barh(y_pos, sorted_df['MDD (%)'], color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy']], fontsize=9)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. Return vs Risk
        ax5 = fig.add_subplot(gs[2, 0])
        for idx, row in metrics_df.iterrows():
            if 'BENCHMARK' in row['Strategy']:
                ax5.scatter(row['MDD (%)'], row['CAGR (%)'], s=500, alpha=0.8,
                           c='red', marker='s', edgecolors='black', linewidths=2,
                           label='Benchmark')
            else:
                ax5.scatter(row['MDD (%)'], row['CAGR (%)'], s=300, alpha=0.7,
                           c='green', edgecolors='black', linewidths=1.5)

        for idx, row in metrics_df.iterrows():
            label = row['Strategy'].replace('_', ' ').replace('BENCHMARK ', 'BM: ')
            ax5.annotate(label, (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=8, ha='center', va='bottom')

        ax5.set_xlabel('MDD (%)', fontsize=11)
        ax5.set_ylabel('CAGR (%)', fontsize=11)
        ax5.set_title('Return vs Risk', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. Total Return
        ax6 = fig.add_subplot(gs[2, 1])
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'green' for s in sorted_df['Strategy']]
        y_pos = range(len(sorted_df))
        ax6.barh(y_pos, sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy']], fontsize=9)
        ax6.set_xlabel('Total Return (%)', fontsize=11)
        ax6.set_title('Total Return', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        # 7. Win Rate
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'steelblue' for s in sorted_df['Strategy']]
        y_pos = range(len(sorted_df))
        ax7.barh(y_pos, sorted_df['Win Rate (%)'], color=colors, alpha=0.7)
        ax7.set_yticks(y_pos)
        ax7.set_yticklabels([s.replace('_', ' ') for s in sorted_df['Strategy']], fontsize=9)
        ax7.set_xlabel('Win Rate (%)', fontsize=11)
        ax7.set_title('Win Rate', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')
        ax7.axvline(x=50, color='black', linestyle='--', linewidth=1, alpha=0.5)

        plt.savefig('bitcoin_final_top5_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nChart saved: bitcoin_final_top5_comparison.png")
        plt.close()

    def print_report(self):
        """최종 보고서 출력"""
        metrics_df = self.create_metrics_df()

        print("\n" + "="*120)
        print(f"{'Bitcoin Trend-Following Strategy - Final Top 5 Report':^120}")
        print("="*120)
        print(f"\nPeriod: 2018-01-01 ~ 2025-11-05 (7.8 years)")
        print(f"Asset: BTC_KRW (Upbit)")
        print(f"Slippage: 0.2%")
        print(f"Position: Long-only, no leverage")

        print("\n" + "-"*120)
        print(f"{'Performance Metrics':^120}")
        print("-"*120)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}')

        # 벤치마크와 비교하여 개선율 추가
        benchmark_sharpe = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0]
        metrics_df['Sharpe Improvement (%)'] = ((metrics_df['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)

        print(metrics_df.to_string(index=False))
        print("\n" + "="*120 + "\n")


def main():
    print("="*80)
    print("Bitcoin Final Top 5 Strategies Analysis")
    print("="*80)

    analyzer = FinalTop5Analysis()
    analyzer.load_data()
    analyzer.run_all()

    metrics_df = analyzer.create_metrics_df()

    analyzer.print_report()
    analyzer.plot_comparison()

    # CSV 저장
    metrics_df.to_csv('bitcoin_final_top5_metrics.csv', index=False)
    print("✓ Metrics saved: bitcoin_final_top5_metrics.csv")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
