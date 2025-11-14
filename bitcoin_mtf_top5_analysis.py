"""
비트코인 멀티 타임프레임 상위 5개 전략 상세 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class MTFTop5Analysis:
    """상위 5개 전략 상세 분석"""

    def __init__(self, slippage=0.002):
        self.slippage = slippage
        self.daily_data = None
        self.weekly_data = None
        self.results = {}

    def load_data(self):
        """데이터 로드"""
        df_daily = pd.read_parquet('chart_day/BTC_KRW.parquet')
        df_daily.columns = [col.capitalize() for col in df_daily.columns]
        df_daily = df_daily[df_daily.index >= '2018-01-01']
        self.daily_data = df_daily

        df_weekly = df_daily.resample('W-MON', label='left', closed='left').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        self.weekly_data = df_weekly

        print(f"Loaded {len(df_daily)} daily bars and {len(df_weekly)} weekly bars")

    def backtest(self, signal, name):
        """백테스트"""
        df = self.daily_data.copy()
        df['signal'] = signal
        df['pos_change'] = df['signal'].diff()
        df['daily_ret'] = df['Close'].pct_change()
        df['strat_ret'] = df['signal'].shift(1) * df['daily_ret']

        slip_cost = pd.Series(0.0, index=df.index)
        slip_cost[df['pos_change'] == 1] = -self.slippage
        slip_cost[df['pos_change'] == -1] = -self.slippage
        df['strat_ret'] = df['strat_ret'] + slip_cost
        df['strat_ret'] = df['strat_ret'].fillna(0)

        df['cumulative'] = (1 + df['strat_ret']).cumprod()

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
            'drawdown': drawdown * 100,
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

    def benchmark_daily_sma30(self):
        """벤치마크"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()
        return (df['Close'] > df['SMA30']).astype(int)

    def strategy_weekly_donchian_daily_sma30(self):
        """1위: Weekly Donchian + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['high_20'] = weekly['High'].rolling(20).max()
        weekly['near_high'] = (weekly['Close'] > weekly['high_20'] * 0.95).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_signal = weekly['near_high'].reindex(daily.index, method='ffill')
        return ((daily['Close'] > daily['SMA30']) & (weekly_signal == 1)).astype(int)

    def strategy_weekly_sma10_daily_sma30(self):
        """2위: Weekly SMA10 + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['SMA10'] = weekly['Close'].rolling(10).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['SMA10']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        return ((daily['Close'] > daily['SMA30']) & (weekly_trend == 1)).astype(int)

    def strategy_weekly_ema20_daily_sma30(self):
        """3위: Weekly EMA20 + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['EMA20'] = weekly['Close'].ewm(span=20, adjust=False).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['EMA20']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        return ((daily['Close'] > daily['SMA30']) & (weekly_trend == 1)).astype(int)

    def strategy_weekly_sma20_daily_sma30(self):
        """4위: Weekly SMA20 + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['SMA20']).astype(int)

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        return ((daily['Close'] > daily['SMA30']) & (weekly_trend == 1)).astype(int)

    def strategy_weekly_ema20_daily_ema30(self):
        """5위: Weekly EMA20 + Daily EMA30"""
        weekly = self.weekly_data.copy()
        weekly['EMA20'] = weekly['Close'].ewm(span=20, adjust=False).mean()
        weekly['trend_up'] = (weekly['Close'] > weekly['EMA20']).astype(int)

        daily = self.daily_data.copy()
        daily['EMA30'] = daily['Close'].ewm(span=30, adjust=False).mean()

        weekly_trend = weekly['trend_up'].reindex(daily.index, method='ffill')
        return ((daily['Close'] > daily['EMA30']) & (weekly_trend == 1)).astype(int)

    def run_top5(self):
        """상위 5개 + 벤치마크 실행"""
        print("\n" + "="*80)
        print("Running Top 5 Multi-Timeframe Strategies")
        print("="*80 + "\n")

        strategies = {
            '0_BENCHMARK_Daily_SMA30': self.benchmark_daily_sma30,
            '1_Weekly_Donchian+Daily_SMA30': self.strategy_weekly_donchian_daily_sma30,
            '2_Weekly_SMA10+Daily_SMA30': self.strategy_weekly_sma10_daily_sma30,
            '3_Weekly_EMA20+Daily_SMA30': self.strategy_weekly_ema20_daily_sma30,
            '4_Weekly_SMA20+Daily_SMA30': self.strategy_weekly_sma20_daily_sma30,
            '5_Weekly_EMA20+Daily_EMA30': self.strategy_weekly_ema20_daily_ema30,
        }

        metrics_list = []
        for name, func in strategies.items():
            print(f"{name}... ", end='')
            signal = func()
            metrics = self.backtest(signal, name)
            metrics_list.append(metrics)
            print(f"Sharpe {metrics['Sharpe Ratio']:.4f}")

        return pd.DataFrame(metrics_list)

    def plot_comparison(self, metrics_df, save_path='bitcoin_mtf_top5_comparison.png'):
        """시각화"""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률
        ax1 = fig.add_subplot(gs[0, :])
        for name in self.results.keys():
            cum = self.results[name]['cumulative']
            if 'BENCHMARK' in name:
                ax1.plot(cum.index, cum, label=name, linewidth=4, linestyle='--', alpha=0.9, color='red')
            else:
                ax1.plot(cum.index, cum, label=name, linewidth=2.5, alpha=0.8)

        ax1.set_title('Multi-Timeframe Strategies: Top 5 vs Benchmark', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        benchmark_sharpe = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0]
        metrics_df['Improvement (%)'] = ((metrics_df['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)

        # 2. 샤프 비율
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'green' for s in sorted_df['Strategy']]
        y_pos = range(len(sorted_df))
        ax2.barh(y_pos, sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_df['Strategy'].str.replace('_', ' '), fontsize=9)
        ax2.set_xlabel('Sharpe Ratio', fontsize=11)
        ax2.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=benchmark_sharpe, color='red', linestyle='--', linewidth=2)

        # 3. CAGR
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'steelblue' for s in sorted_df['Strategy']]
        y_pos = range(len(sorted_df))
        ax3.barh(y_pos, sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(sorted_df['Strategy'].str.replace('_', ' '), fontsize=9)
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
        ax4.set_yticklabels(sorted_df['Strategy'].str.replace('_', ' '), fontsize=9)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 개선율
        ax5 = fig.add_subplot(gs[2, 0])
        improvement_df = metrics_df[~metrics_df['Strategy'].str.contains('BENCHMARK')].sort_values('Improvement (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in improvement_df['Improvement (%)']]
        y_pos = range(len(improvement_df))
        ax5.barh(y_pos, improvement_df['Improvement (%)'], color=colors, alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(improvement_df['Strategy'].str.replace('_', ' '), fontsize=9)
        ax5.set_xlabel('Improvement vs Benchmark (%)', fontsize=11)
        ax5.set_title('Sharpe Ratio Improvement', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)

        # 6. Return vs Risk
        ax6 = fig.add_subplot(gs[2, 1])
        for idx, row in metrics_df.iterrows():
            if 'BENCHMARK' in row['Strategy']:
                ax6.scatter(row['MDD (%)'], row['CAGR (%)'], s=500, alpha=0.8,
                           c='red', marker='s', edgecolors='black', linewidths=2, label='Benchmark')
            else:
                ax6.scatter(row['MDD (%)'], row['CAGR (%)'], s=300, alpha=0.7,
                           c='green', edgecolors='black', linewidths=1.5)

        for idx, row in metrics_df.iterrows():
            label = row['Strategy'].replace('_', ' ').replace('0 BENCHMARK ', 'BM: ')
            ax6.annotate(label, (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=8, ha='center', va='bottom')

        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # 7. Win Rate
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.sort_values('Win Rate (%)', ascending=True)
        colors = ['red' if 'BENCHMARK' in s else 'steelblue' for s in sorted_df['Strategy']]
        y_pos = range(len(sorted_df))
        ax7.barh(y_pos, sorted_df['Win Rate (%)'], color=colors, alpha=0.7)
        ax7.set_yticks(y_pos)
        ax7.set_yticklabels(sorted_df['Strategy'].str.replace('_', ' '), fontsize=9)
        ax7.set_xlabel('Win Rate (%)', fontsize=11)
        ax7.set_title('Win Rate Comparison', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')
        ax7.axvline(x=50, color='black', linestyle='--', linewidth=1)

        # 8. Drawdown 시계열
        ax8 = fig.add_subplot(gs[3, :])
        for name in self.results.keys():
            dd = self.results[name]['drawdown']
            if 'BENCHMARK' in name:
                ax8.plot(dd.index, dd, label=name, linewidth=3, linestyle='--', alpha=0.9, color='red')
            else:
                ax8.plot(dd.index, dd, label=name, linewidth=2, alpha=0.7)

        ax8.fill_between(dd.index, dd, 0, alpha=0.1)
        ax8.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Drawdown (%)', fontsize=12)
        ax8.set_xlabel('Date', fontsize=12)
        ax8.legend(loc='lower right', fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Chart saved: {save_path}")
        plt.close()

    def print_report(self, metrics_df):
        """보고서 출력"""
        benchmark_sharpe = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0]

        print("\n" + "="*120)
        print(f"{'Multi-Timeframe Strategy - Final Top 5 Report':^120}")
        print("="*120)
        print(f"\nPeriod: 2018-01-01 ~ 2025-11-05")
        print(f"Asset: BTC_KRW")
        print(f"Benchmark: Daily SMA30 only (Sharpe {benchmark_sharpe:.4f})")
        print(f"Slippage: 0.2%")

        print("\n" + "-"*120)
        print(f"{'Performance Metrics':^120}")
        print("-"*120)

        metrics_df['Improvement (%)'] = ((metrics_df['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)

        display_df = metrics_df[['Strategy', 'Sharpe Ratio', 'Improvement (%)',
                                  'Total Return (%)', 'CAGR (%)', 'MDD (%)', 'Win Rate (%)', 'Total Trades']]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}')

        print(display_df.to_string(index=False))
        print("\n" + "="*120)


def main():
    """메인"""
    print("\n" + "="*80)
    print("Multi-Timeframe Strategy - Top 5 Final Analysis")
    print("="*80)

    analyzer = MTFTop5Analysis(slippage=0.002)
    analyzer.load_data()

    metrics_df = analyzer.run_top5()
    analyzer.print_report(metrics_df)
    analyzer.plot_comparison(metrics_df)

    # CSV 저장
    metrics_df.to_csv('bitcoin_mtf_top5_final_metrics.csv', index=False)
    print(f"\n✓ Metrics saved: bitcoin_mtf_top5_final_metrics.csv")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)

    return metrics_df


if __name__ == "__main__":
    main()
