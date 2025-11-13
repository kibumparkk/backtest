"""
Per-Symbol SMA Cross Strategy Optimization

Finds the optimal SMA parameters for each individual symbol,
rather than optimizing for the portfolio average.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sma_cross_optimization import SMACrossStrategy

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerSymbolSMAOptimization:
    """Per-symbol SMA optimization analyzer"""

    def __init__(self, symbols, start_date, end_date, slippage=0.002):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.slippage = slippage
        self.results_by_symbol = {}

    def optimize_all_symbols(self, short_range, long_range, metric='sharpe_ratio'):
        """
        Optimize parameters for each symbol individually

        Args:
            short_range: List of short SMA periods
            long_range: List of long SMA periods
            metric: Optimization metric

        Returns:
            Dictionary of results per symbol
        """
        print(f"\n{'#'*80}")
        print(f"# PER-SYMBOL SMA OPTIMIZATION")
        print(f"{'#'*80}")
        print(f"Optimizing {len(self.symbols)} symbols individually")
        print(f"Short range: {short_range}")
        print(f"Long range: {long_range}")
        print(f"Metric: {metric}")
        print(f"{'#'*80}\n")

        for symbol in self.symbols:
            print(f"\n{'='*80}")
            print(f"OPTIMIZING: {symbol}")
            print(f"{'='*80}")

            # Create strategy instance for single symbol
            strategy = SMACrossStrategy([symbol], self.start_date, self.end_date, self.slippage)
            strategy.load_data()

            # Run optimization
            results = []
            total = len([1 for s in short_range for l in long_range if s < l])
            current = 0

            for short in short_range:
                for long in long_range:
                    if short >= long:
                        continue

                    current += 1
                    print(f"  Testing {current}/{total}: SMA({short}) vs SMA({long})...", end='\r')

                    # Get data for this symbol
                    df = strategy.data[symbol]
                    result_df = strategy.backtest_strategy(df, short, long)
                    metrics = strategy.calculate_metrics(result_df)

                    results.append({
                        'symbol': symbol,
                        'short_period': short,
                        'long_period': long,
                        **metrics
                    })

            print(f"\n  ✓ Completed {len(results)} combinations for {symbol}\n")

            # Convert to DataFrame and store
            results_df = pd.DataFrame(results)
            self.results_by_symbol[symbol] = results_df

            # Find and print best parameters
            best_idx = results_df[metric].idxmax()
            best = results_df.loc[best_idx]

            print(f"  BEST PARAMETERS for {symbol}:")
            print(f"    Short={int(best['short_period'])}, Long={int(best['long_period'])}")
            print(f"    Total Return: {best['total_return']:.2f}%")
            print(f"    CAGR: {best['cagr']:.2f}%")
            print(f"    Sharpe Ratio: {best['sharpe_ratio']:.3f}")
            print(f"    Max Drawdown: {best['max_drawdown']:.2f}%")
            print(f"    Win Rate: {best['win_rate']:.2f}%")

        return self.results_by_symbol

    def create_comparison_table(self, metric='sharpe_ratio'):
        """Create comparison table of best parameters per symbol"""

        print(f"\n{'='*80}")
        print(f"BEST PARAMETERS PER SYMBOL (by {metric})")
        print(f"{'='*80}\n")

        comparison_data = []
        for symbol, results_df in self.results_by_symbol.items():
            best_idx = results_df[metric].idxmax()
            best = results_df.loc[best_idx]

            comparison_data.append({
                'Symbol': symbol,
                'Short': int(best['short_period']),
                'Long': int(best['long_period']),
                'Total Return (%)': f"{best['total_return']:.2f}",
                'CAGR (%)': f"{best['cagr']:.2f}",
                'Sharpe': f"{best['sharpe_ratio']:.3f}",
                'MDD (%)': f"{best['max_drawdown']:.2f}",
                'Win Rate (%)': f"{best['win_rate']:.2f}",
                'Trades': int(best['total_trades'])
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print(f"\n{'='*80}\n")

        # Save to CSV
        comparison_df.to_csv('sma_best_params_per_symbol.csv', index=False)
        print("✓ Saved to: sma_best_params_per_symbol.csv\n")

        return comparison_df

    def plot_per_symbol_heatmaps(self, metric='sharpe_ratio',
                                  save_path='sma_per_symbol_heatmaps.png'):
        """Create heatmap for each symbol"""

        n_symbols = len(self.symbols)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        fig.suptitle(f'SMA Cross Strategy - Per-Symbol Optimization Heatmaps ({metric.replace("_", " ").title()})',
                     fontsize=18, fontweight='bold', y=0.995)

        for idx, (symbol, results_df) in enumerate(self.results_by_symbol.items()):
            ax = axes[idx]

            # Create pivot table
            pivot = results_df.pivot_table(
                values=metric,
                index='long_period',
                columns='short_period',
                aggfunc='mean'
            )
            pivot = pivot.sort_index(ascending=False)
            pivot = pivot.sort_index(axis=1)

            # Create heatmap
            cmap = 'RdYlGn' if metric != 'max_drawdown' else 'RdYlGn_r'
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                center=None if metric == 'total_trades' else pivot.mean().mean(),
                cbar_kws={'label': metric.replace('_', ' ').title()},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )

            # Find best value
            best_idx = results_df[metric].idxmax() if metric != 'max_drawdown' else results_df[metric].idxmax()
            best = results_df.loc[best_idx]

            ax.set_title(f'{symbol}\nBest: SMA({int(best["short_period"])}) vs SMA({int(best["long_period"])}) | {metric.replace("_", " ").title()}: {best[metric]:.3f}',
                        fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Short SMA Period', fontsize=12, fontweight='bold')
            ax.set_ylabel('Long SMA Period', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Per-symbol heatmaps saved to: {save_path}")
        plt.close()

    def plot_best_parameters_comparison(self, save_path='sma_best_params_comparison.png'):
        """Create bar charts comparing best parameters across symbols"""

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Per-Symbol Best Parameters Comparison',
                     fontsize=18, fontweight='bold', y=0.995)

        # Collect best parameters
        best_params = []
        for symbol, results_df in self.results_by_symbol.items():
            best_idx = results_df['sharpe_ratio'].idxmax()
            best = results_df.loc[best_idx]
            best_params.append({
                'symbol': symbol,
                'short': int(best['short_period']),
                'long': int(best['long_period']),
                'total_return': best['total_return'],
                'cagr': best['cagr'],
                'sharpe': best['sharpe_ratio'],
                'mdd': best['max_drawdown'],
                'win_rate': best['win_rate'],
                'trades': best['total_trades']
            })

        df = pd.DataFrame(best_params)

        # Plot 1: Best Short Periods
        ax1 = axes[0, 0]
        bars = ax1.bar(df['symbol'], df['short'], color='skyblue', edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, df['short']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax1.set_title('Best Short Period', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Short Period', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Best Long Periods
        ax2 = axes[0, 1]
        bars = ax2.bar(df['symbol'], df['long'], color='lightcoral', edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, df['long']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.set_title('Best Long Period', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Long Period', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Sharpe Ratio
        ax3 = axes[0, 2]
        colors = ['green' if x > 1 else 'orange' if x > 0.5 else 'red' for x in df['sharpe']]
        bars = ax3.bar(df['symbol'], df['sharpe'], color=colors, edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, df['sharpe']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax3.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Sharpe=1.0')
        ax3.set_title('Sharpe Ratio (Best Parameters)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: CAGR
        ax4 = axes[1, 0]
        bars = ax4.bar(df['symbol'], df['cagr'], color='lightgreen', edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, df['cagr']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax4.set_title('CAGR (Best Parameters)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('CAGR (%)', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')

        # Plot 5: Max Drawdown
        ax5 = axes[1, 1]
        bars = ax5.bar(df['symbol'], df['mdd'], color='salmon', edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, df['mdd']):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                    f'{val:.1f}%', ha='center', va='top', fontsize=11, fontweight='bold')
        ax5.set_title('Max Drawdown (Best Parameters)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Max Drawdown (%)', fontsize=12)
        ax5.grid(True, alpha=0.3, axis='y')

        # Plot 6: Win Rate
        ax6 = axes[1, 2]
        bars = ax6.bar(df['symbol'], df['win_rate'], color='gold', edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, df['win_rate']):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax6.axhline(y=50, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='50%')
        ax6.set_title('Win Rate (Best Parameters)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Win Rate (%)', fontsize=12)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Best parameters comparison saved to: {save_path}")
        plt.close()

    def plot_parameter_distribution(self, save_path='sma_parameter_distribution.png'):
        """Plot distribution of optimal parameters"""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Distribution of Optimal SMA Parameters Across Symbols',
                     fontsize=16, fontweight='bold')

        # Collect best parameters
        best_shorts = []
        best_longs = []
        for symbol, results_df in self.results_by_symbol.items():
            best_idx = results_df['sharpe_ratio'].idxmax()
            best = results_df.loc[best_idx]
            best_shorts.append(int(best['short_period']))
            best_longs.append(int(best['long_period']))

        # Plot short period distribution
        ax1 = axes[0]
        unique_shorts, counts_shorts = np.unique(best_shorts, return_counts=True)
        bars = ax1.bar(unique_shorts, counts_shorts, color='skyblue', edgecolor='black', linewidth=2)
        for bar, val in zip(bars, counts_shorts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax1.set_title('Optimal Short Period Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Short SMA Period', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Symbols', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot long period distribution
        ax2 = axes[1]
        unique_longs, counts_longs = np.unique(best_longs, return_counts=True)
        bars = ax2.bar(unique_longs, counts_longs, color='lightcoral', edgecolor='black', linewidth=2)
        for bar, val in zip(bars, counts_longs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.set_title('Optimal Long Period Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Long SMA Period', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Symbols', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Parameter distribution saved to: {save_path}")
        plt.close()

    def save_all_results(self):
        """Save detailed results for all symbols"""

        # Combine all results
        all_results = []
        for symbol, results_df in self.results_by_symbol.items():
            all_results.append(results_df)

        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv('sma_per_symbol_all_results.csv', index=False)
        print(f"✓ All results saved to: sma_per_symbol_all_results.csv")

        return combined_df


def main():
    """Main execution"""

    # Configuration
    SYMBOLS = ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW']
    START_DATE = '2018-01-01'
    END_DATE = '2025-11-07'
    SLIPPAGE = 0.002

    # Parameter ranges
    SHORT_RANGE = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    LONG_RANGE = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200]

    # Initialize optimizer
    optimizer = PerSymbolSMAOptimization(SYMBOLS, START_DATE, END_DATE, SLIPPAGE)

    # Run optimization
    results = optimizer.optimize_all_symbols(SHORT_RANGE, LONG_RANGE, metric='sharpe_ratio')

    # Create comparison table
    comparison_df = optimizer.create_comparison_table(metric='sharpe_ratio')

    # Save all results
    optimizer.save_all_results()

    # Create visualizations
    optimizer.plot_per_symbol_heatmaps(metric='sharpe_ratio')
    optimizer.plot_best_parameters_comparison()
    optimizer.plot_parameter_distribution()

    print(f"\n{'='*80}")
    print("PER-SYMBOL OPTIMIZATION COMPLETE!")
    print(f"{'='*80}")
    print("✓ Best parameters per symbol: sma_best_params_per_symbol.csv")
    print("✓ All detailed results: sma_per_symbol_all_results.csv")
    print("✓ Per-symbol heatmaps: sma_per_symbol_heatmaps.png")
    print("✓ Best parameters comparison: sma_best_params_comparison.png")
    print("✓ Parameter distribution: sma_parameter_distribution.png")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
