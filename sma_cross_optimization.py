"""
SMA Cross Strategy with Parameter Optimization

This module implements a Simple Moving Average (SMA) crossover strategy
where we buy when SMA(short) > SMA(long) and sell otherwise.

Key features:
- Short period includes 1 (previous day close) as an option
- Comprehensive parameter optimization with grid search
- Visualization of optimization curves (heatmap)
- Avoids look-ahead bias and other backtesting pitfalls
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SMACrossStrategy:
    """
    SMA Cross Strategy Backtester with Parameter Optimization

    Strategy Logic:
    - Buy Signal: SMA(short) > SMA(long)
    - Sell Signal: SMA(short) <= SMA(long)
    - Short period can be 1 (previous day close)
    """

    def __init__(self, symbols, start_date, end_date, slippage=0.002):
        """
        Initialize the SMA cross strategy backtester

        Args:
            symbols: List of trading symbols (e.g., ['BTC_KRW', 'ETH_KRW'])
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            slippage: Transaction cost as decimal (default 0.2%)
        """
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.slippage = slippage
        self.data = {}

    def load_data(self):
        """Load OHLCV data from parquet files"""
        print(f"Loading data for {len(self.symbols)} symbols...")

        for symbol in self.symbols:
            file_path = f'chart_day/{symbol}.parquet'
            if Path(file_path).exists():
                df = pd.read_parquet(file_path)
                # Capitalize column names
                df.columns = [col.capitalize() for col in df.columns]
                # Filter by date range
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                self.data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} days loaded")
            else:
                print(f"  ✗ {symbol}: File not found")

        return self.data

    def calculate_sma(self, prices, period):
        """
        Calculate Simple Moving Average

        Args:
            prices: Price series
            period: SMA period (if period=1, returns shifted close price)

        Returns:
            SMA series
        """
        if period == 1:
            # For period=1, use previous day's close
            return prices.shift(1)
        else:
            return prices.rolling(window=period).mean()

    def backtest_strategy(self, df, short_period=10, long_period=30):
        """
        Run backtest for given SMA parameters

        Args:
            df: DataFrame with OHLCV data
            short_period: Short SMA period (can be 1 for previous close)
            long_period: Long SMA period

        Returns:
            DataFrame with strategy results
        """
        df = df.copy()

        # Calculate SMAs
        df['sma_short'] = self.calculate_sma(df['Close'], short_period)
        df['sma_long'] = self.calculate_sma(df['Close'], long_period)

        # Generate signals (1 = long position, 0 = no position)
        # Buy when short SMA > long SMA
        df['signal'] = (df['sma_short'] > df['sma_long']).astype(int)

        # CRITICAL: Shift signal by 1 to avoid look-ahead bias
        # We can only act on tomorrow's open based on today's signal
        df['position'] = df['signal'].shift(1)

        # Calculate daily returns
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'] * df['daily_price_return']

        # Apply slippage costs
        slippage_cost = pd.Series(0.0, index=df.index)
        position_change = df['position'].diff()

        # Apply slippage on position changes (buys and sells)
        slippage_cost[position_change == 1] = -self.slippage   # Buy
        slippage_cost[position_change == -1] = -self.slippage  # Sell

        df['returns'] = df['returns'] + slippage_cost

        # Calculate cumulative returns
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def calculate_metrics(self, df):
        """
        Calculate performance metrics

        Args:
            df: DataFrame with backtest results

        Returns:
            Dictionary of performance metrics
        """
        returns = df['returns'].dropna()
        cumulative = df['cumulative'].dropna()

        if len(returns) == 0 or len(cumulative) == 0:
            return {
                'total_return': 0,
                'cagr': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0
            }

        # Calculate years
        years = (df.index[-1] - df.index[0]).days / 365.25

        # Total return
        total_return = (cumulative.iloc[-1] - 1) * 100

        # CAGR
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # Sharpe ratio (annualized, assuming 365 trading days for crypto)
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0

        # Maximum drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }

    def optimize_parameters(self, short_range, long_range, metric='sharpe_ratio'):
        """
        Perform grid search parameter optimization

        Args:
            short_range: List of short SMA periods to test (e.g., [1, 5, 10, 20])
            long_range: List of long SMA periods to test (e.g., [20, 30, 50, 100])
            metric: Optimization metric ('sharpe_ratio', 'cagr', 'total_return')

        Returns:
            DataFrame with optimization results
        """
        print(f"\n{'='*60}")
        print(f"PARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Short periods: {short_range}")
        print(f"Long periods: {long_range}")
        print(f"Optimization metric: {metric}")
        print(f"Total combinations: {len(short_range) * len(long_range)}")
        print(f"{'='*60}\n")

        results = []
        total_combinations = len(short_range) * len(long_range)
        current = 0

        for short in short_range:
            for long in long_range:
                # Skip if short >= long (doesn't make sense for cross strategy)
                if short >= long:
                    continue

                current += 1
                print(f"Testing {current}/{total_combinations}: SMA({short}) vs SMA({long})...", end='\r')

                # Run backtest for each symbol
                symbol_metrics = []
                for symbol, df in self.data.items():
                    result_df = self.backtest_strategy(df, short, long)
                    metrics = self.calculate_metrics(result_df)
                    symbol_metrics.append(metrics)

                # Calculate average metrics across all symbols
                avg_metrics = {
                    'short_period': short,
                    'long_period': long,
                    'total_return': np.mean([m['total_return'] for m in symbol_metrics]),
                    'cagr': np.mean([m['cagr'] for m in symbol_metrics]),
                    'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in symbol_metrics]),
                    'max_drawdown': np.mean([m['max_drawdown'] for m in symbol_metrics]),
                    'win_rate': np.mean([m['win_rate'] for m in symbol_metrics]),
                    'total_trades': np.mean([m['total_trades'] for m in symbol_metrics])
                }

                results.append(avg_metrics)

        print(f"\n\n✓ Optimization complete! Tested {len(results)} valid combinations.\n")

        results_df = pd.DataFrame(results)

        # Find best parameters
        best_idx = results_df[metric].idxmax()
        best_params = results_df.loc[best_idx]

        print(f"{'='*60}")
        print(f"BEST PARAMETERS (by {metric})")
        print(f"{'='*60}")
        print(f"Short Period: {int(best_params['short_period'])}")
        print(f"Long Period: {int(best_params['long_period'])}")
        print(f"Total Return: {best_params['total_return']:.2f}%")
        print(f"CAGR: {best_params['cagr']:.2f}%")
        print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {best_params['max_drawdown']:.2f}%")
        print(f"Win Rate: {best_params['win_rate']:.2f}%")
        print(f"Avg Trades: {best_params['total_trades']:.1f}")
        print(f"{'='*60}\n")

        return results_df

    def plot_optimization_curves(self, results_df, save_path='sma_optimization_curves.png'):
        """
        Visualize parameter optimization results with multiple heatmaps

        Args:
            results_df: DataFrame with optimization results
            save_path: Path to save the visualization
        """
        print(f"Creating optimization visualizations...")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('SMA Cross Strategy - Parameter Optimization Curves',
                     fontsize=20, fontweight='bold', y=0.995)

        # Metrics to visualize
        metrics = [
            ('sharpe_ratio', 'Sharpe Ratio', 'RdYlGn'),
            ('cagr', 'CAGR (%)', 'RdYlGn'),
            ('total_return', 'Total Return (%)', 'RdYlGn'),
            ('max_drawdown', 'Max Drawdown (%)', 'RdYlGn_r'),  # Reversed: lower is better
            ('win_rate', 'Win Rate (%)', 'RdYlGn'),
            ('total_trades', 'Total Trades', 'viridis')
        ]

        for idx, (metric, title, cmap) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            # Create pivot table for heatmap
            pivot_table = results_df.pivot_table(
                values=metric,
                index='long_period',
                columns='short_period',
                aggfunc='mean'
            )

            # Sort index and columns
            pivot_table = pivot_table.sort_index(ascending=False)
            pivot_table = pivot_table.sort_index(axis=1)

            # Create heatmap
            sns.heatmap(
                pivot_table,
                annot=True,
                fmt='.2f' if metric != 'total_trades' else '.0f',
                cmap=cmap,
                center=None if metric == 'total_trades' else 0 if 'drawdown' in metric else pivot_table.mean().mean(),
                cbar_kws={'label': title},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )

            ax.set_title(f'{title}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Short SMA Period', fontsize=12, fontweight='bold')
            ax.set_ylabel('Long SMA Period', fontsize=12, fontweight='bold')

            # Find and mark best value
            if metric == 'max_drawdown':
                # For drawdown, best is minimum (least negative)
                best_val = results_df.loc[results_df[metric].idxmax()]
            else:
                best_val = results_df.loc[results_df[metric].idxmax()]

            # Add text annotation for best parameters
            ax.text(0.02, 0.98,
                    f"Best: Short={int(best_val['short_period'])}, Long={int(best_val['long_period'])}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Optimization curves saved to: {save_path}")
        plt.close()

        # Create a separate 3D surface plot for Sharpe Ratio
        self._plot_3d_surface(results_df, 'sharpe_ratio',
                             'sma_optimization_3d_sharpe.png')

    def _plot_3d_surface(self, results_df, metric, save_path):
        """Create 3D surface plot for optimization results"""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Prepare data
        pivot_table = results_df.pivot_table(
            values=metric,
            index='long_period',
            columns='short_period',
            aggfunc='mean'
        )

        X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
        Z = pivot_table.values

        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                              edgecolor='none', alpha=0.8)

        # Add contour lines
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5)

        # Labels
        ax.set_xlabel('Short SMA Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('Long SMA Period', fontsize=12, fontweight='bold')
        ax.set_zlabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'3D Optimization Surface - {metric.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold', pad=20)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Find and mark best point
        best_idx = results_df[metric].idxmax()
        best_params = results_df.loc[best_idx]
        ax.scatter([best_params['short_period']],
                  [best_params['long_period']],
                  [best_params[metric]],
                  color='red', s=200, marker='*',
                  label=f"Best: ({int(best_params['short_period'])}, {int(best_params['long_period'])})")
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 3D surface plot saved to: {save_path}")
        plt.close()

    def plot_best_strategy_results(self, best_short, best_long,
                                   save_path='sma_best_strategy_results.png'):
        """
        Plot detailed results for the best parameter combination

        Args:
            best_short: Best short SMA period
            best_long: Best long SMA period
            save_path: Path to save the visualization
        """
        print(f"\nGenerating detailed results for SMA({best_short}) vs SMA({best_long})...")

        n_symbols = len(self.symbols)
        fig, axes = plt.subplots(n_symbols, 2, figsize=(20, 5*n_symbols))

        if n_symbols == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'SMA Cross Strategy - Best Parameters: SMA({best_short}) vs SMA({best_long})',
                    fontsize=18, fontweight='bold', y=0.998)

        for idx, (symbol, df) in enumerate(self.data.items()):
            # Run backtest
            result_df = self.backtest_strategy(df, best_short, best_long)
            metrics = self.calculate_metrics(result_df)

            # Left plot: Price and SMAs with signals
            ax1 = axes[idx, 0]
            ax1.plot(result_df.index, result_df['Close'],
                    label='Close Price', color='black', linewidth=1.5, alpha=0.7)
            ax1.plot(result_df.index, result_df['sma_short'],
                    label=f'SMA({best_short})', color='blue', linewidth=1.5)
            ax1.plot(result_df.index, result_df['sma_long'],
                    label=f'SMA({best_long})', color='red', linewidth=1.5)

            # Mark buy signals
            buy_signals = result_df[result_df['position'].diff() == 1]
            ax1.scatter(buy_signals.index, buy_signals['Close'],
                       color='green', marker='^', s=100,
                       label='Buy Signal', zorder=5, alpha=0.7)

            # Mark sell signals
            sell_signals = result_df[result_df['position'].diff() == -1]
            ax1.scatter(sell_signals.index, sell_signals['Close'],
                       color='red', marker='v', s=100,
                       label='Sell Signal', zorder=5, alpha=0.7)

            ax1.set_title(f'{symbol} - Price & Signals', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Price (KRW)', fontsize=11, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')

            # Right plot: Cumulative returns
            ax2 = axes[idx, 1]
            ax2.plot(result_df.index, result_df['cumulative'],
                    label='Strategy Returns', color='green', linewidth=2)
            ax2.axhline(y=1, color='black', linestyle='--',
                       linewidth=1, alpha=0.5, label='Break Even')

            # Add performance metrics as text
            metrics_text = f"""Performance Metrics:
Total Return: {metrics['total_return']:.2f}%
CAGR: {metrics['cagr']:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
Max Drawdown: {metrics['max_drawdown']:.2f}%
Win Rate: {metrics['win_rate']:.2f}%
Total Trades: {int(metrics['total_trades'])}"""

            ax2.text(0.02, 0.98, metrics_text,
                    transform=ax2.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                    family='monospace')

            ax2.set_title(f'{symbol} - Cumulative Returns', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Cumulative Return (1 = 100%)', fontsize=11, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Best strategy results saved to: {save_path}")
        plt.close()

    def run_optimization_analysis(self, short_range=None, long_range=None,
                                 metric='sharpe_ratio'):
        """
        Run complete optimization analysis workflow

        Args:
            short_range: List of short periods (default: [1, 5, 10, 15, 20, 30])
            long_range: List of long periods (default: [20, 30, 50, 100, 150, 200])
            metric: Optimization metric (default: 'sharpe_ratio')
        """
        # Default ranges
        if short_range is None:
            short_range = [1, 5, 10, 15, 20, 30, 40, 50]
        if long_range is None:
            long_range = [20, 30, 50, 75, 100, 150, 200]

        print(f"\n{'#'*60}")
        print(f"# SMA CROSS STRATEGY OPTIMIZATION")
        print(f"{'#'*60}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Slippage: {self.slippage*100}%")
        print(f"{'#'*60}\n")

        # Step 1: Load data
        self.load_data()

        # Step 2: Optimize parameters
        results_df = self.optimize_parameters(short_range, long_range, metric)

        # Step 3: Save results to CSV
        csv_path = 'sma_optimization_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to: {csv_path}\n")

        # Step 4: Plot optimization curves
        self.plot_optimization_curves(results_df)

        # Step 5: Plot best strategy results
        best_idx = results_df[metric].idxmax()
        best_params = results_df.loc[best_idx]
        best_short = int(best_params['short_period'])
        best_long = int(best_params['long_period'])

        self.plot_best_strategy_results(best_short, best_long)

        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"✓ Optimization results: sma_optimization_results.csv")
        print(f"✓ Optimization heatmaps: sma_optimization_curves.png")
        print(f"✓ 3D surface plot: sma_optimization_3d_sharpe.png")
        print(f"✓ Best strategy details: sma_best_strategy_results.png")
        print(f"{'='*60}\n")

        return results_df


def main():
    """Main execution function"""

    # Configuration
    SYMBOLS = ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW']
    START_DATE = '2018-01-01'
    END_DATE = '2025-11-07'
    SLIPPAGE = 0.002  # 0.2%

    # Short range: includes 1 (previous close) to 50
    SHORT_RANGE = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # Long range: from 5 to 200 (extended to include shorter periods)
    LONG_RANGE = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200]

    # Optimization metric
    METRIC = 'sharpe_ratio'  # Options: 'sharpe_ratio', 'cagr', 'total_return'

    # Initialize and run
    strategy = SMACrossStrategy(SYMBOLS, START_DATE, END_DATE, SLIPPAGE)
    results_df = strategy.run_optimization_analysis(SHORT_RANGE, LONG_RANGE, METRIC)

    return results_df


if __name__ == '__main__':
    main()
