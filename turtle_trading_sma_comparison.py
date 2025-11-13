"""
Turtle Trading vs SMA30 Baseline Comparison
============================================

This script compares the performance of:
1. SMA30 Baseline Strategy (Close > SMA30)
2. Turtle Trading with various window parameters

Outputs:
- CAGR vs MDD scatter plot comparing all strategies
- Optimal parameter curves for each Turtle Trading window size
- Performance metrics CSV file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TurtleTradingSMAComparison:
    """Compares SMA30 baseline with Turtle Trading variations"""

    def __init__(self, data_dir: str = 'chart_day'):
        self.data_dir = Path(data_dir)
        self.symbols = ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW']
        self.data = {}
        self.results = {}

    def load_data(self):
        """Load cryptocurrency price data from parquet files"""
        print("Loading data...")
        for symbol in self.symbols:
            file_path = self.data_dir / f'{symbol}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                # Standardize column names
                df.columns = [col.capitalize() for col in df.columns]
                self.data[symbol] = df
                print(f"  Loaded {symbol}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
            else:
                print(f"  Warning: {file_path} not found")

        if not self.data:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")

    def strategy_sma_baseline(self, df: pd.DataFrame, sma_period: int = 30) -> pd.DataFrame:
        """
        SMA30 Baseline Strategy

        Entry: Close > SMA30
        Exit: Close <= SMA30

        Args:
            df: DataFrame with OHLCV data
            sma_period: SMA period (default: 30)

        Returns:
            DataFrame with strategy signals and returns
        """
        df = df.copy()

        # Calculate SMA
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # Generate signals (1 = long, 0 = cash)
        # Use previous day's close vs SMA to avoid look-ahead bias
        df['prev_close'] = df['Close'].shift(1)
        df['prev_sma'] = df['SMA'].shift(1)
        df['position'] = np.where(df['prev_close'] > df['prev_sma'], 1, 0)

        # Calculate position changes
        df['position_change'] = df['position'].diff()

        # Calculate returns with slippage (0.2%)
        slippage = 0.002
        df['market_return'] = df['Close'].pct_change()
        df['strategy_return'] = df['position'].shift(1) * df['market_return']

        # Apply slippage on position changes
        df.loc[df['position_change'] != 0, 'strategy_return'] -= slippage

        # Fill NaN with 0
        df['strategy_return'] = df['strategy_return'].fillna(0)

        return df

    def strategy_turtle_trading(self, df: pd.DataFrame, entry_period: int = 20,
                                exit_period: int = 10) -> pd.DataFrame:
        """
        Turtle Trading Strategy

        Entry: Price breaks above N-day high
        Exit: Price breaks below M-day low

        Args:
            df: DataFrame with OHLCV data
            entry_period: Period for entry signal (N-day high)
            exit_period: Period for exit signal (M-day low)

        Returns:
            DataFrame with strategy signals and returns
        """
        df = df.copy()

        # Calculate breakout levels (using previous periods to avoid look-ahead bias)
        df['upper_band'] = df['High'].shift(1).rolling(window=entry_period).max()
        df['lower_band'] = df['Low'].shift(1).rolling(window=exit_period).min()

        # Initialize position
        df['position'] = 0

        # Generate signals
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            current_high = df['High'].iloc[i]
            current_low = df['Low'].iloc[i]
            upper = df['upper_band'].iloc[i]
            lower = df['lower_band'].iloc[i]

            # Entry signal: break above upper band
            if pd.notna(upper) and current_high > upper and position == 0:
                position = 1
            # Exit signal: break below lower band
            elif pd.notna(lower) and current_low < lower and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions
        df['position_change'] = df['position'].diff()

        # Calculate returns with slippage (0.2%)
        slippage = 0.002
        df['market_return'] = df['Close'].pct_change()

        # Execute at close price when signal is generated
        df['strategy_return'] = df['position'].shift(1) * df['market_return']

        # Apply slippage on position changes
        df.loc[df['position_change'] != 0, 'strategy_return'] -= slippage

        # Fill NaN with 0
        df['strategy_return'] = df['strategy_return'].fillna(0)

        return df

    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics

        Args:
            returns: Series of daily returns

        Returns:
            Dictionary of performance metrics
        """
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()

        # Total return
        total_return = (cumulative.iloc[-1] - 1) * 100

        # CAGR
        years = len(returns) / 365.0
        cagr = (cumulative.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else 0

        # Maximum Drawdown
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        mdd = drawdown.min()

        # Sharpe Ratio (annualized)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0

        # Win Rate
        trades = returns[returns != 0]
        win_rate = (trades > 0).sum() / len(trades) * 100 if len(trades) > 0 else 0
        total_trades = len(trades)

        # Profit Factor
        gains = trades[trades > 0].sum()
        losses = abs(trades[trades < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.inf

        return {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades,
            'Profit Factor': profit_factor
        }

    def create_portfolio(self, strategy_name: str, strategy_results: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Create equal-weighted portfolio from individual symbol results

        Args:
            strategy_name: Name of the strategy
            strategy_results: Dictionary of symbol -> DataFrame with strategy_return column

        Returns:
            Series of portfolio returns
        """
        # Find common dates across all symbols
        indices = [df.index for df in strategy_results.values()]
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)

        # Equal weight
        weight = 1.0 / len(strategy_results)

        # Combine returns
        portfolio_returns = pd.Series(0.0, index=common_index)
        for symbol, df in strategy_results.items():
            symbol_returns = df.loc[common_index, 'strategy_return']
            portfolio_returns += symbol_returns * weight

        return portfolio_returns

    def run_baseline_strategy(self):
        """Run SMA30 baseline strategy on all symbols"""
        print("\nRunning SMA30 Baseline Strategy...")

        baseline_results = {}
        for symbol in self.symbols:
            df = self.data[symbol].copy()
            result_df = self.strategy_sma_baseline(df, sma_period=30)
            baseline_results[symbol] = result_df
            print(f"  {symbol}: Completed")

        # Create portfolio
        portfolio_returns = self.create_portfolio('SMA30_Baseline', baseline_results)

        # Calculate metrics
        metrics = self.calculate_metrics(portfolio_returns)

        self.results['SMA30_Baseline'] = {
            'returns': portfolio_returns,
            'metrics': metrics,
            'entry_period': None,
            'exit_period': None
        }

        print(f"  Portfolio CAGR: {metrics['CAGR (%)']:.2f}%, MDD: {metrics['MDD (%)']:.2f}%")

    def run_turtle_trading_variations(self):
        """Run Turtle Trading with various window parameters"""
        print("\nRunning Turtle Trading Variations...")

        # Define parameter grid
        entry_periods = [10, 15, 20, 25, 30, 40, 50]
        exit_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]

        for entry_period in entry_periods:
            for exit_ratio in exit_ratios:
                exit_period = int(entry_period * exit_ratio)
                if exit_period < 5:  # Skip very small exit periods
                    continue

                strategy_name = f'Turtle_{entry_period}_{exit_period}'

                # Run strategy on all symbols
                turtle_results = {}
                for symbol in self.symbols:
                    df = self.data[symbol].copy()
                    result_df = self.strategy_turtle_trading(df, entry_period, exit_period)
                    turtle_results[symbol] = result_df

                # Create portfolio
                portfolio_returns = self.create_portfolio(strategy_name, turtle_results)

                # Calculate metrics
                metrics = self.calculate_metrics(portfolio_returns)

                self.results[strategy_name] = {
                    'returns': portfolio_returns,
                    'metrics': metrics,
                    'entry_period': entry_period,
                    'exit_period': exit_period
                }

                print(f"  {strategy_name}: CAGR={metrics['CAGR (%)']:.2f}%, MDD={metrics['MDD (%)']:.2f}%")

    def plot_cagr_vs_mdd(self, save_path: str = 'cagr_vs_mdd_comparison.png'):
        """
        Plot CAGR vs MDD scatter plot

        Args:
            save_path: Path to save the plot
        """
        print(f"\nGenerating CAGR vs MDD plot...")

        # Prepare data
        plot_data = []
        for strategy_name, result in self.results.items():
            metrics = result['metrics']
            entry_period = result['entry_period']

            plot_data.append({
                'Strategy': strategy_name,
                'CAGR (%)': metrics['CAGR (%)'],
                'MDD (%)': abs(metrics['MDD (%)']),  # Use absolute value for better visualization
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Entry Period': entry_period if entry_period else 0,
                'Is Baseline': strategy_name == 'SMA30_Baseline'
            })

        df_plot = pd.DataFrame(plot_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Separate baseline and turtle trading
        baseline = df_plot[df_plot['Is Baseline']]
        turtle = df_plot[~df_plot['Is Baseline']]

        # Plot Turtle Trading strategies (colored by entry period)
        scatter = ax.scatter(
            turtle['MDD (%)'],
            turtle['CAGR (%)'],
            c=turtle['Entry Period'],
            s=turtle['Sharpe Ratio'] * 100,  # Size by Sharpe Ratio
            alpha=0.6,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5,
            label='Turtle Trading'
        )

        # Plot baseline with distinct marker
        ax.scatter(
            baseline['MDD (%)'],
            baseline['CAGR (%)'],
            s=300,
            marker='*',
            c='red',
            edgecolors='darkred',
            linewidth=2,
            label='SMA30 Baseline',
            zorder=5
        )

        # Add colorbar for entry period
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Entry Period (days)', fontsize=12)

        # Add labels and title
        ax.set_xlabel('Maximum Drawdown (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('CAGR (%)', fontsize=14, fontweight='bold')
        ax.set_title('CAGR vs MDD: SMA30 Baseline vs Turtle Trading Variations\n(Bubble size = Sharpe Ratio)',
                     fontsize=16, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add legend
        ax.legend(fontsize=12, loc='best')

        # Add annotations for best strategies
        # Best CAGR
        best_cagr = df_plot.loc[df_plot['CAGR (%)'].idxmax()]
        ax.annotate(
            f"Best CAGR: {best_cagr['Strategy']}\n({best_cagr['CAGR (%)']:.2f}%)",
            xy=(best_cagr['MDD (%)'], best_cagr['CAGR (%)']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        # Best Sharpe (if not baseline)
        turtle_df = df_plot[~df_plot['Is Baseline']]
        if not turtle_df.empty:
            best_sharpe = turtle_df.loc[turtle_df['Sharpe Ratio'].idxmax()]
            ax.annotate(
                f"Best Sharpe: {best_sharpe['Strategy']}\n({best_sharpe['Sharpe Ratio']:.2f})",
                xy=(best_sharpe['MDD (%)'], best_sharpe['CAGR (%)']),
                xytext=(10, -20),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()

    def plot_parameter_curves(self, save_path: str = 'turtle_parameter_curves.png'):
        """
        Plot optimal parameter curves by entry window size

        For each entry period, show how CAGR and MDD vary with exit period

        Args:
            save_path: Path to save the plot
        """
        print(f"\nGenerating parameter curves...")

        # Prepare data
        turtle_data = []
        for strategy_name, result in self.results.items():
            if strategy_name == 'SMA30_Baseline':
                continue

            metrics = result['metrics']
            turtle_data.append({
                'Entry Period': result['entry_period'],
                'Exit Period': result['exit_period'],
                'Exit Ratio': result['exit_period'] / result['entry_period'],
                'CAGR (%)': metrics['CAGR (%)'],
                'MDD (%)': metrics['MDD (%)'],
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Win Rate (%)': metrics['Win Rate (%)']
            })

        df_turtle = pd.DataFrame(turtle_data)

        # Get unique entry periods
        entry_periods = sorted(df_turtle['Entry Period'].unique())

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Turtle Trading: Parameter Optimization Curves by Entry Window Size',
                     fontsize=16, fontweight='bold', y=0.995)

        # Define metrics to plot
        metrics_to_plot = [
            ('CAGR (%)', 'CAGR (%)', axes[0, 0]),
            ('MDD (%)', 'MDD (%)', axes[0, 1]),
            ('Sharpe Ratio', 'Sharpe Ratio', axes[1, 0]),
            ('Win Rate (%)', 'Win Rate (%)', axes[1, 1])
        ]

        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(entry_periods)))

        for metric_col, ylabel, ax in metrics_to_plot:
            for i, entry_period in enumerate(entry_periods):
                # Filter data for this entry period
                data_subset = df_turtle[df_turtle['Entry Period'] == entry_period].sort_values('Exit Ratio')

                if len(data_subset) == 0:
                    continue

                # Plot line
                ax.plot(
                    data_subset['Exit Ratio'],
                    data_subset[metric_col],
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    label=f'Entry={entry_period}d',
                    color=colors[i]
                )

                # Find and annotate optimal point for this entry period
                if metric_col in ['CAGR (%)', 'Sharpe Ratio', 'Win Rate (%)']:
                    best_idx = data_subset[metric_col].idxmax()
                else:  # For MDD, we want minimum absolute value
                    best_idx = data_subset[metric_col].abs().idxmin()

                best_point = data_subset.loc[best_idx]
                ax.scatter(
                    best_point['Exit Ratio'],
                    best_point[metric_col],
                    s=200,
                    marker='*',
                    color=colors[i],
                    edgecolors='black',
                    linewidth=1.5,
                    zorder=5
                )

            # Formatting
            ax.set_xlabel('Exit Ratio (Exit Period / Entry Period)', fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(f'{ylabel} vs Exit Ratio', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=8, loc='best', ncol=2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()

    def save_results(self, save_path: str = 'turtle_sma_comparison_results.csv'):
        """
        Save all results to CSV

        Args:
            save_path: Path to save the CSV file
        """
        print(f"\nSaving results to {save_path}...")

        results_data = []
        for strategy_name, result in self.results.items():
            metrics = result['metrics']
            row = {
                'Strategy': strategy_name,
                'Entry Period': result['entry_period'] if result['entry_period'] else 'N/A',
                'Exit Period': result['exit_period'] if result['exit_period'] else 'N/A',
                **metrics
            }
            results_data.append(row)

        df_results = pd.DataFrame(results_data)

        # Sort by CAGR descending
        df_results = df_results.sort_values('CAGR (%)', ascending=False)

        df_results.to_csv(save_path, index=False)
        print(f"  Saved {len(df_results)} strategies")

        # Print summary
        print("\nTop 10 Strategies by CAGR:")
        print(df_results[['Strategy', 'CAGR (%)', 'MDD (%)', 'Sharpe Ratio']].head(10).to_string(index=False))

    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("=" * 80)
        print("Turtle Trading vs SMA30 Baseline Comparison")
        print("=" * 80)

        # Load data
        self.load_data()

        # Run baseline
        self.run_baseline_strategy()

        # Run turtle trading variations
        self.run_turtle_trading_variations()

        # Generate visualizations
        self.plot_cagr_vs_mdd()
        self.plot_parameter_curves()

        # Save results
        self.save_results()

        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)


if __name__ == '__main__':
    # Create and run analysis
    analysis = TurtleTradingSMAComparison(data_dir='chart_day')
    analysis.run_analysis()
