"""
Adaptive SMA Strategy with Quarterly Rebalancing

This strategy dynamically selects the best performing SMA window (10-60)
based on the last 1 year performance and rebalances quarterly.

Strategy Logic:
- At the start of each quarter, evaluate all SMA windows (10-60) using past 252 trading days
- Select the window with the best Sharpe Ratio
- Use that window for the entire quarter
- Repeat for the next quarter

Author: Claude
Date: 2025-11-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdaptiveSMAQuarterlyBacktest:
    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date='2025-11-07',
                 slippage=0.002, sma_range=(10, 60),
                 lookback_days=252, rebalance_days=63):
        """
        Initialize the adaptive SMA backtest

        Parameters:
        -----------
        symbols : list
            List of cryptocurrency symbols to backtest
        start_date : str
            Start date for backtest
        end_date : str
            End date for backtest
        slippage : float
            Slippage cost per trade (default 0.2%)
        sma_range : tuple
            (min_window, max_window) for SMA testing
        lookback_days : int
            Number of trading days to look back for evaluation (default 252 = ~1 year)
        rebalance_days : int
            Number of trading days between rebalancing (default 63 = ~1 quarter)
        """
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.slippage = slippage
        self.sma_range = sma_range
        self.lookback_days = lookback_days
        self.rebalance_days = rebalance_days

        self.data = {}
        self.results = {}
        self.window_selections = {}

    def load_data(self):
        """Load data from parquet files"""
        print("Loading data...")
        for symbol in self.symbols:
            file_path = f'chart_day/{symbol}.parquet'
            try:
                df = pd.read_parquet(file_path)
                df.columns = [col.capitalize() for col in df.columns]
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                self.data[symbol] = df.copy()
                print(f"  {symbol}: {len(df)} days loaded")
            except Exception as e:
                print(f"  Error loading {symbol}: {e}")
        print(f"Total symbols loaded: {len(self.data)}\n")

    def calculate_sma_strategy_returns(self, df, sma_window):
        """
        Calculate returns for a single SMA strategy

        Parameters:
        -----------
        df : pd.DataFrame
            Price data with 'Close' column
        sma_window : int
            SMA window period

        Returns:
        --------
        pd.Series
            Daily returns including slippage
        """
        df_temp = df.copy()
        df_temp['SMA'] = df_temp['Close'].rolling(window=sma_window).mean()
        df_temp['position'] = np.where(df_temp['Close'] >= df_temp['SMA'], 1, 0)
        df_temp['position_change'] = df_temp['position'].diff()
        df_temp['daily_price_return'] = df_temp['Close'].pct_change()
        df_temp['returns'] = df_temp['position'].shift(1) * df_temp['daily_price_return']

        # Apply slippage on position changes
        slippage_cost = pd.Series(0.0, index=df_temp.index)
        slippage_cost[df_temp['position_change'] == 1] = -self.slippage  # BUY
        slippage_cost[df_temp['position_change'] == -1] = -self.slippage  # SELL
        df_temp['returns'] = df_temp['returns'] + slippage_cost

        return df_temp['returns']

    def calculate_sharpe_ratio(self, returns):
        """Calculate annualized Sharpe Ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return -999  # Return very low value for invalid cases
        mean_return = returns.mean() * 252  # Annualize
        std_return = returns.std() * np.sqrt(252)  # Annualize
        return mean_return / std_return if std_return > 0 else -999

    def select_best_window(self, df, evaluation_start, evaluation_end):
        """
        Evaluate all SMA windows on a specific period and select the best

        Parameters:
        -----------
        df : pd.DataFrame
            Full price data
        evaluation_start : pd.Timestamp
            Start date for evaluation period
        evaluation_end : pd.Timestamp
            End date for evaluation period

        Returns:
        --------
        int
            Best SMA window based on Sharpe Ratio
        """
        eval_df = df[(df.index >= evaluation_start) & (df.index <= evaluation_end)]

        if len(eval_df) < 30:  # Not enough data
            return 30  # Default to SMA 30

        best_window = 30
        best_sharpe = -999

        for window in range(self.sma_range[0], self.sma_range[1] + 1):
            returns = self.calculate_sma_strategy_returns(eval_df, window)
            sharpe = self.calculate_sharpe_ratio(returns.dropna())

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_window = window

        return best_window

    def run_adaptive_strategy(self, symbol):
        """
        Run adaptive SMA strategy with quarterly rebalancing for a single symbol

        Parameters:
        -----------
        symbol : str
            Symbol to backtest

        Returns:
        --------
        pd.DataFrame
            Results with columns: returns, cumulative_returns, selected_window
        """
        print(f"\nRunning adaptive strategy for {symbol}...")
        df = self.data[symbol].copy()

        # Initialize results
        results_df = pd.DataFrame(index=df.index)
        results_df['returns'] = 0.0
        results_df['selected_window'] = 0
        results_df['position'] = 0

        # Get all trading days
        all_dates = df.index.tolist()

        # Need at least lookback_days before we can start
        if len(all_dates) < self.lookback_days:
            print(f"  Not enough data for {symbol}")
            return results_df

        # Start from lookback_days
        current_idx = self.lookback_days
        window_changes = []

        while current_idx < len(all_dates):
            # Define evaluation period (past lookback_days)
            eval_end_date = all_dates[current_idx - 1]
            eval_start_idx = max(0, current_idx - self.lookback_days)
            eval_start_date = all_dates[eval_start_idx]

            # Select best window based on past performance
            best_window = self.select_best_window(df, eval_start_date, eval_end_date)

            # Define trading period (next rebalance_days or until end)
            trade_start_idx = current_idx
            trade_end_idx = min(current_idx + self.rebalance_days, len(all_dates))
            trade_start_date = all_dates[trade_start_idx]
            trade_end_date = all_dates[trade_end_idx - 1]

            window_changes.append({
                'date': trade_start_date,
                'window': best_window,
                'eval_period': f"{eval_start_date.date()} to {eval_end_date.date()}"
            })

            # Calculate returns for this period using selected window
            trade_df = df[(df.index >= trade_start_date) & (df.index <= trade_end_date)]

            # Calculate SMA and position
            # Need to use full history up to this point for SMA calculation
            historical_df = df[df.index <= trade_end_date].copy()
            historical_df['SMA'] = historical_df['Close'].rolling(window=best_window).mean()
            historical_df['position'] = np.where(historical_df['Close'] >= historical_df['SMA'], 1, 0)
            historical_df['position_change'] = historical_df['position'].diff()
            historical_df['daily_price_return'] = historical_df['Close'].pct_change()
            historical_df['returns'] = historical_df['position'].shift(1) * historical_df['daily_price_return']

            # Apply slippage
            slippage_cost = pd.Series(0.0, index=historical_df.index)
            slippage_cost[historical_df['position_change'] == 1] = -self.slippage
            slippage_cost[historical_df['position_change'] == -1] = -self.slippage
            historical_df['returns'] = historical_df['returns'] + slippage_cost

            # Extract only the trading period results
            period_results = historical_df.loc[trade_start_date:trade_end_date]

            # Store results
            for date in period_results.index:
                results_df.loc[date, 'returns'] = period_results.loc[date, 'returns']
                results_df.loc[date, 'selected_window'] = best_window
                results_df.loc[date, 'position'] = period_results.loc[date, 'position']

            print(f"  Quarter {len(window_changes)}: {trade_start_date.date()} | Window={best_window} | "
                  f"Eval: {eval_start_date.date()} to {eval_end_date.date()}")

            # Move to next rebalancing period
            current_idx = trade_end_idx

        # Calculate cumulative returns
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod()

        # Store window selection history
        self.window_selections[symbol] = pd.DataFrame(window_changes)

        print(f"  Total quarters: {len(window_changes)}")
        print(f"  Window usage: {results_df[results_df['selected_window'] > 0]['selected_window'].value_counts().sort_index().to_dict()}")

        return results_df

    def run_all_strategies(self):
        """Run adaptive strategy for all symbols"""
        print("\n" + "="*80)
        print("ADAPTIVE SMA QUARTERLY BACKTEST")
        print("="*80)
        print(f"SMA Range: {self.sma_range[0]} to {self.sma_range[1]}")
        print(f"Lookback Period: {self.lookback_days} days (~{self.lookback_days/252:.1f} years)")
        print(f"Rebalancing: Every {self.rebalance_days} days (~{self.rebalance_days/63:.1f} quarters)")
        print(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        print("="*80)

        for symbol in self.symbols:
            self.results[symbol] = self.run_adaptive_strategy(symbol)

    def create_portfolio(self):
        """Create equal-weighted portfolio across all symbols"""
        print("\n" + "="*80)
        print("CREATING PORTFOLIO")
        print("="*80)

        # Combine all returns
        all_dates = sorted(set([date for symbol in self.results for date in self.results[symbol].index]))
        portfolio_df = pd.DataFrame(index=all_dates)

        # Add returns for each symbol
        for symbol in self.symbols:
            if symbol in self.results:
                portfolio_df[f'{symbol}_returns'] = self.results[symbol]['returns']

        # Equal weight portfolio
        portfolio_df['portfolio_returns'] = portfolio_df[[f'{symbol}_returns' for symbol in self.symbols]].mean(axis=1)
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['portfolio_returns']).cumprod()

        self.portfolio_results = portfolio_df
        print(f"Portfolio created with {len(self.symbols)} symbols")
        print(f"Total trading days: {len(portfolio_df)}")

        return portfolio_df

    def calculate_metrics(self, returns_series, name="Strategy"):
        """Calculate performance metrics"""
        returns = returns_series.dropna()

        if len(returns) == 0:
            return {}

        cum_returns = (1 + returns).cumprod()

        # Total Return
        total_return = (cum_returns.iloc[-1] - 1) * 100

        # CAGR
        days = len(returns)
        years = days / 252
        cagr = ((cum_returns.iloc[-1]) ** (1/years) - 1) * 100 if years > 0 else 0

        # Maximum Drawdown
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100

        # Sharpe Ratio
        sharpe = self.calculate_sharpe_ratio(returns)

        # Win Rate
        winning_days = (returns > 0).sum()
        total_days = len(returns[returns != 0])
        win_rate = (winning_days / total_days * 100) if total_days > 0 else 0

        # Trade count (position changes)
        trades = 0

        metrics = {
            'Name': name,
            'Total Return (%)': f"{total_return:.2f}",
            'CAGR (%)': f"{cagr:.2f}",
            'Max Drawdown (%)': f"{max_drawdown:.2f}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Win Rate (%)': f"{win_rate:.2f}",
            'Trading Days': len(returns)
        }

        return metrics

    def calculate_all_metrics(self):
        """Calculate metrics for all symbols and portfolio"""
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)

        all_metrics = []

        # Individual symbol metrics
        for symbol in self.symbols:
            if symbol in self.results:
                metrics = self.calculate_metrics(self.results[symbol]['returns'], symbol)
                all_metrics.append(metrics)

        # Portfolio metrics
        portfolio_metrics = self.calculate_metrics(
            self.portfolio_results['portfolio_returns'],
            'PORTFOLIO (Equal Weight)'
        )
        all_metrics.append(portfolio_metrics)

        self.metrics_df = pd.DataFrame(all_metrics)
        print(self.metrics_df.to_string(index=False))

        # Save to CSV
        self.metrics_df.to_csv('adaptive_sma_quarterly_metrics.csv', index=False)
        print(f"\nMetrics saved to: adaptive_sma_quarterly_metrics.csv")

        return self.metrics_df

    def plot_results(self):
        """Create comprehensive visualization"""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Cumulative Returns (log scale)
        ax1 = fig.add_subplot(gs[0, :])
        for symbol in self.symbols:
            if symbol in self.results:
                cum_returns = self.results[symbol]['cumulative_returns']
                ax1.plot(cum_returns.index, cum_returns, label=symbol, linewidth=2)
        ax1.plot(self.portfolio_results.index, self.portfolio_results['cumulative_returns'],
                 label='Portfolio', linewidth=3, color='black', linestyle='--')
        ax1.set_yscale('log')
        ax1.set_title('Adaptive SMA Strategy - Cumulative Returns (Log Scale)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Selected SMA Windows Over Time (for each symbol)
        ax2 = fig.add_subplot(gs[1, :])
        for i, symbol in enumerate(self.symbols):
            if symbol in self.results and symbol in self.window_selections:
                selections = self.window_selections[symbol]
                ax2.scatter(selections['date'], selections['window'],
                           label=symbol, s=50, alpha=0.7)
        ax2.set_title('Selected SMA Windows by Quarter', fontsize=14, fontweight='bold')
        ax2.set_ylabel('SMA Window')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(self.sma_range[0]-5, self.sma_range[1]+5)

        # 3. Window Selection Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        all_windows = []
        for symbol in self.symbols:
            if symbol in self.window_selections:
                all_windows.extend(self.window_selections[symbol]['window'].tolist())
        if all_windows:
            ax3.hist(all_windows, bins=range(self.sma_range[0], self.sma_range[1]+2),
                    edgecolor='black', alpha=0.7)
        ax3.set_title('Window Selection Frequency', fontsize=12, fontweight='bold')
        ax3.set_xlabel('SMA Window')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)

        # 4. Total Return Comparison
        ax4 = fig.add_subplot(gs[2, 1])
        returns_data = []
        labels = []
        for symbol in self.symbols:
            if symbol in self.results:
                total_ret = (self.results[symbol]['cumulative_returns'].iloc[-1] - 1) * 100
                returns_data.append(total_ret)
                labels.append(symbol.replace('_KRW', ''))
        portfolio_ret = (self.portfolio_results['cumulative_returns'].iloc[-1] - 1) * 100
        returns_data.append(portfolio_ret)
        labels.append('Portfolio')

        colors = ['skyblue'] * len(self.symbols) + ['darkblue']
        ax4.bar(labels, returns_data, color=colors, edgecolor='black')
        ax4.set_title('Total Return (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        # 5. Sharpe Ratio Comparison
        ax5 = fig.add_subplot(gs[2, 2])
        sharpe_data = []
        labels_sharpe = []
        for symbol in self.symbols:
            if symbol in self.results:
                sharpe = self.calculate_sharpe_ratio(self.results[symbol]['returns'].dropna())
                sharpe_data.append(sharpe)
                labels_sharpe.append(symbol.replace('_KRW', ''))
        portfolio_sharpe = self.calculate_sharpe_ratio(self.portfolio_results['portfolio_returns'].dropna())
        sharpe_data.append(portfolio_sharpe)
        labels_sharpe.append('Portfolio')

        colors = ['lightgreen'] * len(self.symbols) + ['darkgreen']
        ax5.bar(labels_sharpe, sharpe_data, color=colors, edgecolor='black')
        ax5.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

        # 6. Drawdown
        ax6 = fig.add_subplot(gs[3, :])
        cum_returns = self.portfolio_results['cumulative_returns']
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max * 100
        ax6.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax6.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax6.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Drawdown (%)')
        ax6.set_xlabel('Date')
        ax6.grid(True, alpha=0.3)

        plt.savefig('adaptive_sma_quarterly_results.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to: adaptive_sma_quarterly_results.png")
        plt.close()

    def save_detailed_results(self):
        """Save detailed results to CSV"""
        print("\n" + "="*80)
        print("SAVING DETAILED RESULTS")
        print("="*80)

        # Save portfolio results
        self.portfolio_results.to_csv('adaptive_sma_quarterly_portfolio.csv')
        print("Portfolio results saved to: adaptive_sma_quarterly_portfolio.csv")

        # Save individual symbol results
        for symbol in self.symbols:
            if symbol in self.results:
                filename = f'adaptive_sma_quarterly_{symbol}.csv'
                self.results[symbol].to_csv(filename)
                print(f"{symbol} results saved to: {filename}")

        # Save window selection history
        for symbol in self.symbols:
            if symbol in self.window_selections:
                filename = f'adaptive_sma_windows_{symbol}.csv'
                self.window_selections[symbol].to_csv(filename, index=False)
                print(f"{symbol} window selections saved to: {filename}")

    def run_analysis(self):
        """Run complete analysis pipeline"""
        self.load_data()
        self.run_all_strategies()
        self.create_portfolio()
        self.calculate_all_metrics()
        self.plot_results()
        self.save_detailed_results()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)


def main():
    """Main execution function"""
    backtest = AdaptiveSMAQuarterlyBacktest(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date='2025-11-07',
        slippage=0.002,
        sma_range=(10, 60),
        lookback_days=252,  # 1 year
        rebalance_days=63   # 1 quarter
    )

    backtest.run_analysis()


if __name__ == '__main__':
    main()
