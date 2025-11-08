"""
BTC Reversal (Counter-Trend) Strategies Comparison

This module implements and compares 10 different reversal/mean-reversion trading strategies
for Bitcoin (BTC_KRW). Reversal strategies attempt to profit from price extremes by
buying oversold conditions and selling overbought conditions.

Key Features:
- 10 different reversal strategy implementations
- Look-ahead bias prevention with shift(1)
- Realistic execution with slippage
- Comprehensive performance metrics
- Professional visualizations

Author: Claude
Date: 2025-11-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, Tuple
from datetime import datetime

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 24)
plt.rcParams['font.size'] = 10


class BTCReversalStrategiesComparison:
    """
    Backtesting framework for BTC reversal (counter-trend) strategies.

    Reversal strategies bet against the prevailing trend, buying on dips
    and selling on rallies. These strategies profit from mean reversion
    rather than trend continuation.
    """

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01',
                 end_date='2025-12-31', slippage=0.002):
        """
        Initialize the comparison framework.

        Args:
            symbol: Cryptocurrency symbol (default: BTC_KRW)
            start_date: Start date for backtesting
            end_date: End date for backtesting
            slippage: Transaction cost as decimal (0.002 = 0.2%)
        """
        self.symbol = symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.slippage = slippage
        self.data = None
        self.strategy_results = {}

    def load_data(self):
        """Load BTC data from parquet file."""
        data_path = Path(f'chart_day/{self.symbol}.parquet')

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_parquet(data_path)

        # Ensure column names are capitalized
        df.columns = [col.capitalize() for col in df.columns]

        # Filter by date range
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        # Calculate daily price returns
        df['daily_price_return'] = df['Close'].pct_change()

        self.data = df
        print(f"✓ Loaded {len(df)} days of {self.symbol} data ({df.index[0].date()} to {df.index[-1].date()})")

        return df

    # ==================== TECHNICAL INDICATORS ====================

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    @staticmethod
    def calculate_williams_r(high, low, close, period=14):
        """Calculate Williams %R indicator."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r

    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def calculate_cci(high, low, close, period=20):
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci

    @staticmethod
    def calculate_mfi(high, low, close, volume, period=14):
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        # Positive and negative money flow
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
        negative_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()

        mfi_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + mfi_ratio))
        return mfi

    # ==================== STRATEGY IMPLEMENTATIONS ====================

    def strategy_rsi_reversal(self, df, rsi_period=14, oversold=30, overbought=70):
        """
        Strategy 1: RSI Oversold/Overbought Reversal

        Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)
        Classic mean reversion based on momentum extremes.
        """
        df = df.copy()

        # Calculate RSI
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)

        # Generate signals with shift to prevent look-ahead bias
        df['buy_signal'] = (df['RSI'] < oversold).astype(int)
        df['sell_signal'] = (df['RSI'] > overbought).astype(int)

        # Position: 1 when holding, 0 when not
        df['position'] = 0
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            # Use previous day's signals
            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1  # Buy
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0  # Sell

            positions.append(position)

        df['position'] = positions

        # Calculate returns with slippage
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_bollinger_reversal(self, df, period=20, std_dev=2):
        """
        Strategy 2: Bollinger Bands Mean Reversion

        Buy when price touches lower band, sell when price touches upper band.
        Profits from price returning to the mean.
        """
        df = df.copy()

        # Calculate Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'], period, std_dev)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower

        # Generate signals
        df['buy_signal'] = (df['Close'] <= df['BB_Lower']).astype(int)
        df['sell_signal'] = (df['Close'] >= df['BB_Upper']).astype(int)

        # Position management
        df['position'] = 0
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions

        # Calculate returns
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_williams_reversal(self, df, period=14, oversold=-80, overbought=-20):
        """
        Strategy 3: Williams %R Reversal

        Buy when %R < -80 (oversold), sell when %R > -20 (overbought).
        Similar to RSI but uses high/low range.
        """
        df = df.copy()

        # Calculate Williams %R
        df['Williams_R'] = self.calculate_williams_r(df['High'], df['Low'], df['Close'], period)

        # Generate signals
        df['buy_signal'] = (df['Williams_R'] < oversold).astype(int)
        df['sell_signal'] = (df['Williams_R'] > overbought).astype(int)

        # Position management
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions

        # Calculate returns
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_stochastic_reversal(self, df, k_period=14, d_period=3, oversold=20, overbought=80):
        """
        Strategy 4: Stochastic Reversal

        Buy when %K < 20 and crosses above %D, sell when %K > 80 and crosses below %D.
        Confirms reversal with crossover signal.
        """
        df = df.copy()

        # Calculate Stochastic
        k, d = self.calculate_stochastic(df['High'], df['Low'], df['Close'], k_period, d_period)
        df['Stoch_K'] = k
        df['Stoch_D'] = d

        # Generate signals with crossover confirmation
        df['buy_signal'] = ((df['Stoch_K'] < oversold) &
                           (df['Stoch_K'] > df['Stoch_D']) &
                           (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1))).astype(int)

        df['sell_signal'] = ((df['Stoch_K'] > overbought) &
                            (df['Stoch_K'] < df['Stoch_D']) &
                            (df['Stoch_K'].shift(1) >= df['Stoch_D'].shift(1))).astype(int)

        # Position management
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions

        # Calculate returns
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_cci_reversal(self, df, period=20, oversold=-100, overbought=100):
        """
        Strategy 5: CCI (Commodity Channel Index) Reversal

        Buy when CCI < -100, sell when CCI > 100.
        Measures deviation from statistical mean.
        """
        df = df.copy()

        # Calculate CCI
        df['CCI'] = self.calculate_cci(df['High'], df['Low'], df['Close'], period)

        # Generate signals
        df['buy_signal'] = (df['CCI'] < oversold).astype(int)
        df['sell_signal'] = (df['CCI'] > overbought).astype(int)

        # Position management
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions

        # Calculate returns
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_ma_distance_reversal(self, df, ma_period=50, threshold=0.10):
        """
        Strategy 6: Moving Average Distance Reversal

        Buy when price is 10% below MA, sell when 10% above MA.
        Profits from extreme deviations from trend.
        """
        df = df.copy()

        # Calculate MA and distance
        df['MA'] = df['Close'].rolling(window=ma_period).mean()
        df['Distance'] = (df['Close'] - df['MA']) / df['MA']

        # Generate signals
        df['buy_signal'] = (df['Distance'] < -threshold).astype(int)
        df['sell_signal'] = (df['Distance'] > threshold).astype(int)

        # Position management
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions

        # Calculate returns
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_rsi_divergence(self, df, rsi_period=14, lookback=14):
        """
        Strategy 7: RSI Divergence Reversal

        Simplified divergence: Buy when price makes new low but RSI doesn't,
        sell when price makes new high but RSI doesn't.
        """
        df = df.copy()

        # Calculate RSI
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)

        # Rolling min/max for divergence detection
        df['Price_Low'] = df['Close'].rolling(window=lookback).min()
        df['Price_High'] = df['Close'].rolling(window=lookback).max()
        df['RSI_Low'] = df['RSI'].rolling(window=lookback).min()
        df['RSI_High'] = df['RSI'].rolling(window=lookback).max()

        # Bullish divergence: price at low but RSI not at low
        df['bullish_div'] = ((df['Close'] == df['Price_Low']) &
                            (df['RSI'] > df['RSI_Low'] + 5)).astype(int)

        # Bearish divergence: price at high but RSI not at high
        df['bearish_div'] = ((df['Close'] == df['Price_High']) &
                            (df['RSI'] < df['RSI_High'] - 5)).astype(int)

        # Generate signals
        df['buy_signal'] = df['bullish_div']
        df['sell_signal'] = df['bearish_div']

        # Position management
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions

        # Calculate returns
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_mfi_reversal(self, df, mfi_period=14, oversold=20, overbought=80):
        """
        Strategy 8: Money Flow Index (MFI) Reversal

        Buy when MFI < 20, sell when MFI > 80.
        Volume-weighted version of RSI.
        """
        df = df.copy()

        # Calculate MFI
        df['MFI'] = self.calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'], mfi_period)

        # Generate signals
        df['buy_signal'] = (df['MFI'] < oversold).astype(int)
        df['sell_signal'] = (df['MFI'] > overbought).astype(int)

        # Position management
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions

        # Calculate returns
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_zscore_reversal(self, df, period=20, threshold=2):
        """
        Strategy 9: Z-Score Mean Reversion

        Buy when Z-score < -2, sell when Z-score > 2.
        Statistical approach to mean reversion.
        """
        df = df.copy()

        # Calculate Z-score
        df['MA'] = df['Close'].rolling(window=period).mean()
        df['STD'] = df['Close'].rolling(window=period).std()
        df['ZScore'] = (df['Close'] - df['MA']) / df['STD']

        # Generate signals
        df['buy_signal'] = (df['ZScore'] < -threshold).astype(int)
        df['sell_signal'] = (df['ZScore'] > threshold).astype(int)

        # Position management
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions

        # Calculate returns
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_roc_reversal(self, df, period=10, threshold=15):
        """
        Strategy 10: Rate of Change (ROC) Reversal

        Buy when ROC < -15% (sharp decline), sell when ROC > 15% (sharp rally).
        Bets on reversal after extreme moves.
        """
        df = df.copy()

        # Calculate ROC
        df['ROC'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100

        # Generate signals
        df['buy_signal'] = (df['ROC'] < -threshold).astype(int)
        df['sell_signal'] = (df['ROC'] > threshold).astype(int)

        # Position management
        position = 0
        positions = []

        for i in range(len(df)):
            if i == 0:
                positions.append(0)
                continue

            if df['buy_signal'].iloc[i-1] == 1 and position == 0:
                position = 1
            elif df['sell_signal'].iloc[i-1] == 1 and position == 1:
                position = 0

            positions.append(position)

        df['position'] = positions

        # Calculate returns
        df['position_change'] = df['position'].diff()
        df['slippage_cost'] = abs(df['position_change']) * self.slippage
        df['returns'] = df['position'].shift(1) * df['daily_price_return'] - df['slippage_cost']
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    # ==================== EXECUTION & ANALYSIS ====================

    def run_all_strategies(self):
        """Run all 10 reversal strategies on BTC data."""

        strategies = {
            '1. RSI Reversal (30/70)': lambda df: self.strategy_rsi_reversal(df),
            '2. Bollinger Bands Mean Reversion': lambda df: self.strategy_bollinger_reversal(df),
            '3. Williams %R Reversal': lambda df: self.strategy_williams_reversal(df),
            '4. Stochastic Reversal': lambda df: self.strategy_stochastic_reversal(df),
            '5. CCI Reversal': lambda df: self.strategy_cci_reversal(df),
            '6. MA Distance Reversal (10%)': lambda df: self.strategy_ma_distance_reversal(df),
            '7. RSI Divergence': lambda df: self.strategy_rsi_divergence(df),
            '8. MFI Reversal': lambda df: self.strategy_mfi_reversal(df),
            '9. Z-Score Mean Reversion': lambda df: self.strategy_zscore_reversal(df),
            '10. ROC Reversal (15%)': lambda df: self.strategy_roc_reversal(df),
        }

        print(f"\n{'='*80}")
        print(f"Running 10 reversal strategies on {self.symbol}...")
        print(f"{'='*80}\n")

        for strategy_name, strategy_func in strategies.items():
            try:
                result = strategy_func(self.data.copy())
                self.strategy_results[strategy_name] = result

                # Quick stats
                total_return = (result['cumulative'].iloc[-1] - 1) * 100
                num_trades = abs(result['position'].diff()).sum() / 2

                print(f"✓ {strategy_name:40s} | Return: {total_return:>8.2f}% | Trades: {num_trades:>4.0f}")

            except Exception as e:
                print(f"✗ {strategy_name:40s} | Error: {str(e)}")

        print(f"\n{'='*80}\n")

    def calculate_metrics(self, returns_series, name):
        """Calculate comprehensive performance metrics."""

        # Remove NaN values
        returns = returns_series.dropna()

        if len(returns) == 0 or returns.sum() == 0:
            return {
                'Strategy': name,
                'Total Return (%)': 0,
                'CAGR (%)': 0,
                'MDD (%)': 0,
                'Sharpe Ratio': 0,
                'Win Rate (%)': 0,
                'Total Trades': 0,
                'Profit Factor': 0,
            }

        # Cumulative returns
        cumulative = (1 + returns).cumprod()

        # Total return
        total_return = (cumulative.iloc[-1] - 1) * 100

        # CAGR
        years = len(returns) / 252  # Assuming ~252 trading days per year
        cagr = (cumulative.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else 0

        # Maximum Drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Sharpe Ratio (annualized)
        if returns.std() != 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Win Rate
        winning_days = (returns > 0).sum()
        total_trading_days = (returns != 0).sum()
        win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0

        # Profit Factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses != 0 else 0

        # Total trades (estimate from position changes)
        total_trades = total_trading_days / 2  # Rough estimate

        return {
            'Strategy': name,
            'Total Return (%)': round(total_return, 2),
            'CAGR (%)': round(cagr, 2),
            'MDD (%)': round(max_drawdown, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Total Trades': int(total_trades),
            'Profit Factor': round(profit_factor, 2),
        }

    def calculate_all_metrics(self):
        """Calculate metrics for all strategies."""

        metrics_list = []

        for strategy_name, result_df in self.strategy_results.items():
            metrics = self.calculate_metrics(result_df['returns'], strategy_name)
            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df = metrics_df.sort_values('Total Return (%)', ascending=False)

        return metrics_df

    def plot_comparison(self, metrics_df):
        """Create comprehensive comparison visualization."""

        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)

        # Color palette
        colors = sns.color_palette("husl", len(self.strategy_results))

        # 1. Cumulative Returns (Log Scale)
        ax1 = fig.add_subplot(gs[0, :])
        for idx, (strategy_name, result_df) in enumerate(self.strategy_results.items()):
            cumulative = result_df['cumulative']
            ax1.plot(result_df.index, cumulative, label=strategy_name,
                    linewidth=2, alpha=0.8, color=colors[idx])

        ax1.set_yscale('log')
        ax1.set_title(f'BTC Reversal Strategies - Cumulative Returns (Log Scale)',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative Return (Log)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)

        # 2. Total Return Bar Chart
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_metrics = metrics_df.sort_values('Total Return (%)', ascending=True)
        bars = ax2.barh(range(len(sorted_metrics)), sorted_metrics['Total Return (%)'],
                       color=colors[:len(sorted_metrics)])
        ax2.set_yticks(range(len(sorted_metrics)))
        ax2.set_yticklabels(sorted_metrics['Strategy'], fontsize=9)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return by Strategy', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (idx, row) in enumerate(sorted_metrics.iterrows()):
            ax2.text(row['Total Return (%)'], i, f" {row['Total Return (%)']:.1f}%",
                    va='center', fontsize=8)

        # 3. CAGR
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_metrics = metrics_df.sort_values('CAGR (%)', ascending=True)
        bars = ax3.barh(range(len(sorted_metrics)), sorted_metrics['CAGR (%)'],
                       color=colors[:len(sorted_metrics)])
        ax3.set_yticks(range(len(sorted_metrics)))
        ax3.set_yticklabels(sorted_metrics['Strategy'], fontsize=9)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('Compound Annual Growth Rate', fontsize=13, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # 4. Maximum Drawdown
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_metrics = metrics_df.sort_values('MDD (%)', ascending=False)
        bars = ax4.barh(range(len(sorted_metrics)), sorted_metrics['MDD (%)'],
                       color=colors[:len(sorted_metrics)])
        ax4.set_yticks(range(len(sorted_metrics)))
        ax4.set_yticklabels(sorted_metrics['Strategy'], fontsize=9)
        ax4.set_xlabel('Max Drawdown (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown (Risk)', fontsize=13, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)

        # 5. Sharpe Ratio
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_metrics = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        bars = ax5.barh(range(len(sorted_metrics)), sorted_metrics['Sharpe Ratio'],
                       color=colors[:len(sorted_metrics)])
        ax5.set_yticks(range(len(sorted_metrics)))
        ax5.set_yticklabels(sorted_metrics['Strategy'], fontsize=9)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Risk-Adjusted Returns (Sharpe)', fontsize=13, fontweight='bold')
        ax5.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        ax5.axvline(x=2.0, color='green', linestyle='--', alpha=0.5, label='Sharpe = 2.0')
        ax5.legend(fontsize=8)
        ax5.grid(axis='x', alpha=0.3)

        # 6. Win Rate
        ax6 = fig.add_subplot(gs[2, 1])
        sorted_metrics = metrics_df.sort_values('Win Rate (%)', ascending=True)
        bars = ax6.barh(range(len(sorted_metrics)), sorted_metrics['Win Rate (%)'],
                       color=colors[:len(sorted_metrics)])
        ax6.set_yticks(range(len(sorted_metrics)))
        ax6.set_yticklabels(sorted_metrics['Strategy'], fontsize=9)
        ax6.set_xlabel('Win Rate (%)', fontsize=11)
        ax6.set_title('Win Rate', fontsize=13, fontweight='bold')
        ax6.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50%')
        ax6.legend(fontsize=8)
        ax6.grid(axis='x', alpha=0.3)

        # 7. Profit Factor
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_metrics = metrics_df.sort_values('Profit Factor', ascending=True)
        bars = ax7.barh(range(len(sorted_metrics)), sorted_metrics['Profit Factor'],
                       color=colors[:len(sorted_metrics)])
        ax7.set_yticks(range(len(sorted_metrics)))
        ax7.set_yticklabels(sorted_metrics['Strategy'], fontsize=9)
        ax7.set_xlabel('Profit Factor', fontsize=11)
        ax7.set_title('Profit Factor (Gain/Loss Ratio)', fontsize=13, fontweight='bold')
        ax7.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax7.legend(fontsize=8)
        ax7.grid(axis='x', alpha=0.3)

        # 8. Return vs Risk Scatter
        ax8 = fig.add_subplot(gs[3, 0])
        scatter = ax8.scatter(abs(metrics_df['MDD (%)']), metrics_df['CAGR (%)'],
                            s=200, c=metrics_df['Sharpe Ratio'], cmap='RdYlGn',
                            alpha=0.7, edgecolors='black', linewidths=1.5)

        # Add strategy labels
        for idx, row in metrics_df.iterrows():
            strategy_short = row['Strategy'].split('.')[0] + '.'
            ax8.annotate(strategy_short,
                        (abs(row['MDD (%)']), row['CAGR (%)']),
                        fontsize=8, ha='center', va='bottom')

        ax8.set_xlabel('Maximum Drawdown (%) - Lower is Better', fontsize=11)
        ax8.set_ylabel('CAGR (%) - Higher is Better', fontsize=11)
        ax8.set_title('Return vs Risk Profile', fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax8)
        cbar.set_label('Sharpe Ratio', fontsize=10)

        # 9. Total Trades
        ax9 = fig.add_subplot(gs[3, 1])
        sorted_metrics = metrics_df.sort_values('Total Trades', ascending=True)
        bars = ax9.barh(range(len(sorted_metrics)), sorted_metrics['Total Trades'],
                       color=colors[:len(sorted_metrics)])
        ax9.set_yticks(range(len(sorted_metrics)))
        ax9.set_yticklabels(sorted_metrics['Strategy'], fontsize=9)
        ax9.set_xlabel('Total Trades', fontsize=11)
        ax9.set_title('Trading Frequency', fontsize=13, fontweight='bold')
        ax9.grid(axis='x', alpha=0.3)

        # 10. Metrics Table
        ax10 = fig.add_subplot(gs[3, 2])
        ax10.axis('off')

        # Top 3 strategies
        top3 = metrics_df.head(3)
        table_data = []
        table_data.append(['Rank', 'Strategy', 'Return', 'Sharpe'])

        medals = ['🥇', '🥈', '🥉']
        for i, (idx, row) in enumerate(top3.iterrows()):
            strategy_short = row['Strategy'][:25] + '...' if len(row['Strategy']) > 25 else row['Strategy']
            table_data.append([
                medals[i],
                strategy_short,
                f"{row['Total Return (%)']:.1f}%",
                f"{row['Sharpe Ratio']:.2f}"
            ])

        table = ax10.table(cellText=table_data, cellLoc='left', loc='center',
                          colWidths=[0.1, 0.5, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax10.set_title('Top 3 Strategies', fontsize=13, fontweight='bold', pad=20)

        # 11. Drawdown Over Time
        ax11 = fig.add_subplot(gs[4, :])

        for idx, (strategy_name, result_df) in enumerate(self.strategy_results.items()):
            cumulative = result_df['cumulative']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100

            ax11.plot(result_df.index, drawdown, label=strategy_name,
                     linewidth=1.5, alpha=0.7, color=colors[idx])

        ax11.fill_between(ax11.get_xlim(), 0, -100, alpha=0.1, color='red')
        ax11.set_xlabel('Date', fontsize=12)
        ax11.set_ylabel('Drawdown (%)', fontsize=12)
        ax11.set_title('Drawdown Over Time - All Strategies', fontsize=14, fontweight='bold')
        ax11.legend(loc='lower left', fontsize=9, ncol=2)
        ax11.grid(True, alpha=0.3)
        ax11.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add subtitle with date range
        date_range = f"{self.data.index[0].date()} to {self.data.index[-1].date()}"
        fig.suptitle(f'BTC Reversal Strategies Comparison | {date_range} | Slippage: {self.slippage*100}%',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig('btc_reversal_strategies_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved: btc_reversal_strategies_comparison.png")

        return fig

    def export_results(self, metrics_df):
        """Export detailed results to CSV."""

        # Save metrics
        metrics_df.to_csv('btc_reversal_metrics.csv', index=False)
        print(f"✓ Metrics saved: btc_reversal_metrics.csv")

        # Save individual strategy returns
        for strategy_name, result_df in self.strategy_results.items():
            safe_name = strategy_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            filename = f'strategy_{safe_name}.csv'
            result_df[['Close', 'position', 'returns', 'cumulative']].to_csv(filename)

        print(f"✓ Individual strategy results saved (10 CSV files)")

    def print_summary(self, metrics_df):
        """Print formatted summary table."""

        print(f"\n{'='*120}")
        print(f"BTC REVERSAL STRATEGIES - PERFORMANCE SUMMARY")
        print(f"{'='*120}\n")

        print(metrics_df.to_string(index=False))

        print(f"\n{'='*120}")
        print(f"🥇 BEST STRATEGY: {metrics_df.iloc[0]['Strategy']}")
        print(f"   Total Return: {metrics_df.iloc[0]['Total Return (%)']}%")
        print(f"   CAGR: {metrics_df.iloc[0]['CAGR (%)']}%")
        print(f"   Sharpe Ratio: {metrics_df.iloc[0]['Sharpe Ratio']}")
        print(f"   Max Drawdown: {metrics_df.iloc[0]['MDD (%)']}%")
        print(f"{'='*120}\n")


def main():
    """Main execution function."""

    # Initialize comparison
    btc_comparison = BTCReversalStrategiesComparison(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date='2025-12-31',
        slippage=0.002  # 0.2% transaction cost
    )

    # Load data
    btc_comparison.load_data()

    # Run all strategies
    btc_comparison.run_all_strategies()

    # Calculate metrics
    metrics_df = btc_comparison.calculate_all_metrics()

    # Print summary
    btc_comparison.print_summary(metrics_df)

    # Create visualization
    btc_comparison.plot_comparison(metrics_df)

    # Export results
    btc_comparison.export_results(metrics_df)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
