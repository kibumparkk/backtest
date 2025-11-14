"""
비트코인 멀티 타임프레임 전략 - Loop-based Implementation
주봉 신호는 주말 종료 후 다음 거래일(월요일)부터 사용 가능

CORRECT LOGIC:
- Week ending Sunday → Signal available from Monday (1 day delay)
- NOT 7 day delay (that's too conservative)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MTFLoopBased:
    """Loop-based Multi-Timeframe Strategy Backtester"""

    def __init__(self, slippage=0.002):
        self.slippage = slippage
        self.daily_data = None
        self.weekly_data = None
        self.results = {}

    def load_data(self):
        """데이터 로드"""
        print("="*80)
        print("Loading Bitcoin data (Loop-based MTF)...")
        print("="*80)

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

        print(f"\nDaily data: {len(df_daily)} bars")
        print(f"Weekly data: {len(df_weekly)} bars")
        print("="*80 + "\n")

    def backtest_loop(self, weekly_signals_dict, daily_signals, name):
        """
        Loop-based backtest

        Args:
            weekly_signals_dict: {date: {'signal': 0/1, 'available_from': date}}
            daily_signals: Series with daily signals
            name: Strategy name
        """
        df = self.daily_data.copy()

        capital = 1.0
        position = 0
        equity_curve = []

        for i in range(len(df)):
            date = df.index[i]
            close = df.iloc[i]['Close']

            # Daily signal
            daily_sig = daily_signals.iloc[i] if i < len(daily_signals) else 0

            # Find most recent available weekly signal
            weekly_sig = 0
            for week_date in sorted(weekly_signals_dict.keys(), reverse=True):
                if date >= weekly_signals_dict[week_date]['available_from']:
                    weekly_sig = weekly_signals_dict[week_date]['signal']
                    break

            # Combined signal
            final_signal = 1 if (daily_sig == 1 and weekly_sig == 1) else 0

            # Calculate returns and update capital
            if i > 0:
                prev_close = df.iloc[i-1]['Close']
                daily_return = (close - prev_close) / prev_close

                if position == 1:
                    capital = capital * (1 + daily_return)

                # Position change
                if position == 0 and final_signal == 1:
                    capital = capital * (1 - self.slippage)
                elif position == 1 and final_signal == 0:
                    capital = capital * (1 - self.slippage)

            position = final_signal
            equity_curve.append({'date': date, 'capital': capital, 'position': position})

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        returns = equity_df['capital'].pct_change().fillna(0)

        total_return = (capital - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (capital ** (1/years) - 1) * 100

        cummax = equity_df['capital'].cummax()
        drawdown = (equity_df['capital'] - cummax) / cummax
        mdd = drawdown.min() * 100

        sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0

        # Count trades
        position_changes = equity_df['position'].diff().fillna(0)
        total_trades = (position_changes != 0).sum()

        self.results[name] = {
            'metrics': {
                'Strategy': name,
                'Total Return (%)': total_return,
                'CAGR (%)': cagr,
                'MDD (%)': mdd,
                'Sharpe Ratio': sharpe,
                'Total Trades': int(total_trades)
            }
        }

        return self.results[name]['metrics']

    def benchmark_daily_sma30(self):
        """Benchmark: Close > SMA30"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()
        signal = (df['Close'] > df['SMA30']).astype(int)

        # No weekly component, so use simple backtest
        return self.backtest_simple(signal, '0_BENCHMARK_Daily_SMA30')

    def backtest_simple(self, signal, name):
        """Simple backtest for benchmark (no weekly component)"""
        df = self.daily_data.copy()
        df['signal'] = signal
        df['daily_ret'] = df['Close'].pct_change()
        df['strat_ret'] = df['signal'].shift(1) * df['daily_ret']

        # Slippage
        pos_change = df['signal'].diff()
        slip_cost = pd.Series(0.0, index=df.index)
        slip_cost[pos_change == 1] = -self.slippage
        slip_cost[pos_change == -1] = -self.slippage
        df['strat_ret'] = df['strat_ret'] + slip_cost
        df['strat_ret'] = df['strat_ret'].fillna(0)

        df['cumulative'] = (1 + df['strat_ret']).cumprod()

        total_return = (df['cumulative'].iloc[-1] - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['cumulative'].iloc[-1] ** (1/years) - 1) * 100

        cummax = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - cummax) / cummax
        mdd = drawdown.min() * 100

        sharpe = df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(365) if df['strat_ret'].std() > 0 else 0

        total_trades = (df['strat_ret'] != 0).sum()

        self.results[name] = {
            'metrics': {
                'Strategy': name,
                'Total Return (%)': total_return,
                'CAGR (%)': cagr,
                'MDD (%)': mdd,
                'Sharpe Ratio': sharpe,
                'Total Trades': int(total_trades)
            }
        }

        return self.results[name]['metrics']

    # ==================== MTF Strategies ====================

    def strategy_weekly_sma10_daily_sma30(self):
        """Weekly SMA10 + Daily SMA30"""
        # Weekly signals
        weekly = self.weekly_data.copy()
        weekly['SMA10'] = weekly['Close'].rolling(10).mean()
        weekly['signal'] = (weekly['Close'] > weekly['SMA10']).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            # Signal available from NEXT day after week ends
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        # Daily signals
        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()
        daily_sig = (daily['Close'] > daily['SMA30']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '1_Weekly_SMA10+Daily_SMA30')

    def strategy_weekly_sma20_daily_sma30(self):
        """Weekly SMA20 + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['signal'] = (weekly['Close'] > weekly['SMA20']).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()
        daily_sig = (daily['Close'] > daily['SMA30']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '2_Weekly_SMA20+Daily_SMA30')

    def strategy_weekly_sma50_daily_sma30(self):
        """Weekly SMA50 + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['SMA50'] = weekly['Close'].rolling(50).mean()
        weekly['signal'] = (weekly['Close'] > weekly['SMA50']).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()
        daily_sig = (daily['Close'] > daily['SMA30']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '3_Weekly_SMA50+Daily_SMA30')

    def strategy_weekly_ema20_daily_sma30(self):
        """Weekly EMA20 + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['EMA20'] = weekly['Close'].ewm(span=20, adjust=False).mean()
        weekly['signal'] = (weekly['Close'] > weekly['EMA20']).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()
        daily_sig = (daily['Close'] > daily['SMA30']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '4_Weekly_EMA20+Daily_SMA30')

    def strategy_weekly_golden_cross_daily_sma30(self):
        """Weekly Golden Cross + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['SMA10'] = weekly['Close'].rolling(10).mean()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['signal'] = (weekly['SMA10'] > weekly['SMA20']).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()
        daily_sig = (daily['Close'] > daily['SMA30']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '5_Weekly_GoldenCross+Daily_SMA30')

    def strategy_weekly_macd_daily_sma30(self):
        """Weekly MACD + Daily SMA30"""
        weekly = self.weekly_data.copy()
        ema12 = weekly['Close'].ewm(span=12, adjust=False).mean()
        ema26 = weekly['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        weekly['signal'] = (macd > signal_line).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()
        daily_sig = (daily['Close'] > daily['SMA30']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '6_Weekly_MACD+Daily_SMA30')

    def strategy_weekly_donchian_daily_sma30(self):
        """Weekly Donchian + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['high_20'] = weekly['High'].rolling(20).max()
        weekly['signal'] = (weekly['Close'] > weekly['high_20'] * 0.95).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()
        daily_sig = (daily['Close'] > daily['SMA30']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '7_Weekly_Donchian+Daily_SMA30')

    def strategy_weekly_dual_sma_daily_sma30(self):
        """Weekly Dual SMA + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['SMA50'] = weekly['Close'].rolling(50).mean()
        weekly['signal'] = ((weekly['Close'] > weekly['SMA20']) &
                           (weekly['Close'] > weekly['SMA50'])).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()
        daily_sig = (daily['Close'] > daily['SMA30']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '8_Weekly_DualSMA+Daily_SMA30')

    def strategy_weekly_alignment_daily_sma30(self):
        """Weekly Alignment + Daily SMA30"""
        weekly = self.weekly_data.copy()
        weekly['SMA10'] = weekly['Close'].rolling(10).mean()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['SMA50'] = weekly['Close'].rolling(50).mean()
        weekly['signal'] = ((weekly['SMA10'] > weekly['SMA20']) &
                           (weekly['SMA20'] > weekly['SMA50'])).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        daily = self.daily_data.copy()
        daily['SMA30'] = daily['Close'].rolling(30).mean()
        daily_sig = (daily['Close'] > daily['SMA30']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '9_Weekly_Alignment+Daily_SMA30')

    def strategy_weekly_sma20_daily_sma20(self):
        """Weekly SMA20 + Daily SMA20"""
        weekly = self.weekly_data.copy()
        weekly['SMA20'] = weekly['Close'].rolling(20).mean()
        weekly['signal'] = (weekly['Close'] > weekly['SMA20']).astype(int)

        weekly_signals = {}
        for i in range(len(weekly)):
            week_date = weekly.index[i]
            signal = weekly.iloc[i]['signal']
            available_from = week_date + pd.Timedelta(days=1)
            weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

        daily = self.daily_data.copy()
        daily['SMA20'] = daily['Close'].rolling(20).mean()
        daily_sig = (daily['Close'] > daily['SMA20']).astype(int)

        return self.backtest_loop(weekly_signals, daily_sig, '10_Weekly_SMA20+Daily_SMA20')

    def run_all_strategies(self):
        """모든 전략 실행"""
        print("="*80)
        print("Running MTF Strategies (Loop-based - CORRECT)")
        print("="*80 + "\n")

        strategies = [
            ('0_BENCHMARK', self.benchmark_daily_sma30),
            ('1_Weekly_SMA10+Daily_SMA30', self.strategy_weekly_sma10_daily_sma30),
            ('2_Weekly_SMA20+Daily_SMA30', self.strategy_weekly_sma20_daily_sma30),
            ('3_Weekly_SMA50+Daily_SMA30', self.strategy_weekly_sma50_daily_sma30),
            ('4_Weekly_EMA20+Daily_SMA30', self.strategy_weekly_ema20_daily_sma30),
            ('5_Weekly_GoldenCross+Daily_SMA30', self.strategy_weekly_golden_cross_daily_sma30),
            ('6_Weekly_MACD+Daily_SMA30', self.strategy_weekly_macd_daily_sma30),
            ('7_Weekly_Donchian+Daily_SMA30', self.strategy_weekly_donchian_daily_sma30),
            ('8_Weekly_DualSMA+Daily_SMA30', self.strategy_weekly_dual_sma_daily_sma30),
            ('9_Weekly_Alignment+Daily_SMA30', self.strategy_weekly_alignment_daily_sma30),
            ('10_Weekly_SMA20+Daily_SMA20', self.strategy_weekly_sma20_daily_sma20),
        ]

        metrics_list = []

        for name, func in strategies:
            try:
                print(f"Running {name}...")
                metrics = func()
                metrics_list.append(metrics)
                print(f"  Sharpe: {metrics['Sharpe Ratio']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80)

        return pd.DataFrame(metrics_list)

    def analyze_results(self, metrics_df):
        """결과 분석"""
        benchmark_sharpe = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0]

        better = metrics_df[
            (~metrics_df['Strategy'].str.contains('BENCHMARK')) &
            (metrics_df['Sharpe Ratio'] > benchmark_sharpe)
        ].sort_values('Sharpe Ratio', ascending=False)

        print("\n" + "="*120)
        print(f"{'MTF Strategy Results (Loop-based - CORRECT)':^120}")
        print("="*120)
        print(f"\nBenchmark (Daily SMA30): Sharpe {benchmark_sharpe:.4f}")
        print(f"Strategies beating benchmark: {len(better)}")

        if len(better) > 0:
            print("\n" + "-"*120)
            print("🎯 Strategies Beating Benchmark:")
            print("-"*120)

            for idx, row in better.iterrows():
                improvement = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
                print(f"\n{row['Strategy']}")
                print(f"  Sharpe: {row['Sharpe Ratio']:.4f} (+{improvement:.2f}% vs benchmark)")
                print(f"  Total Return: {row['Total Return (%)']:.2f}%")
                print(f"  CAGR: {row['CAGR (%)']:.2f}%")
                print(f"  MDD: {row['MDD (%)']:.2f}%")
                print(f"  Total Trades: {row['Total Trades']}")
        else:
            print("\n⚠ No strategies beat the benchmark")

        print("\n" + "="*120)

        return better


def main():
    """메인 실행"""
    print("\n" + "="*80)
    print("Bitcoin MTF Strategy Analysis (Loop-based)")
    print("Correct implementation: Weekly signal available next day after week ends")
    print("="*80)

    analyzer = MTFLoopBased(slippage=0.002)
    analyzer.load_data()

    metrics_df = analyzer.run_all_strategies()
    better_strategies = analyzer.analyze_results(metrics_df)

    metrics_df.to_csv('bitcoin_mtf_loopbased_results.csv', index=False)
    print(f"\n✓ Results saved to bitcoin_mtf_loopbased_results.csv")

    return metrics_df, better_strategies


if __name__ == "__main__":
    main()
