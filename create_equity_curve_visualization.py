"""
MTF 전략 Equity Curve 및 Drawdown 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bitcoin_mtf_loop_based import MTFLoopBased
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def calculate_equity_curve(analyzer, strategy_func, name):
    """전략의 equity curve 계산"""
    if 'BENCHMARK' in name:
        # 벤치마크는 간단한 백테스트
        df = analyzer.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()
        df['signal'] = (df['Close'] > df['SMA30']).astype(int)
        df['pos_change'] = df['signal'].diff()
        df['daily_ret'] = df['Close'].pct_change()
        df['strat_ret'] = df['signal'].shift(1) * df['daily_ret']

        # 슬리피지
        slip_cost = pd.Series(0.0, index=df.index)
        slip_cost[df['pos_change'] == 1] = -analyzer.slippage
        slip_cost[df['pos_change'] == -1] = -analyzer.slippage
        df['strat_ret'] = df['strat_ret'] + slip_cost
        df['strat_ret'] = df['strat_ret'].fillna(0)

        df['cumulative'] = (1 + df['strat_ret']).cumprod()

        return df[['cumulative']].rename(columns={'cumulative': 'equity'})

    else:
        # MTF 전략은 loop-based로 계산
        # 전략 실행
        strategy_func()

        # 결과에서 equity curve 추출
        result = analyzer.results[name]

        # Daily data에서 equity curve 재구성
        df = analyzer.daily_data.copy()

        # 간단히 하기 위해 전략을 다시 실행하여 일별 equity 계산
        # (실제로는 backtest_loop에서 equity_curve를 반환하도록 수정하는 것이 좋지만,
        #  여기서는 간단히 재계산)

        # 전략별로 다시 계산해야 하므로, 여기서는 strategy_func를 호출하여
        # 내부적으로 equity curve를 반환하도록 수정 필요

        # 임시로 간단한 방법 사용: 결과 metrics에서 재구성
        # 이건 정확하지 않으므로, backtest_loop을 수정해야 함

        return None  # 나중에 수정


def create_equity_drawdown_visualization():
    """Equity curve 및 Drawdown 시각화 생성"""

    print("Loading data and calculating equity curves...")

    analyzer = MTFLoopBased(slippage=0.002)
    analyzer.load_data()

    # 벤치마크 및 Top 5 전략 실행
    strategies = {
        'Benchmark': analyzer.benchmark_daily_sma30,
        'Weekly Donchian + Daily SMA30': analyzer.strategy_weekly_donchian_daily_sma30,
        'Weekly EMA20 + Daily SMA30': analyzer.strategy_weekly_ema20_daily_sma30,
        'Weekly SMA10 + Daily SMA30': analyzer.strategy_weekly_sma10_daily_sma30,
        'Weekly SMA20 + Daily SMA30': analyzer.strategy_weekly_sma20_daily_sma30,
        'Weekly SMA50 + Daily SMA30': analyzer.strategy_weekly_sma50_daily_sma30,
    }

    # 각 전략의 equity curve 계산 (loop-based로 직접 계산)
    equity_curves = {}

    for name, func in strategies.items():
        print(f"Calculating {name}...")

        if name == 'Benchmark':
            # 벤치마크
            df = analyzer.daily_data.copy()
            df['SMA30'] = df['Close'].rolling(30).mean()
            df['signal'] = (df['Close'] > df['SMA30']).astype(int)
            df['pos_change'] = df['signal'].diff()
            df['daily_ret'] = df['Close'].pct_change()
            df['strat_ret'] = df['signal'].shift(1) * df['daily_ret']

            slip_cost = pd.Series(0.0, index=df.index)
            slip_cost[df['pos_change'] == 1] = -analyzer.slippage
            slip_cost[df['pos_change'] == -1] = -analyzer.slippage
            df['strat_ret'] = df['strat_ret'] + slip_cost
            df['strat_ret'] = df['strat_ret'].fillna(0)

            df['equity'] = (1 + df['strat_ret']).cumprod()
            equity_curves[name] = df['equity']

        else:
            # MTF 전략 - loop-based 계산
            # 각 전략의 weekly/daily 신호 생성
            if 'Donchian' in name:
                weekly = analyzer.weekly_data.copy()
                weekly['high_20'] = weekly['High'].rolling(20).max()
                weekly['signal'] = (weekly['Close'] > weekly['high_20'] * 0.95).astype(int)
            elif 'EMA20' in name:
                weekly = analyzer.weekly_data.copy()
                weekly['EMA20'] = weekly['Close'].ewm(span=20, adjust=False).mean()
                weekly['signal'] = (weekly['Close'] > weekly['EMA20']).astype(int)
            elif 'SMA10' in name:
                weekly = analyzer.weekly_data.copy()
                weekly['SMA10'] = weekly['Close'].rolling(10).mean()
                weekly['signal'] = (weekly['Close'] > weekly['SMA10']).astype(int)
            elif 'SMA20' in name:
                weekly = analyzer.weekly_data.copy()
                weekly['SMA20'] = weekly['Close'].rolling(20).mean()
                weekly['signal'] = (weekly['Close'] > weekly['SMA20']).astype(int)
            elif 'SMA50' in name:
                weekly = analyzer.weekly_data.copy()
                weekly['SMA50'] = weekly['Close'].rolling(50).mean()
                weekly['signal'] = (weekly['Close'] > weekly['SMA50']).astype(int)

            # Weekly signals with availability
            weekly_signals = {}
            for i in range(len(weekly)):
                week_date = weekly.index[i]
                signal = weekly.iloc[i]['signal']
                available_from = week_date + pd.Timedelta(days=1)
                weekly_signals[week_date] = {'signal': signal, 'available_from': available_from}

            # Daily signal
            daily = analyzer.daily_data.copy()
            daily['SMA30'] = daily['Close'].rolling(30).mean()
            daily['daily_signal'] = (daily['Close'] > daily['SMA30']).astype(int)

            # Loop-based combination
            capital = 1.0
            position = 0
            equity = []

            for i in range(len(daily)):
                date = daily.index[i]
                close = daily.iloc[i]['Close']
                daily_sig = daily.iloc[i]['daily_signal']

                # Find weekly signal
                weekly_sig = 0
                for week_date in sorted(weekly_signals.keys(), reverse=True):
                    if date >= weekly_signals[week_date]['available_from']:
                        weekly_sig = weekly_signals[week_date]['signal']
                        break

                final_signal = 1 if (daily_sig == 1 and weekly_sig == 1) else 0

                # Calculate returns
                if i > 0:
                    prev_close = daily.iloc[i-1]['Close']
                    daily_return = (close - prev_close) / prev_close

                    if position == 1:
                        capital = capital * (1 + daily_return)

                    # Position change
                    if position == 0 and final_signal == 1:
                        capital = capital * (1 - analyzer.slippage)
                    elif position == 1 and final_signal == 0:
                        capital = capital * (1 - analyzer.slippage)

                position = final_signal
                equity.append(capital)

            equity_series = pd.Series(equity, index=daily.index)
            equity_curves[name] = equity_series

    # Drawdown 계산
    drawdowns = {}
    for name, equity in equity_curves.items():
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100  # Percentage
        drawdowns[name] = drawdown

    # 시각화
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # 1. Equity Curve (Log scale)
    ax1 = axes[0]

    colors = {
        'Benchmark': '#FF6B6B',
        'Weekly Donchian + Daily SMA30': '#4ECDC4',
        'Weekly EMA20 + Daily SMA30': '#95E1D3',
        'Weekly SMA10 + Daily SMA30': '#F38181',
        'Weekly SMA20 + Daily SMA30': '#AA96DA',
        'Weekly SMA50 + Daily SMA30': '#FCBAD3',
    }

    for name, equity in equity_curves.items():
        linewidth = 3 if name == 'Benchmark' else 2.5 if 'Donchian' in name else 1.5
        linestyle = '--' if name == 'Benchmark' else '-'
        alpha = 1.0 if name == 'Benchmark' or 'Donchian' in name else 0.7

        ax1.plot(equity.index, equity,
                label=name.replace('Weekly ', 'W ').replace('Daily ', 'D '),
                color=colors[name], linewidth=linewidth,
                linestyle=linestyle, alpha=alpha)

    ax1.set_yscale('log')
    ax1.set_ylabel('Cumulative Equity (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Equity Curves - Multi-Timeframe Strategies vs Benchmark',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Y축 포맷
    from matplotlib.ticker import FuncFormatter
    def format_func(value, tick_number):
        if value >= 100:
            return f'{int(value)}'
        elif value >= 10:
            return f'{value:.0f}'
        else:
            return f'{value:.1f}'

    ax1.yaxis.set_major_formatter(FuncFormatter(format_func))

    # 2. Drawdown
    ax2 = axes[1]

    for name, dd in drawdowns.items():
        linewidth = 3 if name == 'Benchmark' else 2.5 if 'Donchian' in name else 1.5
        linestyle = '--' if name == 'Benchmark' else '-'
        alpha = 1.0 if name == 'Benchmark' or 'Donchian' in name else 0.7

        ax2.fill_between(dd.index, dd, 0,
                         color=colors[name], alpha=0.3)
        ax2.plot(dd.index, dd,
                label=name.replace('Weekly ', 'W ').replace('Daily ', 'D '),
                color=colors[name], linewidth=linewidth,
                linestyle=linestyle, alpha=alpha)

    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Drawdown Over Time', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # X축 포맷
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('mtf_equity_drawdown.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: mtf_equity_drawdown.png")

    # 통계 요약
    print("\n" + "="*80)
    print("EQUITY CURVE STATISTICS")
    print("="*80)

    stats = []
    for name, equity in equity_curves.items():
        final_value = equity.iloc[-1]
        max_dd = drawdowns[name].min()

        stats.append({
            'Strategy': name,
            'Final Equity': f'{final_value:.2f}',
            'Total Return (%)': f'{(final_value - 1) * 100:.2f}',
            'Max Drawdown (%)': f'{max_dd:.2f}'
        })

    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    create_equity_drawdown_visualization()
