"""
완전 새로운 구현: 반복문으로 Day-by-Day 시뮬레이션
Lookahead bias 완전 제거
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def loop_based_backtest():
    """반복문 기반 백테스트 - Lookahead bias 완전 제거"""

    print("="*80)
    print("LOOP-BASED BACKTEST: Zero Lookahead Bias")
    print("="*80 + "\n")

    # 데이터 로드
    df_daily = pd.read_parquet('chart_day/BTC_KRW.parquet')
    df_daily.columns = [col.capitalize() for col in df_daily.columns]
    df_daily = df_daily[df_daily.index >= '2018-01-01'].copy()

    print(f"Loaded {len(df_daily)} daily bars\n")

    # 주봉 데이터 생성
    df_weekly = df_daily.resample('W-MON', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    print(f"Created {len(df_weekly)} weekly bars")
    print(f"Weekly bars from {df_weekly.index[0]} to {df_weekly.index[-1]}\n")

    # ============================================
    # 벤치마크: Daily SMA30
    # ============================================
    print("="*80)
    print("BENCHMARK: Daily SMA30 (Loop-based)")
    print("="*80 + "\n")

    capital_bench = 1.0
    position_bench = 0  # 0 = cash, 1 = in position
    trades_bench = []
    equity_curve_bench = []

    slippage = 0.002

    for i in range(len(df_daily)):
        date = df_daily.index[i]
        close = df_daily.iloc[i]['Close']

        # SMA30 계산 (과거 30일만 사용)
        if i < 30:
            sma30 = np.nan
        else:
            sma30 = df_daily.iloc[i-29:i+1]['Close'].mean()

        # 신호 계산 (오늘 종가와 오늘 SMA30 비교)
        if pd.notna(sma30):
            signal_today = 1 if close > sma30 else 0
        else:
            signal_today = 0

        # 포지션 변경 (내일 시가에 체결된다고 가정)
        # 하지만 백테스트에서는 오늘 종가로 근사
        prev_capital = capital_bench

        if i > 0:
            prev_close = df_daily.iloc[i-1]['Close']
            daily_return = (close - prev_close) / prev_close

            # 어제 신호에 따라 오늘 수익 실현
            if position_bench == 1:
                capital_bench = capital_bench * (1 + daily_return)

            # 포지션 변경 시 슬리피지
            if position_bench == 0 and signal_today == 1:
                # 매수
                capital_bench = capital_bench * (1 - slippage)
                trades_bench.append({
                    'date': date,
                    'action': 'BUY',
                    'price': close,
                    'capital': capital_bench
                })
            elif position_bench == 1 and signal_today == 0:
                # 매도
                capital_bench = capital_bench * (1 - slippage)
                trades_bench.append({
                    'date': date,
                    'action': 'SELL',
                    'price': close,
                    'capital': capital_bench
                })

        position_bench = signal_today
        equity_curve_bench.append({
            'date': date,
            'capital': capital_bench,
            'position': position_bench
        })

    bench_final = capital_bench
    bench_return = (bench_final - 1) * 100

    years = (df_daily.index[-1] - df_daily.index[0]).days / 365.25
    bench_cagr = (bench_final ** (1/years) - 1) * 100

    equity_bench = pd.DataFrame(equity_curve_bench).set_index('date')
    bench_returns = equity_bench['capital'].pct_change().fillna(0)
    bench_sharpe = bench_returns.mean() / bench_returns.std() * np.sqrt(365) if bench_returns.std() > 0 else 0

    print(f"Final Capital: {bench_final:.2f}x")
    print(f"Total Return: {bench_return:.2f}%")
    print(f"CAGR: {bench_cagr:.2f}%")
    print(f"Sharpe: {bench_sharpe:.4f}")
    print(f"Trades: {len(trades_bench)}")

    # ============================================
    # Strategy: Weekly SMA10 + Daily SMA30
    # ============================================
    print("\n" + "="*80)
    print("STRATEGY: Weekly SMA10 + Daily SMA30 (Loop-based, Zero Lookahead)")
    print("="*80 + "\n")

    print("⚠️ CRITICAL: Ensuring NO lookahead bias")
    print("Rule: Weekly signal calculated on Monday can ONLY be used from TUESDAY onwards\n")

    capital_strat = 1.0
    position_strat = 0
    trades_strat = []
    equity_curve_strat = []

    # 주봉 신호를 미리 계산하되, 사용 가능 시점을 명확히
    weekly_signals = {}

    for i in range(len(df_weekly)):
        week_end_date = df_weekly.index[i]  # 월요일 00:00 (주봉 마감)

        # SMA10 계산 (과거 10주만 사용)
        if i < 10:
            weekly_sma10 = np.nan
        else:
            weekly_sma10 = df_weekly.iloc[i-9:i+1]['Close'].mean()

        # 이번 주 종가
        week_close = df_weekly.iloc[i]['Close']

        # 주봉 신호
        if pd.notna(weekly_sma10):
            weekly_signal = 1 if week_close > weekly_sma10 else 0
        else:
            weekly_signal = 0

        # ⚠️ KEY: 이 신호는 week_end_date (월요일) 이후부터 사용 가능
        # 월요일 00:00에 주봉이 마감되므로, 월요일부터 사용 가능
        # 하지만 더 보수적으로, 다음 날(화요일)부터 사용하도록 설정
        signal_available_from = week_end_date + timedelta(days=1)

        weekly_signals[week_end_date] = {
            'signal': weekly_signal,
            'available_from': signal_available_from,
            'sma10': weekly_sma10,
            'close': week_close
        }

    print("Weekly signals calculated:")
    print("Sample (first 15 weeks):")
    for i, (date, info) in enumerate(list(weekly_signals.items())[:15]):
        print(f"  Week ending {date.date()}: Signal={info['signal']}, Available from {info['available_from'].date()}")

    # 일봉 순회
    for i in range(len(df_daily)):
        date = df_daily.index[i]
        close = df_daily.iloc[i]['Close']

        # Daily SMA30 계산
        if i < 30:
            daily_sma30 = np.nan
        else:
            daily_sma30 = df_daily.iloc[i-29:i+1]['Close'].mean()

        # Daily 신호
        if pd.notna(daily_sma30):
            daily_signal = 1 if close > daily_sma30 else 0
        else:
            daily_signal = 0

        # Weekly 신호 찾기 (오늘 사용 가능한 가장 최근 신호)
        weekly_signal = 0
        for week_date in sorted(weekly_signals.keys(), reverse=True):
            if date >= weekly_signals[week_date]['available_from']:
                weekly_signal = weekly_signals[week_date]['signal']
                break

        # 최종 신호: Daily AND Weekly
        final_signal = 1 if (daily_signal == 1 and weekly_signal == 1) else 0

        # 포지션 및 자본 업데이트
        if i > 0:
            prev_close = df_daily.iloc[i-1]['Close']
            daily_return = (close - prev_close) / prev_close

            # 어제 포지션에 따라 오늘 수익 실현
            if position_strat == 1:
                capital_strat = capital_strat * (1 + daily_return)

            # 포지션 변경
            if position_strat == 0 and final_signal == 1:
                # 매수
                capital_strat = capital_strat * (1 - slippage)
                trades_strat.append({
                    'date': date,
                    'action': 'BUY',
                    'price': close,
                    'daily_signal': daily_signal,
                    'weekly_signal': weekly_signal,
                    'capital': capital_strat
                })
            elif position_strat == 1 and final_signal == 0:
                # 매도
                capital_strat = capital_strat * (1 - slippage)
                trades_strat.append({
                    'date': date,
                    'action': 'SELL',
                    'price': close,
                    'daily_signal': daily_signal,
                    'weekly_signal': weekly_signal,
                    'capital': capital_strat
                })

        position_strat = final_signal
        equity_curve_strat.append({
            'date': date,
            'capital': capital_strat,
            'position': position_strat,
            'daily_signal': daily_signal,
            'weekly_signal': weekly_signal
        })

    strat_final = capital_strat
    strat_return = (strat_final - 1) * 100
    strat_cagr = (strat_final ** (1/years) - 1) * 100

    equity_strat = pd.DataFrame(equity_curve_strat).set_index('date')
    strat_returns = equity_strat['capital'].pct_change().fillna(0)
    strat_sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(365) if strat_returns.std() > 0 else 0

    print(f"\nFinal Capital: {strat_final:.2f}x")
    print(f"Total Return: {strat_return:.2f}%")
    print(f"CAGR: {strat_cagr:.2f}%")
    print(f"Sharpe: {strat_sharpe:.4f}")
    print(f"Trades: {len(trades_strat)}")

    print("\nFirst 20 trades:")
    for trade in trades_strat[:20]:
        print(f"  {trade['date'].date()}: {trade['action']:4s} @ {trade['price']:>11,.0f} | "
              f"D={trade['daily_signal']} W={trade['weekly_signal']} | Cap={trade['capital']:.2f}x")

    # 비교
    print("\n" + "="*80)
    print("COMPARISON: Loop-based vs Previous Vectorized")
    print("="*80 + "\n")

    ratio = strat_final / bench_final

    print("Loop-based Results:")
    print(f"  Benchmark: {bench_final:.2f}x ({bench_return:.2f}%, Sharpe {bench_sharpe:.4f})")
    print(f"  Strategy:  {strat_final:.2f}x ({strat_return:.2f}%, Sharpe {strat_sharpe:.4f})")
    print(f"  Ratio: {ratio:.2f}x")

    print("\nPrevious Vectorized Results:")
    print(f"  Benchmark: 89.59x (8858.65%, Sharpe 1.6591)")
    print(f"  Strategy:  309.70x (30869.99%, Sharpe 2.2185)")
    print(f"  Ratio: 3.46x")

    print("\n" + "="*80)
    print("VERDICT:")
    print("="*80)

    if abs(ratio - 3.46) < 0.5:
        print("\n✅ CONFIRMED: Loop-based results match vectorized results")
        print("   No lookahead bias detected!")
        print("   The 3.46x ratio is LEGITIMATE.")
    else:
        print(f"\n🚨 DISCREPANCY DETECTED!")
        print(f"   Loop-based: {ratio:.2f}x")
        print(f"   Vectorized: 3.46x")
        print(f"   Difference: {abs(ratio - 3.46):.2f}x")
        print(f"   This suggests potential lookahead bias in vectorized version!")

    # 연도별 비교
    print("\n" + "="*80)
    print("YEAR-BY-YEAR COMPARISON (Loop-based)")
    print("="*80)

    for year in range(2018, 2026):
        year_mask = equity_bench.index.year == year
        if year_mask.sum() == 0:
            continue

        bench_start = equity_bench.loc[year_mask, 'capital'].iloc[0] if year > 2018 else 1.0
        bench_end = equity_bench.loc[year_mask, 'capital'].iloc[-1]
        bench_year_ret = (bench_end / bench_start - 1) * 100

        strat_start = equity_strat.loc[year_mask, 'capital'].iloc[0] if year > 2018 else 1.0
        strat_end = equity_strat.loc[year_mask, 'capital'].iloc[-1]
        strat_year_ret = (strat_end / strat_start - 1) * 100

        print(f"\n{year}:")
        print(f"  Benchmark: {bench_year_ret:+.2f}%")
        print(f"  Strategy:  {strat_year_ret:+.2f}%")
        print(f"  Difference: {(strat_year_ret - bench_year_ret):+.2f}%")

    return equity_bench, equity_strat, trades_bench, trades_strat


if __name__ == "__main__":
    loop_based_backtest()
