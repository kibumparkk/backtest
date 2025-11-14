"""
멀티 타임프레임 전략 검증
- 과대평가 오류 확인
- 미래 정보 누출(lookahead bias) 체크
- 수익률 계산 로직 검증
"""

import pandas as pd
import numpy as np
from datetime import datetime

def validate_strategy():
    """전략 검증"""
    print("="*80)
    print("Multi-Timeframe Strategy Validation")
    print("="*80 + "\n")

    # 데이터 로드
    df_daily = pd.read_parquet('chart_day/BTC_KRW.parquet')
    df_daily.columns = [col.capitalize() for col in df_daily.columns]
    df_daily = df_daily[df_daily.index >= '2018-01-01']

    # 주봉 생성
    df_weekly = df_daily.resample('W-MON', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    print(f"Daily bars: {len(df_daily)}")
    print(f"Weekly bars: {len(df_weekly)}\n")

    # ============= 벤치마크: Daily SMA30 =============
    print("="*80)
    print("BENCHMARK: Daily SMA30 only")
    print("="*80)

    df_bench = df_daily.copy()
    df_bench['SMA30'] = df_bench['Close'].rolling(30).mean()
    df_bench['signal'] = (df_bench['Close'] > df_bench['SMA30']).astype(int)

    # 수익률 계산
    df_bench['pos_change'] = df_bench['signal'].diff()
    df_bench['daily_ret'] = df_bench['Close'].pct_change()
    df_bench['strat_ret'] = df_bench['signal'].shift(1) * df_bench['daily_ret']

    # 슬리피지
    slippage = 0.002
    slip_cost = pd.Series(0.0, index=df_bench.index)
    slip_cost[df_bench['pos_change'] == 1] = -slippage
    slip_cost[df_bench['pos_change'] == -1] = -slippage
    df_bench['strat_ret'] = df_bench['strat_ret'] + slip_cost
    df_bench['strat_ret'] = df_bench['strat_ret'].fillna(0)

    df_bench['cumulative'] = (1 + df_bench['strat_ret']).cumprod()

    bench_final = df_bench['cumulative'].iloc[-1]
    bench_total_ret = (bench_final - 1) * 100

    years = (df_bench.index[-1] - df_bench.index[0]).days / 365.25
    bench_cagr = (bench_final ** (1/years) - 1) * 100
    bench_sharpe = df_bench['strat_ret'].mean() / df_bench['strat_ret'].std() * np.sqrt(365)

    print(f"Total Return: {bench_total_ret:.2f}%")
    print(f"CAGR: {bench_cagr:.2f}%")
    print(f"Sharpe: {bench_sharpe:.4f}")
    print(f"Final Cumulative: {bench_final:.2f}")

    # 샘플 거래 출력
    trades = df_bench[df_bench['pos_change'] != 0][['Close', 'SMA30', 'signal', 'pos_change', 'strat_ret']].head(10)
    print("\nSample trades:")
    print(trades)

    # ============= 2위 전략: Weekly SMA10 + Daily SMA30 =============
    print("\n" + "="*80)
    print("STRATEGY 2: Weekly SMA10 + Daily SMA30")
    print("="*80)

    # 주봉 신호
    df_w = df_weekly.copy()
    df_w['SMA10'] = df_w['Close'].rolling(10).mean()
    df_w['weekly_signal'] = (df_w['Close'] > df_w['SMA10']).astype(int)

    print(f"\nWeekly SMA10 calculation:")
    print(df_w[['Close', 'SMA10', 'weekly_signal']].tail(10))

    # 일봉 신호
    df_d = df_daily.copy()
    df_d['SMA30'] = df_d['Close'].rolling(30).mean()
    df_d['daily_signal'] = (df_d['Close'] > df_d['SMA30']).astype(int)

    # 주봉 신호를 일봉에 확장
    print(f"\n⚠️ CRITICAL: Weekly signal expansion to daily")
    print(f"Method: reindex with forward fill (ffill)")

    weekly_signal_expanded = df_w['weekly_signal'].reindex(df_d.index, method='ffill')

    # 확장 검증
    print(f"\nWeekly signal expanded - First 20 days:")
    check_df = pd.DataFrame({
        'date': df_d.index[:20],
        'daily_close': df_d['Close'].iloc[:20].values,
        'weekly_signal': weekly_signal_expanded.iloc[:20].values
    })
    print(check_df)

    # 최종 신호
    df_d['weekly_signal'] = weekly_signal_expanded
    df_d['final_signal'] = (df_d['daily_signal'] & (df_d['weekly_signal'] == 1)).astype(int)

    print(f"\nSignal combination check (first 50 days):")
    signal_check = df_d[['Close', 'SMA30', 'daily_signal', 'weekly_signal', 'final_signal']].head(50)
    print(signal_check[signal_check['final_signal'].diff() != 0])

    # 수익률 계산
    df_d['pos_change'] = df_d['final_signal'].diff()
    df_d['daily_ret'] = df_d['Close'].pct_change()
    df_d['strat_ret'] = df_d['final_signal'].shift(1) * df_d['daily_ret']

    # 슬리피지
    slip_cost = pd.Series(0.0, index=df_d.index)
    slip_cost[df_d['pos_change'] == 1] = -slippage
    slip_cost[df_d['pos_change'] == -1] = -slippage
    df_d['strat_ret'] = df_d['strat_ret'] + slip_cost
    df_d['strat_ret'] = df_d['strat_ret'].fillna(0)

    df_d['cumulative'] = (1 + df_d['strat_ret']).cumprod()

    strat_final = df_d['cumulative'].iloc[-1]
    strat_total_ret = (strat_final - 1) * 100
    strat_cagr = (strat_final ** (1/years) - 1) * 100
    strat_sharpe = df_d['strat_ret'].mean() / df_d['strat_ret'].std() * np.sqrt(365)

    print(f"\nTotal Return: {strat_total_ret:.2f}%")
    print(f"CAGR: {strat_cagr:.2f}%")
    print(f"Sharpe: {strat_sharpe:.4f}")
    print(f"Final Cumulative: {strat_final:.2f}")

    # 샘플 거래 출력
    trades = df_d[df_d['pos_change'] != 0][['Close', 'SMA30', 'weekly_signal', 'final_signal', 'pos_change', 'strat_ret']].head(20)
    print("\nSample trades:")
    print(trades)

    # ============= 비교 분석 =============
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)

    print(f"\nBenchmark:")
    print(f"  Total Return: {bench_total_ret:.2f}%")
    print(f"  CAGR: {bench_cagr:.2f}%")
    print(f"  Sharpe: {bench_sharpe:.4f}")

    print(f"\nStrategy 2 (Weekly SMA10 + Daily SMA30):")
    print(f"  Total Return: {strat_total_ret:.2f}%")
    print(f"  CAGR: {strat_cagr:.2f}%")
    print(f"  Sharpe: {strat_sharpe:.4f}")

    ratio = strat_total_ret / bench_total_ret
    print(f"\nReturn Ratio: {ratio:.2f}x")

    if ratio > 2.0:
        print("⚠️ WARNING: Return ratio > 2.0 is suspicious!")
        print("   Possible issues:")
        print("   1. Lookahead bias (future information leak)")
        print("   2. Data alignment error")
        print("   3. Signal calculation bug")
        print("   4. Return calculation error")

    # ============= 포지션 비교 =============
    print("\n" + "="*80)
    print("POSITION COMPARISON")
    print("="*80)

    bench_in_market = df_bench['signal'].sum() / len(df_bench) * 100
    strat_in_market = df_d['final_signal'].sum() / len(df_d) * 100

    print(f"\nBenchmark in-market days: {bench_in_market:.2f}%")
    print(f"Strategy 2 in-market days: {strat_in_market:.2f}%")

    # 신호 차이 분석
    print("\n" + "="*80)
    print("SIGNAL DIFFERENCE ANALYSIS")
    print("="*80)

    # 같은 인덱스로 정렬
    common_idx = df_bench.index.intersection(df_d.index)
    bench_sig = df_bench.loc[common_idx, 'signal']
    strat_sig = df_d.loc[common_idx, 'final_signal']

    both_in = ((bench_sig == 1) & (strat_sig == 1)).sum()
    bench_only = ((bench_sig == 1) & (strat_sig == 0)).sum()
    strat_only = ((bench_sig == 0) & (strat_sig == 1)).sum()
    both_out = ((bench_sig == 0) & (strat_sig == 0)).sum()

    print(f"\nSignal overlap:")
    print(f"  Both IN:  {both_in} days ({both_in/len(common_idx)*100:.2f}%)")
    print(f"  Bench only: {bench_only} days ({bench_only/len(common_idx)*100:.2f}%)")
    print(f"  Strat only: {strat_only} days ({strat_only/len(common_idx)*100:.2f}%)")
    print(f"  Both OUT: {both_out} days ({both_out/len(common_idx)*100:.2f}%)")

    # 차이나는 구간의 수익률 분석
    if strat_only > 0:
        print("\n⚠️ Strategy-only days (when benchmark is OUT but strategy is IN):")
        strat_only_mask = (bench_sig == 0) & (strat_sig == 1)
        strat_only_dates = df_d.loc[common_idx[strat_only_mask]]

        if len(strat_only_dates) > 0:
            strat_only_ret = strat_only_dates['daily_ret'].mean() * 100
            print(f"  Average daily return: {strat_only_ret:.3f}%")
            print(f"  Sample dates (first 10):")
            print(strat_only_dates[['Close', 'daily_ret', 'weekly_signal']].head(10))

    # ============= 연도별 성과 =============
    print("\n" + "="*80)
    print("YEAR-BY-YEAR PERFORMANCE")
    print("="*80)

    for year in range(2018, 2026):
        year_mask = df_bench.index.year == year
        if year_mask.sum() == 0:
            continue

        bench_year_ret = (1 + df_bench.loc[year_mask, 'strat_ret']).prod() - 1
        strat_year_ret = (1 + df_d.loc[year_mask, 'strat_ret']).prod() - 1

        print(f"\n{year}:")
        print(f"  Benchmark: {bench_year_ret*100:+.2f}%")
        print(f"  Strategy:  {strat_year_ret*100:+.2f}%")
        print(f"  Difference: {(strat_year_ret - bench_year_ret)*100:+.2f}%")

        if abs(strat_year_ret - bench_year_ret) > 2.0:  # 200% 차이
            print(f"  ⚠️ HUGE difference detected in {year}!")

    # 최종 판단
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    if ratio > 3.0:
        print("\n🚨 CRITICAL: Return ratio > 3.0x is HIGHLY SUSPICIOUS")
        print("   This suggests a fundamental error in the backtest")
        print("   Recommend thorough review of:")
        print("   - Data alignment (weekly-daily reindex)")
        print("   - Signal generation logic")
        print("   - Return calculation")
    elif ratio > 2.0:
        print("\n⚠️ WARNING: Return ratio > 2.0x is questionable")
        print("   Further investigation recommended")
    else:
        print("\n✓ Return ratio seems reasonable")

    return df_bench, df_d


if __name__ == "__main__":
    validate_strategy()
