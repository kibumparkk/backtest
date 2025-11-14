"""
Lookahead Bias 원인 찾기
"""

import pandas as pd
import numpy as np

def find_lookahead_source():
    """Vectorized 버전에서 lookahead가 어디서 발생했는지 찾기"""

    print("="*80)
    print("FINDING LOOKAHEAD BIAS SOURCE")
    print("="*80 + "\n")

    # 데이터 로드
    df_daily = pd.read_parquet('chart_day/BTC_KRW.parquet')
    df_daily.columns = [col.capitalize() for col in df_daily.columns]
    df_daily = df_daily[df_daily.index >= '2018-01-01'].copy()

    # 주봉 생성 (vectorized 방식 그대로)
    df_weekly = df_daily.resample('W-MON', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    print("Weekly resampling parameters:")
    print("  label='left' - 주봉 라벨이 주의 시작일")
    print("  closed='left' - 시작일 포함, 종료일 미포함")
    print()

    # 2018년 4월 마지막 주 상세 분석
    print("="*80)
    print("CASE STUDY: Week of 2018-04-23 (First BUY signal)")
    print("="*80 + "\n")

    # 해당 주의 일봉 데이터
    week_daily = df_daily['2018-04-16':'2018-04-30']
    print("Daily data for this period:")
    print(week_daily[['Close']].to_string())

    # 주봉 계산
    print("\nWeekly bars around this date:")
    week_weekly = df_weekly['2018-04-09':'2018-04-30']
    print(week_weekly[['Close']])

    # 문제의 주봉
    problem_week = df_weekly.loc['2018-04-23']
    print(f"\n⚠️ Week ending 2018-04-23:")
    print(f"  Weekly Close: {problem_week['Close']:,.0f}")

    # 이 주봉에는 어떤 일봉들이 포함되나?
    print(f"\n  This weekly bar includes:")
    # W-MON, label='left', closed='left' 의미:
    # 2018-04-23 주봉 = 2018-04-23 (월) ~ 2018-04-29 (일)
    # 하지만 closed='left'이면 2018-04-23만 포함하고 2018-04-29는 다음 주!
    # 아니다, 다시 생각해보니 W-MON은 월요일 기준이므로...

    # 실제로 확인
    test_period = df_daily['2018-04-16':'2018-04-29']
    print("  Daily bars 2018-04-16 to 2018-04-29:")
    print(test_period[['Close']])

    # Resample 테스트
    test_weekly = test_period.resample('W-MON', label='left', closed='left').agg({
        'Close': 'last'
    })
    print("\n  Resampled to weekly:")
    print(test_weekly)

    print("\n" + "="*80)
    print("CRITICAL DISCOVERY:")
    print("="*80)

    print("""
W-MON with label='left', closed='left' means:
- Week label: Monday 00:00
- Includes: Monday 00:00 (inclusive) to next Monday 00:00 (exclusive)

For week labeled 2018-04-23:
- Includes: 2018-04-23 (Mon) to 2018-04-29 (Sun)
- Weekly Close = 2018-04-29 (Sun) close price

⚠️ THE PROBLEM:
In vectorized version with reindex + ffill:
  weekly_signal.reindex(daily.index, method='ffill')

This assigns the weekly signal to ALL days of that week, INCLUDING Monday!

So on 2018-04-23 (Monday), the daily strategy uses:
- Weekly signal calculated WITH 2018-04-23~29 data
- But 2018-04-23 data is USED to calculate that week's signal
- This is LOOKAHEAD BIAS!

CORRECT APPROACH (Loop-based):
- Weekly signal available AFTER the week ends
- Week ending 2018-04-23 signal available from 2018-04-24 (Tuesday) onwards
""")

    # Reindex 문제 시연
    print("\n" + "="*80)
    print("DEMONSTRATING THE REINDEX PROBLEM")
    print("="*80 + "\n")

    # 간단한 예시
    df_w = df_weekly.head(15).copy()
    df_w['SMA10'] = df_w['Close'].rolling(10).mean()
    df_w['signal'] = (df_w['Close'] > df_w['SMA10']).astype(int)

    print("Weekly signals (first 15 weeks):")
    print(df_w[['Close', 'SMA10', 'signal']])

    # Reindex to daily
    df_d = df_daily['2018-01-01':'2018-05-01'].copy()
    weekly_signal_expanded = df_w['signal'].reindex(df_d.index, method='ffill')

    print("\n⚠️ After reindex with ffill:")
    print("Week 2018-04-23:")
    week_mask = (df_d.index >= '2018-04-23') & (df_d.index < '2018-04-30')
    print(df_d[week_mask][['Close']])
    print("\nExpanded weekly signal for these days:")
    print(weekly_signal_expanded[week_mask])

    week_2018_04_23_signal = df_w.loc['2018-04-23', 'signal']
    print(f"\nWeekly signal for 2018-04-23: {week_2018_04_23_signal}")
    print(f"This signal is applied to 2018-04-23 (Monday) immediately!")
    print(f"But it's calculated using data from 2018-04-23 ~ 2018-04-29")
    print(f"→ LOOKAHEAD BIAS on 2018-04-23!")

    print("\n" + "="*80)
    print("SOLUTION")
    print("="*80)
    print("""
Option 1 (Loop-based):
  Manually shift weekly signal by 1 day
  Week ending Monday → available from Tuesday

Option 2 (Vectorized fix):
  weekly_signal_shifted = weekly_signal.shift(1)
  Then reindex

Option 3 (Vectorized fix):
  After reindex, shift by 1 day:
  weekly_signal_expanded = weekly_signal.reindex(...).shift(1)
""")

    # 수정된 vectorized 버전 테스트
    print("\n" + "="*80)
    print("TESTING FIXED VECTORIZED VERSION")
    print("="*80 + "\n")

    df_w_full = df_weekly.copy()
    df_w_full['SMA10'] = df_w_full['Close'].rolling(10).mean()
    df_w_full['signal'] = (df_w_full['Close'] > df_w_full['SMA10']).astype(int)

    # 수정: shift before reindex
    df_w_full['signal_shifted'] = df_w_full['signal'].shift(1)

    df_d_full = df_daily.copy()
    df_d_full['SMA30'] = df_d_full['Close'].rolling(30).mean()
    df_d_full['daily_signal'] = (df_d_full['Close'] > df_d_full['SMA30']).astype(int)

    # Original (buggy)
    weekly_signal_orig = df_w_full['signal'].reindex(df_d_full.index, method='ffill')

    # Fixed
    weekly_signal_fixed = df_w_full['signal_shifted'].reindex(df_d_full.index, method='ffill')

    df_d_full['weekly_signal_orig'] = weekly_signal_orig
    df_d_full['weekly_signal_fixed'] = weekly_signal_fixed

    print("Comparison for 2018-04-23 week:")
    sample = df_d_full['2018-04-23':'2018-04-30'][['Close', 'weekly_signal_orig', 'weekly_signal_fixed']]
    print(sample)

    print("\n✓ With shift, Monday uses PREVIOUS week's signal")
    print("  This eliminates lookahead bias!")

    return df_w_full, df_d_full


if __name__ == "__main__":
    find_lookahead_source()
