"""
벤치마크 전략 검증 스크립트
"""

import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_parquet('chart_day/BTC_KRW.parquet')
df.columns = [col.capitalize() for col in df.columns]
df = df[(df.index >= '2018-01-01')]

print("="*80)
print("벤치마크 전략 구현 비교")
print("="*80)

# 현재 구현 (의심스러운 버전)
df_test = df.copy()
df_test['SMA30'] = df_test['Close'].rolling(window=30).mean()
df_test['prev_close'] = df_test['Close'].shift(1)
df_test['signal_v1'] = (df_test['prev_close'] > df_test['SMA30']).astype(int)

# 올바른 구현 1: 전일 종가 > 전일 SMA30
df_test['prev_sma30'] = df_test['SMA30'].shift(1)
df_test['signal_v2'] = (df_test['prev_close'] > df_test['prev_sma30']).astype(int)

# 올바른 구현 2: 당일 종가 > 당일 SMA30 (가장 일반적)
df_test['signal_v3'] = (df_test['Close'] > df_test['SMA30']).astype(int)

# 샘플 데이터 출력
print("\n샘플 데이터 (최근 10일):")
print("-"*80)
sample = df_test[['Close', 'SMA30', 'prev_close', 'prev_sma30',
                   'signal_v1', 'signal_v2', 'signal_v3']].tail(10)
print(sample.to_string())

# 각 버전별 신호 통계
print("\n" + "="*80)
print("신호 통계 비교:")
print("="*80)
print(f"V1 (현재 구현: 전일종가 vs 당일SMA30): 매수일수 {df_test['signal_v1'].sum()}/{len(df_test)}")
print(f"V2 (전일종가 vs 전일SMA30):          매수일수 {df_test['signal_v2'].sum()}/{len(df_test)}")
print(f"V3 (당일종가 vs 당일SMA30):          매수일수 {df_test['signal_v3'].sum()}/{len(df_test)}")

# 간단한 백테스트 비교
def simple_backtest(signal, df, slippage=0.002):
    df_bt = df.copy()
    df_bt['signal'] = signal
    df_bt['position_change'] = df_bt['signal'].diff()
    df_bt['daily_return'] = df_bt['Close'].pct_change()
    df_bt['strategy_return'] = df_bt['signal'].shift(1) * df_bt['daily_return']

    # 슬리피지
    slippage_cost = pd.Series(0.0, index=df_bt.index)
    slippage_cost[df_bt['position_change'] == 1] = -slippage
    slippage_cost[df_bt['position_change'] == -1] = -slippage
    df_bt['strategy_return'] = df_bt['strategy_return'] + slippage_cost
    df_bt['strategy_return'] = df_bt['strategy_return'].fillna(0)

    cumulative = (1 + df_bt['strategy_return']).cumprod().iloc[-1] - 1
    sharpe = (df_bt['strategy_return'].mean() / df_bt['strategy_return'].std() * np.sqrt(365))

    return cumulative * 100, sharpe

print("\n" + "="*80)
print("백테스트 결과 비교:")
print("="*80)

ret1, sharpe1 = simple_backtest(df_test['signal_v1'], df_test)
print(f"V1 (현재 구현): 수익률 {ret1:.2f}%, 샤프 {sharpe1:.4f}")

ret2, sharpe2 = simple_backtest(df_test['signal_v2'], df_test)
print(f"V2 (전일종가 vs 전일SMA30): 수익률 {ret2:.2f}%, 샤프 {sharpe2:.4f}")

ret3, sharpe3 = simple_backtest(df_test['signal_v3'], df_test)
print(f"V3 (당일종가 vs 당일SMA30): 수익률 {ret3:.2f}%, 샤프 {sharpe3:.4f}")

print("\n" + "="*80)
print("결론:")
print("="*80)
print("V1은 미래 정보(당일 SMA30)를 사용하여 lookahead bias 가능성")
print("V2는 전일 정보만 사용 (가장 엄격)")
print("V3는 당일 종가 기준 (가장 일반적이고 실용적)")
print("\n권장: V3 (당일 종가 > 당일 SMA30) 사용")
print("="*80)
