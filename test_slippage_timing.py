"""
슬리피지 타이밍 오류 검증 스크립트

현재 코드의 문제:
- position.shift(1)로 전일 포지션 사용
- 하지만 slippage는 position_change (shift 안 됨)에 적용
- 결과: 슬리피지가 1일 일찍 적용됨
"""

import pandas as pd
import numpy as np

# 샘플 데이터 생성
data = {
    'Date': pd.date_range('2024-01-01', periods=10),
    'Close': [100, 105, 110, 108, 112, 115, 113, 118, 120, 119],
}
df = pd.DataFrame(data).set_index('Date')

# SMA 계산
df['SMA'] = df['Close'].rolling(window=3).mean()
df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)

print("="*80)
print("샘플 데이터 및 포지션")
print("="*80)
print(df[['Close', 'SMA', 'position']])

# 현재 방식 (잘못된 방식)
print("\n" + "="*80)
print("현재 코드 방식 (슬리피지 타이밍 오류)")
print("="*80)

df_wrong = df.copy()
df_wrong['position_change'] = df_wrong['position'].diff()
df_wrong['daily_return'] = df_wrong['Close'].pct_change()
df_wrong['returns'] = df_wrong['position'].shift(1) * df_wrong['daily_return']

slippage = 0.002
slippage_cost_wrong = pd.Series(0.0, index=df_wrong.index)
slippage_cost_wrong[df_wrong['position_change'] == 1] = -slippage
slippage_cost_wrong[df_wrong['position_change'] == -1] = -slippage

df_wrong['slippage_cost'] = slippage_cost_wrong
df_wrong['returns_with_slippage'] = df_wrong['returns'] + df_wrong['slippage_cost']
df_wrong['cumulative'] = (1 + df_wrong['returns_with_slippage'].fillna(0)).cumprod()

print(df_wrong[['position', 'position_change', 'daily_return', 'returns', 'slippage_cost', 'returns_with_slippage', 'cumulative']])

# 올바른 방식
print("\n" + "="*80)
print("올바른 방식 (슬리피지 타이밍 수정)")
print("="*80)

df_correct = df.copy()
df_correct['position_shifted'] = df_correct['position'].shift(1)
df_correct['position_change_shifted'] = df_correct['position_shifted'].diff()
df_correct['daily_return'] = df_correct['Close'].pct_change()
df_correct['returns'] = df_correct['position_shifted'] * df_correct['daily_return']

slippage_cost_correct = pd.Series(0.0, index=df_correct.index)
slippage_cost_correct[df_correct['position_change_shifted'] == 1] = -slippage
slippage_cost_correct[df_correct['position_change_shifted'] == -1] = -slippage

df_correct['slippage_cost'] = slippage_cost_correct
df_correct['returns_with_slippage'] = df_correct['returns'] + df_correct['slippage_cost']
df_correct['cumulative'] = (1 + df_correct['returns_with_slippage'].fillna(0)).cumprod()

print(df_correct[['position', 'position_shifted', 'position_change_shifted', 'daily_return', 'returns', 'slippage_cost', 'returns_with_slippage', 'cumulative']])

# 비교
print("\n" + "="*80)
print("최종 성과 비교")
print("="*80)
print(f"현재 방식 (잘못됨) - 최종 누적 수익률: {df_wrong['cumulative'].iloc[-1]:.6f}")
print(f"올바른 방식 - 최종 누적 수익률: {df_correct['cumulative'].iloc[-1]:.6f}")
print(f"차이: {(df_correct['cumulative'].iloc[-1] - df_wrong['cumulative'].iloc[-1]):.6f}")
print(f"차이 (%): {((df_correct['cumulative'].iloc[-1] / df_wrong['cumulative'].iloc[-1] - 1) * 100):.4f}%")
