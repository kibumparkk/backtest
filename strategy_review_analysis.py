"""
전략 검토 및 문제점 분석

터틀트레이딩 전략의 잠재적 문제점을 검토하고 상세 거래 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def analyze_turtle_trades():
    """터틀트레이딩 거래 상세 분석"""

    # BTC 데이터 로드
    df = pd.read_parquet('chart_day/BTC_KRW.parquet')
    df.columns = [col.capitalize() for col in df.columns]
    df = df[(df.index >= '2018-01-01') & (df.index <= '2025-11-07')]

    print("="*100)
    print("터틀트레이딩 전략 문제점 분석")
    print("="*100)

    # 터틀 채널 계산
    entry_period = 20
    exit_period = 10
    df['entry_high'] = df['High'].rolling(window=entry_period).max().shift(1)
    df['exit_low'] = df['Low'].rolling(window=exit_period).min().shift(1)

    # 포지션 관리
    df['position'] = 0
    for i in range(1, len(df)):
        df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

        if df.iloc[i]['High'] > df.iloc[i]['entry_high'] and df.iloc[i-1]['position'] == 0:
            df.iloc[i, df.columns.get_loc('position')] = 1
        elif df.iloc[i]['Low'] < df.iloc[i]['exit_low'] and df.iloc[i-1]['position'] == 1:
            df.iloc[i, df.columns.get_loc('position')] = 0

    # 거래 시점 찾기
    df['position_change'] = df['position'].diff()
    buy_dates = df[df['position_change'] == 1].index[:10]  # 처음 10개 매수

    print("\n" + "="*100)
    print("문제점 1: 비현실적인 매수 가격")
    print("="*100)
    print("\n현재 코드는 '20일 최고가(entry_high)'를 돌파했을 때, entry_high 가격에 매수한다고 가정합니다.")
    print("하지만 이것은 불가능합니다! 가격이 entry_high를 '돌파'했다는 것은 이미 그보다 높다는 의미입니다.\n")

    for date in buy_dates[:5]:
        idx = df.index.get_loc(date)
        entry_high = df.iloc[idx]['entry_high']
        open_price = df.iloc[idx]['Open']
        high_price = df.iloc[idx]['High']
        low_price = df.iloc[idx]['Low']
        close_price = df.iloc[idx]['Close']

        print(f"\n날짜: {date.strftime('%Y-%m-%d')}")
        print(f"  20일 최고가 (entry_high): {entry_high:,.0f} KRW")
        print(f"  당일 시가: {open_price:,.0f} KRW")
        print(f"  당일 고가: {high_price:,.0f} KRW")
        print(f"  당일 저가: {low_price:,.0f} KRW")
        print(f"  당일 종가: {close_price:,.0f} KRW")
        print(f"  ❌ 현재 코드: {entry_high:,.0f} KRW에 매수 (불가능!)")
        print(f"  ✅ 현실적 가격: {close_price:,.0f} KRW에 매수 (종가 또는 그 이상)")
        print(f"  📊 가격 차이: {((entry_high - close_price) / close_price * 100):.2f}%")

    # 매도 분석
    sell_dates = df[df['position_change'] == -1].index[:10]

    print("\n" + "="*100)
    print("문제점 2: 비현실적인 매도 가격")
    print("="*100)
    print("\n현재 코드는 '10일 최저가(exit_low)'를 하향 돌파했을 때, exit_low 가격에 매도한다고 가정합니다.")
    print("하지만 이것도 불가능합니다! 가격이 exit_low를 '하향 돌파'했다는 것은 이미 그보다 낮다는 의미입니다.\n")

    for date in sell_dates[:5]:
        idx = df.index.get_loc(date)
        exit_low = df.iloc[idx]['exit_low']
        open_price = df.iloc[idx]['Open']
        high_price = df.iloc[idx]['High']
        low_price = df.iloc[idx]['Low']
        close_price = df.iloc[idx]['Close']

        print(f"\n날짜: {date.strftime('%Y-%m-%d')}")
        print(f"  10일 최저가 (exit_low): {exit_low:,.0f} KRW")
        print(f"  당일 시가: {open_price:,.0f} KRW")
        print(f"  당일 고가: {high_price:,.0f} KRW")
        print(f"  당일 저가: {low_price:,.0f} KRW")
        print(f"  당일 종가: {close_price:,.0f} KRW")
        print(f"  ❌ 현재 코드: {exit_low:,.0f} KRW에 매도 (불가능!)")
        print(f"  ✅ 현실적 가격: {close_price:,.0f} KRW에 매도 (종가 또는 그 이하)")
        print(f"  📊 가격 차이: {((exit_low - close_price) / close_price * 100):.2f}%")

    # 수익률 비교
    print("\n" + "="*100)
    print("수익률 비교: 현재 코드 vs 현실적인 가격")
    print("="*100)

    # 현재 코드의 수익률 계산
    df['returns_wrong'] = 0.0
    df['buy_price_wrong'] = np.nan
    slippage = 0.002

    for i in range(1, len(df)):
        if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
            df.iloc[i, df.columns.get_loc('buy_price_wrong')] = df.iloc[i]['entry_high']
        elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
            buy_price = df.iloc[i-1]['buy_price_wrong'] if pd.notna(df.iloc[i-1]['buy_price_wrong']) else df.iloc[i-1]['Close']
            df.iloc[i, df.columns.get_loc('returns_wrong')] = (df.iloc[i]['exit_low'] / buy_price - 1) - slippage
        elif df.iloc[i]['position'] == 1:
            if pd.notna(df.iloc[i-1]['buy_price_wrong']):
                df.iloc[i, df.columns.get_loc('buy_price_wrong')] = df.iloc[i-1]['buy_price_wrong']

    df['cumulative_wrong'] = (1 + df['returns_wrong']).cumprod()

    # 현실적인 가격의 수익률 계산 (종가 사용)
    df['returns_correct'] = 0.0
    df['buy_price_correct'] = np.nan

    for i in range(1, len(df)):
        if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
            # 매수: 당일 종가에 매수 (슬리피지 포함)
            df.iloc[i, df.columns.get_loc('buy_price_correct')] = df.iloc[i]['Close'] * (1 + slippage)
        elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
            # 매도: 당일 종가에 매도 (슬리피지 포함)
            buy_price = df.iloc[i-1]['buy_price_correct'] if pd.notna(df.iloc[i-1]['buy_price_correct']) else df.iloc[i-1]['Close']
            df.iloc[i, df.columns.get_loc('returns_correct')] = (df.iloc[i]['Close'] * (1 - slippage) / buy_price - 1)
        elif df.iloc[i]['position'] == 1:
            if pd.notna(df.iloc[i-1]['buy_price_correct']):
                df.iloc[i, df.columns.get_loc('buy_price_correct')] = df.iloc[i-1]['buy_price_correct']

    df['cumulative_correct'] = (1 + df['returns_correct']).cumprod()

    # 결과 비교
    total_return_wrong = (df['cumulative_wrong'].iloc[-1] - 1) * 100
    total_return_correct = (df['cumulative_correct'].iloc[-1] - 1) * 100

    # MDD 계산
    cummax_wrong = df['cumulative_wrong'].cummax()
    drawdown_wrong = (df['cumulative_wrong'] - cummax_wrong) / cummax_wrong
    mdd_wrong = drawdown_wrong.min() * 100

    cummax_correct = df['cumulative_correct'].cummax()
    drawdown_correct = (df['cumulative_correct'] - cummax_correct) / cummax_correct
    mdd_correct = drawdown_correct.min() * 100

    print(f"\n❌ 현재 코드 (비현실적):")
    print(f"   총 수익률: {total_return_wrong:.2f}%")
    print(f"   MDD: {mdd_wrong:.2f}%")

    print(f"\n✅ 수정된 코드 (현실적):")
    print(f"   총 수익률: {total_return_correct:.2f}%")
    print(f"   MDD: {mdd_correct:.2f}%")

    print(f"\n📊 차이:")
    print(f"   수익률 차이: {total_return_wrong - total_return_correct:.2f}%p")
    print(f"   과대평가 비율: {((total_return_wrong / total_return_correct - 1) * 100):.2f}%")

    # 시각화
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 누적 수익률 비교
    axes[0].plot(df.index, df['cumulative_wrong'], label='비현실적 가격 (현재 코드)', linewidth=2, color='red', alpha=0.7)
    axes[0].plot(df.index, df['cumulative_correct'], label='현실적 가격 (종가 사용)', linewidth=2, color='green', alpha=0.7)
    axes[0].set_title('Turtle Trading: 비현실적 vs 현실적 체결 가격 비교', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # 드로우다운 비교
    axes[1].fill_between(df.index, drawdown_wrong * 100, 0, alpha=0.3, color='red', label='비현실적 가격')
    axes[1].plot(df.index, drawdown_correct * 100, color='green', linewidth=2, alpha=0.7, label='현실적 가격')
    axes[1].set_title('Drawdown 비교', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('turtle_trading_issue_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n차트 저장: turtle_trading_issue_analysis.png")

    # RSI 55, SMA 30과 비교
    print("\n" + "="*100)
    print("왜 RSI 55와 SMA 30은 문제가 없는가?")
    print("="*100)

    print("\nRSI 55 전략:")
    print("  - 종가 기준으로 RSI 계산")
    print("  - 신호 판단: 종가 시점의 RSI 값 사용")
    print("  - 체결 가격: 다음날 종가 (shift(1) 사용)")
    print("  - ✅ Look-ahead bias 없음")

    print("\nSMA 30 전략:")
    print("  - 종가 기준으로 SMA 계산")
    print("  - 신호 판단: 종가가 SMA보다 높은지 확인")
    print("  - 체결 가격: 다음날 종가 (shift(1) 사용)")
    print("  - ✅ Look-ahead bias 없음")

    print("\n터틀트레이딩 (현재 코드):")
    print("  - 20일 최고가 계산 (shift(1) 사용 - 이 부분은 OK)")
    print("  - 신호 판단: 당일 고가가 entry_high 돌파")
    print("  - ❌ 체결 가격: entry_high (불가능! 이미 돌파했으므로 더 높은 가격)")
    print("  - ❌ 실제로는 최소한 종가 또는 평균가를 사용해야 함")

    print("\n" + "="*100)
    print("결론")
    print("="*100)
    print("""
터틀트레이딩 전략의 과대평가된 성과는 다음 두 가지 문제 때문입니다:

1. **비현실적인 매수 가격**:
   - 20일 최고가를 '돌파'했는데 20일 최고가에 매수
   - 실제로는 돌파 시점의 가격(종가 등)에 매수해야 함

2. **비현실적인 매도 가격**:
   - 10일 최저가를 '하향 돌파'했는데 10일 최저가에 매도
   - 실제로는 하향 돌파 시점의 가격(종가 등)에 매도해야 함

이는 **Perfect Execution Bias**의 일종으로, 최적의 가격에 항상 체결된다고
가정하는 비현실적인 백테스트 오류입니다.

수정 방법:
- 매수/매도 체결 가격을 당일 종가 또는 다음날 시가로 변경
- 슬리피지를 더 보수적으로 적용
    """)

    print("="*100 + "\n")

    return df


if __name__ == "__main__":
    analyze_turtle_trades()
