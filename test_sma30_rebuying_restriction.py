"""
SMA30 전략의 매도 후 재매수 제한 로직 테스트
"""

import pandas as pd
import numpy as np
from datetime import datetime
from crypto_portfolio_strategy_comparison_fixed import CryptoPortfolioComparisonFixed


def test_sma30_rebuying_restriction():
    """SMA30 전략에서 매도 후 7일간 재매수 금지가 정상 작동하는지 테스트"""

    print("\n" + "="*80)
    print("SMA30 재매수 제한 로직 테스트")
    print("="*80 + "\n")

    # 테스트용 백테스트 인스턴스 생성 (짧은 기간)
    comparison = CryptoPortfolioComparisonFixed(
        symbols=['BTC_KRW'],  # 빠른 테스트를 위해 BTC만 사용
        start_date='2023-01-01',
        end_date='2023-12-31',
        slippage=0.002
    )

    # 데이터 로드
    print("데이터 로딩 중...")
    comparison.load_data()

    # SMA30 전략 실행
    print("\nSMA30 전략 실행 중...")
    df = comparison.data['BTC_KRW'].copy()
    result = comparison.strategy_sma_30(df, sma_period=30, rebuying_restriction_days=7)

    # 매도 후 재매수가 발생한 경우 찾기
    print("\n" + "-"*80)
    print("매도 후 재매수 이벤트 분석")
    print("-"*80)

    # 포지션 변화 분석
    position_changes = result[result['position_change'] != 0].copy()

    sell_signals = position_changes[position_changes['position_change'] == -1]
    buy_signals = position_changes[position_changes['position_change'] == 1]

    print(f"\n총 매도 신호: {len(sell_signals)}개")
    print(f"총 매수 신호: {len(buy_signals)}개")

    # 매도 후 재매수 간격 확인
    if len(sell_signals) > 0:
        print("\n매도 후 다음 매수까지의 일수:")
        print("-" * 60)

        sell_dates = sell_signals.index.tolist()
        buy_dates = buy_signals.index.tolist()

        for i, sell_date in enumerate(sell_dates):
            # 해당 매도 이후의 첫 매수 찾기
            next_buys = [buy_date for buy_date in buy_dates if buy_date > sell_date]

            if next_buys:
                next_buy = next_buys[0]
                days_diff = (next_buy - sell_date).days

                status = "✅ OK (>= 7일)" if days_diff >= 7 else "❌ ERROR (< 7일)"
                print(f"{i+1}. 매도: {sell_date.strftime('%Y-%m-%d')} → "
                      f"재매수: {next_buy.strftime('%Y-%m-%d')} "
                      f"(간격: {days_diff}일) {status}")

    # 전체 통계
    print("\n" + "-"*80)
    print("전략 성과 요약")
    print("-"*80)

    total_return = (result['cumulative'].iloc[-1] - 1) * 100
    print(f"총 수익률: {total_return:.2f}%")

    # 포지션 보유 기간 분석
    position_days = result['position'].sum()
    total_days = len(result)
    position_ratio = position_days / total_days * 100

    print(f"포지션 보유 일수: {position_days}/{total_days}일 ({position_ratio:.1f}%)")

    print("\n" + "="*80)
    print("테스트 완료!")
    print("="*80 + "\n")

    return result


if __name__ == "__main__":
    result = test_sma30_rebuying_restriction()
