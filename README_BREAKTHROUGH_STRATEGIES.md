# 10가지 유명한 돌파 전략 백테스트 비교

BTC 데이터를 활용한 10가지 유명한 돌파(Breakthrough) 전략의 성과 비교 분석

## 구현된 전략

1. **Donchian Channel Breakout** - 돈치안 채널 돌파 전략
2. **Volatility Breakout** - 래리 윌리엄스의 변동성 돌파 전략
3. **Range Breakout** - 레인지 돌파 전략
4. **Opening Range Breakout** - 시가 레인지 돌파 전략
5. **ATR Breakout** - ATR 기반 변동성 돌파 전략
6. **Turtle Trading** - 터틀 트레이딩 전략
7. **Bollinger Band Breakout** - 볼린저 밴드 돌파 전략
8. **High/Low Breakout** - 고/저가 돌파 전략
9. **Momentum Breakout** - 모멘텀 돌파 전략
10. **Keltner Channel Breakout** - 켈트너 채널 돌파 전략

## 백테스트 설정

- **종목**: BTC_KRW (비트코인)
- **기간**: 2018-01-01 ~ 2025-11-05 (약 7.8년)
- **슬리피지**: 0.2% (모든 거래에 반영)
- **데이터**: 일봉 (Daily)

## 성과 요약

### 최고 성과 전략 (총 수익률 기준)

| 순위 | 전략 | 총 수익률 | CAGR | MDD | 샤프 비율 | 승률 |
|------|------|-----------|------|-----|-----------|------|
| 1 | High/Low Breakout | 8,208% | 75.67% | -33.49% | 1.16 | 43.15% |
| 2 | Donchian Channel | 6,349% | 70.09% | -31.70% | 0.82 | 55.26% |
| 3 | Turtle Trading | 6,203% | 69.60% | -19.20% | 0.93 | 57.69% |
| 4 | Bollinger Band | 6,006% | 68.91% | -32.97% | 0.78 | 45.45% |
| 5 | Keltner Channel | 3,814% | 59.60% | -40.17% | 0.57 | 41.67% |

### 안정성 기준 (샤프 비율)

| 순위 | 전략 | 샤프 비율 | CAGR | MDD |
|------|------|-----------|------|-----|
| 1 | High/Low Breakout | 1.16 | 75.67% | -33.49% |
| 2 | Turtle Trading | 0.93 | 69.60% | -19.20% |
| 3 | Momentum Breakout | 0.91 | 27.17% | -47.87% |
| 4 | Volatility Breakout | 0.88 | 24.39% | -58.56% |
| 5 | Donchian Channel | 0.82 | 70.09% | -31.70% |

### 리스크 관리 (MDD 기준)

| 순위 | 전략 | MDD | CAGR | 샤프 비율 |
|------|------|-----|------|-----------|
| 1 | ATR Breakout | -7.77% | 6.62% | 0.77 |
| 2 | Turtle Trading | -19.20% | 69.60% | 0.93 |
| 3 | Donchian Channel | -31.70% | 70.09% | 0.82 |
| 4 | Bollinger Band | -32.97% | 68.91% | 0.78 |
| 5 | High/Low Breakout | -33.49% | 75.67% | 1.16 |

## 주요 인사이트

### 1. 최고 성과 전략

**High/Low Breakout** 전략이 8,208%의 총 수익률과 1.16의 샤프 비율로 가장 우수한 성과를 보였습니다.
- CAGR: 75.67% (연평균 수익률)
- MDD: -33.49% (허용 가능한 수준)
- 거래 횟수: 146회 (적절한 거래 빈도)

### 2. 위험 조정 수익률

**Turtle Trading**은 샤프 비율 0.93과 MDD -19.20%로 위험 대비 수익이 가장 안정적입니다.
- 높은 승률: 57.69%
- 적은 거래 횟수: 52회
- 우수한 Profit Factor: 6.24

### 3. 실패한 전략

**Range Breakout** 전략은 거래 신호가 전혀 발생하지 않았습니다.
- 원인: BTC의 높은 변동성으로 인해 횡보 구간이 거의 없음
- 개선 방안: 임계값(threshold) 조정 필요

### 4. 거래 빈도 분석

- **고빈도**: Volatility Breakout (1,139회), Opening Range (1,512회)
- **저빈도**: Turtle Trading (52회), Donchian Channel (38회)
- **결론**: 저빈도 전략이 더 높은 수익률과 낮은 MDD를 보임

### 5. 승률 vs 수익률

승률이 높다고 반드시 수익률이 높은 것은 아닙니다:
- High/Low Breakout: 승률 43.15% → 수익률 8,208%
- Opening Range: 승률 45.70% → 수익률 307%
- **결론**: 평균 수익/손실 비율(Risk/Reward)이 더 중요

## 사용 방법

### 1. 기본 실행

```bash
python breakthrough_strategies_comparison.py
```

### 2. 결과 파일

- `breakthrough_strategies_comparison.png` - 전략 비교 차트
- `breakthrough_strategies_metrics.csv` - 성과 지표 요약
- `strategy_*.csv` - 각 전략의 상세 거래 내역

### 3. 코드 구조

```python
from breakthrough_strategies_comparison import BreakthroughStrategiesBacktest

# 백테스트 인스턴스 생성
backtest = BreakthroughStrategiesBacktest(
    symbol='BTC_KRW',
    start_date='2018-01-01',
    end_date=None,
    slippage=0.002  # 0.2%
)

# 데이터 로드
backtest.load_data()

# 모든 전략 실행
metrics_df = backtest.run_all_strategies()

# 시각화
backtest.plot_comparison(metrics_df)
```

## 전략별 상세 설명

### 1. Donchian Channel Breakout
- N일 최고가/최저가 채널을 이용한 돌파 전략
- 매수: N일 최고가 돌파
- 매도: N일 최저가 하향 돌파
- 파라미터: period=20

### 2. Volatility Breakout
- 래리 윌리엄스의 변동성 돌파 전략
- 목표가 = 시가 + (전일 고가 - 전일 저가) × k
- 당일 청산 전략
- 파라미터: k=0.5

### 3. Range Breakout
- 횡보장에서 레인지 상단/하단 돌파 전략
- 레인지 폭이 좁을 때만 신호 발생
- 파라미터: period=10, threshold=0.02

### 4. Opening Range Breakout
- 평균 시가 대비 현재 시가 비교
- 시가가 높으면 매수, 당일 청산
- 파라미터: lookback=5

### 5. ATR Breakout
- ATR(Average True Range) 기반 변동성 돌파
- 목표가 = 시가 + ATR × multiplier
- 파라미터: period=14, multiplier=2

### 6. Turtle Trading
- 전설적인 터틀 트레이더의 전략
- 매수: N일 최고가 돌파
- 매도: M일 최저가 하향 돌파
- 파라미터: entry_period=20, exit_period=10

### 7. Bollinger Band Breakout
- 볼린저 밴드 상단/하단 돌파 전략
- 상단 돌파 시 매수, 하단 돌파 시 매도
- 파라미터: period=20, std_dev=2

### 8. High/Low Breakout
- 단순 N일 최고가/최저가 돌파 전략
- 빠른 진입/청산
- 파라미터: period=5

### 9. Momentum Breakout
- 모멘텀(N일 수익률) 기반 돌파 전략
- 강한 상승 모멘텀 발생 시 매수
- 파라미터: period=10, threshold=0.05

### 10. Keltner Channel Breakout
- EMA와 ATR을 결합한 채널 전략
- 채널 = EMA ± (ATR × multiplier)
- 파라미터: period=20, atr_period=10, multiplier=2

## 파라미터 최적화

현재 구현은 각 전략의 전통적인 파라미터를 사용합니다. 더 나은 성과를 위해서는:

1. **그리드 서치**: 파라미터 조합을 체계적으로 테스트
2. **워크 포워드 분석**: 과적합 방지를 위한 검증
3. **다양한 시장 환경**: 상승장/하락장/횡보장 구분 분석
4. **동적 파라미터**: 변동성에 따라 파라미터 자동 조정

## 주의사항

1. **과거 성과 ≠ 미래 수익**: 백테스트 결과가 미래를 보장하지 않습니다
2. **슬리피지**: 실제 거래에서는 0.2% 이상의 슬리피지가 발생할 수 있습니다
3. **거래 비용**: 거래소 수수료가 추가로 발생합니다
4. **시장 충격**: 대량 거래 시 가격에 영향을 줄 수 있습니다
5. **과적합 위험**: 파라미터 최적화 시 과적합 주의

## 향후 개선 방향

1. **포지션 사이징**: 켈리 기준, 고정 비율 등 적용
2. **리스크 관리**: 손절매, 익절매 로직 추가
3. **다중 시간프레임**: 일봉 외 다른 시간프레임 분석
4. **기계학습**: 전략 조합 최적화
5. **실시간 거래**: API 연동 및 자동 매매

## 라이선스

MIT License

## 참고 자료

- Larry Williams - Volatility Breakout
- Richard Dennis - Turtle Trading
- John Bollinger - Bollinger Bands
- Chester Keltner - Keltner Channel
- Richard Donchian - Donchian Channel
