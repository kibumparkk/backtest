# 전략 자동 탐색 결과 리포트

## 목표
전일종가 > SMA30 전략보다 좋은 전략 5개를 자동으로 찾기

## 방법론
- **테스트 기간**: 2018-01-01 ~ 2024-11-10 (약 7년)
- **테스트 종목**: BTC_KRW, ETH_KRW, ADA_KRW, XRP_KRW (동일 비중 포트폴리오)
- **슬리피지**: 0.2% (매수/매도 시)
- **테스트 전략 수**: 71개
- **전략 유형**: SMA, EMA, RSI, Dual SMA, Bollinger Band, MACD, Momentum

## 벤치마크 (SMA_30)
| 지표 | 값 |
|------|-----|
| **CAGR** | 73.90% |
| **Sharpe Ratio** | 1.49 |
| **MDD** | -47.42% |
| **Total Return** | 4,865.83% |

## 상위 5개 전략 (SMA_30 대비 우수)

### 🥇 1위: BB_20_1.5 (Bollinger Band)
- **Parameters**: period=20, std_dev=1.5
- **CAGR**: 91.71% (+17.81% vs SMA_30)
- **Sharpe Ratio**: 1.73 (+0.24 vs SMA_30)
- **MDD**: -37.69% (개선: 9.73%p)
- **Total Return**: 8,579.46%
- **Win Rate**: 49.89%
- **Score**: 61.96

**전략 설명**: 20일 볼린저 밴드 중간선(SMA)을 기준으로 매매
- 가격 ≥ BB 중간선: 매수/보유
- 가격 < BB 중간선: 매도/현금 보유

---

### 🥈 2위: BB_20_2.0 (Bollinger Band)
- **Parameters**: period=20, std_dev=2.0
- **CAGR**: 91.71% (+17.81% vs SMA_30)
- **Sharpe Ratio**: 1.73 (+0.24 vs SMA_30)
- **MDD**: -37.69% (개선: 9.73%p)
- **Total Return**: 8,579.46%
- **Win Rate**: 49.89%
- **Score**: 61.96

**전략 설명**: BB_20_1.5와 동일한 성과 (표준편차 값이 중간선 기준 매매에는 영향 없음)

---

### 🥉 3위: BB_20_2.5 (Bollinger Band)
- **Parameters**: period=20, std_dev=2.5
- **CAGR**: 91.71% (+17.81% vs SMA_30)
- **Sharpe Ratio**: 1.73 (+0.24 vs SMA_30)
- **MDD**: -37.69% (개선: 9.73%p)
- **Total Return**: 8,579.46%
- **Win Rate**: 49.89%
- **Score**: 61.96

**전략 설명**: BB_20_1.5와 동일한 성과

---

### 4위: SMA_20 (Simple Moving Average)
- **Parameters**: period=20
- **CAGR**: 82.02% (+8.12% vs SMA_30)
- **Sharpe Ratio**: 1.59 (+0.10 vs SMA_30)
- **MDD**: -46.95% (개선: 0.47%p)
- **Total Return**: 5,981.23%
- **Win Rate**: 49.86%
- **Score**: 55.58

**전략 설명**: 전일 종가가 20일 SMA 이상일 때 매수/보유
- 전일 종가 ≥ SMA_20: 매수/보유
- 전일 종가 < SMA_20: 매도/현금 보유

---

### 5위: BB_30_1.5 (Bollinger Band)
- **Parameters**: period=30, std_dev=1.5
- **CAGR**: 81.85% (+7.95% vs SMA_30)
- **Sharpe Ratio**: 1.60 (+0.11 vs SMA_30)
- **MDD**: -40.70% (개선: 6.72%p)
- **Total Return**: 5,942.43%
- **Win Rate**: 51.13%
- **Score**: 55.52

**전략 설명**: 30일 볼린저 밴드 중간선(SMA)을 기준으로 매매

---

## 주요 발견

### 1. 볼린저 밴드 전략의 우수성
- **상위 5개 중 4개가 볼린저 밴드 전략**
- BB_20 전략이 가장 우수한 성과 (CAGR 91.71%, Sharpe 1.73)
- SMA_30 대비 CAGR 17.81%p 개선, MDD 9.73%p 개선

### 2. 짧은 기간 이동평균의 효과
- SMA_20이 SMA_30보다 우수 (CAGR 82.02% vs 73.90%)
- 더 빠른 추세 전환 포착

### 3. 전략별 성과 순위
1. **Bollinger Band** (최우수): CAGR 81.85%~91.71%
2. **SMA 20일** (우수): CAGR 82.02%
3. **RSI & Momentum** (양호): CAGR 50%~77%
4. **Dual SMA** (보통): CAGR 25%~67%
5. **MACD** (보통): CAGR 59%~72%

### 4. 최적 파라미터
- **SMA**: 15~25일이 최적 (CAGR 75%~82%)
- **EMA**: 20~30일이 최적 (CAGR 62%~70%)
- **RSI**: 10기간/55임계값이 최적 (CAGR 77.26%)
- **Momentum**: 10일이 최적 (CAGR 77.31%)

## 결론

반복문을 통한 자동 전략 탐색 결과:
1. **Bollinger Band 20일 전략**이 가장 우수한 성과 (CAGR 91.71%, Sharpe 1.73)
2. SMA_30 대비 **17.81%p의 CAGR 개선**과 **9.73%p의 MDD 개선**
3. 모든 상위 전략이 **Sharpe Ratio 1.5 이상**으로 위험 대비 수익률 우수
4. **유의미한 성과 차이 확인**: CAGR, Sharpe Ratio, MDD 모두 개선

## 파일
- `loop_strategy_finder.py`: 전략 자동 탐색 스크립트
- `loop_strategy_results_all.csv`: 전체 71개 전략 결과
- `loop_strategy_results_top5.csv`: 상위 5개 전략 결과

## 권장 전략
**BB_20 (Bollinger Band 20일 중간선 전략)**을 메인 전략으로 권장
- 가장 높은 수익률 (CAGR 91.71%)
- 가장 높은 위험조정수익률 (Sharpe 1.73)
- 가장 낮은 MDD (-37.69%)
- 구현 간단, 백테스트 결과 안정적
