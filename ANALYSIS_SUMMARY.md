# 암호화폐 포트폴리오 전략 비교 분석 - 문제점 발견 및 수정

## 🔍 문제 발견

초기 터틀트레이딩 전략 결과가 압도적으로 좋았음:
- **Turtle Trading Portfolio**: 13,895% 수익률, -22.26% MDD
- **SMA 30 Portfolio**: 5,942% 수익률, -40.70% MDD
- **RSI 55 Portfolio**: 3,142% 수익률, -37.74% MDD

## ⚠️ 발견된 문제점

### Perfect Execution Bias (완벽한 체결 가격 편향)

터틀트레이딩 전략이 **비현실적인 체결 가격**을 사용하고 있었음:

#### 1. 비현실적인 매수 가격
```python
# ❌ 잘못된 코드
if df.iloc[i]['High'] > df.iloc[i]['entry_high']:
    buy_price = df.iloc[i]['entry_high']  # 불가능!
```

**문제점**:
- 20일 최고가(entry_high)를 "돌파"했다는 것은 가격이 이미 entry_high보다 높다는 의미
- entry_high 가격에 매수하는 것은 불가능
- 실제로는 돌파 시점의 가격(종가 등)에 매수해야 함

**실제 사례**:
```
날짜: 2018-02-20
  20일 최고가 (entry_high): 12,755,000 KRW
  당일 종가: 13,548,000 KRW
  ❌ 현재 코드: 12,755,000 KRW에 매수 (불가능!)
  ✅ 현실적 가격: 13,548,000 KRW에 매수
  📊 가격 차이: -5.85%
```

#### 2. 비현실적인 매도 가격
```python
# ❌ 잘못된 코드
if df.iloc[i]['Low'] < df.iloc[i]['exit_low']:
    sell_price = df.iloc[i]['exit_low']  # 불가능!
```

**문제점**:
- 10일 최저가(exit_low)를 "하향 돌파"했다는 것은 가격이 이미 exit_low보다 낮다는 의미
- exit_low 가격에 매도하는 것은 불가능
- 실제로는 하향 돌파 시점의 가격(종가 등)에 매도해야 함

## ✅ 수정 사항

### 수정된 코드
```python
# ✅ 수정된 코드
if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
    # 매수: 당일 종가에 매수 (슬리피지 포함)
    buy_price = df.iloc[i]['Close'] * (1 + slippage)

elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
    # 매도: 당일 종가에 매도 (슬리피지 포함)
    sell_price = df.iloc[i]['Close'] * (1 - slippage)
```

## 📊 수정 전후 비교

### BTC 단일 종목 (터틀트레이딩)
| 지표 | 수정 전 (비현실적) | 수정 후 (현실적) | 차이 |
|------|-------------------|-----------------|------|
| 총 수익률 | 6,203% | 3,250% | **-2,953%p** |
| MDD | -19.20% | -31.88% | -12.68%p |
| 과대평가 비율 | - | - | **90.88%** |

### 포트폴리오 비교 (BTC, ETH, ADA, XRP 동일비중)

#### 수정 전 (비현실적)
| 전략 | 총 수익률 | CAGR | MDD | Sharpe |
|------|-----------|------|-----|--------|
| **Turtle Trading** | **13,895%** ⭐ | **105.54%** | **-22.26%** ✅ | 1.28 |
| SMA 30 | 5,942% | 81.85% | -40.70% | **1.60** ⭐ |
| RSI 55 | 3,142% | 66.07% | -37.74% | 1.45 |

#### 수정 후 (현실적) ✅
| 전략 | 총 수익률 | CAGR | MDD | Sharpe |
|------|-----------|------|-----|--------|
| **SMA 30** | **5,942%** ⭐ | **81.85%** | -40.70% | **1.60** ⭐ |
| Turtle Trading | 5,646% | 80.52% | **-29.83%** ✅ | 1.12 |
| RSI 55 | 3,142% | 66.07% | -37.74% | 1.45 |

## 🎯 결론

### 왜 RSI 55와 SMA 30은 문제가 없었나?

**RSI 55 전략:**
- 종가 기준으로 RSI 계산
- 신호 판단: 종가 시점의 RSI 값 사용
- 체결 가격: 다음날 종가 (shift(1) 사용)
- ✅ **Look-ahead bias 없음**

**SMA 30 전략:**
- 종가 기준으로 SMA 계산
- 신호 판단: 종가가 SMA보다 높은지 확인
- 체결 가격: 다음날 종가 (shift(1) 사용)
- ✅ **Look-ahead bias 없음**

**터틀트레이딩 (기존):**
- 20일 최고가 계산 (shift(1) 사용 - 이 부분은 OK)
- 신호 판단: 당일 고가가 entry_high 돌파
- ❌ **체결 가격: entry_high (불가능!)**
- ❌ **실제로는 종가 또는 평균가 사용해야 함**

### 최종 권장사항

수정 후 결과를 기준으로:

1. **최고 수익률**: SMA 30 (5,942%)
2. **최저 MDD**: Turtle Trading (-29.83%)
3. **최고 Sharpe**: SMA 30 (1.60)

**종합 평가**: **SMA 30 전략**이 수익률과 위험조정수익률(Sharpe) 모두에서 가장 우수

## 📁 생성된 파일

### 분석 파일
- `strategy_review_analysis.py` - 문제점 상세 분석 스크립트
- `turtle_trading_issue_analysis.png` - 수정 전후 비교 차트

### 수정 전 (비현실적)
- `crypto_portfolio_strategy_comparison.py`
- `crypto_portfolio_comparison.png`
- `crypto_portfolio_metrics.csv`

### 수정 후 (현실적) ✅ 권장
- `crypto_portfolio_strategy_comparison_fixed.py`
- `crypto_portfolio_comparison_fixed.png`
- `crypto_portfolio_metrics_fixed.csv`

## 💡 교훈

백테스팅에서 주의해야 할 편향(Bias):

1. **Look-ahead Bias**: 미래 정보를 사용하는 문제
2. **Perfect Execution Bias**: 최적의 가격에 항상 체결된다고 가정 ⚠️ (이번에 발견된 문제)
3. **Survivorship Bias**: 생존한 종목만 분석
4. **Data Snooping Bias**: 과적합

**결과가 너무 좋으면 의심하라!**
