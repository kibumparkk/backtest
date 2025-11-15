# 암호화폐 백테스팅 프로젝트

> 터틀트레이딩, RSI 55, SMA 30 전략의 포트폴리오 백테스팅

## 🎯 프로젝트 개요

BTC, ETH, ADA, XRP 4개 암호화폐에 대해 3가지 트레이딩 전략을 적용하고, 동일 비중 포트폴리오로 구성하여 성과를 비교합니다.

### 구현된 전략
1. **Turtle Trading** (터틀트레이딩) - 20일 돌파 전략
2. **RSI 55** - RSI 지표 기반 추세 추종
3. **SMA 30** - 이동평균 교차 전략

---

## 📁 프로젝트 구조

```
backtest/
├── 📊 백테스팅 스크립트
│   └── crypto_portfolio_strategy_comparison_fixed.py  # 메인 백테스트 (✅ 편향 수정 완료)
│
├── 📈 결과 파일
│   ├── crypto_portfolio_comparison_fixed.png          # 포트폴리오 비교 차트
│   ├── crypto_portfolio_metrics_fixed.csv             # 성과 지표
│   ├── portfolio_turtle_trading_fixed.csv             # 터틀트레이딩 상세
│   ├── portfolio_rsi_55.csv                           # RSI 55 상세
│   └── portfolio_sma_30.csv                           # SMA 30 상세
│
├── 📚 문서
│   ├── README.md                                      # 이 파일
│   ├── BACKTESTING_BIAS_GUIDE.md                     # ⭐ 백테스팅 편향 완벽 가이드
│   └── BACKTESTING_CHECKLIST.md                      # ⭐ 빠른 참조 체크리스트
│
└── 📂 데이터
    └── chart_day/                                     # 일봉 데이터 (parquet)
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필수 패키지 설치
pip install pandas numpy matplotlib seaborn pyarrow
```

### 2. 백테스트 실행

```bash
# 포트폴리오 전략 비교
python crypto_portfolio_strategy_comparison_fixed.py
```

### 3. 결과 확인

실행 후 다음 파일들이 생성됩니다:
- `crypto_portfolio_comparison_fixed.png` - 시각화 차트
- `crypto_portfolio_metrics_fixed.csv` - 성과 지표
- `portfolio_*.csv` - 각 전략의 상세 데이터

---

## 📊 백테스팅 결과

### 포트폴리오 성과 (BTC, ETH, ADA, XRP 동일비중 25%)

| 순위 | 전략 | 총 수익률 | CAGR | MDD | Sharpe |
|------|------|-----------|------|-----|--------|
| 🥇 | **SMA 30** | **5,942%** | **81.85%** | -40.70% | **1.60** |
| 🥈 | Turtle Trading | 5,646% | 80.52% | **-29.83%** | 1.12 |
| 🥉 | RSI 55 | 3,142% | 66.07% | -37.74% | 1.45 |

**기간**: 2018-01-01 ~ 2025-11-07
**슬리피지**: 0.2%

### 승자
**SMA 30 전략**이 수익률과 샤프 비율 모두 최고 성능을 보였습니다.

---

## 🛡️ 백테스팅 품질 보증

### 체크된 편향들

| 편향 | 상태 | 확인 내용 |
|------|------|-----------|
| **Look-ahead Bias** | ✅ 해결 | `shift(1)` 사용, 미래 정보 차단 |
| **Perfect Execution Bias** | ✅ 해결 | 현실적 종가 체결, 슬리피지 0.2% |
| **Transaction Cost** | ✅ 반영 | 매수/매도 양방향 슬리피지 적용 |
| **Data Snooping** | ✅ 최소화 | 표준 파라미터 사용 |

### 레드 플래그 체크

모든 결과가 합리적 범위 내:
- ✅ 샤프 비율: 1.12 ~ 1.60 (< 3.0)
- ✅ 승률: 43% ~ 51% (< 70%)
- ✅ MDD: -29% ~ -40% (합리적)

---

## 📖 백테스팅 가이드

### 🔴 백테스트 전 필독!

새로운 전략을 개발하거나 이 코드를 수정하기 전에 **반드시** 다음 문서를 읽어야 합니다:

1. **[BACKTESTING_CHECKLIST.md](BACKTESTING_CHECKLIST.md)** (5분)
   - 백테스트 전 필수 체크리스트
   - 30초 빠른 체크

2. **[BACKTESTING_BIAS_GUIDE.md](BACKTESTING_BIAS_GUIDE.md)** (20분)
   - 백테스팅 4대 편향 완벽 가이드
   - Look-ahead, Perfect Execution, Survivorship, Data Snooping
   - 실제 사례와 해결 방법

### 핵심 원칙

```python
# ✅ 권장 템플릿
df['indicator'] = calculate_indicator(df['Close'])
df['signal'] = df['indicator'] > threshold
df['position'] = df['signal'].shift(1)  # 필수! 미래 정보 차단

# 수익률 계산
df['returns'] = df['position'] * df['Close'].pct_change()

# 슬리피지 적용 (필수!)
slippage_cost = pd.Series(0.0, index=df.index)
slippage_cost[df['position'].diff() == 1] = -0.002   # 매수
slippage_cost[df['position'].diff() == -1] = -0.002  # 매도
df['returns'] = df['returns'] + slippage_cost
```

### 📊 필수 시각화 요구사항

백테스트 결과는 다음 차트를 반드시 포함해야 합니다:

1. **누적 자산 곡선 (Cumulative Returns)**
   - 초기 자본: **1원**에서 시작
   - Y축 스케일: **로그 스케일 (log-y)** 사용
   - 복리 수익률 반영
   - 이유: 로그 스케일은 수익률의 비율 변화를 선형으로 표현하여 장기 성과 비교에 적합

2. **Drawdown 차트**
   - 단위: **퍼센트 (%)** 표시
   - 최고점 대비 하락폭 계산
   - MDD (Maximum Drawdown) 명시

```python
# 누적 자산 계산 (1원 시작)
df['cumulative_returns'] = (1 + df['returns']).cumprod()

# Drawdown 계산 (%)
df['cumulative_max'] = df['cumulative_returns'].cummax()
df['drawdown'] = (df['cumulative_returns'] - df['cumulative_max']) / df['cumulative_max'] * 100

# 시각화
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 누적 자산 (log-y)
axes[0].plot(df.index, df['cumulative_returns'])
axes[0].set_yscale('log')
axes[0].set_title('Cumulative Returns (Starting from 1 KRW)')
axes[0].set_ylabel('Cumulative Returns (log scale)')
axes[0].grid(True)

# Drawdown (%)
axes[1].fill_between(df.index, df['drawdown'], 0, alpha=0.3, color='red')
axes[1].set_title('Drawdown (%)')
axes[1].set_ylabel('Drawdown (%)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('backtest_results.png')
```

### 📈 성과 지표 계산

모든 백테스트는 다음 지표를 계산하고 보고해야 합니다:

```python
import numpy as np

# CAGR (Compound Annual Growth Rate)
total_days = (df.index[-1] - df.index[0]).days
years = total_days / 365.25
total_return = df['cumulative_returns'].iloc[-1] - 1
cagr = (1 + total_return) ** (1 / years) - 1

# Sharpe Ratio (연율화)
returns_mean = df['returns'].mean() * 252  # 일간 → 연간
returns_std = df['returns'].std() * np.sqrt(252)
sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0

# Maximum Drawdown
mdd = df['drawdown'].min()

# Win Rate
winning_trades = (df['returns'] > 0).sum()
total_trades = (df['returns'] != 0).sum()
win_rate = winning_trades / total_trades if total_trades > 0 else 0

print(f"CAGR: {cagr*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"MDD: {mdd:.2f}%")
print(f"Win Rate: {win_rate*100:.2f}%")
```

---

## ⚠️ 중요: Perfect Execution Bias 수정 완료

이 프로젝트의 초기 버전에서 **심각한 Perfect Execution Bias**를 발견하고 수정했습니다.

### 발견된 문제

```python
# ❌ 잘못된 코드 (91% 과대평가!)
if df['High'] > entry_high:
    buy_price = entry_high  # 불가능! 이미 돌파했는데 돌파선에 매수?
```

**결과**: BTC 터틀트레이딩 6,203% → 실제는 3,250% (과대평가 90.88%)

### 수정된 코드

```python
# ✅ 수정된 코드 (현실적)
if df['High'] > entry_high:
    buy_price = df['Close'] * (1 + slippage)  # 당일 종가 + 슬리피지
```

**현재 코드는 이 문제가 수정된 버전입니다.**

---

## 🎓 교훈 및 베스트 프랙티스

### 핵심 교훈
1. **"결과가 너무 좋으면 의심하라"**
2. **항상 shift(1) 사용**
3. **슬리피지는 필수**
4. **체결 가격은 현실적으로**

### ⚠️ 주의사항

백테스트 결과가 다음과 같이 비현실적으로 좋다면 코드를 재검토하세요:
- 샤프 비율 > 3.0
- 승률 > 70%
- MDD < 10%
- 연속 손실 0회

→ **Look-ahead bias, 슬리피지 누락 등을 체크리스트로 확인**

---

## 💻 코드 상세 설명

### 전략 구현 특징

1. **터틀 트레이딩**
   - 20일 최고가 돌파 시 매수 신호
   - 10일 최저가 하향 돌파 시 매도 신호
   - 당일 종가에 체결 + 슬리피지

2. **RSI 55**
   - RSI >= 55 시 매수/보유
   - RSI < 55 시 매도/현금
   - 전일 신호로 당일 거래

3. **SMA 30**
   - 가격 >= SMA 30 시 매수/보유
   - 가격 < SMA 30 시 매도/현금
   - 전일 신호로 당일 거래

### 포트폴리오 구성
- 각 종목 25% 동일 비중
- 리밸런싱 없음
- 슬리피지 전략별 독립 적용

---

## 📝 데이터 요구사항

### 데이터 형식
- **위치**: `chart_day/`
- **형식**: Parquet
- **컬럼**: `open`, `high`, `low`, `close`, `volume`
- **인덱스**: DatetimeIndex
- **종목**: `{SYMBOL}_KRW.parquet` (예: BTC_KRW.parquet)

### 지원 종목
- BTC_KRW (비트코인)
- ETH_KRW (이더리움)
- ADA_KRW (카르다노)
- XRP_KRW (리플)

---

## 🔬 실전 적용 가이드

### ✅ 사용 가능한 경우
- 소액 투자 (시장 충격 무시 가능)
- 페이퍼 트레이딩으로 시작
- Whipsaw 리스크 인지

### ⚠️ 추가 검증 필요
- 별도 테스트 기간 검증 (2022-2024 등)
- 파라미터 민감도 분석
- 다른 종목군에서 테스트

### ❌ 피해야 할 상황
- 대량 투자 (시장 충격 발생)
- 검증 없이 실전 투자

---

## 🤝 기여 가이드

새 전략 추가 또는 개선 시:
1. `BACKTESTING_CHECKLIST.md` 준수
2. 모든 편향 체크
3. 문서화 (전략 로직, 파라미터 근거)
4. Pull Request 제출

---

## 📞 문의 및 피드백

- Issue: GitHub Issues
- 문서 개선 제안: Pull Request 환영

---

## 📜 라이선스

MIT License

---

## 🙏 감사의 글

이 프로젝트는 다음 원칙을 따릅니다:

> **"In God we trust, all others must bring data. But verify that data first."**

백테스팅은 전략의 **가능성**을 보는 것이지 **보장**이 아닙니다.
항상 소액으로 실전 테스트를 거쳐야 합니다.

---

*최종 수정: 2025-11-07*
*버전: 2.0 (Perfect Execution Bias 수정, Fixed 버전만 유지)*
