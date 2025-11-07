# 암호화폐 백테스팅 프로젝트

> 다양한 트레이딩 전략을 백테스팅하고 비교하는 프로젝트

## 📁 프로젝트 구조

```
backtest/
├── 📊 전략 스크립트
│   ├── breakthrough_strategies_comparison.py    # 10가지 돌파 전략 비교
│   ├── rsi_55_backtest.py                      # RSI 55 전략
│   ├── sma_crossover_backtest.py               # SMA 교차 전략
│   ├── volatility_breakout_backtest.py         # 변동성 돌파 전략
│   ├── crypto_portfolio_strategy_comparison.py  # 포트폴리오 비교 (초기 버전)
│   └── crypto_portfolio_strategy_comparison_fixed.py ✅ # 수정된 버전
│
├── 📈 분석 스크립트
│   └── strategy_review_analysis.py             # 전략 문제점 분석
│
├── 📚 문서
│   ├── README.md                               # 이 파일
│   ├── BACKTESTING_BIAS_GUIDE.md              # ⭐ 백테스팅 편향 완벽 가이드
│   ├── BACKTESTING_CHECKLIST.md               # ⭐ 빠른 참조 체크리스트
│   ├── ANALYSIS_SUMMARY.md                     # 터틀트레이딩 문제 분석 요약
│   └── VISUALIZATION_GUIDE.md                  # 시각화 가이드
│
└── 📂 데이터
    └── chart_day/                              # 일봉 데이터 (parquet)
```

---

## 🚀 빠른 시작

### 1. 권장 스크립트 실행 (편향 수정 완료)

```bash
# 포트폴리오 전략 비교 (✅ 권장)
python crypto_portfolio_strategy_comparison_fixed.py

# 개별 전략 실행
python rsi_55_backtest.py
python sma_crossover_backtest.py
python volatility_breakout_backtest.py
```

### 2. 자신의 전략 개발 전 필독

**⚠️ 백테스팅 전에 반드시 읽어야 할 문서:**

1. **[BACKTESTING_CHECKLIST.md](BACKTESTING_CHECKLIST.md)** (5분 소요)
   - 백테스트 전 필수 체크리스트
   - 빠른 참조용

2. **[BACKTESTING_BIAS_GUIDE.md](BACKTESTING_BIAS_GUIDE.md)** (20분 소요)
   - 백테스팅에서 흔한 편향 4가지 완벽 가이드
   - 실제 사례와 해결 방법 포함

---

## ⚠️ 중요: Perfect Execution Bias 발견 및 수정

### 발견된 문제

초기 터틀트레이딩 전략에서 **심각한 Perfect Execution Bias** 발견:

```python
# ❌ 잘못된 코드 (91% 과대평가!)
if df['High'] > entry_high:
    buy_price = entry_high  # 불가능! 이미 돌파했는데 돌파선에 매수?
```

**결과**:
- BTC 터틀트레이딩: 6,203% → 3,250% (실제 성능은 절반)
- 과대평가 비율: **90.88%**

### 수정된 코드

```python
# ✅ 수정된 코드 (현실적)
if df['High'] > entry_high:
    buy_price = df['Close'] * (1 + slippage)  # 당일 종가 + 슬리피지
```

### 파일 버전

| 파일 | 상태 | 설명 |
|------|------|------|
| `crypto_portfolio_strategy_comparison.py` | ❌ 비권장 | 편향 포함 버전 |
| `crypto_portfolio_strategy_comparison_fixed.py` | ✅ **권장** | 편향 수정 버전 |

**📊 자세한 분석**: [ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md)

---

## 📊 전략 성과 비교 (수정 후)

### 포트폴리오 전략 (BTC, ETH, ADA, XRP 동일비중 25%)

| 순위 | 전략 | 총 수익률 | CAGR | MDD | Sharpe |
|------|------|-----------|------|-----|--------|
| 🥇 | **SMA 30** | **5,942%** | **81.85%** | -40.70% | **1.60** |
| 🥈 | Turtle Trading | 5,646% | 80.52% | **-29.83%** | 1.12 |
| 🥉 | RSI 55 | 3,142% | 66.07% | -37.74% | 1.45 |

**승자**: SMA 30 (수익률과 샤프 비율 모두 최고)

---

## 📚 구현된 전략

### 1. 돌파(Breakthrough) 전략 (10가지)
- Donchian Channel
- Volatility Breakout (래리 윌리엄스)
- Range Breakout
- Opening Range Breakout
- ATR Breakout
- **Turtle Trading** (터틀 트레이딩) ✅ 수정됨
- Bollinger Band Breakout
- High/Low Breakout
- Momentum Breakout
- Keltner Channel Breakout

### 2. 추세 추종 전략
- **SMA 30 Crossover** (가격 vs SMA 30)
- **RSI 55** (RSI >= 55 시 매수)

### 3. 기타 전략
- Mean Reversion
- Disparity Index

---

## 🛡️ 백테스팅 품질 보증

### 체크된 편향들

| 편향 | 상태 | 확인 방법 |
|------|------|-----------|
| Look-ahead Bias | ✅ 해결 | `shift(1)` 사용 확인 |
| Perfect Execution Bias | ✅ 해결 | 슬리피지 0.2% 적용 |
| Data Snooping Bias | ✅ 해결 | Train/Test 분할 (해당 시) |
| Transaction Cost | ✅ 해결 | 수수료 + 슬리피지 반영 |

### 적용된 현실적 가정
- 슬리피지: 0.2%
- 체결 가격: 종가 (돌파 전략) / 다음날 종가 (지표 전략)
- 매수/매도 시 양방향 슬리피지 적용

---

## 📖 문서 가이드

### 초보자용
1. **[BACKTESTING_CHECKLIST.md](BACKTESTING_CHECKLIST.md)** - 30초 체크리스트
2. 기존 스크립트 실행 및 결과 확인
3. 파라미터 약간 수정해보기

### 중급자용
1. **[BACKTESTING_BIAS_GUIDE.md](BACKTESTING_BIAS_GUIDE.md)** - 완벽 이해
2. **[ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md)** - 실제 사례 학습
3. 자신만의 전략 개발

### 고급자용
1. `strategy_review_analysis.py` - 상세 분석 코드 리뷰
2. Walk-Forward Analysis 구현
3. Multi-timeframe 전략 개발

---

## 🎯 다음 작업 시 주의사항

### ✅ 새 전략 개발 전 체크리스트

```bash
# 1. 체크리스트 읽기 (필수!)
cat BACKTESTING_CHECKLIST.md

# 2. 코드 작성

# 3. 실행 전 자가 점검
# - shift(1) 있나?
# - 슬리피지 있나?
# - 당일 종가로 당일 매수 안 하나?

# 4. 실행 및 검증
# - 샤프 > 3.0이면 버그 의심
# - 승률 > 70%이면 버그 의심
# - MDD < 10%이면 버그 의심
```

### 🚨 레드 플래그

다음 결과가 나오면 **99% 버그**:
- 샤프 비율 > 3.0
- 승률 > 70%
- MDD < 10%
- 연속 손실 0회

→ **즉시 `BACKTESTING_CHECKLIST.md` 참조하여 재검토**

---

## 💻 개발 환경

### 필수 패키지
```bash
pip install pandas numpy matplotlib seaborn pyarrow
```

### 데이터 형식
- 위치: `chart_day/`
- 형식: Parquet
- 컬럼: `open`, `high`, `low`, `close`, `volume`
- 인덱스: DatetimeIndex

---

## 📊 결과 파일

### 생성되는 파일들
```
crypto_portfolio_comparison_fixed.png      # 포트폴리오 비교 차트
crypto_portfolio_metrics_fixed.csv         # 성과 지표
portfolio_*.csv                            # 포트폴리오 상세 데이터
turtle_trading_issue_analysis.png          # 문제점 분석 차트
```

---

## 🎓 교훈 및 베스트 프랙티스

### 핵심 교훈
1. **"결과가 너무 좋으면 의심하라"**
2. **항상 shift(1) 사용**
3. **슬리피지는 필수**
4. **체결 가격은 현실적으로**

### 코딩 템플릿

```python
# ✅ 권장 템플릿
df['indicator'] = calculate_indicator(df['Close'])
df['signal'] = df['indicator'] > threshold
df['position'] = df['signal'].shift(1)  # 필수!

# 수익률 계산
df['returns'] = df['position'] * df['Close'].pct_change()

# 슬리피지 적용
slippage_cost = pd.Series(0.0, index=df.index)
slippage_cost[df['position'].diff() == 1] = -0.002   # 매수
slippage_cost[df['position'].diff() == -1] = -0.002  # 매도
df['returns'] = df['returns'] + slippage_cost
```

---

## 🤝 기여 가이드

새 전략 추가 시:
1. `BACKTESTING_CHECKLIST.md` 준수
2. 테스트 데이터셋 분리
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
*버전: 2.0 (Perfect Execution Bias 수정)*
