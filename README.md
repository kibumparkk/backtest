# 비트코인 추세추종 전략 Top 5 발굴 - 최종 보고서

## 📋 Executive Summary

**목표**: Close > SMA30 벤치마크(Sharpe 1.6591)를 능가하는 추세추종 전략 5개 발굴

**결과**: ✅ **성공 - 멀티 타임프레임(MTF) 전략으로 7개 발견, Top 5 선정**

**핵심 성과**: 최고 전략 **Sharpe 2.3170 (+39.65%)** 달성

---

## 🏆 Top 5 Multi-Timeframe Strategies

### 벤치마크
- **전략**: Close > SMA30 (일봉 단순이동평균선 30일)
- **Sharpe Ratio**: 1.6591
- **Total Return**: 8,859%
- **CAGR**: 65.44%
- **MDD**: -29.07%
- **기간**: 2018-01-01 ~ 2025-11-05 (7.8년)
- **슬리피지**: 0.2% (매매 시 양방향)

---

### 🥇 #1: Weekly Donchian + Daily SMA30

**전략 로직**:
```python
# Weekly filter
weekly_high_20 = weekly['High'].rolling(20).max()
weekly_signal = (weekly['Close'] > weekly_high_20 * 0.95)

# Daily timing
daily_signal = (daily['Close'] > daily['SMA30'])

# Combined (Weekly signal available from next day after week ends)
final_signal = weekly_signal AND daily_signal
```

**성과**:
- **Sharpe Ratio**: **2.3170** (+39.65% vs benchmark) 🏆
- **Total Return**: 5,149%
- **CAGR**: 65.69%
- **MDD**: **-16.10%** (벤치마크 -29.07% 대비 **45% 감소**) 🔥
- **Total Trades**: 76 (7.8년간 매우 선택적 진입)

**전략 설명**:
- **주봉 필터**: 20주 최고가의 95% 이상 (강력한 상승 추세만 선택)
- **일봉 타이밍**: SMA30 위에서 진입
- **핵심 강점**:
  - 약세장에서 조기 이탈 (2018, 2022년 손실 최소화)
  - MDD -16.10%로 **매우 안전한 리스크 프로필**
  - 거래 횟수 76회로 비용 최소화

**왜 이 전략이 최고인가**:
- Donchian 채널은 브레이크아웃 시스템의 정석
- 주봉 20주 신고가 = 강력한 모멘텀 확인
- 95% 기준으로 과도한 진입 방지
- 일봉 SMA30으로 단기 노이즈 필터링

---

### 🥈 #2: Weekly EMA20 + Daily SMA30

**전략 로직**:
```python
weekly_signal = (weekly['Close'] > weekly['EMA20'])
daily_signal = (daily['Close'] > daily['SMA30'])
final_signal = weekly_signal AND daily_signal
```

**성과**:
- **Sharpe Ratio**: **2.0780** (+25.25% vs benchmark)
- **Total Return**: 15,961%
- **CAGR**: 91.08% (가장 높음)
- **MDD**: -31.87%
- **Total Trades**: 106

**전략 설명**:
- EMA는 SMA보다 최근 가격에 민감
- 주봉 EMA20으로 중기 추세 확인
- 총 수익률 15,961%로 **Top 5 중 가장 높음**

---

### 🥉 #3: Weekly SMA10 + Daily SMA30

**전략 로직**:
```python
weekly_signal = (weekly['Close'] > weekly['SMA10'])
daily_signal = (daily['Close'] > daily['SMA30'])
final_signal = weekly_signal AND daily_signal
```

**성과**:
- **Sharpe Ratio**: **2.0425** (+23.11% vs benchmark)
- **Total Return**: 18,500%
- **CAGR**: 94.69% (가장 높음)
- **MDD**: -27.18%
- **Total Trades**: 112

**전략 설명**:
- 주봉 SMA10으로 빠른 추세 전환 포착
- CAGR 94.69%로 **연평균 수익률 최고**
- 총 수익률 18,500% = 초기 투자금 186배

---

### 4위: Weekly SMA20 + Daily SMA30

**성과**:
- **Sharpe Ratio**: 1.9095 (+15.10% vs benchmark)
- **Total Return**: 10,487%
- **CAGR**: 81.19%
- **MDD**: -31.87%
- **Total Trades**: 108

**전략 설명**:
- 주봉 SMA20으로 중기 추세 확인
- 균형잡힌 진입/퇴출 빈도

---

### 5위: Weekly SMA50 + Daily SMA30

**성과**:
- **Sharpe Ratio**: 1.7954 (+8.22% vs benchmark)
- **Total Return**: 7,437%
- **CAGR**: 73.51%
- **MDD**: -36.18%
- **Total Trades**: 118

**전략 설명**:
- 주봉 SMA50으로 장기 추세 확인
- 가장 보수적인 접근
- 여전히 벤치마크보다 8.22% 우수

---

## 📊 Top 5 종합 비교

| Rank | Strategy | Sharpe | Improve | Total Return | CAGR | MDD | Trades |
|------|----------|--------|---------|--------------|------|-----|--------|
| - | **Benchmark** | **1.6591** | - | **8,859%** | **65.44%** | **-29.07%** | **1,527** |
| 🥇 | **Weekly Donchian + Daily SMA30** | **2.3170** | **+39.7%** | **5,149%** | **65.69%** | **-16.10%** | **76** |
| 🥈 | Weekly EMA20 + Daily SMA30 | 2.0780 | +25.3% | 15,961% | 91.08% | -31.87% | 106 |
| 🥉 | Weekly SMA10 + Daily SMA30 | 2.0425 | +23.1% | 18,500% | 94.69% | -27.18% | 112 |
| 4 | Weekly SMA20 + Daily SMA30 | 1.9095 | +15.1% | 10,487% | 81.19% | -31.87% | 108 |
| 5 | Weekly SMA50 + Daily SMA30 | 1.7954 | +8.2% | 7,437% | 73.51% | -36.18% | 118 |

### 시각화

![Strategy Comparison](mtf_strategies_comparison.png)

*그림 1: Top 5 전략 성과 비교 (Sharpe, CAGR, MDD, Risk-Return Profile)*

![Detailed Metrics](mtf_strategies_table.png)

*그림 2: 상세 성과 지표 테이블*

---

## 💡 핵심 발견 (Key Insights)

### 1. **멀티 타임프레임의 위력**
- 단일 타임프레임 최고: SMA31 (Sharpe 1.70, +2.3%)
- 멀티 타임프레임 최고: **Weekly Donchian (Sharpe 2.32, +39.7%)**
- **17배 더 나은 성과 개선**

### 2. **주봉 필터의 효과**
주봉 추세 확인으로:
- 약세장 조기 회피 (2018, 2022)
- 강세장 완전 참여 (2020-2021, 2023-2024)
- MDD 대폭 감소 (29.07% → 16.10%)

### 3. **Donchian 채널의 우수성**
- SMA/EMA보다 **더 명확한 브레이크아웃 신호**
- 20주 최고가 = 시장의 강력한 합의
- 95% 기준 = 과도한 고점 매수 방지

### 4. **거래 빈도와 성과**
- 벤치마크: 1,527회
- #1 전략: 76회 (**95% 감소**)
- 적은 거래 = 낮은 비용 + 높은 성과
- "Less is More" 실증

### 5. **리스크 조정 수익률의 중요성**
- #1 전략: 총 수익 5,149% (5위)
- 하지만 **Sharpe 2.32로 1위**
- MDD -16.10%로 **가장 안전**
- 안정적 수익 > 변동성 큰 고수익

---

## 📈 투자 시뮬레이션

### 100만원 투자 시 (7.8년)

| Strategy | Final Amount | vs Benchmark |
|----------|-------------|--------------|
| Benchmark | 89.6M | - |
| #1 Donchian | 62.5M | -27.1M |
| #2 EMA20 | 160.6M | +71.0M |
| #3 SMA10 | 186.0M | +96.4M |

**주목**: #1 전략이 총 금액은 적지만 **MDD -16.10%로 가장 안전**

### 2018년 약세장 성과

| Year | Benchmark | #1 Donchian | Difference |
|------|-----------|-------------|------------|
| 2018 | -22.74% | -8.63% | **+14.11%p** |

**핵심**: 약세장에서 손실을 절반 이하로 줄임

---

## 🎯 전략 선택 가이드

### 안전성 최우선 (리스크 회피형)
**추천**: **#1 Weekly Donchian + Daily SMA30**
- Sharpe 2.32 (최고)
- MDD -16.10% (최저)
- 76회 거래 (관리 용이)
- 연 65.69% 수익

### 최대 수익 추구 (공격형)
**추천**: **#3 Weekly SMA10 + Daily SMA30**
- CAGR 94.69% (최고)
- 총 수익 18,500% (최고)
- MDD -27.18% (감수 가능)

### 균형잡힌 선택
**추천**: **#2 Weekly EMA20 + Daily SMA30**
- Sharpe 2.08
- CAGR 91.08%
- 총 수익 15,961%

---

## ⚠️ 실전 적용 시 주의사항

### 1. 백테스트 한계
- 과거 성과 ≠ 미래 성과
- 생존 편향 (비트코인이 살아남은 시나리오)
- 최적화 편향 (특정 기간에 과적합 가능성)

### 2. 실전 비용
- 거래소 수수료: 0.05~0.25% 추가
- 슬리피지: 시장 상황에 따라 0.2% 이상 가능
- 세금: 한국 기준 22% 또는 27.5% (250만원 공제)

### 3. 신호 지연
- 백테스트: 종가 확정 후 다음 날 진입
- 실전: 종가 예측 또는 익일 시가 진입
- 차이로 인한 성과 저하 가능

### 4. 심리적 요인
- MDD 구간(최대 -16% ~ -36%)에서 전략 이탈 위험
- 기계적 실행 필수
- 감정 배제

### 5. 시장 환경 변화
- 규제 변화 (ETF 승인, 규제 강화 등)
- 시장 성숙도 증가로 효율성 향상
- 전략 효과 감소 가능성
- 정기적 재평가 필요

---

## 💻 실행 방법

### 코드 실행

```bash
# 전략 실행
python bitcoin_mtf_loop_based.py

# 시각화 생성
python create_mtf_visualization.py
```

### 전략 구현 예시

```python
import pandas as pd

# 데이터 로드
df_daily = pd.read_parquet('chart_day/BTC_KRW.parquet')
df_weekly = df_daily.resample('W-MON', label='left', closed='left').agg({
    'High': 'max',
    'Close': 'last'
}).dropna()

# Weekly Donchian 신호
df_weekly['high_20'] = df_weekly['High'].rolling(20).max()
df_weekly['weekly_signal'] = (df_weekly['Close'] > df_weekly['high_20'] * 0.95).astype(int)

# Daily SMA30 신호
df_daily['SMA30'] = df_daily['Close'].rolling(30).mean()
df_daily['daily_signal'] = (df_daily['Close'] > df_daily['SMA30']).astype(int)

# Loop-based combination
for each day:
    # 1. Get daily signal for today
    # 2. Find most recent completed weekly signal
    # 3. Combine: final_signal = weekly_signal AND daily_signal
    # 4. Trade based on final_signal
```

### 일일 체크리스트

1. **주말 (일요일 밤)**:
   - 주봉 종가 확인
   - Donchian 20주 최고가 계산
   - 종가 > 최고가 * 0.95? → 주봉 신호 생성

2. **매일 (종가 후)**:
   - 일봉 종가 확인
   - SMA30 계산
   - 종가 > SMA30? → 일봉 신호 생성

3. **신호 결합**:
   - 주봉 신호 AND 일봉 신호 = 1 → 매수/보유
   - 하나라도 0 → 매도/대기

---

## 📁 파일 구조

```
backtest/
├── README.md                          # 본 보고서
├── bitcoin_mtf_loop_based.py         # MTF 전략 구현 (Loop-based)
├── bitcoin_mtf_loopbased_results.csv # 전략 성과 데이터
├── create_mtf_visualization.py       # 시각화 생성 코드
├── mtf_strategies_comparison.png     # 성과 비교 차트
├── mtf_strategies_table.png          # 상세 지표 테이블
└── chart_day/                         # 비트코인 일봉 데이터
    └── BTC_KRW.parquet
```

---

## 🎓 결론

### ✅ 목표 달성

**100% 성공**: 벤치마크(Sharpe 1.6591)를 능가하는 전략 **7개 발견**, Top 5 선정 완료

### 🏆 최우수 전략

**Weekly Donchian + Daily SMA30**
- Sharpe 2.3170 (벤치마크 대비 **+39.65%**)
- MDD -16.10% (벤치마크 -29.07% 대비 **45% 감소**)
- 76회 거래 (관리 용이)
- **리스크 조정 수익률 최고**

### 💡 핵심 교훈

1. **멀티 타임프레임의 위력**: 단일 대비 17배 더 나은 성과
2. **주봉 필터 효과**: 약세장 조기 회피로 MDD 대폭 감소
3. **Donchian 우수성**: 명확한 브레이크아웃 신호
4. **거래 빈도**: 적은 거래가 더 나은 성과 (76 vs 1,527회)
5. **Loop-based 검증**: 벡터화 구현의 lookahead bias 방지

---

**작성일**: 2025-11-14
**분석 기간**: 2018-01-01 ~ 2025-11-05 (7.8년)
**데이터**: Upbit BTC/KRW 일봉
**슬리피지**: 0.2%
**Git Branch**: `claude/bitcoin-trend-strategy-benchmark-015LzBQYd1BxPjBA7caaBZ8y`
