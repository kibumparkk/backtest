# 백테스팅 체크리스트 (Quick Reference)

> 백테스트 전에 반드시 확인하세요! 하나라도 미확인 시 결과 신뢰 불가

## 🚨 필수 체크 (하나라도 실패하면 STOP)

### 1️⃣ Look-ahead Bias (미래 정보 사용)
```python
# ❌ 금지
df['signal'] = df['RSI'] > 55
df['returns'] = df['signal'] * df['Close'].pct_change()

# ✅ 필수
df['signal'] = df['RSI'] > 55
df['returns'] = df['signal'].shift(1) * df['Close'].pct_change()
#                            ^^^^^^^^^ 필수!
```

- [ ] 모든 지표에 `shift(1)` 적용
- [ ] 당일 종가로 계산한 지표로 당일 매수 금지
- [ ] 신호 생성(T-1) → 체결(T) 시간차 확인

### 2️⃣ Perfect Execution Bias (완벽한 체결)
```python
# ❌ 금지
if df['High'] > entry_high:
    buy_price = entry_high  # 돌파했는데 돌파선에서 산다? 불가능!

# ✅ 필수
if df['High'] > entry_high:
    buy_price = df['Close'] * (1 + slippage)  # 현실적 가격
```

- [ ] 모든 거래에 슬리피지 적용 (최소 0.1%)
- [ ] 돌파/이탈 시 돌파선 가격 체결 금지
- [ ] 당일 최고가/최저가 체결 금지
- [ ] 체결 가격: 종가, 시가, 평균가만 사용

### 3️⃣ Data Snooping Bias (과적합)
- [ ] 데이터를 Train(70%) / Test(30%) 분할
- [ ] Test 데이터는 단 1회만 사용
- [ ] 백테스트 결과 본 후 파라미터 수정 금지
- [ ] 파라미터 조합 100개 이하

### 4️⃣ Transaction Cost (거래 비용)
```python
# ✅ 필수
slippage = 0.002  # 0.2%
fee = 0.001       # 0.1%
buy_cost = price * (1 + slippage + fee)
sell_revenue = price * (1 - slippage - fee)
```

- [ ] 슬리피지 반영
- [ ] 거래 수수료 반영
- [ ] 세금 반영 (해당 시)

---

## ⚠️ 레드 플래그 (하나라도 해당하면 재검토)

결과가 다음 중 하나라도 해당하면 **99% 버그**:

- [ ] ❌ 샤프 비율 > 3.0
- [ ] ❌ 승률 > 70%
- [ ] ❌ MDD < 10%
- [ ] ❌ Train/Test 성능 차이 > 50%
- [ ] ❌ CAGR > 100% (암호화폐 제외)
- [ ] ❌ 모든 거래가 수익
- [ ] ❌ 연속 손실 0회

---

## 📊 코드 리뷰 체크리스트

### 신호 생성 부분
```python
# 이 부분 중점 체크!
df['indicator'] = calculate_indicator(df['Close'])
df['signal'] = df['indicator'] > threshold

# ✅ shift(1) 있는가?
df['position'] = df['signal'].shift(1)  # 필수!
```

- [ ] `shift(1)` 확인
- [ ] 당일 데이터로 당일 거래 금지 확인

### 수익률 계산 부분
```python
# 이 부분 중점 체크!
df['returns'] = (sell_price / buy_price - 1)

# ✅ 슬리피지 있는가?
slippage_cost = 2 * slippage  # 매수+매도
df['returns'] = (sell_price / buy_price - 1) - slippage_cost
```

- [ ] 슬리피지 적용 확인
- [ ] 수수료 적용 확인

### 체결 가격 부분
```python
# 이 부분 중점 체크!
buy_price = ???  # 어떤 가격 사용?

# ✅ 현실적 가격인가?
# OK: df['Close'], df['Open'], (High+Low)/2
# NG: df['High'], df['Low'], entry_high, exit_low
```

- [ ] 최고가/최저가 직접 사용 금지
- [ ] 돌파선 가격 직접 사용 금지

---

## 🎯 결과 검증

### Train vs Test 비교
| 지표 | 허용 범위 |
|------|-----------|
| 수익률 차이 | < 30% |
| 샤프 비율 차이 | < 30% |
| MDD 차이 | < 50% |

- [ ] Train/Test 성능이 비슷한가?
- [ ] Test가 Train보다 훨씬 나쁘지 않은가?

### 합리성 검증
- [ ] 샤프 비율: 0.5 ~ 2.0 범위
- [ ] 승률: 40% ~ 60% 범위
- [ ] MDD: -20% ~ -50% 범위
- [ ] 최대 연속 손실: 5~10회 수준

### 강건성 검증
- [ ] 파라미터 ±20% 변경 시 성능 유지
- [ ] 다른 기간에도 작동
- [ ] 다른 종목에도 적용 가능

---

## 🔍 디버깅 체크리스트

백테스트 결과가 이상할 때:

### 1단계: 타임라인 검증
```python
# 샘플 거래 5개 출력
trades = df[df['position_change'] != 0].head(5)
for idx, row in trades.iterrows():
    print(f"날짜: {idx}")
    print(f"  신호 계산에 사용된 데이터: {idx - 1}일")  # 전일
    print(f"  체결 날짜: {idx}")  # 당일
    print(f"  체결 가격: {row['buy_price']}")
    print(f"  종가: {row['Close']}")
    print(f"  차이: {(row['buy_price'] - row['Close']) / row['Close'] * 100:.2f}%")
```

- [ ] 신호 생성과 체결이 다른 날짜인가?
- [ ] 체결 가격이 현실적인가?

### 2단계: 가격 검증
```python
# 매수가 검증
buy_trades = df[df['buy_signal'] == True]
for idx, row in buy_trades.iterrows():
    if row['buy_price'] < row['Low'] or row['buy_price'] > row['High']:
        print(f"⚠️ {idx}: 불가능한 체결가! {row['buy_price']}")
        print(f"   High: {row['High']}, Low: {row['Low']}")
```

- [ ] 모든 체결가가 High/Low 범위 내인가?

### 3단계: 수익률 검증
```python
# 비정상적으로 큰 수익 확인
big_wins = df[df['returns'] > 0.1]  # 10% 이상
print(f"10% 이상 수익 거래: {len(big_wins)}건")
for idx, row in big_wins.iterrows():
    print(f"{idx}: {row['returns']*100:.1f}% - 검토 필요")
```

- [ ] 비정상적인 수익이 있는가?
- [ ] 그 원인이 설명 가능한가?

---

## 📝 최소 리포팅 항목

백테스트 완료 시 반드시 기록:

```markdown
## 백테스트 요약

### 전략
- 이름: [전략명]
- 로직: [간단한 설명]
- 파라미터: [key=value]

### 데이터
- 종목: [심볼]
- 기간: [시작~종료]
- Train/Test: [분할 날짜]

### 결과
|  | Train | Test |
|--|-------|------|
| 총 수익률 | X% | X% |
| CAGR | X% | X% |
| Sharpe | X.X | X.X |
| MDD | -X% | -X% |

### 체크
- [x] Look-ahead: shift(1) 사용
- [x] Execution: 슬리피지 0.2%
- [x] Snooping: Test 1회만
- [x] Cost: 수수료 0.1%

### 판정
- 🟢 GREEN / 🟡 YELLOW / 🔴 RED
- 사유: [...]
```

---

## 🚀 배포 전 최종 체크

실전 투입 전 마지막 확인:

- [ ] **독립 데이터** 1회 테스트 (절대 본 적 없는 데이터)
- [ ] **소액 실전** 테스트 (최소 1개월)
- [ ] **심리 테스트**: MDD 견딜 수 있는가?
- [ ] **자금 관리**: 포지션 사이징 계획
- [ ] **비상 계획**: 손절 기준, 전략 중단 조건

---

## ⏱️ 빠른 검토 (30초)

급할 때 이것만 확인:

1. ✅ `shift(1)` 있나? → 없으면 STOP
2. ✅ 슬리피지 있나? → 없으면 STOP
3. ✅ 샤프 < 3.0? → 크면 STOP
4. ✅ Train/Test 비슷? → 차이 크면 STOP
5. ✅ 체결가 현실적? → 최고가/최저가면 STOP

**5개 모두 통과해야 진행 가능!**

---

*참고: 자세한 내용은 `BACKTESTING_BIAS_GUIDE.md` 참조*
