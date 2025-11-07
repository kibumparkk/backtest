# 백테스팅 편향(Bias) 완벽 가이드

> **"백테스트 결과가 너무 좋으면 의심하라!"**
>
> 이 문서는 백테스팅에서 흔히 발생하는 편향들을 정의하고, 감지하고, 방지하는 방법을 제시합니다.

---

## 📚 목차

1. [Look-ahead Bias (미래 정보 사용)](#1-look-ahead-bias-미래-정보-사용)
2. [Perfect Execution Bias (완벽한 체결 가격)](#2-perfect-execution-bias-완벽한-체결-가격)
3. [Survivorship Bias (생존 편향)](#3-survivorship-bias-생존-편향)
4. [Data Snooping Bias (데이터 스누핑)](#4-data-snooping-bias-데이터-스누핑)
5. [기타 편향들](#5-기타-편향들)
6. [백테스팅 체크리스트](#6-백테스팅-체크리스트)

---

## 1. Look-ahead Bias (미래 정보 사용)

### 📖 정의
백테스팅 시점에서는 알 수 없는 **미래의 정보**를 사용하여 거래 결정을 내리는 오류

### ⚠️ 왜 위험한가?
- 실제 트레이딩에서는 절대 달성 불가능한 성과
- 가장 흔하게 발생하는 편향
- 발견하기 어려울 수 있음

### 🔍 실제 사례

#### ❌ 잘못된 예시 1: 당일 데이터 사용
```python
# 문제: 당일 종가를 이용해 신호 생성 후 당일에 매수
df['SMA'] = df['Close'].rolling(20).mean()
df['signal'] = df['Close'] > df['SMA']  # 당일 종가 사용

# 당일에 매수 - 문제!
df['buy_price'] = np.where(df['signal'], df['Close'], np.nan)
```

**문제점**: 종가는 장 마감 후에만 알 수 있는데, 종가로 신호를 판단하고 종가에 매수한다는 것은 불가능

#### ✅ 올바른 예시 1
```python
# 해결: 전일 종가로 신호 생성, 다음날 매수
df['SMA'] = df['Close'].rolling(20).mean()
df['signal'] = df['Close'] > df['SMA']

# 다음날 시가/종가에 매수
df['signal_shifted'] = df['signal'].shift(1)  # 전일 신호 사용
df['buy_price'] = np.where(df['signal_shifted'], df['Close'], np.nan)
```

#### ❌ 잘못된 예시 2: shift() 누락
```python
# 문제: 당일 계산된 지표를 당일에 사용
df['RSI'] = calculate_rsi(df['Close'], 14)
df['position'] = (df['RSI'] >= 55).astype(int)
df['returns'] = df['position'] * df['Close'].pct_change()
```

#### ✅ 올바른 예시 2
```python
# 해결: 전일 지표로 판단, 당일 수익
df['RSI'] = calculate_rsi(df['Close'], 14)
df['position'] = (df['RSI'] >= 55).astype(int)
df['returns'] = df['position'].shift(1) * df['Close'].pct_change()
#                                  ^^^^^^ shift 필수!
```

### 🛡️ 방지 방법

1. **타임라인 체크**
   ```python
   # 신호 생성 시점: T-1 (전일)
   # 체결 시점: T (당일)
   # 수익 실현: T (당일) ~ T+N
   ```

2. **shift() 사용 습관화**
   ```python
   # 모든 지표는 shift(1) 후 사용
   df['signal'] = (df['MA_fast'] > df['MA_slow']).shift(1)
   ```

3. **체결 가격 현실화**
   ```python
   # 시가, 종가, 또는 다음날 시가 사용
   # 절대 당일 계산된 값으로 당일 매수 금지
   ```

### ✅ 체크리스트
- [ ] 모든 지표가 최소 1일 이전 데이터만 사용하는가?
- [ ] shift(1)이 적절히 사용되었는가?
- [ ] 신호 생성 시점과 체결 시점이 명확히 분리되어 있는가?
- [ ] 당일 종가로 계산한 지표로 당일 매수하지 않는가?

---

## 2. Perfect Execution Bias (완벽한 체결 가격)

### 📖 정의
항상 **최적의 가격**에 체결된다고 가정하는 오류. 실제로는 불가능한 가격에 매수/매도하는 것.

### ⚠️ 왜 위험한가?
- 수익률을 크게 과대평가 (50~100%까지도 가능)
- 실전에서 큰 실망을 초래
- **이번 터틀트레이딩 분석에서 발견!**

### 🔍 실제 사례 (이번 프로젝트)

#### ❌ 잘못된 예시: 터틀트레이딩
```python
# 문제: 돌파한 가격에 체결한다고 가정
df['entry_high'] = df['High'].rolling(20).max().shift(1)

# 고가가 entry_high를 돌파했을 때
if df.iloc[i]['High'] > df.iloc[i]['entry_high']:
    buy_price = df.iloc[i]['entry_high']  # ❌ 불가능!
```

**문제점**:
- "돌파"했다는 것은 이미 entry_high보다 **높은** 가격이라는 의미
- entry_high 가격에 매수하는 것은 시간 역행과 같음
- 실제 사례: entry_high=12,755,000원인데 실제 종가=13,548,000원 (5.85% 차이)

#### ✅ 올바른 예시
```python
# 해결 1: 당일 종가에 체결
if df.iloc[i]['High'] > df.iloc[i]['entry_high']:
    buy_price = df.iloc[i]['Close'] * (1 + slippage)  # ✅ 현실적

# 해결 2: 다음날 시가에 체결
if df.iloc[i]['High'] > df.iloc[i]['entry_high']:
    buy_price = df.iloc[i+1]['Open'] * (1 + slippage)  # ✅ 더 현실적
```

### 🔍 다른 사례들

#### ❌ 잘못된 예시: 지정가 체결 가정
```python
# 문제: 정확히 지지선/저항선에서 체결된다고 가정
support_level = 50000
if df['Low'] <= support_level:
    buy_price = support_level  # ❌ 너무 이상적
```

#### ✅ 올바른 예시
```python
# 해결: 현실적인 체결 가격
support_level = 50000
if df['Low'] <= support_level:
    # 종가 또는 평균가 사용
    buy_price = df['Close'] * (1 + slippage)
    # 또는
    buy_price = (df['High'] + df['Low']) / 2 * (1 + slippage)
```

### 🛡️ 방지 방법

1. **슬리피지 필수 적용**
   ```python
   slippage = 0.002  # 0.2% (암호화폐 기준)
   slippage = 0.0005 # 0.05% (주식 기준)

   buy_price = target_price * (1 + slippage)
   sell_price = target_price * (1 - slippage)
   ```

2. **현실적인 체결 가격 사용**
   ```python
   # 우선순위:
   # 1. 다음날 시가 (가장 현실적)
   # 2. 당일 종가 + 슬리피지
   # 3. 당일 평균가 (OHLC/4 또는 HL/2)
   ```

3. **극단가 회피**
   ```python
   # ❌ 금지: 당일 최고가에 매도, 최저가에 매수
   # ✅ 권장: 종가, 시가, 평균가 사용
   ```

### ✅ 체크리스트
- [ ] 모든 거래에 슬리피지가 적용되었는가?
- [ ] 돌파/이탈 신호에서 돌파선 가격에 체결하지 않는가?
- [ ] 당일 최고가/최저가를 체결 가격으로 사용하지 않는가?
- [ ] 체결 가격이 현실적인가? (종가, 시가, 평균가)

---

## 3. Survivorship Bias (생존 편향)

### 📖 정의
현재까지 **살아남은** 자산만을 대상으로 분석하여, 실패한(상장폐지된) 자산을 제외하는 오류

### ⚠️ 왜 위험한가?
- 실제 수익률을 과대평가 (20~40% 과대평가 가능)
- 특히 장기 백테스트에서 심각
- 실전에서는 상장폐지 리스크가 존재

### 🔍 실제 사례

#### ❌ 잘못된 예시
```python
# 문제: 현재 거래 중인 코인만 분석
symbols = ['BTC', 'ETH', 'ADA', 'XRP']  # 살아남은 코인만

# 2018년부터 백테스트
for symbol in symbols:
    backtest(symbol, start='2018-01-01')
```

**문제점**:
- 2018년에 존재했지만 현재 사라진 코인들 제외
- 예: BitConnect, Tera Luna Classic 등 수많은 실패 코인 누락
- 실제로 투자했다면 이런 코인에도 일부 투자했을 것

#### ✅ 올바른 예시
```python
# 해결 1: 과거 시점의 전체 종목 포함
def get_symbols_at_date(date):
    """특정 시점에 거래 가능했던 모든 종목 반환"""
    # 상장폐지된 종목도 포함
    return historical_symbols_db.query(date)

# 해결 2: 상장폐지 이벤트 반영
def backtest_with_delisting(symbol, start, end):
    df = load_data(symbol)

    # 상장폐지 시 손실 반영
    if is_delisted(symbol):
        delisting_date = get_delisting_date(symbol)
        # 상장폐지 시 -100% 수익률
        df.loc[delisting_date:, 'returns'] = -1.0
```

### 🔍 암호화폐 특화 사례

**2018년 Top 10 vs 현재**:
```
2018년 시총 순위:
1. BTC  ✅ 생존
2. ETH  ✅ 생존
3. XRP  ✅ 생존
4. BCH  ⚠️ 순위 하락
5. EOS  ⚠️ 순위 급락
6. LTC  ✅ 생존
7. ADA  ✅ 생존
8. XLM  ⚠️ 순위 하락
9. TRX  ⚠️ 순위 하락
10. MIOTA ⚠️ 순위 급락

→ Top 10중 50%가 순위 급락
→ Top 50 중 많은 수가 사실상 사장
```

### 🛡️ 방지 방법

1. **시점별 유니버스 구성**
   ```python
   def get_universe(date, criterion='top_100_by_volume'):
       """특정 시점의 투자 가능 종목 반환"""
       # 해당 시점의 시가총액, 거래량 기준
       return get_top_coins(date, n=100, by='volume')
   ```

2. **리밸런싱 시 종목 교체 반영**
   ```python
   # 매 분기 리밸런싱
   for quarter in quarters:
       current_universe = get_universe(quarter)
       # 신규 진입 종목 매수
       # 탈락 종목 매도
       rebalance_portfolio(current_universe)
   ```

3. **상장폐지 손실 명시적 반영**
   ```python
   def apply_delisting_loss(returns, symbol):
       if is_delisted(symbol):
           delisting_date = get_delisting_date(symbol)
           # 상장폐지 시 -90% 손실 (일부 회수 가능하다고 가정)
           returns.loc[delisting_date] = -0.9
       return returns
   ```

### ✅ 체크리스트
- [ ] 과거 시점에 존재했던 모든 종목을 포함하는가?
- [ ] 상장폐지된 종목의 손실을 반영하는가?
- [ ] 리밸런싱 시 종목 교체를 반영하는가?
- [ ] "현재 살아있는 종목"만 선택하지 않았는가?

---

## 4. Data Snooping Bias (데이터 스누핑)

### 📖 정의
같은 데이터를 반복적으로 분석하여 **우연히** 잘 맞는 파라미터를 찾는 오류 (과적합의 일종)

### ⚠️ 왜 위험한가?
- 백테스트에서는 완벽, 실전에서는 실패
- 통계적으로 유의미해 보이지만 실제로는 노이즈
- 발견하기 어려움

### 🔍 실제 사례

#### ❌ 잘못된 예시 1: 과도한 파라미터 최적화
```python
# 문제: 수백 가지 조합을 시도해서 최적값 찾기
best_return = -999
best_params = None

for sma_period in range(5, 200, 5):      # 40가지
    for rsi_period in range(5, 30, 1):   # 25가지
        for threshold in range(30, 70, 1): # 40가지
            # 총 40,000가지 조합!
            result = backtest(sma_period, rsi_period, threshold)
            if result > best_return:
                best_return = result
                best_params = (sma_period, rsi_period, threshold)

# 최적 파라미터: SMA=73, RSI=17, threshold=63
# → 이것은 과적합!
```

**문제점**:
- 40,000번 시도하면 우연히 좋은 결과가 나올 수밖에 없음
- p-value < 0.05를 40,000번 시도하면 2,000개는 우연히 통과
- 실전에서는 작동하지 않음

#### ✅ 올바른 예시 1: Train/Test 분할
```python
# 해결: 데이터 분할 및 검증
train_data = df['2018':'2021']  # 훈련 데이터
test_data = df['2022':'2024']   # 테스트 데이터 (절대 건드리지 않음)

# 1단계: 훈련 데이터에서만 최적화
best_params = optimize_on_train_data(train_data)

# 2단계: 테스트 데이터로 검증 (1회만!)
test_result = backtest(test_data, best_params)

# 3단계: 성능 비교
if test_result < train_result * 0.7:  # 30% 이상 하락
    print("⚠️ 과적합 의심!")
```

#### ❌ 잘못된 예시 2: 결과 보고 전략 수정
```python
# 문제: 백테스트 결과를 보고 전략 수정
result1 = backtest(strategy_v1)  # 수익률 50%
# → "음, RSI 조건을 추가하면 어떨까?"

result2 = backtest(strategy_v2)  # 수익률 80%
# → "이번엔 볼륨 필터를 추가해볼까?"

result3 = backtest(strategy_v3)  # 수익률 120%
# → "완벽해!" → ❌ 과적합!
```

#### ✅ 올바른 예시 2: 사전 가설 수립
```python
# 해결: 백테스트 전에 가설과 전략을 완전히 정의

# 1단계: 가설 수립 (데이터 보지 않고)
hypothesis = """
시장이 과열되면(RSI > 70) 조정이 온다.
따라서 RSI < 30일 때 매수하면 수익이 날 것이다.
"""

# 2단계: 전략 완전히 정의
strategy = {
    'entry': 'RSI < 30',
    'exit': 'RSI > 70',
    'rsi_period': 14  # 표준 값 사용
}

# 3단계: 백테스트 (1회만)
result = backtest(strategy)

# 4단계: 성공/실패 판정
# 결과가 나쁘더라도 파라미터 변경 금지!
```

### 🛡️ 방지 방법

1. **Walk-Forward Analysis**
   ```python
   # 시간에 따라 순차적으로 검증
   periods = [
       ('2018', '2019'),  # 훈련
       ('2020', '2020'),  # 검증
       ('2021', '2021'),  # 훈련
       ('2022', '2022'),  # 검증
       ('2023', '2024'),  # 최종 테스트
   ]
   ```

2. **Cross-Validation (시계열용)**
   ```python
   from sklearn.model_selection import TimeSeriesSplit

   tscv = TimeSeriesSplit(n_splits=5)
   for train_idx, test_idx in tscv.split(data):
       train = data.iloc[train_idx]
       test = data.iloc[test_idx]
       # 각 fold에서 성능 측정
   ```

3. **파라미터 제한**
   ```python
   # 표준 파라미터 사용
   STANDARD_PARAMS = {
       'SMA': [20, 50, 200],        # 3가지만
       'RSI': [14],                 # 표준값만
       'RSI_threshold': [30, 70]    # 전통적 값만
   }
   # 임의 값 탐색 금지!
   ```

4. **Out-of-Sample 필수**
   ```python
   # 최소 20%는 테스트용으로 남겨두기
   train_size = int(len(df) * 0.8)
   train_data = df[:train_size]
   test_data = df[train_size:]  # 절대 훈련에 사용 금지
   ```

### ✅ 체크리스트
- [ ] 데이터를 Train/Test로 분할했는가?
- [ ] Test 데이터는 1회만 사용했는가?
- [ ] 백테스트 결과를 보고 파라미터를 수정하지 않았는가?
- [ ] 파라미터 최적화를 과도하게 하지 않았는가? (조합 100개 이하)
- [ ] Walk-Forward 또는 Cross-Validation을 수행했는가?

---

## 5. 기타 편향들

### 5.1 Transaction Cost Bias (거래비용 무시)

#### 문제
```python
# ❌ 수수료, 슬리피지 무시
returns = (sell_price / buy_price - 1)
```

#### 해결
```python
# ✅ 모든 비용 반영
fee_rate = 0.001  # 0.1% 수수료
slippage = 0.002  # 0.2% 슬리피지

buy_cost = buy_price * (1 + fee_rate + slippage)
sell_revenue = sell_price * (1 - fee_rate - slippage)
returns = (sell_revenue / buy_cost - 1)
```

### 5.2 Market Impact Bias (시장 충격 무시)

#### 문제
대량 주문 시 가격에 미치는 영향 무시

```python
# ❌ 무한한 유동성 가정
position_size = 1_000_000_000  # 10억 원
buy_price = df['Close']  # 가격 변동 없다고 가정
```

#### 해결
```python
# ✅ 거래량 대비 주문 크기 제한
daily_volume = df['Volume'] * df['Close']
max_position = daily_volume * 0.01  # 일일 거래량의 1% 이하

if position_size > max_position:
    # 시장 충격 반영 또는 주문 크기 제한
    price_impact = (position_size / daily_volume) * 0.1
    buy_price = df['Close'] * (1 + price_impact)
```

### 5.3 Psychological Bias (심리적 편향)

#### 문제
백테스트대로 실행하지 못하는 심리적 요인 무시

**실제 사례**:
- 연속 5번 손실 후 전략 중단
- 큰 하락장에서 패닉 매도
- 수익 나면 너무 일찍 매도

#### 해결
```python
# MDD, 최대 연속 손실 등 심리적 지표 계산
def calculate_psychological_metrics(returns):
    # 최대 연속 손실
    consecutive_losses = 0
    max_consecutive_losses = 0

    for r in returns:
        if r < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses,
                                        consecutive_losses)
        else:
            consecutive_losses = 0

    return {
        'max_consecutive_losses': max_consecutive_losses,
        'max_drawdown_duration_days': calculate_max_dd_duration(returns)
    }

# 자신이 견딜 수 있는 수준인지 평가
metrics = calculate_psychological_metrics(returns)
if metrics['max_consecutive_losses'] > 10:
    print("⚠️ 연속 10번 손실 - 실전에서 견디기 어려울 수 있음")
```

---

## 6. 백테스팅 체크리스트

### 📋 전체 체크리스트

#### Phase 1: 전략 설계
- [ ] 명확한 가설이 있는가?
- [ ] 백테스트 전에 전략이 완전히 정의되었는가?
- [ ] 파라미터가 논리적 근거를 가지고 있는가?

#### Phase 2: 데이터 준비
- [ ] 데이터를 Train/Test로 분할했는가?
- [ ] 상장폐지된 종목도 포함되었는가?
- [ ] 데이터 품질을 검증했는가? (결측치, 이상치)

#### Phase 3: 코드 구현
- [ ] **Look-ahead Bias**: shift(1)을 적절히 사용했는가?
- [ ] **Perfect Execution**: 현실적인 체결 가격을 사용하는가?
- [ ] **Transaction Cost**: 수수료와 슬리피지를 반영했는가?
- [ ] 모든 지표가 전일 데이터만 사용하는가?

#### Phase 4: 백테스트 실행
- [ ] Test 데이터는 1회만 사용하는가?
- [ ] 결과를 보고 파라미터를 수정하지 않았는가?
- [ ] 과도한 최적화를 하지 않았는가?

#### Phase 5: 결과 검증
- [ ] Train과 Test 성능이 비슷한가?
- [ ] 샤프 비율이 2.0 이하인가? (너무 높으면 의심)
- [ ] 승률이 90% 이상은 아닌가? (너무 높으면 의심)
- [ ] MDD가 현실적인가?
- [ ] 심리적으로 견딜 수 있는 수준인가?

#### Phase 6: 민감도 분석
- [ ] 파라미터를 약간 변경해도 결과가 크게 바뀌지 않는가?
- [ ] 다른 기간에도 작동하는가?
- [ ] 다른 종목에도 적용 가능한가?

---

## 7. 실전 적용 가이드

### 🚦 신호등 시스템

백테스팅 결과에 대한 신뢰도 평가:

#### 🟢 GREEN (신뢰 가능)
- Train/Test 성능 차이 < 30%
- 샤프 비율: 0.5 ~ 2.0
- 승률: 45% ~ 60%
- MDD: -20% ~ -50%
- 파라미터에 강건함 (±20% 변경 시 성능 유지)

#### 🟡 YELLOW (주의 필요)
- Train/Test 성능 차이 30% ~ 50%
- 샤프 비율: 2.0 ~ 3.0
- 승률: 60% ~ 70%
- MDD: -10% ~ -20% 또는 -50% ~ -70%
- 파라미터에 약간 민감

#### 🔴 RED (의심 필요)
- Train/Test 성능 차이 > 50%
- 샤프 비율 > 3.0
- 승률 > 70%
- MDD < -10% (너무 낮음) 또는 > -70% (너무 높음)
- 파라미터에 매우 민감

### 📊 리포트 템플릿

```markdown
# 백테스팅 리포트

## 1. 전략 정의
- **가설**: [전략의 논리적 근거]
- **진입 조건**: [구체적 조건]
- **청산 조건**: [구체적 조건]
- **파라미터**: [값과 선택 근거]

## 2. 데이터
- **종목**: BTC, ETH, ADA, XRP
- **기간**: 2018-01-01 ~ 2024-12-31
- **Train**: 2018-01-01 ~ 2021-12-31
- **Test**: 2022-01-01 ~ 2024-12-31
- **생존 편향**: ✅ 상장폐지 종목 포함

## 3. 성과 지표
| 지표 | Train | Test | 차이 |
|------|-------|------|------|
| 총 수익률 | 450% | 380% | -15.6% ✅ |
| CAGR | 55% | 48% | -12.7% ✅ |
| Sharpe | 1.2 | 1.0 | -16.7% ✅ |
| MDD | -35% | -42% | +20% ⚠️ |

## 4. 편향 체크
- [x] Look-ahead Bias: shift(1) 사용 확인
- [x] Perfect Execution: 슬리피지 0.2% 적용
- [x] Survivorship Bias: N/A (암호화폐 4종목만)
- [x] Data Snooping: Test 데이터 1회만 사용
- [x] Transaction Cost: 수수료 0.1% 적용

## 5. 민감도 분석
- SMA 기간 30→35: 수익률 -8%
- SMA 기간 30→25: 수익률 -12%
→ ✅ 파라미터에 강건함

## 6. 리스크 평가
- 최대 연속 손실: 7회 ✅
- 최대 DD 기간: 180일 ⚠️
- 심리적 부담: 중간 수준

## 7. 최종 판정
🟢 GREEN - 실전 적용 가능
```

---

## 8. 참고 자료

### 📚 추천 도서
1. "Advances in Financial Machine Learning" - Marcos Lopez de Prado
2. "Evidence-Based Technical Analysis" - David Aronson
3. "Quantitative Trading" - Ernest Chan

### 🔗 유용한 링크
- [백테스팅 함정들 (QuantStart)](https://www.quantstart.com)
- [과적합 방지 방법](https://www.machinelearningplus.com)

### 💻 도구
- `backtrader`: 파이썬 백테스팅 프레임워크
- `zipline`: Quantopian의 백테스팅 엔진
- `vectorbt`: 빠른 벡터화 백테스팅

---

## 9. 요약

### 🎯 핵심 원칙

1. **의심하라**: 결과가 너무 좋으면 버그가 있을 확률이 높다
2. **검증하라**: Train/Test 분할은 필수
3. **현실화하라**: 슬리피지, 수수료, 시장 충격 모두 반영
4. **단순하라**: 복잡한 전략일수록 과적합 위험 증가

### ⚠️ 레드 플래그

다음 중 하나라도 해당하면 **즉시 재검토**:

- ✋ 샤프 비율 > 3.0
- ✋ 승률 > 70%
- ✋ MDD < 10%
- ✋ Train/Test 성능 차이 > 50%
- ✋ 파라미터 1% 변경 시 성능 20% 이상 변화
- ✋ shift(1) 없이 당일 지표로 당일 매수

### ✅ 체크 순서

```
1. Look-ahead Bias 체크
   └─> shift(1) 확인

2. Perfect Execution Bias 체크
   └─> 슬리피지 확인

3. Data Snooping Bias 체크
   └─> Train/Test 분할 확인

4. Survivorship Bias 체크
   └─> 상장폐지 종목 포함 확인

5. 전체 재검증
   └─> 독립적인 데이터셋에서 1회 테스트
```

---

**마지막 조언**: 백테스트는 전략의 **가능성**을 보는 것이지, **보장**이 아닙니다. 항상 소액으로 실전 테스트를 거쳐야 합니다.

**"In God we trust, all others must bring data. But verify that data first."**

---

*문서 버전: 1.0*
*최종 수정: 2025-11-07*
*작성자: Backtesting Team*
