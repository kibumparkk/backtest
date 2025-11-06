# 백테스팅 시각화 가이드

## 개요
백테스팅 결과를 효과적으로 시각화하기 위한 가이드입니다. matplotlib의 subplot을 사용하여 전략 성과와 리스크 지표를 한눈에 확인할 수 있습니다.

## 시각화 구성

### SUBPLOT 구조
2개의 서브플롯을 세로로 배치하여 다음 정보를 표시합니다:

#### 그림 1: 누적 수익률 (Cumulative Return)
- **데이터**:
  - 전략의 누적 수익률 (1에서 시작)
  - Buy and Hold 벤치마크 (1에서 시작)
- **Y축**: 로그 스케일 (log scale)
  - 장기간 데이터의 변화를 효과적으로 표현
  - 수익률의 비율 변화를 시각적으로 비교 가능
- **목적**: 전략 성과와 벤치마크 비교

#### 그림 2: Drawdown
- **데이터**:
  - 전략의 Drawdown (%)
  - Buy and Hold의 Drawdown (%)
- **Y축**: 퍼센트 단위 (%)
  - 최대 낙폭을 직관적으로 확인
  - 리스크 관리 지표로 활용
- **목적**: 손실 구간과 회복 패턴 분석

## 코드 예제

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# subplot 생성 (2행 1열, x축 공유)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, dpi=150)

# 그림 1: 누적 수익률 (LOG Y축)
ax1.plot(df.index, df['btc_hold'], label='BTC Hold', linewidth=2)
ax1.plot(df_strategy.index, df_strategy['cumulative_return'],
         label='Strategy', linewidth=2)
ax1.set_ylabel('Cumulative Return (Starting from 1)', fontsize=10)
ax1.set_title('Strategy Performance vs. Buy & Hold', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.set_yscale('log')  # LOG 스케일 적용
ax1.grid(True, alpha=0.3)

# 그림 2: Drawdown (% 표시)
ax2.plot(df.index, df['drawdown_btc'] * 100,
         label='BTC Hold Drawdown', linewidth=2)
ax2.plot(df_strategy.index, df_strategy['drawdown_strategy'] * 100,
         label='Strategy Drawdown', linewidth=2)
ax2.set_ylabel('Drawdown (%)', fontsize=10)
ax2.set_xlabel('Date', fontsize=10)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Y축을 % 형식으로 포맷팅
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

plt.tight_layout()
plt.show()
```

## 주요 설정 설명

### 1. Log Y축 (그림 1)
```python
ax1.set_yscale('log')
```
- 누적 수익이 기하급수적으로 증가할 때 유용
- 초기와 후기 변화를 동일한 비율로 표현

### 2. Drawdown 계산
```python
# 전략 Drawdown
df['cumulative_max'] = df['cumulative_return'].cummax()
df['drawdown'] = (df['cumulative_return'] - df['cumulative_max']) / df['cumulative_max']
```
- cummax(): 누적 최고값
- 현재값과 최고값의 차이를 비율로 계산

### 3. % 포맷팅 (그림 2)
```python
# 방법 1: 데이터를 100 곱하기
ax2.plot(df.index, df['drawdown'] * 100)

# 방법 2: 포맷터 사용
import matplotlib.ticker as mticker
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
```

## 시각화 개선 팁

### 1. 색상 및 스타일
```python
# 색상 지정
ax1.plot(df.index, df['btc_hold'], label='BTC Hold',
         color='#FF9800', linewidth=2, alpha=0.8)
ax1.plot(df.index, df['strategy'], label='Strategy',
         color='#2196F3', linewidth=2, alpha=0.8)
```

### 2. 그리드 스타일
```python
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
```

### 3. 레이아웃 최적화
```python
plt.tight_layout()  # 자동 여백 조정
plt.subplots_adjust(hspace=0.1)  # subplot 간격 조정
```

### 4. 주요 이벤트 표시
```python
# 매수/매도 신호 표시
buy_signals = df[df['buy_condition'] == True]
sell_signals = df[df['sell_condition'] == True]

ax1.scatter(buy_signals.index, buy_signals['cumulative_return'],
           marker='^', color='green', s=50, alpha=0.7, label='Buy')
ax1.scatter(sell_signals.index, sell_signals['cumulative_return'],
           marker='v', color='red', s=50, alpha=0.7, label='Sell')
```

## 해석 가이드

### 누적 수익률 그래프 해석
- **상승 추세**: 전략이 수익을 내고 있음
- **벤치마크 대비 위치**: 전략의 상대적 성과
- **변동성**: 선의 기울기 변화로 파악

### Drawdown 그래프 해석
- **0% 근처**: 최고점 갱신 중
- **음수 값**: 현재 손실 구간
- **최대 Drawdown**: 그래프의 최저점
- **회복 속도**: 골짜기에서 0%로 돌아오는 시간

## 성과 지표 추가

### 그래프에 통계 정보 표시
```python
# 주요 지표 계산
total_return = df['cumulative_return'].iloc[-1] - 1
max_drawdown = df['drawdown'].min()
sharpe_ratio = calculate_sharpe_ratio(df['return'])

# 텍스트 박스로 표시
textstr = f'Total Return: {total_return:.2%}\n'
textstr += f'Max Drawdown: {max_drawdown:.2%}\n'
textstr += f'Sharpe Ratio: {sharpe_ratio:.2f}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)
```

## 저장 및 출력

### 고해상도 이미지 저장
```python
# PNG로 저장
plt.savefig('backtest_result.png', dpi=300, bbox_inches='tight')

# PDF로 저장 (벡터 형식)
plt.savefig('backtest_result.pdf', bbox_inches='tight')
```

### 인터랙티브 플롯 (선택사항)
```python
# plotly를 사용한 인터랙티브 차트
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1,
                    subplot_titles=('Cumulative Return', 'Drawdown'))

# 데이터 추가
fig.add_trace(go.Scatter(x=df.index, y=df['btc_hold'],
                         name='BTC Hold'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['cumulative_return'],
                         name='Strategy'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['drawdown_btc']*100,
                         name='BTC DD'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['drawdown_strategy']*100,
                         name='Strategy DD'), row=2, col=1)

fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
fig.show()
```

## 체크리스트
- [ ] 누적 수익률이 1에서 시작하는가?
- [ ] Y축이 로그 스케일로 설정되어 있는가?
- [ ] Drawdown이 %로 표시되는가?
- [ ] 범례가 명확한가?
- [ ] 그리드가 가독성을 높이는가?
- [ ] 제목과 라벨이 명확한가?
- [ ] 색상이 구분하기 쉬운가?

## 참고 사항
- Drawdown은 항상 0 이하의 값을 가집니다
- 로그 스케일에서는 음수 값을 표시할 수 없으므로 Drawdown 그래프에는 사용하지 않습니다
- 매수/매도 시점의 Drawdown은 실제 고점 대비 Drawdown보다 과소평가될 수 있습니다
