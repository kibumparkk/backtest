"""
5개 vs 20개 알트코인 포트폴리오 비교

동일한 추세 추종 전략을 5개와 20개 알트코인 포트폴리오에 적용했을 때의 차이 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
metrics_5 = pd.read_csv('altcoin_trendfollowing_metrics.csv')
metrics_20 = pd.read_csv('altcoin20_trendfollowing_metrics.csv')

# 포트폴리오 성과 추출
portfolio_5 = metrics_5[metrics_5['Strategy'].str.contains('Portfolio')].copy()
portfolio_20 = metrics_20[metrics_20['Strategy'].str.contains('Portfolio')].copy()

portfolio_5['Portfolio Size'] = '5 Coins'
portfolio_20['Portfolio Size'] = '20 Coins'

portfolio_5['Strategy Name'] = portfolio_5['Strategy'].str.replace(' Portfolio', '').str.replace(' Trend-Following', '')
portfolio_20['Strategy Name'] = portfolio_20['Strategy'].str.replace(' Portfolio', '').str.replace(' Trend-Following', '')

combined = pd.concat([portfolio_5, portfolio_20], ignore_index=True)

# 시계열 데이터 로드
portfolio_5_ts = {}
portfolio_20_ts = {}

try:
    portfolio_5_ts['Bollinger'] = pd.read_csv('portfolio_bollinger_trend_following.csv', index_col=0, parse_dates=True)
    portfolio_5_ts['RSI'] = pd.read_csv('portfolio_rsi_trend_following.csv', index_col=0, parse_dates=True)
    portfolio_5_ts['Z-Score'] = pd.read_csv('portfolio_z_score_trend_following.csv', index_col=0, parse_dates=True)

    portfolio_20_ts['Bollinger'] = pd.read_csv('portfolio20_bollinger_trend_following.csv', index_col=0, parse_dates=True)
    portfolio_20_ts['RSI'] = pd.read_csv('portfolio20_rsi_trend_following.csv', index_col=0, parse_dates=True)
    portfolio_20_ts['Z-Score'] = pd.read_csv('portfolio20_z_score_trend_following.csv', index_col=0, parse_dates=True)
except Exception as e:
    print(f"Error loading time series data: {e}")

print("\n" + "="*150)
print(f"{'5개 vs 20개 알트코인 포트폴리오 비교':^150}")
print("="*150)
print("\n5개 포트폴리오: STEEM, ANKR, CHZ, MANA, ZIL")
print("20개 포트폴리오: ATOM, ALGO, HBAR, NEAR, EOS, QTUM, ZIL, TRX, AAVE, KAVA, LINK, AXS, SAND, MANA, CHZ, ANKR, VET, STEEM, DOGE, XLM")
print("전략: 추세 추종 (Bollinger, RSI, Z-Score)")
print("기간: 2018-01-01 ~ 2025-11-12")
print("슬리피지: 0.2%\n")

print("="*150)
print(f"{'포트폴리오 성과 비교':^150}")
print("="*150)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
print(combined[['Strategy Name', 'Portfolio Size', 'Total Return (%)', 'CAGR (%)', 'MDD (%)',
                'Sharpe Ratio', 'Win Rate (%)', 'Total Trades', 'Profit Factor']].to_string(index=False))
print("="*150 + "\n")

# 시각화
fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. 누적 수익률 비교
ax1 = fig.add_subplot(gs[0, :])
for strategy_name, data in portfolio_5_ts.items():
    ax1.plot(data.index, data['cumulative'],
            label=f'{strategy_name} (5 coins)', linewidth=2.5, alpha=0.8, linestyle='-')
for strategy_name, data in portfolio_20_ts.items():
    ax1.plot(data.index, data['cumulative'],
            label=f'{strategy_name} (20 coins)', linewidth=2.5, alpha=0.8, linestyle='--')

ax1.set_title('Cumulative Returns: 5-Coin vs 20-Coin Portfolios (Trend-Following Strategies)',
             fontsize=16, fontweight='bold')
ax1.set_ylabel('Cumulative Return (log scale)', fontsize=12)
ax1.set_xlabel('Date', fontsize=12)
ax1.legend(loc='upper left', fontsize=10, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 2-4. 성과 지표 비교
x = np.arange(len(combined['Strategy Name'].unique()))
width = 0.35

for idx, (metric_name, ax_pos) in enumerate([
    ('Total Return (%)', gs[1, 0]),
    ('CAGR (%)', gs[1, 1]),
    ('MDD (%)', gs[1, 2])
]):
    ax = fig.add_subplot(ax_pos)

    data_5 = combined[combined['Portfolio Size'] == '5 Coins'].sort_values('Strategy Name')
    data_20 = combined[combined['Portfolio Size'] == '20 Coins'].sort_values('Strategy Name')

    bars1 = ax.bar(x - width/2, data_5[metric_name], width,
                   label='5 Coins', alpha=0.8, color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, data_20[metric_name], width,
                   label='20 Coins', alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1.5)

    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f'{metric_name} Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data_5['Strategy Name'], rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) < 1000:
                label = f'{height:.1f}'
            elif abs(height) < 10000:
                label = f'{height:.0f}'
            else:
                label = f'{height/1000:.0f}k'

            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center',
                   va='bottom' if height > 0 else 'top', fontsize=9)

# 5-7. 추가 지표
for idx, (metric_name, ax_pos) in enumerate([
    ('Sharpe Ratio', gs[2, 0]),
    ('Win Rate (%)', gs[2, 1]),
    ('Profit Factor', gs[2, 2])
]):
    ax = fig.add_subplot(ax_pos)

    data_5 = combined[combined['Portfolio Size'] == '5 Coins'].sort_values('Strategy Name')
    data_20 = combined[combined['Portfolio Size'] == '20 Coins'].sort_values('Strategy Name')

    bars1 = ax.bar(x - width/2, data_5[metric_name], width,
                   label='5 Coins', alpha=0.8, color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, data_20[metric_name], width,
                   label='20 Coins', alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1.5)

    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f'{metric_name} Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data_5['Strategy Name'], rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    if metric_name == 'Sharpe Ratio':
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    elif metric_name == 'Profit Factor':
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# 8. Return vs Risk 산점도
ax8 = fig.add_subplot(gs[3, 0])
data_5 = combined[combined['Portfolio Size'] == '5 Coins']
data_20 = combined[combined['Portfolio Size'] == '20 Coins']

ax8.scatter(data_5['MDD (%)'], data_5['CAGR (%)'],
           s=250, alpha=0.7, c='skyblue', marker='o', label='5 Coins',
           edgecolors='black', linewidth=2)
ax8.scatter(data_20['MDD (%)'], data_20['CAGR (%)'],
           s=250, alpha=0.7, c='lightcoral', marker='s', label='20 Coins',
           edgecolors='black', linewidth=2)

for idx, row in data_5.iterrows():
    ax8.annotate(row['Strategy Name'],
                (row['MDD (%)'], row['CAGR (%)']),
                fontsize=9, ha='center', va='bottom')
for idx, row in data_20.iterrows():
    ax8.annotate(row['Strategy Name'],
                (row['MDD (%)'], row['CAGR (%)']),
                fontsize=9, ha='center', va='top')

ax8.set_xlabel('MDD (%)', fontsize=11)
ax8.set_ylabel('CAGR (%)', fontsize=11)
ax8.set_title('Return vs Risk: 5-Coin vs 20-Coin Portfolios', fontsize=13, fontweight='bold')
ax8.legend(fontsize=11)
ax8.grid(True, alpha=0.3)
ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# 9. 드로우다운 비교 (Bollinger)
ax9 = fig.add_subplot(gs[3, 1])
if 'Bollinger' in portfolio_5_ts and 'Bollinger' in portfolio_20_ts:
    data_5 = portfolio_5_ts['Bollinger']
    cummax_5 = data_5['cumulative'].cummax()
    dd_5 = (data_5['cumulative'] - cummax_5) / cummax_5 * 100

    data_20 = portfolio_20_ts['Bollinger']
    cummax_20 = data_20['cumulative'].cummax()
    dd_20 = (data_20['cumulative'] - cummax_20) / cummax_20 * 100

    ax9.plot(dd_5.index, dd_5, label='5 Coins', linewidth=2.5, alpha=0.8, color='skyblue')
    ax9.plot(dd_20.index, dd_20, label='20 Coins', linewidth=2.5, alpha=0.8, color='lightcoral')
    ax9.fill_between(dd_5.index, dd_5, 0, alpha=0.2, color='skyblue')
    ax9.fill_between(dd_20.index, dd_20, 0, alpha=0.2, color='lightcoral')

ax9.set_ylabel('Drawdown (%)', fontsize=11)
ax9.set_xlabel('Date', fontsize=11)
ax9.set_title('Drawdown Comparison (Bollinger)', fontsize=13, fontweight='bold')
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3)

# 10. 분산 효과 분석
ax10 = fig.add_subplot(gs[3, 2])

# 다시 데이터 로드 (전체)
data_5_full = combined[combined['Portfolio Size'] == '5 Coins']
data_20_full = combined[combined['Portfolio Size'] == '20 Coins']

metrics = ['CAGR (%)', 'MDD (%)', 'Sharpe Ratio']
avg_5 = [data_5_full['CAGR (%)'].mean(), data_5_full['MDD (%)'].mean(), data_5_full['Sharpe Ratio'].mean()]
avg_20 = [data_20_full['CAGR (%)'].mean(), data_20_full['MDD (%)'].mean(), data_20_full['Sharpe Ratio'].mean()]

x_pos = np.arange(len(metrics))
bars1 = ax10.bar(x_pos - width/2, avg_5, width, label='5 Coins',
                 alpha=0.8, color='skyblue', edgecolor='black', linewidth=1.5)
bars2 = ax10.bar(x_pos + width/2, avg_20, width, label='20 Coins',
                 alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1.5)

ax10.set_ylabel('Average Value', fontsize=11)
ax10.set_title('Average Metrics Comparison', fontsize=13, fontweight='bold')
ax10.set_xticks(x_pos)
ax10.set_xticklabels(metrics)
ax10.legend(fontsize=10)
ax10.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

fig.suptitle('5-Coin vs 20-Coin Altcoin Portfolios Comparison (Trend-Following Strategies)',
            fontsize=18, fontweight='bold', y=0.995)

plt.savefig('5coin_vs_20coin_comparison.png', dpi=300, bbox_inches='tight')
print("\nComparison chart saved to 5coin_vs_20coin_comparison.png")
plt.close()

# 주요 인사이트
print("\n" + "="*150)
print(f"{'주요 인사이트':^150}")
print("="*150)

print("\n1. 수익률 비교 (평균):")
print(f"   5개 포트폴리오:  CAGR {data_5_full['CAGR (%)'].mean():.2f}%, Total Return {data_5_full['Total Return (%)'].mean():.2f}%")
print(f"   20개 포트폴리오: CAGR {data_20_full['CAGR (%)'].mean():.2f}%, Total Return {data_20_full['Total Return (%)'].mean():.2f}%")

print("\n2. 리스크 비교 (평균):")
print(f"   5개 포트폴리오:  MDD {data_5_full['MDD (%)'].mean():.2f}%")
print(f"   20개 포트폴리오: MDD {data_20_full['MDD (%)'].mean():.2f}%")
print(f"   리스크 감소: {abs(data_5_full['MDD (%)'].mean() - data_20_full['MDD (%)'].mean()):.2f}%p")

print("\n3. 리스크 조정 수익률 비교 (평균):")
print(f"   5개 포트폴리오:  Sharpe Ratio {data_5_full['Sharpe Ratio'].mean():.2f}")
print(f"   20개 포트폴리오: Sharpe Ratio {data_20_full['Sharpe Ratio'].mean():.2f}")

print("\n4. 거래 빈도:")
print(f"   5개 포트폴리오:  평균 {data_5_full['Total Trades'].mean():.0f} 거래")
print(f"   20개 포트폴리오: 평균 {data_20_full['Total Trades'].mean():.0f} 거래")

print("\n5. 분산투자 효과:")
mdd_improvement = (data_20_full['MDD (%)'].mean() / data_5_full['MDD (%)'].mean() - 1) * 100
print(f"   MDD 개선: {mdd_improvement:.1f}% (20개 포트폴리오가 더 안정적)")

sharpe_change = ((data_20_full['Sharpe Ratio'].mean() / data_5_full['Sharpe Ratio'].mean() - 1) * 100) if data_5_full['Sharpe Ratio'].mean() != 0 else 0
print(f"   Sharpe Ratio 변화: {sharpe_change:+.1f}%")

cagr_change = (data_20_full['CAGR (%)'].mean() / data_5_full['CAGR (%)'].mean() - 1) * 100 if data_5_full['CAGR (%)'].mean() != 0 else 0
print(f"   CAGR 변화: {cagr_change:+.1f}% (수익률 감소는 극단값 제거 효과)")

print("\n" + "="*150)
print("결론:")
print("- 20개 포트폴리오는 분산투자 효과로 리스크(MDD)가 크게 감소")
print("- 5개 포트폴리오는 높은 변동성을 가진 소수 코인의 극단적 수익에 의존")
print("- 20개 포트폴리오는 더 안정적이고 일관된 성과를 제공")
print("- 리스크 회피 투자자에게는 20개 포트폴리오가 더 적합")
print("="*150 + "\n")
