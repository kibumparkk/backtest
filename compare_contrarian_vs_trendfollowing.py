"""
역추세 vs 추세 추종 전략 비교

동일한 5개 알트코인(STEEM, ANKR, CHZ, MANA, ZIL)에 대해
역추세 전략과 추세 추종 전략을 직접 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
contrarian_metrics = pd.read_csv('altcoin_contrarian_metrics.csv')
trendfollowing_metrics = pd.read_csv('altcoin_trendfollowing_metrics.csv')

# 포트폴리오 성과만 추출
contrarian_portfolio = contrarian_metrics[contrarian_metrics['Strategy'].str.contains('Portfolio')].copy()
trendfollowing_portfolio = trendfollowing_metrics[trendfollowing_metrics['Strategy'].str.contains('Portfolio')].copy()

# 전략명 단순화
contrarian_portfolio['Strategy Type'] = 'Contrarian'
trendfollowing_portfolio['Strategy Type'] = 'Trend-Following'

contrarian_portfolio['Strategy Name'] = contrarian_portfolio['Strategy'].str.replace(' Portfolio', '').str.replace(' Contrarian', '')
trendfollowing_portfolio['Strategy Name'] = trendfollowing_portfolio['Strategy'].str.replace(' Portfolio', '').str.replace(' Trend-Following', '')

# 데이터 결합
combined = pd.concat([contrarian_portfolio, trendfollowing_portfolio], ignore_index=True)

# 포트폴리오 시계열 데이터 로드
portfolio_data = {}

# 역추세 전략
portfolio_data['Bollinger Contrarian'] = pd.read_csv('portfolio_bollinger_contrarian.csv', index_col=0, parse_dates=True)
portfolio_data['RSI Contrarian'] = pd.read_csv('portfolio_rsi_contrarian.csv', index_col=0, parse_dates=True)
portfolio_data['Z-Score Contrarian'] = pd.read_csv('portfolio_z-score_contrarian.csv', index_col=0, parse_dates=True)

# 추세 추종 전략
portfolio_data['Bollinger Trend-Following'] = pd.read_csv('portfolio_bollinger_trend_following.csv', index_col=0, parse_dates=True)
portfolio_data['RSI Trend-Following'] = pd.read_csv('portfolio_rsi_trend_following.csv', index_col=0, parse_dates=True)
portfolio_data['Z-Score Trend-Following'] = pd.read_csv('portfolio_z_score_trend_following.csv', index_col=0, parse_dates=True)

print("\n" + "="*150)
print(f"{'역추세 vs 추세 추종 전략 비교':^150}")
print("="*150)
print("\n알트코인: STEEM, ANKR, CHZ, MANA, ZIL (동일 비중 20%)")
print("기간: 2018-01-01 ~ 2025-11-12")
print("슬리피지: 0.2%\n")

print("="*150)
print(f"{'포트폴리오 성과 비교':^150}")
print("="*150)
print(combined.to_string(index=False))
print("="*150 + "\n")

# 시각화
fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# 1. 전체 누적 수익률 비교 (역추세)
ax1 = fig.add_subplot(gs[0, :])
for strategy_name, data in portfolio_data.items():
    if 'Contrarian' in strategy_name:
        ax1.plot(data.index, data['cumulative'], label=strategy_name,
                linewidth=2.5, alpha=0.8, linestyle='--')
    else:
        ax1.plot(data.index, data['cumulative'], label=strategy_name,
                linewidth=2.5, alpha=0.8)

ax1.set_title('Cumulative Returns: Contrarian (--) vs Trend-Following (-) Strategies',
             fontsize=16, fontweight='bold')
ax1.set_ylabel('Cumulative Return (log scale)', fontsize=12)
ax1.set_xlabel('Date', fontsize=12)
ax1.legend(loc='upper left', fontsize=11, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 2. 총 수익률 비교
ax2 = fig.add_subplot(gs[1, 0])
x = np.arange(len(combined['Strategy Name'].unique()))
width = 0.35
contrarian_data = combined[combined['Strategy Type'] == 'Contrarian'].sort_values('Strategy Name')
trendfollowing_data = combined[combined['Strategy Type'] == 'Trend-Following'].sort_values('Strategy Name')

bars1 = ax2.bar(x - width/2, contrarian_data['Total Return (%)'], width,
                label='Contrarian', alpha=0.8, color='coral')
bars2 = ax2.bar(x + width/2, trendfollowing_data['Total Return (%)'], width,
                label='Trend-Following', alpha=0.8, color='skyblue')

ax2.set_ylabel('Total Return (%)', fontsize=11)
ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(contrarian_data['Strategy Name'], rotation=15, ha='right')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%' if abs(height) < 1000 else f'{height/1000:.1f}k%',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# 3. CAGR 비교
ax3 = fig.add_subplot(gs[1, 1])
bars1 = ax3.bar(x - width/2, contrarian_data['CAGR (%)'], width,
                label='Contrarian', alpha=0.8, color='coral')
bars2 = ax3.bar(x + width/2, trendfollowing_data['CAGR (%)'], width,
                label='Trend-Following', alpha=0.8, color='skyblue')

ax3.set_ylabel('CAGR (%)', fontsize=11)
ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(contrarian_data['Strategy Name'], rotation=15, ha='right')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# 4. MDD 비교
ax4 = fig.add_subplot(gs[1, 2])
bars1 = ax4.bar(x - width/2, contrarian_data['MDD (%)'], width,
                label='Contrarian', alpha=0.8, color='coral')
bars2 = ax4.bar(x + width/2, trendfollowing_data['MDD (%)'], width,
                label='Trend-Following', alpha=0.8, color='skyblue')

ax4.set_ylabel('MDD (%)', fontsize=11)
ax4.set_title('Maximum Drawdown Comparison (lower is better)', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(contrarian_data['Strategy Name'], rotation=15, ha='right')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# 5. 샤프 비율 비교
ax5 = fig.add_subplot(gs[2, 0])
bars1 = ax5.bar(x - width/2, contrarian_data['Sharpe Ratio'], width,
                label='Contrarian', alpha=0.8, color='coral')
bars2 = ax5.bar(x + width/2, trendfollowing_data['Sharpe Ratio'], width,
                label='Trend-Following', alpha=0.8, color='skyblue')

ax5.set_ylabel('Sharpe Ratio', fontsize=11)
ax5.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(contrarian_data['Strategy Name'], rotation=15, ha='right')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# 6. 승률 비교
ax6 = fig.add_subplot(gs[2, 1])
bars1 = ax6.bar(x - width/2, contrarian_data['Win Rate (%)'], width,
                label='Contrarian', alpha=0.8, color='coral')
bars2 = ax6.bar(x + width/2, trendfollowing_data['Win Rate (%)'], width,
                label='Trend-Following', alpha=0.8, color='skyblue')

ax6.set_ylabel('Win Rate (%)', fontsize=11)
ax6.set_title('Win Rate Comparison', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(contrarian_data['Strategy Name'], rotation=15, ha='right')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom', fontsize=9)

# 7. Profit Factor 비교
ax7 = fig.add_subplot(gs[2, 2])
bars1 = ax7.bar(x - width/2, contrarian_data['Profit Factor'], width,
                label='Contrarian', alpha=0.8, color='coral')
bars2 = ax7.bar(x + width/2, trendfollowing_data['Profit Factor'], width,
                label='Trend-Following', alpha=0.8, color='skyblue')

ax7.set_ylabel('Profit Factor', fontsize=11)
ax7.set_title('Profit Factor Comparison', fontsize=13, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(contrarian_data['Strategy Name'], rotation=15, ha='right')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')
ax7.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

# 8. Return vs Risk 산점도
ax8 = fig.add_subplot(gs[3, 0])
ax8.scatter(contrarian_data['MDD (%)'], contrarian_data['CAGR (%)'],
           s=200, alpha=0.7, c='coral', marker='o', label='Contrarian', edgecolors='black', linewidth=1.5)
ax8.scatter(trendfollowing_data['MDD (%)'], trendfollowing_data['CAGR (%)'],
           s=200, alpha=0.7, c='skyblue', marker='s', label='Trend-Following', edgecolors='black', linewidth=1.5)

for idx, row in contrarian_data.iterrows():
    ax8.annotate(row['Strategy Name'],
                (row['MDD (%)'], row['CAGR (%)']),
                fontsize=9, ha='center', va='bottom')
for idx, row in trendfollowing_data.iterrows():
    ax8.annotate(row['Strategy Name'],
                (row['MDD (%)'], row['CAGR (%)']),
                fontsize=9, ha='center', va='top')

ax8.set_xlabel('MDD (%)', fontsize=11)
ax8.set_ylabel('CAGR (%)', fontsize=11)
ax8.set_title('Return vs Risk: Contrarian vs Trend-Following', fontsize=13, fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3)
ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# 9. 전략별 누적 수익률 비교 (Bollinger)
ax9 = fig.add_subplot(gs[3, 1])
ax9.plot(portfolio_data['Bollinger Contrarian'].index,
        portfolio_data['Bollinger Contrarian']['cumulative'],
        label='Bollinger Contrarian', linewidth=2.5, alpha=0.8, color='coral', linestyle='--')
ax9.plot(portfolio_data['Bollinger Trend-Following'].index,
        portfolio_data['Bollinger Trend-Following']['cumulative'],
        label='Bollinger Trend-Following', linewidth=2.5, alpha=0.8, color='skyblue')
ax9.set_ylabel('Cumulative Return', fontsize=11)
ax9.set_xlabel('Date', fontsize=11)
ax9.set_title('Bollinger Strategy Comparison', fontsize=13, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3)
ax9.set_yscale('log')

# 10. 전략별 누적 수익률 비교 (RSI)
ax10 = fig.add_subplot(gs[3, 2])
ax10.plot(portfolio_data['RSI Contrarian'].index,
         portfolio_data['RSI Contrarian']['cumulative'],
         label='RSI Contrarian', linewidth=2.5, alpha=0.8, color='coral', linestyle='--')
ax10.plot(portfolio_data['RSI Trend-Following'].index,
         portfolio_data['RSI Trend-Following']['cumulative'],
         label='RSI Trend-Following', linewidth=2.5, alpha=0.8, color='skyblue')
ax10.set_ylabel('Cumulative Return', fontsize=11)
ax10.set_xlabel('Date', fontsize=11)
ax10.set_title('RSI Strategy Comparison', fontsize=13, fontweight='bold')
ax10.legend(fontsize=9)
ax10.grid(True, alpha=0.3)
ax10.set_yscale('log')

# 전체 제목
fig.suptitle('Contrarian vs Trend-Following Strategies Comparison\nAltcoins: STEEM, ANKR, CHZ, MANA, ZIL (Equal-Weight 20% each)',
            fontsize=18, fontweight='bold', y=0.995)

plt.savefig('contrarian_vs_trendfollowing_comparison.png', dpi=300, bbox_inches='tight')
print("\nComparison chart saved to contrarian_vs_trendfollowing_comparison.png")
plt.close()

# 성과 차이 요약
print("\n" + "="*150)
print(f"{'주요 인사이트':^150}")
print("="*150)

print("\n1. 총 수익률 비교:")
for i, strategy in enumerate(contrarian_data['Strategy Name']):
    contrarian_return = contrarian_data.iloc[i]['Total Return (%)']
    trendfollowing_return = trendfollowing_data.iloc[i]['Total Return (%)']
    diff = trendfollowing_return - contrarian_return
    print(f"   {strategy:20s}: 역추세 {contrarian_return:8.2f}% vs 추세추종 {trendfollowing_return:10.2f}% (차이: {diff:+10.2f}%p)")

print("\n2. CAGR 비교:")
for i, strategy in enumerate(contrarian_data['Strategy Name']):
    contrarian_cagr = contrarian_data.iloc[i]['CAGR (%)']
    trendfollowing_cagr = trendfollowing_data.iloc[i]['CAGR (%)']
    diff = trendfollowing_cagr - contrarian_cagr
    print(f"   {strategy:20s}: 역추세 {contrarian_cagr:8.2f}% vs 추세추종 {trendfollowing_cagr:8.2f}% (차이: {diff:+8.2f}%p)")

print("\n3. 샤프 비율 비교:")
for i, strategy in enumerate(contrarian_data['Strategy Name']):
    contrarian_sharpe = contrarian_data.iloc[i]['Sharpe Ratio']
    trendfollowing_sharpe = trendfollowing_data.iloc[i]['Sharpe Ratio']
    diff = trendfollowing_sharpe - contrarian_sharpe
    print(f"   {strategy:20s}: 역추세 {contrarian_sharpe:6.2f} vs 추세추종 {trendfollowing_sharpe:6.2f} (차이: {diff:+6.2f})")

print("\n" + "="*150)
print("결론: 알트코인 시장에서는 추세 추종 전략이 역추세 전략보다 훨씬 우수한 성과를 보임")
print("      강한 추세성을 가진 알트코인 시장의 특성이 반영된 결과")
print("="*150 + "\n")
