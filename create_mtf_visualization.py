"""
MTF 전략 시각화 생성
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bitcoin_mtf_loop_based import MTFLoopBased
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 12)

# 데이터 로드 및 전략 실행
analyzer = MTFLoopBased(slippage=0.002)
analyzer.load_data()
metrics_df = analyzer.run_all_strategies()

# Top 5 추출
benchmark_sharpe = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')]['Sharpe Ratio'].iloc[0]
top5 = metrics_df[~metrics_df['Strategy'].str.contains('BENCHMARK')].nlargest(5, 'Sharpe Ratio')
all_strategies = pd.concat([
    metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')],
    top5
])

# 1. Sharpe Ratio 비교 차트
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Sharpe Ratio
ax1 = axes[0, 0]
colors = ['#FF6B6B' if 'BENCHMARK' in s else '#4ECDC4' for s in all_strategies['Strategy']]
bars = ax1.barh(range(len(all_strategies)), all_strategies['Sharpe Ratio'], color=colors)
ax1.set_yticks(range(len(all_strategies)))
ax1.set_yticklabels([s.replace('_', ' ') for s in all_strategies['Strategy']], fontsize=10)
ax1.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax1.set_title('Sharpe Ratio Comparison (Top 5 MTF Strategies)', fontsize=14, fontweight='bold')
ax1.axvline(x=benchmark_sharpe, color='red', linestyle='--', linewidth=2, label='Benchmark')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# 값 표시
for i, (bar, val) in enumerate(zip(bars, all_strategies['Sharpe Ratio'])):
    improvement = ((val / benchmark_sharpe - 1) * 100) if 'BENCHMARK' not in all_strategies.iloc[i]['Strategy'] else 0
    label = f'{val:.4f}' if improvement == 0 else f'{val:.4f} (+{improvement:.1f}%)'
    ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, label,
             va='center', fontsize=9, fontweight='bold')

# CAGR 비교
ax2 = axes[0, 1]
colors = ['#FF6B6B' if 'BENCHMARK' in s else '#95E1D3' for s in all_strategies['Strategy']]
bars = ax2.barh(range(len(all_strategies)), all_strategies['CAGR (%)'], color=colors)
ax2.set_yticks(range(len(all_strategies)))
ax2.set_yticklabels([s.replace('_', ' ') for s in all_strategies['Strategy']], fontsize=10)
ax2.set_xlabel('CAGR (%)', fontsize=12, fontweight='bold')
ax2.set_title('CAGR Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, all_strategies['CAGR (%)']):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
             va='center', fontsize=9, fontweight='bold')

# MDD 비교
ax3 = axes[1, 0]
colors = ['#FF6B6B' if 'BENCHMARK' in s else '#F38181' for s in all_strategies['Strategy']]
bars = ax3.barh(range(len(all_strategies)), all_strategies['MDD (%)'], color=colors)
ax3.set_yticks(range(len(all_strategies)))
ax3.set_yticklabels([s.replace('_', ' ') for s in all_strategies['Strategy']], fontsize=10)
ax3.set_xlabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
ax3.set_title('Maximum Drawdown (Lower is Better)', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, all_strategies['MDD (%)']):
    ax3.text(val - 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
             va='center', ha='right', fontsize=9, fontweight='bold')

# Risk-Return Scatter
ax4 = axes[1, 1]
for idx, row in all_strategies.iterrows():
    if 'BENCHMARK' in row['Strategy']:
        ax4.scatter(row['MDD (%)'], row['CAGR (%)'], s=300, c='red', marker='s',
                   label='Benchmark', zorder=5, edgecolors='black', linewidths=2)
        ax4.annotate('Benchmark', (row['MDD (%)'], row['CAGR (%)']),
                    xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')
    else:
        ax4.scatter(row['MDD (%)'], row['CAGR (%)'], s=200, c='#4ECDC4',
                   marker='o', zorder=3, edgecolors='black', linewidths=1.5)
        # 전략명 간소화
        label = row['Strategy'].split('+')[0].replace('_', ' ')
        ax4.annotate(label, (row['MDD (%)'], row['CAGR (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.set_xlabel('Maximum Drawdown (%) - Lower is Better', fontsize=12, fontweight='bold')
ax4.set_ylabel('CAGR (%)', fontsize=12, fontweight='bold')
ax4.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('mtf_strategies_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mtf_strategies_comparison.png")

# 2. Top 5 상세 비교 테이블 이미지
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

# 테이블 데이터 준비
table_data = []
for idx, row in top5.iterrows():
    improvement = ((row['Sharpe Ratio'] / benchmark_sharpe - 1) * 100)
    table_data.append([
        row['Strategy'].replace('_', ' '),
        f"{row['Sharpe Ratio']:.4f}",
        f"+{improvement:.2f}%",
        f"{row['Total Return (%)']:.0f}%",
        f"{row['CAGR (%)']:.2f}%",
        f"{row['MDD (%)']:.2f}%",
        f"{row['Total Trades']}"
    ])

# 벤치마크 추가
bench_row = metrics_df[metrics_df['Strategy'].str.contains('BENCHMARK')].iloc[0]
table_data.insert(0, [
    'BENCHMARK (Close > SMA30)',
    f"{bench_row['Sharpe Ratio']:.4f}",
    '-',
    f"{bench_row['Total Return (%)']:.0f}%",
    f"{bench_row['CAGR (%)']:.2f}%",
    f"{bench_row['MDD (%)']:.2f}%",
    f"{bench_row['Total Trades']}"
])

headers = ['Strategy', 'Sharpe', 'Improvement', 'Total Return', 'CAGR', 'MDD', 'Trades']

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                colWidths=[0.35, 0.1, 0.12, 0.12, 0.1, 0.1, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# 헤더 스타일
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 벤치마크 행 강조
for i in range(len(headers)):
    table[(1, i)].set_facecolor('#FFE5E5')
    table[(1, i)].set_text_props(weight='bold')

# Top 5 행 색상
for row in range(2, 7):
    for col in range(len(headers)):
        if row == 2:  # #1
            table[(row, col)].set_facecolor('#FFD700')
        elif row == 3:  # #2
            table[(row, col)].set_facecolor('#C0C0C0')
        elif row == 4:  # #3
            table[(row, col)].set_facecolor('#CD7F32')
        else:
            table[(row, col)].set_facecolor('#F0F0F0')

plt.title('Top 5 Multi-Timeframe Strategies - Detailed Metrics',
         fontsize=16, fontweight='bold', pad=20)
plt.savefig('mtf_strategies_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mtf_strategies_table.png")

print("\n✅ All visualizations created successfully!")
