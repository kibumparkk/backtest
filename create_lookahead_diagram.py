"""
Lookahead Bias 설명 다이어그램 생성
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# ============================================================================
# 1. 시간 흐름 다이어그램
# ============================================================================
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 5)
ax1.axis('off')
ax1.set_title('Weekly Signal Timeline - How Lookahead Bias Occurs',
              fontsize=16, fontweight='bold', pad=20)

# 시간축
y_timeline = 3.5
ax1.plot([0.5, 9.5], [y_timeline, y_timeline], 'k-', linewidth=2)

# 날짜 표시
dates = ['Mon\n4/23', 'Tue\n4/24', 'Wed\n4/25', 'Thu\n4/26',
         'Fri\n4/27', 'Sat\n4/28', 'Sun\n4/29', 'Mon\n4/30']
x_positions = np.linspace(1, 8.5, len(dates))

for x, date in zip(x_positions, dates):
    ax1.plot([x, x], [y_timeline-0.1, y_timeline+0.1], 'k-', linewidth=2)
    ax1.text(x, y_timeline-0.4, date, ha='center', va='top', fontsize=9)

# Week boundary
week_x = x_positions[0]
week_end_x = x_positions[-2]
ax1.add_patch(FancyBboxPatch((week_x-0.3, y_timeline+0.3), week_end_x-week_x+0.6, 0.7,
                             boxstyle="round,pad=0.1", linewidth=2,
                             edgecolor='blue', facecolor='lightblue', alpha=0.3))
ax1.text((week_x+week_end_x)/2, y_timeline+0.7, 'Week labeled 2018-04-23',
         ha='center', va='center', fontsize=11, fontweight='bold', color='blue')

# Weekly Close
ax1.plot([x_positions[-2]], [y_timeline+1.5], 'ro', markersize=15)
ax1.text(x_positions[-2], y_timeline+1.9, 'Weekly Close\n(Sun 24:00)',
         ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')

# Buggy: Signal used from Monday
ax1.annotate('', xy=(x_positions[0], y_timeline-0.8),
             xytext=(x_positions[0], y_timeline+1.3),
             arrowprops=dict(arrowstyle='->', lw=3, color='red'))
ax1.text(x_positions[0], y_timeline-1.2, 'BUGGY:\nSignal used\nfrom Mon!',
         ha='center', va='top', fontsize=10, fontweight='bold',
         color='red', bbox=dict(boxstyle='round', facecolor='#FFE5E5', edgecolor='red', linewidth=2))

# Correct: Signal available from Tuesday
ax1.annotate('', xy=(x_positions[1], y_timeline-0.8),
             xytext=(x_positions[-2], y_timeline+1.3),
             arrowprops=dict(arrowstyle='->', lw=3, color='green'))
ax1.text(x_positions[1], y_timeline-1.2, 'CORRECT:\nSignal available\nfrom Tue',
         ha='center', va='top', fontsize=10, fontweight='bold',
         color='green', bbox=dict(boxstyle='round', facecolor='#E5FFE5', edgecolor='green', linewidth=2))

# Lookahead region
ax1.add_patch(mpatches.Rectangle((x_positions[0]-0.2, y_timeline-0.6),
                                 x_positions[-2]-x_positions[0]+0.4, 2.5,
                                 linewidth=3, edgecolor='red', facecolor='none',
                                 linestyle='--', alpha=0.7))
ax1.text((x_positions[0]+x_positions[-2])/2, y_timeline+2.3,
         'LOOKAHEAD BIAS REGION\n(Using future info!)',
         ha='center', va='bottom', fontsize=11, fontweight='bold',
         color='red', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================================
# 2. Loop-based 구현 플로우차트
# ============================================================================
ax2 = axes[1]
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 8)
ax2.axis('off')
ax2.set_title('Loop-based Implementation - How to Avoid Lookahead Bias',
              fontsize=16, fontweight='bold', pad=20)

# Step 1: Calculate weekly signals
box1_x, box1_y = 1, 6.5
ax2.add_patch(FancyBboxPatch((box1_x, box1_y), 3.5, 1,
                             boxstyle="round,pad=0.1", linewidth=2,
                             edgecolor='blue', facecolor='lightblue'))
ax2.text(box1_x+1.75, box1_y+0.7, 'Step 1:', ha='center', fontweight='bold', fontsize=10)
ax2.text(box1_x+1.75, box1_y+0.3, 'Calculate weekly signals', ha='center', fontsize=9)

# Code snippet 1
code1 = "weekly_signals[week_date] = {\n  'signal': signal,\n  'available_from': week_date + 1 day\n}"
ax2.text(box1_x+1.75, box1_y-0.8, code1, ha='center', va='top',
         fontsize=8, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#F0F0F0', edgecolor='gray'))

# Arrow
ax2.annotate('', xy=(6, 7), xytext=(4.5, 7),
             arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Step 2: Daily loop
box2_x, box2_y = 6, 6.5
ax2.add_patch(FancyBboxPatch((box2_x, box2_y), 5, 1,
                             boxstyle="round,pad=0.1", linewidth=2,
                             edgecolor='green', facecolor='lightgreen'))
ax2.text(box2_x+2.5, box2_y+0.7, 'Step 2:', ha='center', fontweight='bold', fontsize=10)
ax2.text(box2_x+2.5, box2_y+0.3, 'Loop through each day sequentially', ha='center', fontsize=9)

# Arrow down
ax2.annotate('', xy=(8.5, 6.2), xytext=(8.5, 5.5),
             arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Step 3: Check availability
box3_x, box3_y = 6, 4.5
ax2.add_patch(FancyBboxPatch((box3_x, box3_y), 5, 0.8,
                             boxstyle="round,pad=0.1", linewidth=2,
                             edgecolor='orange', facecolor='#FFE5CC'))
ax2.text(box3_x+2.5, box3_y+0.5, 'Step 3: Check signal availability', ha='center', fontweight='bold', fontsize=10)
ax2.text(box3_x+2.5, box3_y+0.1, 'if date >= available_from:', ha='center', fontsize=9, family='monospace')

# Code snippet 2
code2 = "for date in daily_dates:\n  # Find weekly signal\n  for week in weekly_signals:\n    if date >= week['available_from']:\n      use_signal = week['signal']\n      break"
ax2.text(box3_x+2.5, box3_y-1, code2, ha='center', va='top',
         fontsize=8, family='monospace',
         bbox=dict(boxstyle='round', facecolor='#F0F0F0', edgecolor='gray'))

# Arrow down
ax2.annotate('', xy=(8.5, 3.3), xytext=(8.5, 2.5),
             arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Step 4: Combine signals
box4_x, box4_y = 6, 1.5
ax2.add_patch(FancyBboxPatch((box4_x, box4_y), 5, 0.8,
                             boxstyle="round,pad=0.1", linewidth=2,
                             edgecolor='purple', facecolor='#E5D5FF'))
ax2.text(box4_x+2.5, box4_y+0.5, 'Step 4: Combine signals (NO future info!)', ha='center', fontweight='bold', fontsize=10)
ax2.text(box4_x+2.5, box4_y+0.1, 'final = weekly_signal AND daily_signal', ha='center', fontsize=9, family='monospace')

# Key principle box
key_x, key_y = 0.5, 0
ax2.add_patch(FancyBboxPatch((key_x, key_y), 11, 1.2,
                             boxstyle="round,pad=0.15", linewidth=3,
                             edgecolor='red', facecolor='#FFFACD'))
ax2.text(key_x+5.5, key_y+0.8, 'KEY PRINCIPLE:', ha='center', fontweight='bold', fontsize=12, color='red')
ax2.text(key_x+5.5, key_y+0.4, 'Each day ONLY uses information available UP TO that day', ha='center', fontsize=11)
ax2.text(key_x+5.5, key_y+0.05, 'Weekly signal calculated with Sunday close → Available from Monday (or Tuesday, conservatively)', ha='center', fontsize=9, style='italic')

# Left side: Summary
summary_x, summary_y = 0.5, 4
ax2.add_patch(FancyBboxPatch((summary_x, summary_y), 4.5, 2.5,
                             boxstyle="round,pad=0.15", linewidth=2,
                             edgecolor='black', facecolor='white'))
ax2.text(summary_x+2.25, summary_y+2.2, 'WHY IT WORKS:', ha='center', fontweight='bold', fontsize=11)

reasons = [
    '✓ Explicit time control',
    '✓ Sequential processing',
    '✓ Available_from check',
    '✓ No future info possible'
]
for i, reason in enumerate(reasons):
    ax2.text(summary_x+0.3, summary_y+1.6-i*0.4, reason,
             ha='left', fontsize=9, family='monospace')

ax2.text(summary_x+2.25, summary_y+0.1, 'Result: Lookahead Bias = IMPOSSIBLE',
         ha='center', fontweight='bold', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))

plt.tight_layout()
plt.savefig('lookahead_bias_explanation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lookahead_bias_explanation.png")
