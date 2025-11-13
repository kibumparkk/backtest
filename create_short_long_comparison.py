"""
Create comparison chart for short long period combinations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
df = pd.read_csv('sma_optimization_results.csv')

# Filter for specific combinations to compare
short_periods = [1, 5, 10]
long_periods = [5, 10, 15, 20, 25, 30, 40, 50]

# Filter data
filtered_df = df[df['short_period'].isin(short_periods) &
                 df['long_period'].isin(long_periods)].copy()

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('SMA Cross Strategy - Short vs Long Period Comparison\n(Focus on Short Long Periods)',
             fontsize=18, fontweight='bold', y=0.995)

metrics = [
    ('sharpe_ratio', 'Sharpe Ratio', 'Higher is Better'),
    ('cagr', 'CAGR (%)', 'Higher is Better'),
    ('total_return', 'Total Return (%)', 'Higher is Better'),
    ('max_drawdown', 'Max Drawdown (%)', 'Closer to 0 is Better'),
    ('win_rate', 'Win Rate (%)', 'Higher is Better'),
    ('total_trades', 'Total Trades', 'Information')
]

for idx, (metric, title, note) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    # Create line plot for each short period
    for short in short_periods:
        data = filtered_df[filtered_df['short_period'] == short].sort_values('long_period')
        ax.plot(data['long_period'], data[metric],
               marker='o', linewidth=2.5, markersize=8,
               label=f'Short={short}', alpha=0.8)

    ax.set_title(f'{title}\n{note}', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Long SMA Period', fontsize=11, fontweight='bold')
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Mark the best value
    if metric == 'max_drawdown':
        best_idx = filtered_df[metric].idxmax()  # Closest to 0
    else:
        best_idx = filtered_df[metric].idxmax()

    best_val = filtered_df.loc[best_idx]
    ax.axvline(x=best_val['long_period'], color='red',
              linestyle=':', linewidth=1.5, alpha=0.5)
    ax.text(best_val['long_period'], ax.get_ylim()[1],
           f"Best Long={int(best_val['long_period'])}\nShort={int(best_val['short_period'])}",
           fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('sma_short_long_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Short-long comparison chart saved to: sma_short_long_comparison.png")
plt.close()

# Create a summary table
print("\n" + "="*80)
print("SUMMARY: Top 10 Combinations (by Sharpe Ratio)")
print("="*80)
top_10 = df.nlargest(10, 'sharpe_ratio')[['short_period', 'long_period',
                                           'total_return', 'cagr',
                                           'sharpe_ratio', 'max_drawdown',
                                           'win_rate', 'total_trades']]
print(top_10.to_string(index=False))
print("="*80)

# Create a focused table for short long periods
print("\n" + "="*80)
print("FOCUS: Combinations with Long Period 5-20")
print("="*80)
short_long_df = df[df['long_period'] <= 20].sort_values('sharpe_ratio', ascending=False)
print(short_long_df[['short_period', 'long_period', 'total_return', 'cagr',
                     'sharpe_ratio', 'max_drawdown', 'win_rate']].head(15).to_string(index=False))
print("="*80)

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. Best Overall: Short={int(df.loc[df['sharpe_ratio'].idxmax(), 'short_period'])}, " +
      f"Long={int(df.loc[df['sharpe_ratio'].idxmax(), 'long_period'])}, " +
      f"Sharpe={df['sharpe_ratio'].max():.3f}")

short_long = df[df['long_period'] <= 20]
print(f"2. Best with Long≤20: Short={int(short_long.loc[short_long['sharpe_ratio'].idxmax(), 'short_period'])}, " +
      f"Long={int(short_long.loc[short_long['sharpe_ratio'].idxmax(), 'long_period'])}, " +
      f"Sharpe={short_long['sharpe_ratio'].max():.3f}")

print(f"3. Combinations with Long=5: {len(df[df['long_period']==5])} tested, " +
      f"Best Sharpe={df[df['long_period']==5]['sharpe_ratio'].max():.3f}")

print(f"4. Combinations with Long=10: {len(df[df['long_period']==10])} tested, " +
      f"Best Sharpe={df[df['long_period']==10]['sharpe_ratio'].max():.3f}")

print(f"5. Combinations with Long=15: {len(df[df['long_period']==15])} tested, " +
      f"Best Sharpe={df[df['long_period']==15]['sharpe_ratio'].max():.3f}")

print(f"6. Combinations with Long=20: {len(df[df['long_period']==20])} tested, " +
      f"Best Sharpe={df[df['long_period']==20]['sharpe_ratio'].max():.3f}")

print("\n7. Pattern Analysis:")
print(f"   - Long=5 performs poorly (avg Sharpe: {df[df['long_period']==5]['sharpe_ratio'].mean():.3f})")
print(f"   - Long=10 shows improvement (avg Sharpe: {df[df['long_period']==10]['sharpe_ratio'].mean():.3f})")
print(f"   - Long=15-30 range is optimal (avg Sharpe 15-30: {df[df['long_period'].between(15,30)]['sharpe_ratio'].mean():.3f})")
print(f"   - Very long periods (>100) decline (avg Sharpe >100: {df[df['long_period']>100]['sharpe_ratio'].mean():.3f})")
print("="*80)
