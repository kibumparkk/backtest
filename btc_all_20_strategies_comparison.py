"""
BTC 전체 20가지 전략 백테스트 최종 비교 분석

기존 10개 전략과 새로운 10개 전략을 모두 비교하여
최고 성과 전략들을 선별하고 실패한 전략들을 제거
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_and_merge_results():
    """두 백테스트 결과를 로드하고 병합"""

    # 기존 10개 전략 결과 로드
    old_metrics = pd.read_csv('btc_10_strategies_metrics.csv')
    old_metrics['Group'] = 'Original 10'

    # 새로운 10개 전략 결과 로드
    new_metrics = pd.read_csv('btc_10_new_strategies_metrics.csv')
    new_metrics['Group'] = 'New 10'

    # 병합
    all_metrics = pd.concat([old_metrics, new_metrics], ignore_index=True)

    return all_metrics, old_metrics, new_metrics


def filter_best_strategies(all_metrics, min_cagr=30, max_mdd=-70):
    """
    우수한 전략만 필터링

    Args:
        min_cagr: 최소 CAGR (%)
        max_mdd: 최대 MDD (%, 음수)
    """
    filtered = all_metrics[
        (all_metrics['CAGR (%)'] >= min_cagr) &
        (all_metrics['MDD (%)'] >= max_mdd)
    ].copy()

    return filtered


def create_comprehensive_comparison(all_metrics, save_path='btc_all_20_strategies_final_comparison.png'):
    """전체 20개 전략 종합 비교 시각화"""

    fig = plt.figure(figsize=(28, 20))
    gs = fig.add_gridspec(6, 4, hspace=0.4, wspace=0.35)

    # 색상 맵 정의
    colors_dict = {
        'Original 10': '#FF6B6B',
        'New 10': '#4ECDC4'
    }

    # 1. 총 수익률 비교 (전체 20개)
    ax1 = fig.add_subplot(gs[0, :2])
    sorted_df = all_metrics.sort_values('Total Return (%)', ascending=True)
    colors = [colors_dict[g] for g in sorted_df['Group']]
    bars = ax1.barh(range(len(sorted_df)), sorted_df['Total Return (%)'],
                   color=colors, alpha=0.7)
    ax1.set_yticks(range(len(sorted_df)))
    ax1.set_yticklabels(sorted_df['Strategy'], fontsize=9)
    ax1.set_xlabel('Total Return (%)', fontsize=12)
    ax1.set_title('All 20 Strategies - Total Return Comparison',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_dict['Original 10'], alpha=0.7, label='Original 10'),
                      Patch(facecolor=colors_dict['New 10'], alpha=0.7, label='New 10')]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # 2. CAGR 비교 (전체 20개)
    ax2 = fig.add_subplot(gs[0, 2:])
    sorted_df = all_metrics.sort_values('CAGR (%)', ascending=True)
    colors = [colors_dict[g] for g in sorted_df['Group']]
    bars = ax2.barh(range(len(sorted_df)), sorted_df['CAGR (%)'],
                   color=colors, alpha=0.7)
    ax2.set_yticks(range(len(sorted_df)))
    ax2.set_yticklabels(sorted_df['Strategy'], fontsize=9)
    ax2.set_xlabel('CAGR (%)', fontsize=12)
    ax2.set_title('All 20 Strategies - CAGR Comparison',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # 3. Return vs Risk (전체)
    ax3 = fig.add_subplot(gs[1, :2])
    for group_name, group_color in colors_dict.items():
        group_data = all_metrics[all_metrics['Group'] == group_name]
        ax3.scatter(group_data['MDD (%)'], group_data['CAGR (%)'],
                   s=200, alpha=0.6, c=group_color, edgecolors='black',
                   linewidth=1.5, label=group_name)

    # 상위 5개 전략 강조
    top5 = all_metrics.nlargest(5, 'CAGR (%)')
    ax3.scatter(top5['MDD (%)'], top5['CAGR (%)'],
               s=400, facecolors='none', edgecolors='gold',
               linewidth=3, label='Top 5 CAGR', zorder=10)

    for idx, row in all_metrics.iterrows():
        label = row['Strategy'].split('.')[0]
        ax3.annotate(label,
                    (row['MDD (%)'], row['CAGR (%)']),
                    fontsize=8, ha='center', va='center')

    ax3.set_xlabel('MDD (%)', fontsize=12)
    ax3.set_ylabel('CAGR (%)', fontsize=12)
    ax3.set_title('Return vs Risk - All 20 Strategies',
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 4. 샤프 비율 비교
    ax4 = fig.add_subplot(gs[1, 2:])
    sorted_df = all_metrics.sort_values('Sharpe Ratio', ascending=True)
    colors = [colors_dict[g] for g in sorted_df['Group']]
    bars = ax4.barh(range(len(sorted_df)), sorted_df['Sharpe Ratio'],
                   color=colors, alpha=0.7)
    ax4.set_yticks(range(len(sorted_df)))
    ax4.set_yticklabels(sorted_df['Strategy'], fontsize=9)
    ax4.set_xlabel('Sharpe Ratio', fontsize=12)
    ax4.set_title('All 20 Strategies - Sharpe Ratio Comparison',
                 fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.axvline(x=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1')
    ax4.legend(fontsize=9)

    # 5. MDD 비교
    ax5 = fig.add_subplot(gs[2, 0])
    sorted_df = all_metrics.sort_values('MDD (%)', ascending=False).head(10)
    colors = [colors_dict[g] for g in sorted_df['Group']]
    bars = ax5.barh(range(len(sorted_df)), sorted_df['MDD (%)'],
                   color=colors, alpha=0.7)
    ax5.set_yticks(range(len(sorted_df)))
    ax5.set_yticklabels(sorted_df['Strategy'], fontsize=9)
    ax5.set_xlabel('MDD (%)', fontsize=11)
    ax5.set_title('Top 10 - Best MDD (Lower Risk)',
                 fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Profit Factor 비교
    ax6 = fig.add_subplot(gs[2, 1])
    filtered_metrics = all_metrics[all_metrics['Profit Factor'] != np.inf].copy()
    sorted_df = filtered_metrics.sort_values('Profit Factor', ascending=True).tail(10)
    colors = [colors_dict[g] for g in sorted_df['Group']]
    bars = ax6.barh(range(len(sorted_df)), sorted_df['Profit Factor'],
                   color=colors, alpha=0.7)
    ax6.set_yticks(range(len(sorted_df)))
    ax6.set_yticklabels(sorted_df['Strategy'], fontsize=9)
    ax6.set_xlabel('Profit Factor', fontsize=11)
    ax6.set_title('Top 10 - Profit Factor',
                 fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 7. 승률 비교
    ax7 = fig.add_subplot(gs[2, 2])
    sorted_df = all_metrics.sort_values('Win Rate (%)', ascending=True).tail(10)
    colors = [colors_dict[g] for g in sorted_df['Group']]
    bars = ax7.barh(range(len(sorted_df)), sorted_df['Win Rate (%)'],
                   color=colors, alpha=0.7)
    ax7.set_yticks(range(len(sorted_df)))
    ax7.set_yticklabels(sorted_df['Strategy'], fontsize=9)
    ax7.set_xlabel('Win Rate (%)', fontsize=11)
    ax7.set_title('Top 10 - Win Rate',
                 fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')
    ax7.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 8. 거래 횟수
    ax8 = fig.add_subplot(gs[2, 3])
    sorted_df = all_metrics.sort_values('Total Trades', ascending=True)
    colors = [colors_dict[g] for g in sorted_df['Group']]
    ax8.scatter(sorted_df['Total Trades'], range(len(sorted_df)),
               c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax8.set_yticks(range(len(sorted_df)))
    ax8.set_yticklabels(sorted_df['Strategy'], fontsize=9)
    ax8.set_xlabel('Total Trades', fontsize=11)
    ax8.set_title('Trading Frequency',
                 fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='x')

    # 9. Top 10 전략 (CAGR 기준)
    ax9 = fig.add_subplot(gs[3, :2])
    top10 = all_metrics.nlargest(10, 'CAGR (%)')
    x = np.arange(len(top10))
    width = 0.35

    ax9_twin = ax9.twinx()

    bars1 = ax9.bar(x - width/2, top10['CAGR (%)'], width,
                    label='CAGR (%)', color='green', alpha=0.7)
    bars2 = ax9_twin.bar(x + width/2, top10['Sharpe Ratio'], width,
                         label='Sharpe Ratio', color='blue', alpha=0.7)

    ax9.set_xlabel('Strategy', fontsize=11)
    ax9.set_ylabel('CAGR (%)', fontsize=11, color='green')
    ax9_twin.set_ylabel('Sharpe Ratio', fontsize=11, color='blue')
    ax9.set_title('Top 10 Strategies by CAGR (with Sharpe Ratio)',
                 fontsize=13, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels([s.split('.')[0] for s in top10['Strategy']], rotation=45, ha='right')
    ax9.tick_params(axis='y', labelcolor='green')
    ax9_twin.tick_params(axis='y', labelcolor='blue')
    ax9.grid(True, alpha=0.3, axis='y')

    # 범례
    lines1, labels1 = ax9.get_legend_handles_labels()
    lines2, labels2 = ax9_twin.get_legend_handles_labels()
    ax9.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # 10. Bottom 5 전략 (CAGR 기준)
    ax10 = fig.add_subplot(gs[3, 2:])
    bottom5 = all_metrics.nsmallest(5, 'CAGR (%)')
    colors = ['red' if x < 0 else 'orange' for x in bottom5['CAGR (%)']]
    bars = ax10.barh(bottom5['Strategy'], bottom5['CAGR (%)'],
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax10.set_xlabel('CAGR (%)', fontsize=11)
    ax10.set_title('Bottom 5 Strategies (Worst CAGR) - TO AVOID',
                  fontsize=13, fontweight='bold', color='red')
    ax10.grid(True, alpha=0.3, axis='x')
    ax10.axvline(x=0, color='black', linestyle='-', linewidth=1)

    for bar in bars:
        width = bar.get_width()
        ax10.text(width, bar.get_y() + bar.get_height()/2,
                 f'{width:.1f}%',
                 ha='right' if width < 0 else 'left',
                 va='center', fontsize=10, fontweight='bold')

    # 11. 성과 지표 상관관계
    ax11 = fig.add_subplot(gs[4, :2])

    # 상관계수 계산
    corr_data = all_metrics[['CAGR (%)', 'MDD (%)', 'Sharpe Ratio',
                             'Win Rate (%)', 'Profit Factor', 'Total Trades']].copy()
    # Profit Factor가 inf인 경우 처리
    corr_data = corr_data[corr_data['Profit Factor'] != np.inf]

    correlation = corr_data.corr()

    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, ax=ax11, linewidths=1, square=True,
               cbar_kws={'label': 'Correlation'})
    ax11.set_title('Performance Metrics Correlation',
                  fontsize=13, fontweight='bold')

    # 12. 그룹별 평균 성과
    ax12 = fig.add_subplot(gs[4, 2:])

    group_stats = all_metrics.groupby('Group').agg({
        'CAGR (%)': 'mean',
        'MDD (%)': 'mean',
        'Sharpe Ratio': 'mean',
        'Win Rate (%)': 'mean'
    })

    x = np.arange(len(group_stats.columns))
    width = 0.35

    bars1 = ax12.bar(x - width/2, group_stats.loc['Original 10'],
                    width, label='Original 10', color=colors_dict['Original 10'], alpha=0.7)
    bars2 = ax12.bar(x + width/2, group_stats.loc['New 10'],
                    width, label='New 10', color=colors_dict['New 10'], alpha=0.7)

    ax12.set_ylabel('Average Value', fontsize=11)
    ax12.set_title('Average Performance by Group',
                  fontsize=13, fontweight='bold')
    ax12.set_xticks(x)
    ax12.set_xticklabels(group_stats.columns, rotation=45, ha='right')
    ax12.legend(fontsize=10)
    ax12.grid(True, alpha=0.3, axis='y')
    ax12.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax12.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}',
                     ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=9)

    # 13. 전략 유형별 분석
    ax13 = fig.add_subplot(gs[5, :])

    # 전략 유형 분류
    trend_following = ['Turtle Trading', 'Momentum', 'Dual Momentum', 'MACD',
                      'EMA Crossover', 'SMA Crossover', 'Parabolic SAR',
                      'SuperTrend', 'Donchian Channel', 'ADX Trend',
                      'Triple MA', 'Golden/Death Cross']

    oscillator_based = ['RSI Oversold/Overbought', 'RSI+MACD Combined',
                       'Williams %R', 'Stochastic']

    channel_based = ['Bollinger Bands', 'Keltner Channel', 'Ichimoku Cloud']

    mean_reversion = ['Mean Reversion']

    # 각 전략의 유형 분류
    def classify_strategy(name):
        for trend in trend_following:
            if trend in name:
                return 'Trend Following'
        for osc in oscillator_based:
            if osc in name:
                return 'Oscillator'
        for channel in channel_based:
            if channel in name:
                return 'Channel'
        for mr in mean_reversion:
            if mr in name:
                return 'Mean Reversion'
        return 'Other'

    all_metrics['Type'] = all_metrics['Strategy'].apply(classify_strategy)

    type_stats = all_metrics.groupby('Type').agg({
        'CAGR (%)': ['mean', 'std', 'count'],
        'Sharpe Ratio': 'mean'
    })

    # 박스플롯
    types = all_metrics['Type'].unique()
    data_to_plot = [all_metrics[all_metrics['Type'] == t]['CAGR (%)'].values
                   for t in types]

    bp = ax13.boxplot(data_to_plot, labels=types, patch_artist=True,
                      showmeans=True, meanline=True)

    # 색상 설정
    colors_box = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightgray']
    for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax13.set_ylabel('CAGR (%)', fontsize=12)
    ax13.set_title('Strategy Performance by Type (CAGR Distribution)',
                  fontsize=14, fontweight='bold')
    ax13.grid(True, alpha=0.3, axis='y')
    ax13.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive comparison chart saved to {save_path}")
    plt.close()


def print_final_summary(all_metrics):
    """최종 요약 출력"""
    print("\n" + "="*150)
    print(f"{'BTC 전체 20가지 전략 최종 비교 분석':^150}")
    print("="*150)

    print("\n" + "🏆 "*30)
    print(f"{'TOP 10 BEST STRATEGIES (by CAGR)':^150}")
    print("🏆 "*30)

    top10 = all_metrics.nlargest(10, 'CAGR (%)')[
        ['Strategy', 'Group', 'Total Return (%)', 'CAGR (%)',
         'MDD (%)', 'Sharpe Ratio', 'Win Rate (%)']
    ]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    print(top10.to_string(index=False))

    print("\n" + "⚠️ "*30)
    print(f"{'BOTTOM 5 WORST STRATEGIES (by CAGR) - AVOID THESE':^150}")
    print("⚠️ "*30)

    bottom5 = all_metrics.nsmallest(5, 'CAGR (%)')[
        ['Strategy', 'Group', 'Total Return (%)', 'CAGR (%)',
         'MDD (%)', 'Sharpe Ratio']
    ]
    print(bottom5.to_string(index=False))

    print("\n" + "="*150)
    print(f"{'BEST STRATEGY BY CATEGORY':^150}")
    print("="*150)

    print(f"\n🥇 Highest CAGR: {all_metrics.loc[all_metrics['CAGR (%)'].idxmax(), 'Strategy']}")
    print(f"   → {all_metrics['CAGR (%)'].max():.2f}% CAGR")

    print(f"\n🛡️ Best MDD (Lowest Risk): {all_metrics.loc[all_metrics['MDD (%)'].idxmax(), 'Strategy']}")
    print(f"   → {all_metrics['MDD (%)'].max():.2f}% MDD")

    print(f"\n📊 Highest Sharpe Ratio: {all_metrics.loc[all_metrics['Sharpe Ratio'].idxmax(), 'Strategy']}")
    print(f"   → {all_metrics['Sharpe Ratio'].max():.2f} Sharpe")

    filtered_pf = all_metrics[all_metrics['Profit Factor'] != np.inf]
    print(f"\n💰 Highest Profit Factor: {filtered_pf.loc[filtered_pf['Profit Factor'].idxmax(), 'Strategy']}")
    print(f"   → {filtered_pf['Profit Factor'].max():.2f}")

    print("\n" + "="*150)
    print(f"{'RECOMMENDED STRATEGIES (CAGR > 50% AND MDD > -60%)':^150}")
    print("="*150)

    recommended = all_metrics[
        (all_metrics['CAGR (%)'] > 50) &
        (all_metrics['MDD (%)'] > -60)
    ].sort_values('CAGR (%)', ascending=False)

    print(recommended[['Strategy', 'Group', 'CAGR (%)', 'MDD (%)',
                      'Sharpe Ratio']].to_string(index=False))

    print("\n" + "="*150 + "\n")


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("BTC 전체 20가지 전략 최종 비교 분석")
    print("="*80 + "\n")

    # 결과 로드 및 병합
    all_metrics, old_metrics, new_metrics = load_and_merge_results()

    # 최종 요약 출력
    print_final_summary(all_metrics)

    # 종합 비교 시각화
    create_comprehensive_comparison(all_metrics)

    # 최종 결과 저장
    all_metrics.to_csv('btc_all_20_strategies_final_metrics.csv', index=False)
    print("Final combined metrics saved to btc_all_20_strategies_final_metrics.csv")

    # 추천 전략만 필터링하여 저장
    recommended = all_metrics[
        (all_metrics['CAGR (%)'] > 50) &
        (all_metrics['MDD (%)'] > -60)
    ].sort_values('CAGR (%)', ascending=False)

    recommended.to_csv('btc_recommended_strategies.csv', index=False)
    print("Recommended strategies saved to btc_recommended_strategies.csv")

    print("\n" + "="*80)
    print("최종 분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
