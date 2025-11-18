"""
볼린저 밴드 전략 파라미터 최적화 스크립트

전일 종가가 Upper Band 돌파 시 매수, Middle Band에서 매도하는 전략의
최적 window와 k 값을 탐색합니다.
"""

from crypto_portfolio_strategy_comparison_fixed import CryptoPortfolioComparisonFixed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_optimization_heatmap(results_df, save_path='bb_optimization_heatmap.png'):
    """최적화 결과를 히트맵으로 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bollinger Band Parameter Optimization Results',
                 fontsize=16, fontweight='bold')

    # 1. Total Return 히트맵
    pivot_return = results_df.pivot(index='k', columns='window', values='Total Return (%)')
    sns.heatmap(pivot_return, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, ax=axes[0, 0], cbar_kws={'label': 'Total Return (%)'})
    axes[0, 0].set_title('Total Return (%) by Parameters', fontweight='bold')
    axes[0, 0].set_xlabel('Window')
    axes[0, 0].set_ylabel('K Value')

    # 2. Sharpe Ratio 히트맵
    pivot_sharpe = results_df.pivot(index='k', columns='window', values='Sharpe Ratio')
    sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=axes[0, 1], cbar_kws={'label': 'Sharpe Ratio'})
    axes[0, 1].set_title('Sharpe Ratio by Parameters', fontweight='bold')
    axes[0, 1].set_xlabel('Window')
    axes[0, 1].set_ylabel('K Value')

    # 3. CAGR 히트맵
    pivot_cagr = results_df.pivot(index='k', columns='window', values='CAGR (%)')
    sns.heatmap(pivot_cagr, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, ax=axes[1, 0], cbar_kws={'label': 'CAGR (%)'})
    axes[1, 0].set_title('CAGR (%) by Parameters', fontweight='bold')
    axes[1, 0].set_xlabel('Window')
    axes[1, 0].set_ylabel('K Value')

    # 4. MDD 히트맵
    pivot_mdd = results_df.pivot(index='k', columns='window', values='MDD (%)')
    sns.heatmap(pivot_mdd, annot=True, fmt='.1f', cmap='RdYlGn_r',
                center=0, ax=axes[1, 1], cbar_kws={'label': 'MDD (%)'})
    axes[1, 1].set_title('Maximum Drawdown (%) by Parameters', fontweight='bold')
    axes[1, 1].set_xlabel('Window')
    axes[1, 1].set_ylabel('K Value')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to {save_path}")
    plt.close()


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("볼린저 밴드 전략 파라미터 최적화")
    print("="*80)
    print("\n전략 설명:")
    print("  - 매수: 전일 종가가 Upper Band 돌파")
    print("  - 매도: Middle Band(이동평균선)에서 청산")
    print("="*80 + "\n")

    # 백테스트 객체 생성
    comparison = CryptoPortfolioComparisonFixed(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 데이터 로드
    comparison.load_data()

    # 파라미터 최적화 실행
    print("\n" + "="*80)
    print("Starting parameter optimization...")
    print("="*80 + "\n")

    # 더 넓은 범위로 테스트
    window_range = [10, 15, 20, 25, 30, 40, 50]
    k_range = [1.0, 1.5, 2.0, 2.5, 3.0]

    optimization_results = comparison.optimize_bollinger_band_parameters(
        window_range=window_range,
        k_range=k_range
    )

    # 결과 저장
    print("\n" + "="*80)
    print("Saving optimization results...")
    print("="*80 + "\n")

    optimization_results.to_csv('bollinger_band_optimization_results.csv', index=False)
    print("Results saved to bollinger_band_optimization_results.csv")

    # 히트맵 생성
    create_optimization_heatmap(optimization_results)

    # 최적 파라미터 추출 (Sharpe Ratio 기준)
    best_idx = optimization_results['Sharpe Ratio'].idxmax()
    best_params = optimization_results.iloc[best_idx]
    best_window = int(best_params['window'])
    best_k = best_params['k']

    print("\n" + "="*80)
    print("Running full analysis with best parameters...")
    print("="*80)
    print(f"Best Window: {best_window}")
    print(f"Best K: {best_k}")
    print("="*80 + "\n")

    # 최적 파라미터로 전체 분석 실행
    comparison_best = CryptoPortfolioComparisonFixed(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )

    metrics_df = comparison_best.run_analysis(
        create_individual_charts=True,
        bb_window=best_window,
        bb_k=best_k
    )

    # 결과 저장
    print("\nSaving final results...")
    metrics_df.to_csv('bollinger_band_best_params_metrics.csv', index=False)
    print("Metrics saved to bollinger_band_best_params_metrics.csv")

    # 각 포트폴리오 상세 결과 저장
    for strategy_name in comparison_best.portfolio_results.keys():
        filename = f"portfolio_{strategy_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '').lower()}.csv"
        comparison_best.portfolio_results[strategy_name].to_csv(filename)
        print(f"Portfolio details saved to {filename}")

    print("\n" + "="*80)
    print("✅ 최적화 완료!")
    print("="*80)
    print(f"\n최종 선택 파라미터:")
    print(f"  Window: {best_window}")
    print(f"  K: {best_k}")
    print(f"\n성과:")
    print(f"  Total Return: {best_params['Total Return (%)']:.2f}%")
    print(f"  CAGR: {best_params['CAGR (%)']:.2f}%")
    print(f"  Sharpe Ratio: {best_params['Sharpe Ratio']:.2f}")
    print(f"  MDD: {best_params['MDD (%)']:.2f}%")
    print(f"  Win Rate: {best_params['Win Rate (%)']:.2f}%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
