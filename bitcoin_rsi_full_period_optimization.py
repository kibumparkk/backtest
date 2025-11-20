"""
비트코인 RSI 전략 파라미터 최적화 (전체 구간)

전체 구간: 2018-01-01 ~ 현재
Train/Test 분할 없이 전체 데이터로 최적 파라미터 탐색

최적화 파라미터:
- RSI 기간: 5, 7, 10, 14, 17, 20, 25, 30, 35, 40
- RSI 임계값: 45, 50, 55, 60, 65, 70

총 조합: 60개
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from itertools import product

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class BitcoinRSIFullPeriodOptimizer:
    """비트코인 RSI 전략 전체 구간 파라미터 최적화"""

    def __init__(self, symbol='BTC_KRW',
                 start_date='2018-01-01', end_date=None,
                 slippage=0.002):
        """
        Args:
            symbol: 종목 심볼
            start_date: 시작일
            end_date: 종료일 (None이면 오늘까지)
            slippage: 슬리피지
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage

        self.data = None
        self.optimization_results = None
        self.best_params = None

    def load_data(self):
        """데이터 로드"""
        print("="*80)
        print(f"Loading {self.symbol} data...")
        print("="*80)

        file_path = f'chart_day/{self.symbol}.parquet'
        print(f"\nLoading from {file_path}...")

        df = pd.read_parquet(file_path)
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df

        print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        print("="*80 + "\n")

    def calculate_rsi_ewm(self, prices, period=14):
        """RSI 계산 (EWM 사용)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def backtest_strategy(self, df, rsi_period, rsi_threshold):
        """특정 파라미터로 백테스트 실행"""
        df = df.copy()

        # RSI 계산
        df['RSI'] = self.calculate_rsi_ewm(df['Close'], rsi_period)

        # 시그널 생성
        df['signal'] = (df['RSI'] >= rsi_threshold).astype(int)
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        # 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['strategy_return'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['strategy_return'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def calculate_metrics(self, df):
        """성과 지표 계산"""
        returns = df['returns']
        cumulative = df['cumulative']

        # 기간
        years = (df.index[-1] - df.index[0]).days / 365.25

        # 총 수익률
        total_return = (cumulative.iloc[-1] - 1) * 100

        # CAGR
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # MDD
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100

        # 샤프 비율
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0

        # 승률
        total_trades = (returns != 0).sum()
        winning_trades = (returns > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit Factor
        total_profit = returns[returns > 0].sum()
        total_loss = abs(returns[returns < 0].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        # Calmar Ratio (CAGR / abs(MDD))
        calmar = abs(cagr / mdd) if mdd != 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = (returns.mean() / downside_std * np.sqrt(365)) if downside_std > 0 else 0

        return {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades),
            'Profit Factor': profit_factor
        }

    def optimize_parameters(self):
        """파라미터 최적화 (전체 구간)"""
        print("\n" + "="*80)
        print("Parameter Optimization on Full Period")
        print("="*80)

        # 파라미터 그리드
        rsi_periods = [5, 7, 10, 14, 17, 20, 25, 30, 35, 40]
        rsi_thresholds = [45, 50, 55, 60, 65, 70]

        total_combinations = len(rsi_periods) * len(rsi_thresholds)
        print(f"\nTesting {total_combinations} parameter combinations...")
        print(f"RSI Periods: {rsi_periods}")
        print(f"RSI Thresholds: {rsi_thresholds}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print()

        results = []

        for i, (period, threshold) in enumerate(product(rsi_periods, rsi_thresholds), 1):
            if i % 10 == 0 or i == 1:
                print(f"Progress: {i}/{total_combinations} - Testing RSI({period}, {threshold})...")

            # 백테스트 실행
            result_df = self.backtest_strategy(self.data, period, threshold)

            # 지표 계산
            metrics = self.calculate_metrics(result_df)
            metrics['RSI Period'] = period
            metrics['RSI Threshold'] = threshold

            results.append(metrics)

        self.optimization_results = pd.DataFrame(results)

        print("\n" + "="*80)
        print("Optimization completed!")
        print("="*80 + "\n")

        return self.optimization_results

    def find_best_parameters(self, metric='Sharpe Ratio', top_n=10):
        """최적 파라미터 찾기

        Args:
            metric: 최적화 기준 지표 (default: 'Sharpe Ratio')
            top_n: 상위 N개 파라미터 출력
        """
        print("\n" + "="*80)
        print(f"Finding Best Parameters (optimized for {metric})")
        print("="*80)

        # 최고 성과 파라미터
        best_idx = self.optimization_results[metric].idxmax()
        self.best_params = self.optimization_results.loc[best_idx]

        print(f"\n🏆 Best Parameters (by {metric}):")
        print(f"  RSI Period: {int(self.best_params['RSI Period'])}")
        print(f"  RSI Threshold: {int(self.best_params['RSI Threshold'])}")
        print(f"\n📊 Performance:")
        for key in ['Total Return (%)', 'CAGR (%)', 'MDD (%)', 'Sharpe Ratio',
                    'Sortino Ratio', 'Calmar Ratio', 'Win Rate (%)', 'Total Trades', 'Profit Factor']:
            value = self.best_params[key]
            if key == 'Total Trades':
                print(f"  {key}: {int(value)}")
            elif key == 'Profit Factor' and value == np.inf:
                print(f"  {key}: INF")
            else:
                print(f"  {key}: {value:.2f}")

        # 상위 N개 파라미터
        print(f"\n📈 Top {top_n} Parameter Combinations (by {metric}):")
        top_n_results = self.optimization_results.nlargest(top_n, metric)
        print("\n" + "-"*120)
        print(f"{'Rank':<6} {'Period':<8} {'Threshold':<11} {metric:<15} {'CAGR (%)':<12} {'MDD (%)':<10} "
              f"{'Sortino':<10} {'Calmar':<10} {'Trades':<8}")
        print("-"*120)
        for rank, (idx, row) in enumerate(top_n_results.iterrows(), 1):
            print(f"{rank:<6} {int(row['RSI Period']):<8} {int(row['RSI Threshold']):<11} "
                  f"{row[metric]:<15.2f} {row['CAGR (%)']:<12.2f} {row['MDD (%)']:<10.2f} "
                  f"{row['Sortino Ratio']:<10.2f} {row['Calmar Ratio']:<10.2f} {int(row['Total Trades']):<8}")
        print("-"*120)

        # 다양한 기준으로 최고 파라미터
        print(f"\n🎯 Best Parameters by Different Metrics:")
        metrics_to_check = ['CAGR (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        for m in metrics_to_check:
            best = self.optimization_results.loc[self.optimization_results[m].idxmax()]
            print(f"\n  Best by {m}: RSI({int(best['RSI Period'])}, {int(best['RSI Threshold'])})")
            print(f"    {m}: {best[m]:.2f}")
            print(f"    CAGR: {best['CAGR (%)']:.2f}%, MDD: {best['MDD (%)']:.2f}%, Sharpe: {best['Sharpe Ratio']:.2f}")

        return self.best_params

    def plot_optimization_results(self, save_path='bitcoin_rsi_full_period_optimization.png'):
        """최적화 결과 시각화"""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        # 1. 샤프 비율 히트맵
        ax1 = fig.add_subplot(gs[0, 0])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='Sharpe Ratio'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax1,
                   cbar_kws={'label': 'Sharpe Ratio'})
        ax1.set_title('Sharpe Ratio by Parameters', fontsize=12, fontweight='bold')

        # 2. CAGR 히트맵
        ax2 = fig.add_subplot(gs[0, 1])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='CAGR (%)'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2,
                   cbar_kws={'label': 'CAGR (%)'})
        ax2.set_title('CAGR by Parameters', fontsize=12, fontweight='bold')

        # 3. MDD 히트맵
        ax3 = fig.add_subplot(gs[0, 2])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='MDD (%)'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax3,
                   cbar_kws={'label': 'MDD (%)'})
        ax3.set_title('MDD by Parameters', fontsize=12, fontweight='bold')

        # 4. Sortino Ratio 히트맵
        ax4 = fig.add_subplot(gs[0, 3])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='Sortino Ratio'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4,
                   cbar_kws={'label': 'Sortino Ratio'})
        ax4.set_title('Sortino Ratio by Parameters', fontsize=12, fontweight='bold')

        # 5. Calmar Ratio 히트맵
        ax5 = fig.add_subplot(gs[1, 0])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='Calmar Ratio'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax5,
                   cbar_kws={'label': 'Calmar Ratio'})
        ax5.set_title('Calmar Ratio by Parameters', fontsize=12, fontweight='bold')

        # 6. 승률 히트맵
        ax6 = fig.add_subplot(gs[1, 1])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='Win Rate (%)'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax6,
                   cbar_kws={'label': 'Win Rate (%)'})
        ax6.set_title('Win Rate by Parameters', fontsize=12, fontweight='bold')

        # 7. 거래 횟수 히트맵
        ax7 = fig.add_subplot(gs[1, 2])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='Total Trades'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='Blues', ax=ax7,
                   cbar_kws={'label': 'Total Trades'})
        ax7.set_title('Total Trades by Parameters', fontsize=12, fontweight='bold')

        # 8. Profit Factor 히트맵
        ax8 = fig.add_subplot(gs[1, 3])
        pf_data = self.optimization_results.copy()
        pf_data.loc[pf_data['Profit Factor'] == np.inf, 'Profit Factor'] = pf_data[pf_data['Profit Factor'] != np.inf]['Profit Factor'].max() * 1.2
        heatmap_data = pf_data.pivot(
            index='RSI Threshold', columns='RSI Period', values='Profit Factor'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax8,
                   cbar_kws={'label': 'Profit Factor'})
        ax8.set_title('Profit Factor by Parameters', fontsize=12, fontweight='bold')

        # 9. Sharpe vs CAGR 산점도
        ax9 = fig.add_subplot(gs[2, 0])
        scatter = ax9.scatter(self.optimization_results['Sharpe Ratio'],
                            self.optimization_results['CAGR (%)'],
                            c=self.optimization_results['MDD (%)'],
                            s=100, alpha=0.6, cmap='RdYlGn')

        # 최적 파라미터 강조
        best_point = self.best_params
        ax9.scatter(best_point['Sharpe Ratio'], best_point['CAGR (%)'],
                   s=500, color='red', marker='*', edgecolors='black', linewidths=2,
                   label=f"Best: RSI({int(best_point['RSI Period'])}, {int(best_point['RSI Threshold'])})")

        ax9.set_xlabel('Sharpe Ratio', fontsize=11)
        ax9.set_ylabel('CAGR (%)', fontsize=11)
        ax9.set_title('Sharpe vs CAGR (colored by MDD)', fontsize=12, fontweight='bold')
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax9, label='MDD (%)')

        # 10. CAGR vs MDD 산점도
        ax10 = fig.add_subplot(gs[2, 1])
        scatter = ax10.scatter(self.optimization_results['MDD (%)'],
                            self.optimization_results['CAGR (%)'],
                            c=self.optimization_results['Sharpe Ratio'],
                            s=100, alpha=0.6, cmap='RdYlGn')

        ax10.scatter(best_point['MDD (%)'], best_point['CAGR (%)'],
                   s=500, color='red', marker='*', edgecolors='black', linewidths=2,
                   label=f"Best: RSI({int(best_point['RSI Period'])}, {int(best_point['RSI Threshold'])})")

        ax10.set_xlabel('MDD (%)', fontsize=11)
        ax10.set_ylabel('CAGR (%)', fontsize=11)
        ax10.set_title('Return vs Risk (colored by Sharpe)', fontsize=12, fontweight='bold')
        ax10.legend(fontsize=9)
        ax10.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax10, label='Sharpe Ratio')

        # 11. Sharpe vs Sortino
        ax11 = fig.add_subplot(gs[2, 2])
        scatter = ax11.scatter(self.optimization_results['Sharpe Ratio'],
                             self.optimization_results['Sortino Ratio'],
                             c=self.optimization_results['CAGR (%)'],
                             s=100, alpha=0.6, cmap='RdYlGn')

        ax11.scatter(best_point['Sharpe Ratio'], best_point['Sortino Ratio'],
                    s=500, color='red', marker='*', edgecolors='black', linewidths=2,
                    label=f"Best: RSI({int(best_point['RSI Period'])}, {int(best_point['RSI Threshold'])})")

        ax11.set_xlabel('Sharpe Ratio', fontsize=11)
        ax11.set_ylabel('Sortino Ratio', fontsize=11)
        ax11.set_title('Sharpe vs Sortino (colored by CAGR)', fontsize=12, fontweight='bold')
        ax11.legend(fontsize=9)
        ax11.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax11, label='CAGR (%)')

        # 12. Calmar Ratio vs CAGR
        ax12 = fig.add_subplot(gs[2, 3])
        scatter = ax12.scatter(self.optimization_results['Calmar Ratio'],
                             self.optimization_results['CAGR (%)'],
                             c=self.optimization_results['Sharpe Ratio'],
                             s=100, alpha=0.6, cmap='RdYlGn')

        ax12.scatter(best_point['Calmar Ratio'], best_point['CAGR (%)'],
                    s=500, color='red', marker='*', edgecolors='black', linewidths=2,
                    label=f"Best: RSI({int(best_point['RSI Period'])}, {int(best_point['RSI Threshold'])})")

        ax12.set_xlabel('Calmar Ratio', fontsize=11)
        ax12.set_ylabel('CAGR (%)', fontsize=11)
        ax12.set_title('Calmar vs CAGR (colored by Sharpe)', fontsize=12, fontweight='bold')
        ax12.legend(fontsize=9)
        ax12.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax12, label='Sharpe Ratio')

        # 13. Top 15 파라미터 by Sharpe
        ax13 = fig.add_subplot(gs[3, :2])
        top_15 = self.optimization_results.nlargest(15, 'Sharpe Ratio')
        labels = [f"({int(row['RSI Period'])}, {int(row['RSI Threshold'])})"
                 for _, row in top_15.iterrows()]
        colors = ['red' if i == 0 else 'steelblue' for i in range(len(top_15))]

        ax13.barh(range(len(top_15)), top_15['Sharpe Ratio'], color=colors, alpha=0.7)
        ax13.set_yticks(range(len(top_15)))
        ax13.set_yticklabels(labels, fontsize=9)
        ax13.set_xlabel('Sharpe Ratio', fontsize=11)
        ax13.set_title('Top 15 Parameters by Sharpe Ratio', fontsize=12, fontweight='bold')
        ax13.invert_yaxis()
        ax13.grid(True, alpha=0.3, axis='x')

        # 14. 성과 지표 요약 테이블
        ax14 = fig.add_subplot(gs[3, 2:])
        ax14.axis('off')

        summary_text = "Best Parameters Summary\n" + "="*50 + "\n\n"
        summary_text += f"RSI Period: {int(best_point['RSI Period'])}\n"
        summary_text += f"RSI Threshold: {int(best_point['RSI Threshold'])}\n\n"
        summary_text += "Performance Metrics:\n" + "-"*50 + "\n"
        summary_text += f"Total Return: {best_point['Total Return (%)']:.2f}%\n"
        summary_text += f"CAGR: {best_point['CAGR (%)']:.2f}%\n"
        summary_text += f"MDD: {best_point['MDD (%)']:.2f}%\n"
        summary_text += f"Sharpe Ratio: {best_point['Sharpe Ratio']:.2f}\n"
        summary_text += f"Sortino Ratio: {best_point['Sortino Ratio']:.2f}\n"
        summary_text += f"Calmar Ratio: {best_point['Calmar Ratio']:.2f}\n"
        summary_text += f"Win Rate: {best_point['Win Rate (%)']:.2f}%\n"
        summary_text += f"Total Trades: {int(best_point['Total Trades'])}\n"

        if best_point['Profit Factor'] != np.inf:
            summary_text += f"Profit Factor: {best_point['Profit Factor']:.2f}\n"
        else:
            summary_text += f"Profit Factor: INF\n"

        ax14.text(0.05, 0.95, summary_text, transform=ax14.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        # 전체 제목
        fig.suptitle(f'Bitcoin RSI Strategy - Full Period Parameter Optimization\n'
                    f'Period: {self.start_date} to {self.end_date} ({len(self.data)} days)',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nOptimization chart saved to {save_path}")
        plt.close()

    def run_full_optimization(self, metric='Sharpe Ratio'):
        """전체 최적화 프로세스 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 파라미터 최적화
        self.optimize_parameters()

        # 3. 최적 파라미터 찾기
        self.find_best_parameters(metric=metric, top_n=10)

        # 4. 시각화
        self.plot_optimization_results()

        # 5. 결과 저장
        print("\nSaving optimization results...")
        self.optimization_results.to_csv('bitcoin_rsi_full_period_optimization_results.csv', index=False)
        print("Results saved to bitcoin_rsi_full_period_optimization_results.csv")

        return self.best_params


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("비트코인 RSI 전략 파라미터 최적화 (전체 구간)")
    print("="*80)
    print("\n⚠️  주의:")
    print("  - 전체 구간 데이터 사용 (Train/Test 분할 없음)")
    print("  - 과적합 위험이 있을 수 있음")
    print("  - 실전 투자 전 추가 검증 필요\n")

    # 최적화 실행
    optimizer = BitcoinRSIFullPeriodOptimizer(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )

    # 전체 최적화 실행
    best_params = optimizer.run_full_optimization(metric='Sharpe Ratio')

    print("\n" + "="*80)
    print("최적화 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
