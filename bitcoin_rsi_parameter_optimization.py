"""
비트코인 RSI 전략 파라미터 최적화

Data Snooping Bias 방지를 위한 Train/Test 분할:
- Train: 2018-01-01 ~ 2021-12-31 (파라미터 최적화)
- Test: 2022-01-01 ~ 현재 (성능 검증)

최적화 파라미터:
- RSI 기간: 5, 7, 10, 14, 17, 20, 25, 30
- RSI 임계값: 45, 50, 55, 60, 65

총 조합: 40개 (< 100개 제한)
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


class BitcoinRSIOptimizer:
    """비트코인 RSI 전략 파라미터 최적화"""

    def __init__(self, symbol='BTC_KRW',
                 train_start='2018-01-01', train_end='2021-12-31',
                 test_start='2022-01-01', test_end=None,
                 slippage=0.002):
        """
        Args:
            symbol: 종목 심볼
            train_start: 훈련 데이터 시작일
            train_end: 훈련 데이터 종료일
            test_start: 테스트 데이터 시작일
            test_end: 테스트 데이터 종료일
            slippage: 슬리피지
        """
        self.symbol = symbol
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end if test_end else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage

        self.data = None
        self.train_data = None
        self.test_data = None
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

        self.data = df

        # Train/Test 분할
        self.train_data = df[(df.index >= self.train_start) & (df.index <= self.train_end)]
        self.test_data = df[(df.index >= self.test_start) & (df.index <= self.test_end)]

        print(f"\nTrain data: {len(self.train_data)} points from {self.train_data.index[0]} to {self.train_data.index[-1]}")
        print(f"Test data: {len(self.test_data)} points from {self.test_data.index[0]} to {self.test_data.index[-1]}")
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

        return {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades),
            'Profit Factor': profit_factor,
            'Calmar Ratio': calmar
        }

    def optimize_parameters(self):
        """파라미터 최적화 (Train 데이터 사용)"""
        print("\n" + "="*80)
        print("Parameter Optimization on Training Data")
        print("="*80)

        # 파라미터 그리드
        rsi_periods = [5, 7, 10, 14, 17, 20, 25, 30]
        rsi_thresholds = [45, 50, 55, 60, 65]

        total_combinations = len(rsi_periods) * len(rsi_thresholds)
        print(f"\nTesting {total_combinations} parameter combinations...")
        print(f"RSI Periods: {rsi_periods}")
        print(f"RSI Thresholds: {rsi_thresholds}")
        print()

        results = []

        for i, (period, threshold) in enumerate(product(rsi_periods, rsi_thresholds), 1):
            if i % 10 == 0 or i == 1:
                print(f"Progress: {i}/{total_combinations} - Testing RSI({period}, {threshold})...")

            # 백테스트 실행
            result_df = self.backtest_strategy(self.train_data, period, threshold)

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

    def find_best_parameters(self, metric='Sharpe Ratio'):
        """최적 파라미터 찾기

        Args:
            metric: 최적화 기준 지표 (default: 'Sharpe Ratio')
                    옵션: 'Sharpe Ratio', 'CAGR (%)', 'Calmar Ratio', 'Total Return (%)'
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
        print(f"\n📊 Training Performance:")
        for key in ['Total Return (%)', 'CAGR (%)', 'MDD (%)', 'Sharpe Ratio',
                    'Calmar Ratio', 'Win Rate (%)', 'Total Trades', 'Profit Factor']:
            value = self.best_params[key]
            if key == 'Total Trades':
                print(f"  {key}: {int(value)}")
            elif key == 'Profit Factor' and value == np.inf:
                print(f"  {key}: INF")
            else:
                print(f"  {key}: {value:.2f}")

        # 상위 5개 파라미터
        print(f"\n📈 Top 5 Parameter Combinations (by {metric}):")
        top_5 = self.optimization_results.nlargest(5, metric)
        print("\n" + "-"*100)
        print(f"{'Rank':<6} {'Period':<8} {'Threshold':<11} {metric:<15} {'CAGR (%)':<12} {'MDD (%)':<10} {'Trades':<8}")
        print("-"*100)
        for rank, (idx, row) in enumerate(top_5.iterrows(), 1):
            print(f"{rank:<6} {int(row['RSI Period']):<8} {int(row['RSI Threshold']):<11} "
                  f"{row[metric]:<15.2f} {row['CAGR (%)']:<12.2f} {row['MDD (%)']:<10.2f} {int(row['Total Trades']):<8}")
        print("-"*100)

        return self.best_params

    def validate_on_test(self):
        """Test 데이터로 최적 파라미터 검증"""
        print("\n" + "="*80)
        print("Validating Best Parameters on Test Data")
        print("="*80)

        period = int(self.best_params['RSI Period'])
        threshold = int(self.best_params['RSI Threshold'])

        print(f"\nTesting RSI({period}, {threshold}) on unseen data...")

        # Test 데이터로 백테스트
        test_result = self.backtest_strategy(self.test_data, period, threshold)
        test_metrics = self.calculate_metrics(test_result)

        print("\n📊 Test Performance:")
        for key in ['Total Return (%)', 'CAGR (%)', 'MDD (%)', 'Sharpe Ratio',
                    'Calmar Ratio', 'Win Rate (%)', 'Total Trades', 'Profit Factor']:
            value = test_metrics[key]
            if key == 'Total Trades':
                print(f"  {key}: {int(value)}")
            elif key == 'Profit Factor' and value == np.inf:
                print(f"  {key}: INF")
            else:
                print(f"  {key}: {value:.2f}")

        # Train vs Test 비교
        print("\n" + "="*80)
        print("Train vs Test Comparison")
        print("="*80)
        print(f"\n{'Metric':<25} {'Train':>15} {'Test':>15} {'Difference':>15} {'% Change':>12}")
        print("-"*85)

        for key in ['Total Return (%)', 'CAGR (%)', 'MDD (%)', 'Sharpe Ratio',
                    'Win Rate (%)', 'Total Trades']:
            train_val = self.best_params[key]
            test_val = test_metrics[key]

            if key == 'Total Trades':
                diff = test_val - train_val
                print(f"{key:<25} {int(train_val):>15} {int(test_val):>15} {int(diff):>+15} {'N/A':>12}")
            else:
                diff = test_val - train_val
                pct_change = (diff / train_val * 100) if train_val != 0 else 0
                print(f"{key:<25} {train_val:>15.2f} {test_val:>15.2f} {diff:>+15.2f} {pct_change:>+11.1f}%")

        print("-"*85)

        # 과적합 경고
        train_sharpe = self.best_params['Sharpe Ratio']
        test_sharpe = test_metrics['Sharpe Ratio']
        sharpe_diff_pct = abs((test_sharpe - train_sharpe) / train_sharpe * 100)

        print("\n🔍 Overfitting Check:")
        if sharpe_diff_pct > 50:
            print(f"  ⚠️  WARNING: Sharpe Ratio difference > 50% ({sharpe_diff_pct:.1f}%)")
            print("  Possible overfitting detected!")
        elif sharpe_diff_pct > 30:
            print(f"  ⚠️  CAUTION: Sharpe Ratio difference > 30% ({sharpe_diff_pct:.1f}%)")
            print("  Moderate overfitting risk")
        else:
            print(f"  ✅ PASS: Sharpe Ratio difference < 30% ({sharpe_diff_pct:.1f}%)")
            print("  Parameters appear robust")

        print("="*80 + "\n")

        return test_metrics, test_result

    def plot_optimization_results(self, save_path='bitcoin_rsi_optimization_results.png'):
        """최적화 결과 시각화"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 샤프 비율 히트맵
        ax1 = fig.add_subplot(gs[0, 0])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='Sharpe Ratio'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax1,
                   cbar_kws={'label': 'Sharpe Ratio'})
        ax1.set_title('Sharpe Ratio by Parameters (Train)', fontsize=12, fontweight='bold')

        # 2. CAGR 히트맵
        ax2 = fig.add_subplot(gs[0, 1])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='CAGR (%)'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2,
                   cbar_kws={'label': 'CAGR (%)'})
        ax2.set_title('CAGR by Parameters (Train)', fontsize=12, fontweight='bold')

        # 3. MDD 히트맵
        ax3 = fig.add_subplot(gs[0, 2])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='MDD (%)'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax3,
                   cbar_kws={'label': 'MDD (%)'})
        ax3.set_title('MDD by Parameters (Train)', fontsize=12, fontweight='bold')

        # 4. Calmar Ratio 히트맵
        ax4 = fig.add_subplot(gs[1, 0])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='Calmar Ratio'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4,
                   cbar_kws={'label': 'Calmar Ratio'})
        ax4.set_title('Calmar Ratio by Parameters (Train)', fontsize=12, fontweight='bold')

        # 5. 승률 히트맵
        ax5 = fig.add_subplot(gs[1, 1])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='Win Rate (%)'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax5,
                   cbar_kws={'label': 'Win Rate (%)'})
        ax5.set_title('Win Rate by Parameters (Train)', fontsize=12, fontweight='bold')

        # 6. 거래 횟수 히트맵
        ax6 = fig.add_subplot(gs[1, 2])
        heatmap_data = self.optimization_results.pivot(
            index='RSI Threshold', columns='RSI Period', values='Total Trades'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='Blues', ax=ax6,
                   cbar_kws={'label': 'Total Trades'})
        ax6.set_title('Total Trades by Parameters (Train)', fontsize=12, fontweight='bold')

        # 7. Sharpe vs CAGR 산점도
        ax7 = fig.add_subplot(gs[2, 0])
        scatter = ax7.scatter(self.optimization_results['Sharpe Ratio'],
                            self.optimization_results['CAGR (%)'],
                            c=self.optimization_results['MDD (%)'],
                            s=100, alpha=0.6, cmap='RdYlGn')

        # 최적 파라미터 강조
        best_point = self.best_params
        ax7.scatter(best_point['Sharpe Ratio'], best_point['CAGR (%)'],
                   s=300, color='red', marker='*', edgecolors='black', linewidths=2,
                   label=f"Best: RSI({int(best_point['RSI Period'])}, {int(best_point['RSI Threshold'])})")

        ax7.set_xlabel('Sharpe Ratio', fontsize=11)
        ax7.set_ylabel('CAGR (%)', fontsize=11)
        ax7.set_title('Risk-Adjusted Return (colored by MDD)', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax7, label='MDD (%)')

        # 8. CAGR vs MDD 산점도
        ax8 = fig.add_subplot(gs[2, 1])
        scatter = ax8.scatter(self.optimization_results['MDD (%)'],
                            self.optimization_results['CAGR (%)'],
                            c=self.optimization_results['Sharpe Ratio'],
                            s=100, alpha=0.6, cmap='RdYlGn')

        # 최적 파라미터 강조
        ax8.scatter(best_point['MDD (%)'], best_point['CAGR (%)'],
                   s=300, color='red', marker='*', edgecolors='black', linewidths=2,
                   label=f"Best: RSI({int(best_point['RSI Period'])}, {int(best_point['RSI Threshold'])})")

        ax8.set_xlabel('MDD (%)', fontsize=11)
        ax8.set_ylabel('CAGR (%)', fontsize=11)
        ax8.set_title('Return vs Risk (colored by Sharpe)', fontsize=12, fontweight='bold')
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax8, label='Sharpe Ratio')

        # 9. Top 10 파라미터 바차트
        ax9 = fig.add_subplot(gs[2, 2])
        top_10 = self.optimization_results.nlargest(10, 'Sharpe Ratio')
        labels = [f"({int(row['RSI Period'])}, {int(row['RSI Threshold'])})"
                 for _, row in top_10.iterrows()]
        colors = ['red' if i == 0 else 'steelblue' for i in range(len(top_10))]

        ax9.barh(range(len(top_10)), top_10['Sharpe Ratio'], color=colors, alpha=0.7)
        ax9.set_yticks(range(len(top_10)))
        ax9.set_yticklabels(labels, fontsize=9)
        ax9.set_xlabel('Sharpe Ratio', fontsize=11)
        ax9.set_title('Top 10 Parameters by Sharpe Ratio', fontsize=12, fontweight='bold')
        ax9.invert_yaxis()
        ax9.grid(True, alpha=0.3, axis='x')

        # 전체 제목
        fig.suptitle(f'Bitcoin RSI Strategy - Parameter Optimization Results\n'
                    f'Train: {self.train_start} to {self.train_end} ({len(self.train_data)} days)',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nOptimization chart saved to {save_path}")
        plt.close()

    def run_full_optimization(self, metric='Sharpe Ratio'):
        """전체 최적화 프로세스 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 파라미터 최적화 (Train)
        self.optimize_parameters()

        # 3. 최적 파라미터 찾기
        self.find_best_parameters(metric=metric)

        # 4. Test 데이터로 검증
        test_metrics, test_result = self.validate_on_test()

        # 5. 시각화
        self.plot_optimization_results()

        # 6. 결과 저장
        print("\nSaving optimization results...")
        self.optimization_results.to_csv('bitcoin_rsi_optimization_results.csv', index=False)
        print("Results saved to bitcoin_rsi_optimization_results.csv")

        return self.best_params, test_metrics


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("비트코인 RSI 전략 파라미터 최적화")
    print("="*80)
    print("\n⚠️  Data Snooping Bias 방지:")
    print("  - Train: 2018-01-01 ~ 2021-12-31 (파라미터 최적화)")
    print("  - Test: 2022-01-01 ~ 현재 (성능 검증)")
    print("  - Test 데이터는 단 1회만 사용!\n")

    # 최적화 실행
    optimizer = BitcoinRSIOptimizer(
        symbol='BTC_KRW',
        train_start='2018-01-01',
        train_end='2021-12-31',
        test_start='2022-01-01',
        test_end=None,
        slippage=0.002
    )

    # 전체 최적화 실행 (Sharpe Ratio 기준)
    best_params, test_metrics = optimizer.run_full_optimization(metric='Sharpe Ratio')

    print("\n" + "="*80)
    print("최적화 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
