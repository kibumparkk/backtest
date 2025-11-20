"""
상위 파라미터들의 강건성(Robustness) 검증

Train 데이터에서 상위 5개 파라미터를 모두 Test 데이터로 검증하여
가장 안정적인 파라미터를 찾습니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class TopParametersValidator:
    """상위 파라미터 검증"""

    def __init__(self, symbol='BTC_KRW',
                 train_start='2018-01-01', train_end='2021-12-31',
                 test_start='2022-01-01', test_end=None,
                 slippage=0.002):
        self.symbol = symbol
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end if test_end else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage

        self.data = None
        self.train_data = None
        self.test_data = None

    def load_data(self):
        """데이터 로드"""
        print("="*80)
        print("Loading data...")
        print("="*80)

        file_path = f'chart_day/{self.symbol}.parquet'
        df = pd.read_parquet(file_path)
        df.columns = [col.capitalize() for col in df.columns]

        self.data = df
        self.train_data = df[(df.index >= self.train_start) & (df.index <= self.train_end)]
        self.test_data = df[(df.index >= self.test_start) & (df.index <= self.test_end)]

        print(f"\nTrain: {len(self.train_data)} days ({self.train_start} ~ {self.train_end})")
        print(f"Test: {len(self.test_data)} days ({self.test_start} ~ {self.test_end})")
        print("="*80 + "\n")

    def calculate_rsi_ewm(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def backtest_strategy(self, df, rsi_period, rsi_threshold):
        """백테스트 실행"""
        df = df.copy()

        df['RSI'] = self.calculate_rsi_ewm(df['Close'], rsi_period)
        df['signal'] = (df['RSI'] >= rsi_threshold).astype(int)
        df['position'] = df['signal'].shift(1)
        df['position_change'] = df['position'].diff()

        df['daily_price_return'] = df['Close'].pct_change()
        df['strategy_return'] = df['position'] * df['daily_price_return']

        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['strategy_return'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def calculate_metrics(self, df):
        """성과 지표 계산"""
        returns = df['returns']
        cumulative = df['cumulative']

        years = (df.index[-1] - df.index[0]).days / 365.25

        total_return = (cumulative.iloc[-1] - 1) * 100
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100

        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0

        total_trades = (returns != 0).sum()
        winning_trades = (returns > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_profit = returns[returns > 0].sum()
        total_loss = abs(returns[returns < 0].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

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

    def validate_top_parameters(self):
        """상위 5개 파라미터 검증"""
        print("\n" + "="*80)
        print("Validating Top 5 Parameters on Test Data")
        print("="*80)

        # 최적화 결과 로드
        opt_results = pd.read_csv('bitcoin_rsi_optimization_results.csv')

        # 상위 5개 파라미터
        top_5 = opt_results.nlargest(5, 'Sharpe Ratio')

        print(f"\nTesting top 5 parameter combinations from training...")
        print()

        validation_results = []

        for rank, (idx, row) in enumerate(top_5.iterrows(), 1):
            period = int(row['RSI Period'])
            threshold = int(row['RSI Threshold'])

            print(f"Rank {rank}: RSI({period}, {threshold})")
            print(f"  Train Sharpe: {row['Sharpe Ratio']:.2f}")

            # Train 성과
            train_result = self.backtest_strategy(self.train_data, period, threshold)
            train_metrics = self.calculate_metrics(train_result)

            # Test 성과
            test_result = self.backtest_strategy(self.test_data, period, threshold)
            test_metrics = self.calculate_metrics(test_result)

            print(f"  Test Sharpe: {test_metrics['Sharpe Ratio']:.2f}")

            # 성과 차이
            sharpe_diff = test_metrics['Sharpe Ratio'] - train_metrics['Sharpe Ratio']
            sharpe_diff_pct = (sharpe_diff / train_metrics['Sharpe Ratio'] * 100) if train_metrics['Sharpe Ratio'] != 0 else 0

            cagr_diff_pct = ((test_metrics['CAGR (%)'] - train_metrics['CAGR (%)']) / train_metrics['CAGR (%)'] * 100) if train_metrics['CAGR (%)'] != 0 else 0

            print(f"  Sharpe Diff: {sharpe_diff:+.2f} ({sharpe_diff_pct:+.1f}%)")
            print(f"  Test CAGR: {test_metrics['CAGR (%)']:.2f}% (Train: {train_metrics['CAGR (%)']:.2f}%)")
            print(f"  Test MDD: {test_metrics['MDD (%)']:.2f}% (Train: {train_metrics['MDD (%)']:.2f}%)")

            # 과적합 판정
            if abs(sharpe_diff_pct) > 50:
                overfitting = "⚠️ HIGH"
            elif abs(sharpe_diff_pct) > 30:
                overfitting = "⚠️ MODERATE"
            else:
                overfitting = "✅ LOW"

            print(f"  Overfitting Risk: {overfitting}")
            print()

            # 결과 저장
            validation_results.append({
                'Rank': rank,
                'RSI Period': period,
                'RSI Threshold': threshold,
                'Train CAGR (%)': train_metrics['CAGR (%)'],
                'Test CAGR (%)': test_metrics['CAGR (%)'],
                'Train Sharpe': train_metrics['Sharpe Ratio'],
                'Test Sharpe': test_metrics['Sharpe Ratio'],
                'Train MDD (%)': train_metrics['MDD (%)'],
                'Test MDD (%)': test_metrics['MDD (%)'],
                'Train Calmar': train_metrics['Calmar Ratio'],
                'Test Calmar': test_metrics['Calmar Ratio'],
                'Train Win Rate (%)': train_metrics['Win Rate (%)'],
                'Test Win Rate (%)': test_metrics['Win Rate (%)'],
                'Sharpe Diff (%)': sharpe_diff_pct,
                'CAGR Diff (%)': cagr_diff_pct,
                'Overfitting Risk': overfitting
            })

        results_df = pd.DataFrame(validation_results)

        print("="*80)
        print("Validation Summary")
        print("="*80)
        print()

        # 가장 강건한 파라미터 찾기
        # 기준: Test Sharpe가 높으면서 Sharpe Diff가 작은 것
        results_df['Robustness Score'] = results_df['Test Sharpe'] - abs(results_df['Sharpe Diff (%)']) / 100

        best_robust_idx = results_df['Robustness Score'].idxmax()
        best_robust = results_df.loc[best_robust_idx]

        print(f"🏆 Most Robust Parameters:")
        print(f"  RSI({int(best_robust['RSI Period'])}, {int(best_robust['RSI Threshold'])})")
        print(f"  Test Sharpe: {best_robust['Test Sharpe']:.2f}")
        print(f"  Test CAGR: {best_robust['Test CAGR (%)']:.2f}%")
        print(f"  Test MDD: {best_robust['Test MDD (%)']:.2f}%")
        print(f"  Sharpe Diff: {best_robust['Sharpe Diff (%)']:+.1f}%")
        print(f"  Overfitting Risk: {best_robust['Overfitting Risk']}")
        print()

        # 가장 높은 Test 성과
        best_test_idx = results_df['Test Sharpe'].idxmax()
        best_test = results_df.loc[best_test_idx]

        print(f"🎯 Best Test Performance:")
        print(f"  RSI({int(best_test['RSI Period'])}, {int(best_test['RSI Threshold'])})")
        print(f"  Test Sharpe: {best_test['Test Sharpe']:.2f}")
        print(f"  Test CAGR: {best_test['Test CAGR (%)']:.2f}%")
        print(f"  Test MDD: {best_test['Test MDD (%)']:.2f}%")
        print()

        print("="*80)
        print("Detailed Comparison Table")
        print("="*80)
        print()
        print(results_df.to_string(index=False))
        print()

        # 저장
        results_df.to_csv('top_parameters_validation.csv', index=False)
        print("\nResults saved to top_parameters_validation.csv")

        return results_df, best_robust

    def plot_validation_results(self, results_df, save_path='top_parameters_validation.png'):
        """검증 결과 시각화"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # 1. Train vs Test Sharpe 비교
        ax1 = fig.add_subplot(gs[0, 0])
        x = range(len(results_df))
        width = 0.35
        ax1.bar([i - width/2 for i in x], results_df['Train Sharpe'],
               width, label='Train', color='steelblue', alpha=0.7)
        ax1.bar([i + width/2 for i in x], results_df['Test Sharpe'],
               width, label='Test', color='orange', alpha=0.7)
        ax1.set_xlabel('Parameter Rank', fontsize=11)
        ax1.set_ylabel('Sharpe Ratio', fontsize=11)
        ax1.set_title('Train vs Test Sharpe Ratio', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Rank {i+1}" for i in x])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Train vs Test CAGR 비교
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar([i - width/2 for i in x], results_df['Train CAGR (%)'],
               width, label='Train', color='steelblue', alpha=0.7)
        ax2.bar([i + width/2 for i in x], results_df['Test CAGR (%)'],
               width, label='Test', color='orange', alpha=0.7)
        ax2.set_xlabel('Parameter Rank', fontsize=11)
        ax2.set_ylabel('CAGR (%)', fontsize=11)
        ax2.set_title('Train vs Test CAGR', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Rank {i+1}" for i in x])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Train vs Test MDD 비교
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar([i - width/2 for i in x], results_df['Train MDD (%)'],
               width, label='Train', color='steelblue', alpha=0.7)
        ax3.bar([i + width/2 for i in x], results_df['Test MDD (%)'],
               width, label='Test', color='orange', alpha=0.7)
        ax3.set_xlabel('Parameter Rank', fontsize=11)
        ax3.set_ylabel('MDD (%)', fontsize=11)
        ax3.set_title('Train vs Test MDD', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"Rank {i+1}" for i in x])
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Sharpe 차이 비율
        ax4 = fig.add_subplot(gs[1, 0])
        colors = ['red' if abs(x) > 50 else 'orange' if abs(x) > 30 else 'green'
                 for x in results_df['Sharpe Diff (%)']]
        ax4.barh(range(len(results_df)), results_df['Sharpe Diff (%)'],
                color=colors, alpha=0.7)
        ax4.set_yticks(range(len(results_df)))
        ax4.set_yticklabels([f"RSI({int(row['RSI Period'])}, {int(row['RSI Threshold'])})"
                            for _, row in results_df.iterrows()])
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(x=-30, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axvline(x=30, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axvline(x=-50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Sharpe Diff (%)', fontsize=11)
        ax4.set_title('Sharpe Ratio Difference (Test - Train)', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. Test Sharpe vs Overfitting
        ax5 = fig.add_subplot(gs[1, 1])
        colors_map = {'✅ LOW': 'green', '⚠️ MODERATE': 'orange', '⚠️ HIGH': 'red'}
        colors = [colors_map[risk] for risk in results_df['Overfitting Risk']]
        scatter = ax5.scatter(abs(results_df['Sharpe Diff (%)']),
                            results_df['Test Sharpe'],
                            c=colors, s=200, alpha=0.7, edgecolors='black', linewidths=2)

        for idx, row in results_df.iterrows():
            ax5.annotate(f"RSI({int(row['RSI Period'])},{int(row['RSI Threshold'])})",
                        (abs(row['Sharpe Diff (%)']), row['Test Sharpe']),
                        fontsize=9, ha='center', va='bottom')

        ax5.set_xlabel('|Sharpe Diff %|', fontsize=11)
        ax5.set_ylabel('Test Sharpe Ratio', fontsize=11)
        ax5.set_title('Test Performance vs Overfitting Risk', fontsize=12, fontweight='bold')
        ax5.axvline(x=30, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='30% threshold')
        ax5.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # 6. Robustness Score
        ax6 = fig.add_subplot(gs[1, 2])
        colors = ['green' if i == results_df['Robustness Score'].idxmax() else 'steelblue'
                 for i in range(len(results_df))]
        ax6.barh(range(len(results_df)), results_df['Robustness Score'],
                color=colors, alpha=0.7)
        ax6.set_yticks(range(len(results_df)))
        ax6.set_yticklabels([f"RSI({int(row['RSI Period'])}, {int(row['RSI Threshold'])})"
                            for _, row in results_df.iterrows()])
        ax6.set_xlabel('Robustness Score', fontsize=11)
        ax6.set_title('Robustness Score (Test Sharpe - |Diff%|/100)', fontsize=12, fontweight='bold')
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.3, axis='x')

        # 7. Train vs Test Calmar 비교
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.bar([i - width/2 for i in x], results_df['Train Calmar'],
               width, label='Train', color='steelblue', alpha=0.7)
        ax7.bar([i + width/2 for i in x], results_df['Test Calmar'],
               width, label='Test', color='orange', alpha=0.7)
        ax7.set_xlabel('Parameter Rank', fontsize=11)
        ax7.set_ylabel('Calmar Ratio', fontsize=11)
        ax7.set_title('Train vs Test Calmar Ratio', fontsize=12, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels([f"Rank {i+1}" for i in x])
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. Test CAGR vs MDD
        ax8 = fig.add_subplot(gs[2, 1])
        colors_map = {'✅ LOW': 'green', '⚠️ MODERATE': 'orange', '⚠️ HIGH': 'red'}
        colors = [colors_map[risk] for risk in results_df['Overfitting Risk']]
        scatter = ax8.scatter(results_df['Test MDD (%)'],
                            results_df['Test CAGR (%)'],
                            c=colors, s=200, alpha=0.7, edgecolors='black', linewidths=2)

        for idx, row in results_df.iterrows():
            ax8.annotate(f"RSI({int(row['RSI Period'])},{int(row['RSI Threshold'])})",
                        (row['Test MDD (%)'], row['Test CAGR (%)']),
                        fontsize=9, ha='center', va='bottom')

        ax8.set_xlabel('Test MDD (%)', fontsize=11)
        ax8.set_ylabel('Test CAGR (%)', fontsize=11)
        ax8.set_title('Test: Return vs Risk (colored by overfitting)', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # 9. 파라미터 정보 테이블
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        table_text = "Parameter Details\n" + "="*40 + "\n\n"
        for idx, row in results_df.iterrows():
            table_text += f"Rank {int(row['Rank'])}: RSI({int(row['RSI Period'])}, {int(row['RSI Threshold'])})\n"
            table_text += f"  Test Sharpe: {row['Test Sharpe']:.2f}\n"
            table_text += f"  Test CAGR: {row['Test CAGR (%)']:.2f}%\n"
            table_text += f"  Test MDD: {row['Test MDD (%)']:.2f}%\n"
            table_text += f"  Risk: {row['Overfitting Risk']}\n\n"

        ax9.text(0.05, 0.95, table_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 전체 제목
        fig.suptitle('Top 5 Parameters Validation: Train vs Test Performance\n'
                    f'Train: {self.train_start} to {self.train_end} | Test: {self.test_start} to {self.test_end}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nValidation chart saved to {save_path}")
        plt.close()

    def run_validation(self):
        """전체 검증 프로세스 실행"""
        self.load_data()
        results_df, best_robust = self.validate_top_parameters()
        self.plot_validation_results(results_df)

        return results_df, best_robust


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("상위 파라미터 강건성 검증")
    print("="*80)

    validator = TopParametersValidator(
        symbol='BTC_KRW',
        train_start='2018-01-01',
        train_end='2021-12-31',
        test_start='2022-01-01',
        test_end=None,
        slippage=0.002
    )

    results_df, best_robust = validator.run_validation()

    print("\n" + "="*80)
    print("검증 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
