"""
ADA SMA 돌파전략 최적화
다양한 SMA 윈도우 파라미터를 테스트하여 최적 파라미터를 찾습니다.
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


class ADASMAOptimization:
    """ADA SMA 전략 최적화 클래스"""

    def __init__(self, symbol='ADA_KRW', start_date='2018-01-01',
                 end_date=None, slippage=0.002):
        """
        Args:
            symbol: 종목 심볼 (default: ADA_KRW)
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = None
        self.optimization_results = {}

    def load_data(self):
        """데이터 로드"""
        print("="*80)
        print(f"Loading {self.symbol} data...")
        print("="*80)

        file_path = f'chart_day/{self.symbol}.parquet'
        print(f"\nLoading from {file_path}...")
        df = pd.read_parquet(file_path)

        # 컬럼명 변경 (소문자 -> 대문자)
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df
        print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        print("="*80 + "\n")

    def strategy_sma_breakout(self, df, sma_period):
        """
        SMA 돌파전략
        - 가격이 SMA 위로 돌파: 매수
        - 가격이 SMA 아래로 하락: 매도

        Args:
            df: 가격 데이터
            sma_period: SMA 기간

        Returns:
            백테스트 결과 데이터프레임
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 포지션 계산 (가격 >= SMA: 매수, 가격 < SMA: 매도)
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 매수/매도 시 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage  # 매수
        slippage_cost[df['position_change'] == -1] = -self.slippage  # 매도

        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def calculate_metrics(self, returns_series, sma_period):
        """성과 지표 계산"""
        # 누적 수익률
        cumulative = (1 + returns_series).cumprod()

        # 총 수익률
        total_return = (cumulative.iloc[-1] - 1) * 100

        # 연간 수익률 (CAGR)
        years = (returns_series.index[-1] - returns_series.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # MDD
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100

        # 샤프 비율
        sharpe = (returns_series.mean() / returns_series.std() * np.sqrt(365)) if returns_series.std() > 0 else 0

        # 승률
        total_trades = (returns_series != 0).sum()
        winning_trades = (returns_series > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit Factor
        total_profit = returns_series[returns_series > 0].sum()
        total_loss = abs(returns_series[returns_series < 0].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        # Calmar Ratio (CAGR / abs(MDD))
        calmar = abs(cagr / mdd) if mdd != 0 else 0

        return {
            'SMA_Period': sma_period,
            'Total_Return_%': total_return,
            'CAGR_%': cagr,
            'MDD_%': mdd,
            'Sharpe_Ratio': sharpe,
            'Calmar_Ratio': calmar,
            'Win_Rate_%': win_rate,
            'Total_Trades': int(total_trades),
            'Profit_Factor': profit_factor
        }

    def optimize_sma_window(self, sma_range=None):
        """
        다양한 SMA 윈도우로 백테스트 실행하여 최적화

        Args:
            sma_range: 테스트할 SMA 기간 리스트 (None이면 기본값 사용)

        Returns:
            최적화 결과 데이터프레임
        """
        if sma_range is None:
            # 기본 SMA 범위: 5일부터 200일까지
            sma_range = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100, 120, 150, 200]

        print("\n" + "="*80)
        print(f"Optimizing SMA parameters for {self.symbol}")
        print(f"Testing {len(sma_range)} different SMA periods: {sma_range}")
        print("="*80 + "\n")

        results = []

        for sma_period in sma_range:
            print(f"Testing SMA {sma_period}...", end=' ')

            # 백테스트 실행
            result_df = self.strategy_sma_breakout(self.data.copy(), sma_period)

            # 성과 지표 계산
            metrics = self.calculate_metrics(result_df['returns'], sma_period)
            results.append(metrics)

            # 결과 저장
            self.optimization_results[sma_period] = result_df

            print(f"CAGR: {metrics['CAGR_%']:.2f}%, MDD: {metrics['MDD_%']:.2f}%, Sharpe: {metrics['Sharpe_Ratio']:.2f}")

        results_df = pd.DataFrame(results)

        print("\n" + "="*80)
        print("Optimization completed!")
        print("="*80 + "\n")

        return results_df

    def find_best_parameters(self, results_df):
        """최적 파라미터 찾기"""
        print("\n" + "="*80)
        print("Finding Best Parameters")
        print("="*80 + "\n")

        # 여러 기준으로 최적 파라미터 찾기
        best_by_total_return = results_df.loc[results_df['Total_Return_%'].idxmax()]
        best_by_cagr = results_df.loc[results_df['CAGR_%'].idxmax()]
        best_by_sharpe = results_df.loc[results_df['Sharpe_Ratio'].idxmax()]
        best_by_calmar = results_df.loc[results_df['Calmar_Ratio'].idxmax()]
        best_by_mdd = results_df.loc[results_df['MDD_%'].idxmax()]  # MDD는 음수이므로 최대값이 최소 손실

        print("Best Parameters by Different Metrics:\n")
        print(f"1. Best by Total Return: SMA {int(best_by_total_return['SMA_Period'])}")
        print(f"   - Total Return: {best_by_total_return['Total_Return_%']:.2f}%")
        print(f"   - CAGR: {best_by_total_return['CAGR_%']:.2f}%")
        print(f"   - MDD: {best_by_total_return['MDD_%']:.2f}%")
        print(f"   - Sharpe: {best_by_total_return['Sharpe_Ratio']:.2f}")

        print(f"\n2. Best by CAGR: SMA {int(best_by_cagr['SMA_Period'])}")
        print(f"   - Total Return: {best_by_cagr['Total_Return_%']:.2f}%")
        print(f"   - CAGR: {best_by_cagr['CAGR_%']:.2f}%")
        print(f"   - MDD: {best_by_cagr['MDD_%']:.2f}%")
        print(f"   - Sharpe: {best_by_cagr['Sharpe_Ratio']:.2f}")

        print(f"\n3. Best by Sharpe Ratio: SMA {int(best_by_sharpe['SMA_Period'])}")
        print(f"   - Total Return: {best_by_sharpe['Total_Return_%']:.2f}%")
        print(f"   - CAGR: {best_by_sharpe['CAGR_%']:.2f}%")
        print(f"   - MDD: {best_by_sharpe['MDD_%']:.2f}%")
        print(f"   - Sharpe: {best_by_sharpe['Sharpe_Ratio']:.2f}")

        print(f"\n4. Best by Calmar Ratio: SMA {int(best_by_calmar['SMA_Period'])}")
        print(f"   - Total Return: {best_by_calmar['Total_Return_%']:.2f}%")
        print(f"   - CAGR: {best_by_calmar['CAGR_%']:.2f}%")
        print(f"   - MDD: {best_by_calmar['MDD_%']:.2f}%")
        print(f"   - Calmar: {best_by_calmar['Calmar_Ratio']:.2f}")

        print(f"\n5. Best by MDD (Minimum Drawdown): SMA {int(best_by_mdd['SMA_Period'])}")
        print(f"   - Total Return: {best_by_mdd['Total_Return_%']:.2f}%")
        print(f"   - CAGR: {best_by_mdd['CAGR_%']:.2f}%")
        print(f"   - MDD: {best_by_mdd['MDD_%']:.2f}%")
        print(f"   - Sharpe: {best_by_mdd['Sharpe_Ratio']:.2f}")

        print("\n" + "="*80 + "\n")

        # 종합 점수 계산 (Sharpe Ratio와 Calmar Ratio의 평균을 기준으로)
        results_df['composite_score'] = (results_df['Sharpe_Ratio'] + results_df['Calmar_Ratio']) / 2
        best_overall = results_df.loc[results_df['composite_score'].idxmax()]

        print("RECOMMENDED: Best Overall (by composite score of Sharpe + Calmar):")
        print(f"SMA Period: {int(best_overall['SMA_Period'])}")
        print(f"   - Total Return: {best_overall['Total_Return_%']:.2f}%")
        print(f"   - CAGR: {best_overall['CAGR_%']:.2f}%")
        print(f"   - MDD: {best_overall['MDD_%']:.2f}%")
        print(f"   - Sharpe: {best_overall['Sharpe_Ratio']:.2f}")
        print(f"   - Calmar: {best_overall['Calmar_Ratio']:.2f}")
        print(f"   - Composite Score: {best_overall['composite_score']:.2f}")
        print("="*80 + "\n")

        return int(best_overall['SMA_Period'])

    def plot_optimization_results(self, results_df, save_path='ada_sma_optimization_results.png'):
        """최적화 결과 시각화"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. CAGR vs SMA Period
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results_df['SMA_Period'], results_df['CAGR_%'],
                marker='o', linewidth=2, markersize=6, color='blue')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        best_cagr_idx = results_df['CAGR_%'].idxmax()
        ax1.scatter([results_df.loc[best_cagr_idx, 'SMA_Period']],
                   [results_df.loc[best_cagr_idx, 'CAGR_%']],
                   color='red', s=200, zorder=5, marker='*', label='Best')
        ax1.set_xlabel('SMA Period', fontsize=11)
        ax1.set_ylabel('CAGR (%)', fontsize=11)
        ax1.set_title('CAGR vs SMA Period', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. MDD vs SMA Period
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results_df['SMA_Period'], results_df['MDD_%'],
                marker='o', linewidth=2, markersize=6, color='red')
        best_mdd_idx = results_df['MDD_%'].idxmax()
        ax2.scatter([results_df.loc[best_mdd_idx, 'SMA_Period']],
                   [results_df.loc[best_mdd_idx, 'MDD_%']],
                   color='green', s=200, zorder=5, marker='*', label='Best (Min DD)')
        ax2.set_xlabel('SMA Period', fontsize=11)
        ax2.set_ylabel('MDD (%)', fontsize=11)
        ax2.set_title('Maximum Drawdown vs SMA Period', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Sharpe Ratio vs SMA Period
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(results_df['SMA_Period'], results_df['Sharpe_Ratio'],
                marker='o', linewidth=2, markersize=6, color='green')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Sharpe=1')
        best_sharpe_idx = results_df['Sharpe_Ratio'].idxmax()
        ax3.scatter([results_df.loc[best_sharpe_idx, 'SMA_Period']],
                   [results_df.loc[best_sharpe_idx, 'Sharpe_Ratio']],
                   color='red', s=200, zorder=5, marker='*', label='Best')
        ax3.set_xlabel('SMA Period', fontsize=11)
        ax3.set_ylabel('Sharpe Ratio', fontsize=11)
        ax3.set_title('Sharpe Ratio vs SMA Period', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Calmar Ratio vs SMA Period
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(results_df['SMA_Period'], results_df['Calmar_Ratio'],
                marker='o', linewidth=2, markersize=6, color='purple')
        best_calmar_idx = results_df['Calmar_Ratio'].idxmax()
        ax4.scatter([results_df.loc[best_calmar_idx, 'SMA_Period']],
                   [results_df.loc[best_calmar_idx, 'Calmar_Ratio']],
                   color='red', s=200, zorder=5, marker='*', label='Best')
        ax4.set_xlabel('SMA Period', fontsize=11)
        ax4.set_ylabel('Calmar Ratio', fontsize=11)
        ax4.set_title('Calmar Ratio vs SMA Period', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # 5. Win Rate vs SMA Period
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(results_df['SMA_Period'], results_df['Win_Rate_%'],
                marker='o', linewidth=2, markersize=6, color='orange')
        ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax5.set_xlabel('SMA Period', fontsize=11)
        ax5.set_ylabel('Win Rate (%)', fontsize=11)
        ax5.set_title('Win Rate vs SMA Period', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # 6. Total Trades vs SMA Period
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.bar(results_df['SMA_Period'], results_df['Total_Trades'],
               color='steelblue', alpha=0.7)
        ax6.set_xlabel('SMA Period', fontsize=11)
        ax6.set_ylabel('Total Trades', fontsize=11)
        ax6.set_title('Total Trades vs SMA Period', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        # 7. Return vs Risk Scatter (CAGR vs MDD)
        ax7 = fig.add_subplot(gs[2, 0])
        scatter = ax7.scatter(results_df['MDD_%'], results_df['CAGR_%'],
                             c=results_df['Sharpe_Ratio'], s=200,
                             cmap='RdYlGn', alpha=0.7, edgecolors='black')
        for idx, row in results_df.iterrows():
            ax7.annotate(f"{int(row['SMA_Period'])}",
                        (row['MDD_%'], row['CAGR_%']),
                        fontsize=8, ha='center', va='center')
        ax7.set_xlabel('MDD (%)', fontsize=11)
        ax7.set_ylabel('CAGR (%)', fontsize=11)
        ax7.set_title('Return vs Risk (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax7, label='Sharpe Ratio')

        # 8. Profit Factor vs SMA Period
        ax8 = fig.add_subplot(gs[2, 1])
        pf_data = results_df[results_df['Profit_Factor'] != np.inf].copy()
        if len(pf_data) > 0:
            ax8.plot(pf_data['SMA_Period'], pf_data['Profit_Factor'],
                    marker='o', linewidth=2, markersize=6, color='darkgreen')
            ax8.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='PF=1')
            ax8.set_xlabel('SMA Period', fontsize=11)
            ax8.set_ylabel('Profit Factor', fontsize=11)
            ax8.set_title('Profit Factor vs SMA Period', fontsize=13, fontweight='bold')
            ax8.grid(True, alpha=0.3)
            ax8.legend()

        # 9. Performance Metrics Table (Top 5)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        # Composite score로 정렬
        results_df['composite_score'] = (results_df['Sharpe_Ratio'] + results_df['Calmar_Ratio']) / 2
        top5 = results_df.nlargest(5, 'composite_score')[['SMA_Period', 'CAGR_%', 'MDD_%', 'Sharpe_Ratio']].copy()
        top5['SMA_Period'] = top5['SMA_Period'].astype(int)

        table_text = "Top 5 SMA Periods\n" + "="*35 + "\n\n"
        table_text += f"{'Rank':<6}{'SMA':<8}{'CAGR':<10}{'MDD':<10}{'Sharpe':<8}\n"
        table_text += "-"*35 + "\n"

        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            table_text += f"{i:<6}{int(row['SMA_Period']):<8}{row['CAGR_%']:>6.1f}%  {row['MDD_%']:>6.1f}%  {row['Sharpe_Ratio']:>6.2f}\n"

        ax9.text(0.1, 0.95, table_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        # 전체 제목
        fig.suptitle(f'ADA SMA Breakout Strategy Optimization Results\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nOptimization chart saved to {save_path}")
        plt.close()

    def plot_best_strategy(self, best_sma_period, save_path='ada_best_sma_strategy.png'):
        """최적 SMA 전략 상세 분석 시각화"""
        result_df = self.optimization_results[best_sma_period].copy()

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 가격 + SMA + 포지션
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(result_df.index, result_df['Close'],
                label='ADA Price', color='gray', linewidth=1.5, alpha=0.6)
        ax1.plot(result_df.index, result_df['SMA'],
                label=f'SMA {best_sma_period}', color='blue', linewidth=2)

        # 매수/매도 신호
        position_changes = result_df['position'].diff()
        buy_signals = result_df[position_changes == 1].index
        sell_signals = result_df[position_changes == -1].index

        ax1.scatter(buy_signals, result_df.loc[buy_signals, 'Close'],
                   color='green', marker='^', s=100, label='Buy', zorder=5, alpha=0.7)
        ax1.scatter(sell_signals, result_df.loc[sell_signals, 'Close'],
                   color='red', marker='v', s=100, label='Sell', zorder=5, alpha=0.7)

        ax1.set_title(f'ADA Price with SMA {best_sma_period} Strategy (Best)',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (KRW)', fontsize=11)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 누적 수익률
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(result_df.index, result_df['cumulative'],
                color='blue', linewidth=2.5, label=f'SMA {best_sma_period} Strategy')

        # Buy & Hold 비교
        buy_hold_cumulative = self.data['Close'] / self.data['Close'].iloc[0]
        ax2.plot(buy_hold_cumulative.index, buy_hold_cumulative,
                color='orange', linewidth=2, alpha=0.7, label='Buy & Hold', linestyle='--')

        ax2.set_title('Cumulative Returns Comparison', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Cumulative Return', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 3. 성과 지표 요약
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.axis('off')

        metrics = self.calculate_metrics(result_df['returns'], best_sma_period)

        # Buy & Hold 성과도 계산
        bh_returns = buy_hold_cumulative.pct_change().fillna(0)
        bh_metrics = self.calculate_metrics(bh_returns, 0)

        metrics_text = f"Strategy Performance\n{'='*30}\n\n"
        metrics_text += f"SMA {best_sma_period} Strategy:\n"
        metrics_text += f"  Total Return: {metrics['Total_Return_%']:.2f}%\n"
        metrics_text += f"  CAGR: {metrics['CAGR_%']:.2f}%\n"
        metrics_text += f"  MDD: {metrics['MDD_%']:.2f}%\n"
        metrics_text += f"  Sharpe: {metrics['Sharpe_Ratio']:.2f}\n"
        metrics_text += f"  Calmar: {metrics['Calmar_Ratio']:.2f}\n"
        metrics_text += f"  Win Rate: {metrics['Win_Rate_%']:.2f}%\n"
        metrics_text += f"  Total Trades: {metrics['Total_Trades']}\n\n"

        metrics_text += f"Buy & Hold:\n"
        metrics_text += f"  Total Return: {bh_metrics['Total_Return_%']:.2f}%\n"
        metrics_text += f"  CAGR: {bh_metrics['CAGR_%']:.2f}%\n"
        metrics_text += f"  MDD: {bh_metrics['MDD_%']:.2f}%\n"
        metrics_text += f"  Sharpe: {bh_metrics['Sharpe_Ratio']:.2f}\n"

        ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 4. Drawdown
        ax4 = fig.add_subplot(gs[2, :2])
        cummax = result_df['cumulative'].cummax()
        drawdown = (result_df['cumulative'] - cummax) / cummax * 100
        ax4.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax4.plot(drawdown.index, drawdown, color='darkred', linewidth=2)

        # MDD 표시
        mdd_value = drawdown.min()
        mdd_date = drawdown.idxmin()
        ax4.scatter([mdd_date], [mdd_value], color='red', s=200, zorder=5, marker='X')
        ax4.annotate(f'MDD: {mdd_value:.2f}%',
                    xy=(mdd_date, mdd_value),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        ax4.set_title('Drawdown Over Time', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Drawdown (%)', fontsize=11)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 5. 월별 수익률 히트맵
        ax5 = fig.add_subplot(gs[2, 2])
        monthly_returns = result_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_returns_pivot = monthly_returns.to_frame('returns')
        monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
        heatmap_data = monthly_returns_pivot.pivot(index='year', columns='month', values='returns')

        sns.heatmap(heatmap_data, annot=False, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax5, linewidths=0.5)
        ax5.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Month', fontsize=10)
        ax5.set_ylabel('Year', fontsize=10)

        # 전체 제목
        fig.suptitle(f'ADA Best SMA Strategy (SMA {best_sma_period}) Detailed Analysis\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Best strategy chart saved to {save_path}")
        plt.close()

    def run_optimization(self, sma_range=None):
        """전체 최적화 프로세스 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. SMA 최적화 실행
        results_df = self.optimize_sma_window(sma_range)

        # 3. 결과 출력
        print("\n" + "="*80)
        print("Optimization Results Summary")
        print("="*80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}')
        print(results_df.to_string(index=False))
        print("="*80 + "\n")

        # 4. 최적 파라미터 찾기
        best_sma = self.find_best_parameters(results_df)

        # 5. 시각화
        self.plot_optimization_results(results_df)
        self.plot_best_strategy(best_sma)

        # 6. 결과 저장
        results_df.to_csv('ada_sma_optimization_results.csv', index=False)
        print(f"\nResults saved to ada_sma_optimization_results.csv")

        return results_df, best_sma


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("ADA SMA Breakout Strategy Optimization")
    print("="*80)

    # 최적화 실행
    optimizer = ADASMAOptimization(
        symbol='ADA_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # SMA 범위 설정 (5일부터 200일까지)
    sma_range = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100, 120, 150, 200]

    # 최적화 실행
    results_df, best_sma = optimizer.run_optimization(sma_range)

    print("\n" + "="*80)
    print(f"OPTIMIZATION COMPLETED!")
    print(f"Best SMA Period: {best_sma}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
