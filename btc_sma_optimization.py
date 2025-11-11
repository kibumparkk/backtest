"""
BTC 단일 종목 SMA 전략 최적화 백테스트

매수 조건: 전일 종가 > SMA n일
SMA 기간(n)을 5일부터 120일까지 1단위로 변경하면서 최적 성과와 최소 MDD 도출
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


class BTCSMAOptimization:
    """BTC SMA 전략 최적화 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01',
                 end_date=None, slippage=0.002):
        """
        Args:
            symbol: 종목 심볼 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = None
        self.results = {}
        self.metrics_summary = None

    def load_data(self):
        """BTC 데이터 로드"""
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

    def strategy_sma(self, df, sma_period):
        """
        SMA 전략
        - 전일 종가 > SMA n일: 매수 (보유)
        - 전일 종가 <= SMA n일: 매도 후 현금 보유

        Args:
            df: 가격 데이터
            sma_period: SMA 기간
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 포지션 계산 - 전일 종가 기준
        df['position'] = np.where(df['Close'].shift(1) > df['SMA'].shift(1), 1, 0)

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

        # NaN 값 처리
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

        # 최종 자산
        final_value = cumulative.iloc[-1]

        return {
            'SMA_Period': sma_period,
            'Total_Return_Pct': total_return,
            'CAGR_Pct': cagr,
            'MDD_Pct': mdd,
            'Sharpe_Ratio': sharpe,
            'Win_Rate_Pct': win_rate,
            'Total_Trades': int(total_trades),
            'Profit_Factor': profit_factor,
            'Final_Value': final_value
        }

    def run_optimization(self, sma_range=(5, 120)):
        """
        SMA 기간 최적화

        Args:
            sma_range: SMA 기간 범위 (start, end) - inclusive
        """
        print("\n" + "="*80)
        print(f"Running SMA optimization for {self.symbol}")
        print(f"SMA period range: {sma_range[0]} to {sma_range[1]} days")
        print("="*80 + "\n")

        metrics_list = []

        for sma_period in range(sma_range[0], sma_range[1] + 1):
            if sma_period % 10 == 0 or sma_period == sma_range[0]:
                print(f"Testing SMA {sma_period}...")

            # 전략 실행
            result_df = self.strategy_sma(self.data, sma_period)

            # 결과 저장
            self.results[sma_period] = result_df

            # 성과 지표 계산
            metrics = self.calculate_metrics(result_df['returns'], sma_period)
            metrics_list.append(metrics)

        # 메트릭 요약 DataFrame 생성
        self.metrics_summary = pd.DataFrame(metrics_list)

        print("\n" + "="*80)
        print("Optimization completed!")
        print("="*80 + "\n")

        return self.metrics_summary

    def find_best_strategies(self):
        """최적 전략 찾기"""
        if self.metrics_summary is None:
            print("Please run optimization first!")
            return None

        # 최고 수익률
        best_return_idx = self.metrics_summary['Total_Return_Pct'].idxmax()
        best_return = self.metrics_summary.loc[best_return_idx]

        # 최고 CAGR
        best_cagr_idx = self.metrics_summary['CAGR_Pct'].idxmax()
        best_cagr = self.metrics_summary.loc[best_cagr_idx]

        # 최고 샤프 비율
        best_sharpe_idx = self.metrics_summary['Sharpe_Ratio'].idxmax()
        best_sharpe = self.metrics_summary.loc[best_sharpe_idx]

        # 최소 MDD (MDD는 음수이므로 max를 찾음)
        best_mdd_idx = self.metrics_summary['MDD_Pct'].idxmax()
        best_mdd = self.metrics_summary.loc[best_mdd_idx]

        print("\n" + "="*80)
        print("BEST STRATEGIES SUMMARY")
        print("="*80)

        print(f"\n[1] Best Total Return:")
        print(f"    SMA Period: {int(best_return['SMA_Period'])} days")
        print(f"    Total Return: {best_return['Total_Return_Pct']:.2f}%")
        print(f"    CAGR: {best_return['CAGR_Pct']:.2f}%")
        print(f"    MDD: {best_return['MDD_Pct']:.2f}%")
        print(f"    Sharpe: {best_return['Sharpe_Ratio']:.2f}")

        print(f"\n[2] Best CAGR:")
        print(f"    SMA Period: {int(best_cagr['SMA_Period'])} days")
        print(f"    Total Return: {best_cagr['Total_Return_Pct']:.2f}%")
        print(f"    CAGR: {best_cagr['CAGR_Pct']:.2f}%")
        print(f"    MDD: {best_cagr['MDD_Pct']:.2f}%")
        print(f"    Sharpe: {best_cagr['Sharpe_Ratio']:.2f}")

        print(f"\n[3] Best Sharpe Ratio:")
        print(f"    SMA Period: {int(best_sharpe['SMA_Period'])} days")
        print(f"    Total Return: {best_sharpe['Total_Return_Pct']:.2f}%")
        print(f"    CAGR: {best_sharpe['CAGR_Pct']:.2f}%")
        print(f"    MDD: {best_sharpe['MDD_Pct']:.2f}%")
        print(f"    Sharpe: {best_sharpe['Sharpe_Ratio']:.2f}")

        print(f"\n[4] Minimum MDD (Best Risk Control):")
        print(f"    SMA Period: {int(best_mdd['SMA_Period'])} days")
        print(f"    Total Return: {best_mdd['Total_Return_Pct']:.2f}%")
        print(f"    CAGR: {best_mdd['CAGR_Pct']:.2f}%")
        print(f"    MDD: {best_mdd['MDD_Pct']:.2f}%")
        print(f"    Sharpe: {best_mdd['Sharpe_Ratio']:.2f}")

        print("\n" + "="*80 + "\n")

        return {
            'best_return': best_return,
            'best_cagr': best_cagr,
            'best_sharpe': best_sharpe,
            'best_mdd': best_mdd
        }

    def plot_optimization_results(self, save_path='btc_sma_optimization_results.png'):
        """최적화 결과 시각화"""
        if self.metrics_summary is None:
            print("Please run optimization first!")
            return

        # 최적 전략 찾기
        best_strategies = self.find_best_strategies()

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. SMA Period vs Total Return
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.metrics_summary['SMA_Period'],
                self.metrics_summary['Total_Return_Pct'],
                linewidth=2, color='blue', alpha=0.7)
        best_idx = self.metrics_summary['Total_Return_Pct'].idxmax()
        ax1.scatter(self.metrics_summary.loc[best_idx, 'SMA_Period'],
                   self.metrics_summary.loc[best_idx, 'Total_Return_Pct'],
                   color='red', s=200, zorder=5, marker='*',
                   label=f"Best: SMA {int(self.metrics_summary.loc[best_idx, 'SMA_Period'])}")
        ax1.set_xlabel('SMA Period (days)', fontsize=11)
        ax1.set_ylabel('Total Return (%)', fontsize=11)
        ax1.set_title('SMA Period vs Total Return', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        # 2. SMA Period vs CAGR
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.metrics_summary['SMA_Period'],
                self.metrics_summary['CAGR_Pct'],
                linewidth=2, color='green', alpha=0.7)
        best_idx = self.metrics_summary['CAGR_Pct'].idxmax()
        ax2.scatter(self.metrics_summary.loc[best_idx, 'SMA_Period'],
                   self.metrics_summary.loc[best_idx, 'CAGR_Pct'],
                   color='red', s=200, zorder=5, marker='*',
                   label=f"Best: SMA {int(self.metrics_summary.loc[best_idx, 'SMA_Period'])}")
        ax2.set_xlabel('SMA Period (days)', fontsize=11)
        ax2.set_ylabel('CAGR (%)', fontsize=11)
        ax2.set_title('SMA Period vs CAGR', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        # 3. SMA Period vs MDD
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.metrics_summary['SMA_Period'],
                self.metrics_summary['MDD_Pct'],
                linewidth=2, color='red', alpha=0.7)
        best_idx = self.metrics_summary['MDD_Pct'].idxmax()  # 최소 MDD (덜 음수)
        ax3.scatter(self.metrics_summary.loc[best_idx, 'SMA_Period'],
                   self.metrics_summary.loc[best_idx, 'MDD_Pct'],
                   color='darkgreen', s=200, zorder=5, marker='*',
                   label=f"Best: SMA {int(self.metrics_summary.loc[best_idx, 'SMA_Period'])}")
        ax3.set_xlabel('SMA Period (days)', fontsize=11)
        ax3.set_ylabel('MDD (%)', fontsize=11)
        ax3.set_title('SMA Period vs Maximum Drawdown', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)

        # 4. SMA Period vs Sharpe Ratio
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.metrics_summary['SMA_Period'],
                self.metrics_summary['Sharpe_Ratio'],
                linewidth=2, color='purple', alpha=0.7)
        best_idx = self.metrics_summary['Sharpe_Ratio'].idxmax()
        ax4.scatter(self.metrics_summary.loc[best_idx, 'SMA_Period'],
                   self.metrics_summary.loc[best_idx, 'Sharpe_Ratio'],
                   color='red', s=200, zorder=5, marker='*',
                   label=f"Best: SMA {int(self.metrics_summary.loc[best_idx, 'SMA_Period'])}")
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axhline(y=1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1')
        ax4.set_xlabel('SMA Period (days)', fontsize=11)
        ax4.set_ylabel('Sharpe Ratio', fontsize=11)
        ax4.set_title('SMA Period vs Sharpe Ratio', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)

        # 5. Return vs Risk (Scatter)
        ax5 = fig.add_subplot(gs[1, 1])
        scatter = ax5.scatter(self.metrics_summary['MDD_Pct'],
                            self.metrics_summary['CAGR_Pct'],
                            c=self.metrics_summary['SMA_Period'],
                            cmap='viridis', s=50, alpha=0.6)

        # 최적 포인트 표시
        for key, strategy in best_strategies.items():
            ax5.scatter(strategy['MDD_Pct'], strategy['CAGR_Pct'],
                       s=300, marker='*', edgecolors='red',
                       linewidths=2, facecolors='none', zorder=10)
            ax5.annotate(f"SMA {int(strategy['SMA_Period'])}",
                        xy=(strategy['MDD_Pct'], strategy['CAGR_Pct']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                            facecolor='yellow', alpha=0.5))

        plt.colorbar(scatter, ax=ax5, label='SMA Period')
        ax5.set_xlabel('MDD (%)', fontsize=11)
        ax5.set_ylabel('CAGR (%)', fontsize=11)
        ax5.set_title('Return vs Risk (colored by SMA Period)', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. Win Rate
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(self.metrics_summary['SMA_Period'],
                self.metrics_summary['Win_Rate_Pct'],
                linewidth=2, color='orange', alpha=0.7)
        ax6.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50%')
        ax6.set_xlabel('SMA Period (days)', fontsize=11)
        ax6.set_ylabel('Win Rate (%)', fontsize=11)
        ax6.set_title('SMA Period vs Win Rate', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=9)

        # 7. 최고 성과 전략의 Cumulative Return
        ax7 = fig.add_subplot(gs[2, :])
        best_return_period = int(best_strategies['best_return']['SMA_Period'])
        best_mdd_period = int(best_strategies['best_mdd']['SMA_Period'])
        best_sharpe_period = int(best_strategies['best_sharpe']['SMA_Period'])

        ax7.plot(self.results[best_return_period].index,
                self.results[best_return_period]['cumulative'],
                label=f"Best Return (SMA {best_return_period})",
                linewidth=2.5, alpha=0.8)

        ax7.plot(self.results[best_mdd_period].index,
                self.results[best_mdd_period]['cumulative'],
                label=f"Best MDD (SMA {best_mdd_period})",
                linewidth=2.5, alpha=0.8)

        ax7.plot(self.results[best_sharpe_period].index,
                self.results[best_sharpe_period]['cumulative'],
                label=f"Best Sharpe (SMA {best_sharpe_period})",
                linewidth=2.5, alpha=0.8)

        # Buy & Hold 추가
        buy_hold_cumulative = self.data['Close'] / self.data['Close'].iloc[0]
        ax7.plot(self.data.index, buy_hold_cumulative,
                label='Buy & Hold', linewidth=2, alpha=0.6, linestyle='--', color='gray')

        ax7.set_title('Cumulative Returns: Best Strategies Comparison',
                     fontsize=14, fontweight='bold')
        ax7.set_ylabel('Cumulative Return', fontsize=12)
        ax7.set_xlabel('Date', fontsize=12)
        ax7.legend(loc='upper left', fontsize=11)
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale('log')

        # 8. 최소 MDD 전략의 Drawdown
        ax8 = fig.add_subplot(gs[3, :])
        best_mdd_result = self.results[best_mdd_period]
        cummax = best_mdd_result['cumulative'].cummax()
        drawdown = (best_mdd_result['cumulative'] - cummax) / cummax * 100

        ax8.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax8.plot(drawdown.index, drawdown, color='darkred', linewidth=2,
                label=f"Best MDD Strategy (SMA {best_mdd_period})")

        # MDD 포인트 표시
        mdd_value = drawdown.min()
        mdd_date = drawdown.idxmin()
        ax8.scatter([mdd_date], [mdd_value], color='red', s=200, zorder=5, marker='X')
        ax8.annotate(f'MDD: {mdd_value:.2f}%',
                    xy=(mdd_date, mdd_value),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        ax8.set_title('Drawdown: Best MDD Strategy', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Drawdown (%)', fontsize=12)
        ax8.set_xlabel('Date', fontsize=12)
        ax8.legend(loc='lower right', fontsize=11)
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 전체 제목
        fig.suptitle(f'{self.symbol} SMA Strategy Optimization Results\n'
                    f'Period: {self.start_date} to {self.end_date} | SMA Range: 5-120 days',
                    fontsize=16, fontweight='bold', y=0.995)

        # 저장
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

        return save_path

    def save_results(self, csv_path='btc_sma_optimization_metrics.csv'):
        """결과를 CSV로 저장"""
        if self.metrics_summary is None:
            print("Please run optimization first!")
            return

        self.metrics_summary.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")

        return csv_path

    def run_full_analysis(self, sma_range=(5, 120)):
        """전체 분석 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 최적화 실행
        metrics_df = self.run_optimization(sma_range)

        # 3. 최적 전략 출력
        best_strategies = self.find_best_strategies()

        # 4. 결과 저장
        self.save_results()

        # 5. 시각화
        self.plot_optimization_results()

        return metrics_df, best_strategies


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("BTC SMA Strategy Optimization Analysis")
    print("="*80)

    # 백테스트 실행
    optimizer = BTCSMAOptimization(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 전체 분석 실행
    metrics_df, best_strategies = optimizer.run_full_analysis(sma_range=(5, 120))

    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
