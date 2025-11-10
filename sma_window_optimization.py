#!/usr/bin/env python3
"""
SMA Window Optimization Backtest

다양한 SMA 윈도우를 테스트하여 최적 조합을 찾고,
포트폴리오 성과와 비교합니다.

테스트할 SMA 윈도우: 5, 10, 20, 30, 50, 100, 200일
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class SMAWindowOptimization:
    """SMA 윈도우 최적화 백테스트"""

    def __init__(self, symbols, start_date, end_date, sma_windows, slippage=0.002):
        """
        Parameters:
        -----------
        symbols : list
            백테스트할 심볼 리스트
        start_date : str
            시작일 (YYYY-MM-DD)
        end_date : str
            종료일 (YYYY-MM-DD)
        sma_windows : list
            테스트할 SMA 윈도우 기간 리스트 (예: [5, 10, 20, 30, 50, 100, 200])
        slippage : float
            슬리피지 (기본값: 0.2%)
        """
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.sma_windows = sorted(sma_windows)
        self.slippage = slippage

        self.data = {}
        self.strategy_results = {}  # {window: {symbol: DataFrame}}
        self.portfolio_results = {}  # {window: DataFrame}
        self.metrics = {}  # {window: {metric: value}}

    def load_data(self):
        """데이터 로드"""
        print("=" * 80)
        print("데이터 로딩 중...")
        print("=" * 80)

        data_dir = Path('chart_day')

        for symbol in self.symbols:
            file_path = data_dir / f"{symbol}.parquet"

            if not file_path.exists():
                print(f"경고: {symbol} 데이터 파일을 찾을 수 없습니다: {file_path}")
                continue

            df = pd.read_parquet(file_path)
            df.index = pd.to_datetime(df.index)

            # 날짜 필터링
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            # 컬럼명 표준화
            df.columns = [col.capitalize() for col in df.columns]

            self.data[symbol] = df
            print(f"✓ {symbol}: {len(df):,} 일 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")

        print(f"\n총 {len(self.data)}개 심볼 로드 완료\n")

    def strategy_sma(self, df, sma_period):
        """
        SMA 크로스오버 전략

        - Price >= SMA: Buy/Hold
        - Price < SMA: Sell/Cash

        Parameters:
        -----------
        df : DataFrame
            가격 데이터
        sma_period : int
            SMA 기간

        Returns:
        --------
        DataFrame
            전략 결과 (returns, cumulative, position 등)
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 포지션: 1 if price >= SMA, 0 otherwise
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)

        # 포지션 변경 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 (전날 시그널이 오늘 적용됨 - look-ahead bias 방지)
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage   # Buy
        slippage_cost[df['position_change'] == -1] = -self.slippage  # Sell

        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def run_all_sma_windows(self):
        """모든 SMA 윈도우에 대해 백테스트 실행"""
        print("=" * 80)
        print("SMA 윈도우 최적화 백테스트 실행 중...")
        print("=" * 80)

        total_combinations = len(self.sma_windows) * len(self.symbols)
        current = 0

        for window in self.sma_windows:
            self.strategy_results[window] = {}

            for symbol in self.symbols:
                current += 1
                print(f"[{current}/{total_combinations}] SMA {window} - {symbol} 처리 중...")

                df = self.data[symbol].copy()
                result = self.strategy_sma(df, sma_period=window)
                self.strategy_results[window][symbol] = result

        print(f"\n총 {total_combinations}개 조합 백테스트 완료\n")

    def create_portfolios(self):
        """각 SMA 윈도우별 포트폴리오 생성 (equal-weight)"""
        print("=" * 80)
        print("포트폴리오 생성 중...")
        print("=" * 80)

        weight = 1.0 / len(self.symbols)

        for window in self.sma_windows:
            # 모든 심볼의 공통 날짜 찾기
            all_indices = [self.strategy_results[window][symbol].index
                          for symbol in self.symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            # Equal-weight 포트폴리오 수익률 계산
            portfolio_returns = pd.Series(0.0, index=common_index)

            for symbol in self.symbols:
                symbol_returns = self.strategy_results[window][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight

            # 포트폴리오 결과 저장
            portfolio_df = pd.DataFrame({
                'returns': portfolio_returns,
                'cumulative': (1 + portfolio_returns).cumprod()
            })

            self.portfolio_results[window] = portfolio_df

            print(f"✓ SMA {window:3d}: {len(portfolio_df):,} 일")

        print(f"\n총 {len(self.portfolio_results)}개 포트폴리오 생성 완료\n")

    def calculate_metrics(self, returns_series):
        """성과 지표 계산"""
        # NaN 제거
        returns = returns_series.dropna()

        if len(returns) == 0:
            return {
                'Total Return (%)': 0,
                'CAGR (%)': 0,
                'MDD (%)': 0,
                'Sharpe Ratio': 0,
                'Total Trades': 0
            }

        # 누적 수익률
        cumulative = (1 + returns).cumprod()
        total_return = (cumulative.iloc[-1] - 1) * 100

        # CAGR (연평균 성장률)
        years = len(returns) / 252
        cagr = ((cumulative.iloc[-1] ** (1 / years)) - 1) * 100 if years > 0 else 0

        # MDD (Maximum Drawdown)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        mdd = drawdown.min()

        # Sharpe Ratio
        excess_returns = returns
        sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0

        # 총 거래 횟수 (매수 횟수)
        total_trades = (returns != 0).sum()

        return {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Total Trades': total_trades
        }

    def calculate_all_metrics(self):
        """모든 포트폴리오의 성과 지표 계산"""
        print("=" * 80)
        print("성과 지표 계산 중...")
        print("=" * 80)

        for window in self.sma_windows:
            portfolio_returns = self.portfolio_results[window]['returns']
            self.metrics[window] = self.calculate_metrics(portfolio_returns)

            print(f"SMA {window:3d}: "
                  f"Return={self.metrics[window]['Total Return (%)']:>8,.1f}%, "
                  f"CAGR={self.metrics[window]['CAGR (%)']:>6,.1f}%, "
                  f"Sharpe={self.metrics[window]['Sharpe Ratio']:>5.2f}")

        print()

    def print_metrics_table(self):
        """성과 지표 테이블 출력"""
        print("=" * 80)
        print("SMA 윈도우별 포트폴리오 성과 비교")
        print("=" * 80)

        # 데이터프레임 생성
        metrics_data = []
        for window in self.sma_windows:
            metrics_data.append({
                'SMA Window': f"SMA {window}",
                'Total Return (%)': self.metrics[window]['Total Return (%)'],
                'CAGR (%)': self.metrics[window]['CAGR (%)'],
                'MDD (%)': self.metrics[window]['MDD (%)'],
                'Sharpe Ratio': self.metrics[window]['Sharpe Ratio'],
                'Total Trades': self.metrics[window]['Total Trades']
            })

        df = pd.DataFrame(metrics_data)

        # 포맷팅하여 출력
        print("\n" + df.to_string(index=False))
        print()

        # 최적 조합 찾기
        best_return_idx = df['Total Return (%)'].idxmax()
        best_sharpe_idx = df['Sharpe Ratio'].idxmax()
        best_cagr_idx = df['CAGR (%)'].idxmax()

        print("=" * 80)
        print("최적 조합")
        print("=" * 80)
        print(f"최고 수익률: {df.loc[best_return_idx, 'SMA Window']} "
              f"({df.loc[best_return_idx, 'Total Return (%)']:,.1f}%)")
        print(f"최고 샤프비율: {df.loc[best_sharpe_idx, 'SMA Window']} "
              f"({df.loc[best_sharpe_idx, 'Sharpe Ratio']:.2f})")
        print(f"최고 CAGR: {df.loc[best_cagr_idx, 'SMA Window']} "
              f"({df.loc[best_cagr_idx, 'CAGR (%)']:.1f}%)")
        print()

        # CSV 저장
        csv_path = 'sma_window_optimization_metrics.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"성과 지표를 {csv_path}에 저장했습니다.\n")

        return df

    def plot_optimization_results(self):
        """SMA 윈도우 최적화 결과 시각화"""
        print("=" * 80)
        print("시각화 생성 중...")
        print("=" * 80)

        fig = plt.figure(figsize=(20, 14))

        # 색상 팔레트
        colors = sns.color_palette("husl", len(self.sma_windows))

        # 1. 포트폴리오 누적 수익률 비교
        ax1 = plt.subplot(3, 3, 1)
        for idx, window in enumerate(self.sma_windows):
            cumulative = self.portfolio_results[window]['cumulative']
            ax1.plot(cumulative.index, cumulative.values,
                    label=f'SMA {window}', linewidth=2, color=colors[idx])
        ax1.set_yscale('log')
        ax1.set_title('Portfolio Cumulative Returns (Log Scale)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. 총 수익률 비교
        ax2 = plt.subplot(3, 3, 2)
        total_returns = [self.metrics[w]['Total Return (%)'] for w in self.sma_windows]
        bars = ax2.bar(range(len(self.sma_windows)), total_returns, color=colors)
        ax2.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Return (%)')
        ax2.set_xticks(range(len(self.sma_windows)))
        ax2.set_xticklabels([f'SMA {w}' for w in self.sma_windows], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}%', ha='center', va='bottom', fontsize=8)

        # 3. CAGR 비교
        ax3 = plt.subplot(3, 3, 3)
        cagrs = [self.metrics[w]['CAGR (%)'] for w in self.sma_windows]
        bars = ax3.bar(range(len(self.sma_windows)), cagrs, color=colors)
        ax3.set_title('CAGR Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('CAGR (%)')
        ax3.set_xticks(range(len(self.sma_windows)))
        ax3.set_xticklabels([f'SMA {w}' for w in self.sma_windows], rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        # 4. 샤프 비율 비교
        ax4 = plt.subplot(3, 3, 4)
        sharpes = [self.metrics[w]['Sharpe Ratio'] for w in self.sma_windows]
        bars = ax4.bar(range(len(self.sma_windows)), sharpes, color=colors)
        ax4.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_xticks(range(len(self.sma_windows)))
        ax4.set_xticklabels([f'SMA {w}' for w in self.sma_windows], rotation=45)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 5. MDD 비교
        ax5 = plt.subplot(3, 3, 5)
        mdds = [self.metrics[w]['MDD (%)'] for w in self.sma_windows]
        bars = ax5.bar(range(len(self.sma_windows)), mdds, color=colors)
        ax5.set_title('Maximum Drawdown Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('MDD (%)')
        ax5.set_xticks(range(len(self.sma_windows)))
        ax5.set_xticklabels([f'SMA {w}' for w in self.sma_windows], rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        # 6. Return vs Risk (Sharpe로 색상 구분)
        ax6 = plt.subplot(3, 3, 6)
        returns = [self.metrics[w]['CAGR (%)'] for w in self.sma_windows]
        risks = [abs(self.metrics[w]['MDD (%)']) for w in self.sma_windows]
        scatter = ax6.scatter(risks, returns, c=sharpes, cmap='RdYlGn',
                             s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        for i, window in enumerate(self.sma_windows):
            ax6.annotate(f'SMA {window}', (risks[i], returns[i]),
                        fontsize=9, ha='center', va='center')
        ax6.set_title('Return vs Risk (colored by Sharpe)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Risk (|MDD| %)')
        ax6.set_ylabel('Return (CAGR %)')
        plt.colorbar(scatter, ax=ax6, label='Sharpe Ratio')
        ax6.grid(True, alpha=0.3)

        # 7. 거래 횟수 비교
        ax7 = plt.subplot(3, 3, 7)
        trades = [self.metrics[w]['Total Trades'] for w in self.sma_windows]
        bars = ax7.bar(range(len(self.sma_windows)), trades, color=colors)
        ax7.set_title('Total Trades Comparison', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Total Trades')
        ax7.set_xticks(range(len(self.sma_windows)))
        ax7.set_xticklabels([f'SMA {w}' for w in self.sma_windows], rotation=45)
        ax7.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=8)

        # 8. Drawdown 비교 (시계열)
        ax8 = plt.subplot(3, 3, 8)
        for idx, window in enumerate(self.sma_windows):
            cumulative = self.portfolio_results[window]['cumulative']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            ax8.plot(drawdown.index, drawdown.values,
                    label=f'SMA {window}', linewidth=1.5, alpha=0.7, color=colors[idx])
        ax8.set_title('Portfolio Drawdown Over Time', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Drawdown (%)')
        ax8.legend(loc='best', fontsize=8)
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 9. 성과 요약 테이블
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        # 최고 성과 찾기
        best_return_window = self.sma_windows[np.argmax(total_returns)]
        best_sharpe_window = self.sma_windows[np.argmax(sharpes)]
        best_cagr_window = self.sma_windows[np.argmax(cagrs)]
        best_mdd_window = self.sma_windows[np.argmax(mdds)]  # MDD는 음수이므로 max가 최소 손실

        summary_text = f"""
OPTIMIZATION SUMMARY
{'=' * 40}

Best Total Return:
  SMA {best_return_window}: {max(total_returns):,.1f}%

Best Sharpe Ratio:
  SMA {best_sharpe_window}: {max(sharpes):.2f}

Best CAGR:
  SMA {best_cagr_window}: {max(cagrs):.1f}%

Best MDD (Lowest Drawdown):
  SMA {best_mdd_window}: {max(mdds):.1f}%

Portfolio: Equal-weight
Assets: {', '.join(self.symbols)}
Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
Slippage: {self.slippage * 100:.2f}%
"""

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('SMA Window Optimization Results',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # 저장
        output_path = 'sma_window_optimization_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 시각화를 {output_path}에 저장했습니다.")
        plt.close()

    def plot_individual_window_comparison(self):
        """각 SMA 윈도우별 개별 자산 성과 비교"""
        print("\n개별 자산별 성과 비교 차트 생성 중...")

        # 각 윈도우별로 개별 자산 성과 차트 생성
        for window in self.sma_windows:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            for idx, symbol in enumerate(self.symbols):
                ax = axes[idx]

                result = self.strategy_results[window][symbol]

                # 성과 지표 계산
                metrics = self.calculate_metrics(result['returns'])

                # 누적 수익률 플롯
                ax.plot(result.index, result['cumulative'],
                       linewidth=2, label='Strategy')

                ax.set_title(f'{symbol} - SMA {window}',
                           fontsize=12, fontweight='bold')
                ax.set_ylabel('Cumulative Return')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)

                # 성과 지표 텍스트 박스
                metrics_text = f"""Return: {metrics['Total Return (%)']:,.1f}%
CAGR: {metrics['CAGR (%)']:.1f}%
MDD: {metrics['MDD (%)']:.1f}%
Sharpe: {metrics['Sharpe Ratio']:.2f}
Trades: {int(metrics['Total Trades']):,}"""

                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                       fontfamily='monospace')

            plt.suptitle(f'Individual Asset Performance - SMA {window}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            # 저장
            output_path = f'individual_results/sma_{window}_individual_analysis.png'
            Path('individual_results').mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"  ✓ SMA {window} 개별 분석 저장: {output_path}")
            plt.close()

    def save_portfolio_results(self):
        """포트폴리오 결과를 CSV로 저장"""
        print("\n포트폴리오 결과 저장 중...")

        for window in self.sma_windows:
            csv_path = f'portfolio_sma_{window}.csv'
            self.portfolio_results[window].to_csv(csv_path, encoding='utf-8-sig')
            print(f"  ✓ SMA {window} 포트폴리오: {csv_path}")

    def run_analysis(self):
        """전체 분석 실행"""
        print("\n" + "=" * 80)
        print("SMA 윈도우 최적화 백테스트 시작")
        print("=" * 80)
        print(f"심볼: {', '.join(self.symbols)}")
        print(f"기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        print(f"SMA 윈도우: {self.sma_windows}")
        print(f"슬리피지: {self.slippage * 100:.2f}%")
        print()

        # 1. 데이터 로드
        self.load_data()

        # 2. 모든 SMA 윈도우에 대해 백테스트 실행
        self.run_all_sma_windows()

        # 3. 포트폴리오 생성
        self.create_portfolios()

        # 4. 성과 지표 계산
        self.calculate_all_metrics()

        # 5. 결과 출력
        self.print_metrics_table()

        # 6. 시각화
        self.plot_optimization_results()

        # 7. 개별 윈도우별 자산 성과 비교
        self.plot_individual_window_comparison()

        # 8. 포트폴리오 결과 저장
        self.save_portfolio_results()

        print("=" * 80)
        print("분석 완료!")
        print("=" * 80)


def main():
    """메인 함수"""
    # 설정
    symbols = ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW']
    start_date = '2018-01-01'
    end_date = '2025-11-07'
    sma_windows = [5, 10, 20, 30, 50, 100, 200]
    slippage = 0.002  # 0.2%

    # 분석 실행
    optimizer = SMAWindowOptimization(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        sma_windows=sma_windows,
        slippage=slippage
    )

    optimizer.run_analysis()


if __name__ == '__main__':
    main()
