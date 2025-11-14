"""
SMA 크로스오버 전략 - 여러 윈도우 파라미터 분산투자 분석

단일 최적 파라미터 vs 여러 파라미터 조합에 분산투자하는 포트폴리오 비교

전략:
- 단일 최적: 최고 Sharpe Ratio 파라미터 1개
- Top 3 분산: 상위 3개 파라미터에 동일 비중 투자
- Top 5 분산: 상위 5개 파라미터에 동일 비중 투자
- Top 10 분산: 상위 10개 파라미터에 동일 비중 투자
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class SMADiversifiedPortfolio:
    """여러 SMA 파라미터에 분산투자하는 포트폴리오 분석 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            slippage: 슬리피지
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = {}
        self.optimization_results = {}
        self.portfolio_results = {}

    def load_data(self):
        """모든 종목 데이터 로드"""
        print("="*80)
        print("Loading data for all symbols...")
        print("="*80)

        for symbol in self.symbols:
            file_path = f'chart_day/{symbol}.parquet'
            print(f"\nLoading {symbol} from {file_path}...")
            df = pd.read_parquet(file_path)

            # 컬럼명 변경
            df.columns = [col.capitalize() for col in df.columns]

            # 날짜 필터링
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            self.data[symbol] = df
            print(f"  Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

        print("\n" + "="*80)
        print("Data loading completed!")
        print("="*80 + "\n")

    def load_optimization_results(self, result_dir='optimization_results'):
        """최적화 결과 CSV 파일 로드"""
        print("\n" + "="*80)
        print("Loading optimization results...")
        print("="*80 + "\n")

        for symbol in self.symbols:
            symbol_clean = symbol.split('_')[0]
            csv_path = f"{result_dir}/sma_optimization_full_{symbol_clean}.csv"
            df = pd.read_csv(csv_path)
            self.optimization_results[symbol] = df
            print(f"  Loaded {len(df)} parameter combinations for {symbol}")

        print("\n" + "="*80)
        print("Optimization results loaded!")
        print("="*80 + "\n")

    def strategy_sma_crossover(self, df, short_window, long_window):
        """SMA 크로스오버 전략 실행"""
        df = df.copy()

        # SMA 계산
        df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window).mean()

        # 포지션
        df['position'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)

        # 포지션 변화
        df['position_change'] = df['position'].diff()

        # 일일 수익률
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def create_diversified_portfolio(self, symbol, top_n=5, metric='sharpe'):
        """
        여러 파라미터 조합에 분산투자하는 포트폴리오 생성

        Args:
            symbol: 종목 심볼
            top_n: 상위 몇 개 파라미터 사용
            metric: 선정 기준 ('sharpe', 'cagr', 'total_return')

        Returns:
            포트폴리오 수익률 데이터프레임
        """
        opt_results = self.optimization_results[symbol]

        # 상위 N개 파라미터 선정
        top_params = opt_results.nlargest(top_n, metric)

        print(f"\n>>> Creating Top-{top_n} diversified portfolio for {symbol}...")
        print(f"    Selection metric: {metric.upper()}")
        print(f"    Selected parameters:")

        # 각 파라미터로 전략 실행
        df = self.data[symbol]
        all_returns = []

        for idx, row in top_params.iterrows():
            short_win = int(row['short_window'])
            long_win = int(row['long_window'])

            print(f"      - Short={short_win:2d}, Long={long_win:3d} | "
                  f"Sharpe={row['sharpe']:.2f}, CAGR={row['cagr']:.1f}%")

            # 전략 실행
            result_df = self.strategy_sma_crossover(df, short_win, long_win)
            all_returns.append(result_df['returns'])

        # 동일 비중으로 포트폴리오 구성
        returns_df = pd.concat(all_returns, axis=1)
        returns_df.columns = [f'Param_{i+1}' for i in range(top_n)]

        # 포트폴리오 수익률 = 평균
        portfolio_returns = returns_df.mean(axis=1)
        portfolio_cumulative = (1 + portfolio_returns).cumprod()

        result = pd.DataFrame({
            'returns': portfolio_returns,
            'cumulative': portfolio_cumulative
        }, index=df.index)

        return result

    def run_diversified_analysis(self, top_n_list=[1, 3, 5, 10], metric='sharpe'):
        """
        모든 종목에 대해 다양한 분산 수준의 포트폴리오 분석

        Args:
            top_n_list: 분산 수준 리스트 (예: [1, 3, 5, 10])
            metric: 파라미터 선정 기준
        """
        print("\n" + "="*80)
        print("Running diversified portfolio analysis...")
        print("="*80)

        for symbol in self.symbols:
            self.portfolio_results[symbol] = {}

            for top_n in top_n_list:
                portfolio_name = f'Top-{top_n}'
                result = self.create_diversified_portfolio(symbol, top_n, metric)
                self.portfolio_results[symbol][portfolio_name] = result

        print("\n" + "="*80)
        print("Diversified portfolio analysis completed!")
        print("="*80 + "\n")

    def calculate_metrics(self, returns_series):
        """성과 지표 계산"""
        cumulative = (1 + returns_series).cumprod()

        # 총 수익률
        total_return = (cumulative.iloc[-1] - 1) * 100

        # CAGR
        years = (returns_series.index[-1] - returns_series.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # MDD
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100

        # Sharpe
        sharpe = (returns_series.mean() / returns_series.std() * np.sqrt(365)) if returns_series.std() > 0 else 0

        # 승률
        total_trades = (returns_series != 0).sum()
        winning_trades = (returns_series > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': int(total_trades)
        }

    def plot_comparison(self, symbol, save_dir='optimization_results'):
        """
        단일 최적 vs 분산 포트폴리오 비교 시각화

        Args:
            symbol: 종목 심볼
            save_dir: 저장 디렉토리
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        symbol_clean = symbol.split('_')[0]
        portfolios = self.portfolio_results[symbol]

        # 시각화
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률 비교 (Log Scale)
        ax1 = fig.add_subplot(gs[0, :])
        colors = ['red', 'orange', 'green', 'blue']
        for i, (portfolio_name, result) in enumerate(portfolios.items()):
            ax1.plot(result.index, result['cumulative'],
                    linewidth=2.5, alpha=0.8, label=portfolio_name,
                    color=colors[i] if i < len(colors) else None)

        ax1.set_yscale('log')
        ax1.set_title(f'{symbol_clean} - Single vs Diversified Parameter Portfolios (Log Scale)',
                     fontsize=15, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (Log Scale)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown 비교
        ax2 = fig.add_subplot(gs[1, :])
        for i, (portfolio_name, result) in enumerate(portfolios.items()):
            cummax = result['cumulative'].cummax()
            drawdown = (result['cumulative'] - cummax) / cummax * 100
            ax2.plot(drawdown.index, drawdown, linewidth=2, alpha=0.7,
                    label=portfolio_name, color=colors[i] if i < len(colors) else None)

        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.1)
        ax2.set_title('Drawdown Comparison (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='lower right', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 성과 지표 계산
        metrics_list = []
        for portfolio_name, result in portfolios.items():
            metrics = self.calculate_metrics(result['returns'])
            metrics['Portfolio'] = portfolio_name
            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list)

        # 3. Total Return 비교
        ax3 = fig.add_subplot(gs[2, 0])
        sorted_df = metrics_df.sort_values('total_return', ascending=True)
        colors_bar = ['green' if x > 0 else 'red' for x in sorted_df['total_return']]
        ax3.barh(sorted_df['Portfolio'], sorted_df['total_return'], color=colors_bar, alpha=0.7)
        ax3.set_xlabel('Total Return (%)', fontsize=11)
        ax3.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. CAGR 비교
        ax4 = fig.add_subplot(gs[2, 1])
        sorted_df = metrics_df.sort_values('cagr', ascending=True)
        colors_bar = ['green' if x > 0 else 'red' for x in sorted_df['cagr']]
        ax4.barh(sorted_df['Portfolio'], sorted_df['cagr'], color=colors_bar, alpha=0.7)
        ax4.set_xlabel('CAGR (%)', fontsize=11)
        ax4.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. MDD 비교
        ax5 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.sort_values('mdd', ascending=False)
        ax5.barh(sorted_df['Portfolio'], sorted_df['mdd'], color='crimson', alpha=0.7)
        ax5.set_xlabel('MDD (%)', fontsize=11)
        ax5.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Sharpe Ratio 비교
        ax6 = fig.add_subplot(gs[3, 0])
        sorted_df = metrics_df.sort_values('sharpe', ascending=True)
        colors_bar = ['green' if x > 0 else 'red' for x in sorted_df['sharpe']]
        ax6.barh(sorted_df['Portfolio'], sorted_df['sharpe'], color=colors_bar, alpha=0.7)
        ax6.set_xlabel('Sharpe Ratio', fontsize=11)
        ax6.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        # 7. Risk-Return 산점도
        ax7 = fig.add_subplot(gs[3, 1])
        ax7.scatter(metrics_df['mdd'], metrics_df['cagr'],
                   s=300, alpha=0.6, c=metrics_df['sharpe'], cmap='RdYlGn')
        for idx, row in metrics_df.iterrows():
            ax7.annotate(row['Portfolio'],
                        (row['mdd'], row['cagr']),
                        fontsize=10, ha='center', va='bottom')
        ax7.set_xlabel('MDD (%)', fontsize=11)
        ax7.set_ylabel('CAGR (%)', fontsize=11)
        ax7.set_title('Risk-Return Scatter (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 8. 성과 지표 테이블
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')

        table_text = f"Performance Metrics\n{'='*40}\n\n"
        for idx, row in metrics_df.iterrows():
            table_text += f"{row['Portfolio']}\n"
            table_text += f"  Total Return: {row['total_return']:.2f}%\n"
            table_text += f"  CAGR: {row['cagr']:.2f}%\n"
            table_text += f"  MDD: {row['mdd']:.2f}%\n"
            table_text += f"  Sharpe: {row['sharpe']:.2f}\n"
            table_text += f"  Win Rate: {row['win_rate']:.2f}%\n\n"

        ax8.text(0.1, 0.95, table_text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 전체 제목
        fig.suptitle(f'{symbol_clean} - Diversified Parameter Portfolio Analysis\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=16, fontweight='bold', y=0.995)

        # 저장
        save_path = f"{save_dir}/sma_diversified_{symbol_clean}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Chart saved: {save_path}")
        plt.close()

        return save_path, metrics_df

    def plot_all_comparisons(self, save_dir='optimization_results'):
        """모든 종목의 비교 차트 생성"""
        print("\n" + "="*80)
        print("Creating diversified portfolio comparison charts...")
        print("="*80 + "\n")

        all_metrics = {}
        saved_files = []

        for symbol in self.symbols:
            print(f"Creating chart for {symbol}...")
            save_path, metrics_df = self.plot_comparison(symbol, save_dir)
            saved_files.append(save_path)
            all_metrics[symbol] = metrics_df

        print("\n" + "="*80)
        print(f"Comparison charts completed! {len(saved_files)} charts saved.")
        print(f"Location: {save_dir}/")
        print("="*80 + "\n")

        return saved_files, all_metrics

    def save_summary(self, all_metrics, save_dir='optimization_results'):
        """요약 결과 저장"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 모든 티커의 결과를 하나의 CSV로 저장
        summary_data = []
        for symbol, metrics_df in all_metrics.items():
            symbol_clean = symbol.split('_')[0]
            for idx, row in metrics_df.iterrows():
                summary_data.append({
                    'Symbol': symbol_clean,
                    'Portfolio': row['Portfolio'],
                    'Total Return (%)': row['total_return'],
                    'CAGR (%)': row['cagr'],
                    'MDD (%)': row['mdd'],
                    'Sharpe Ratio': row['sharpe'],
                    'Win Rate (%)': row['win_rate'],
                    'Total Trades': row['total_trades']
                })

        summary_df = pd.DataFrame(summary_data)
        csv_path = f"{save_dir}/sma_diversified_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSummary saved: {csv_path}")

        return summary_df

    def print_summary(self, all_metrics):
        """요약 결과 출력"""
        print("\n" + "="*140)
        print(f"{'DIVERSIFIED PARAMETER PORTFOLIO SUMMARY':^140}")
        print("="*140)
        print(f"\nPeriod: {self.start_date} ~ {self.end_date}")
        print(f"Symbols: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"Slippage: {self.slippage*100}%")

        for symbol, metrics_df in all_metrics.items():
            symbol_clean = symbol.split('_')[0]
            print("\n" + "-"*140)
            print(f"{symbol_clean} - Performance by Diversification Level:")
            print("-"*140)
            print(f"{'Portfolio':<12} {'Total Return':<15} {'CAGR':<10} {'MDD':<10} {'Sharpe':<10} {'Win Rate':<12} {'Trades':<10}")
            print("-"*140)

            for idx, row in metrics_df.iterrows():
                print(f"{row['Portfolio']:<12} "
                      f"{row['total_return']:>12.2f}%  "
                      f"{row['cagr']:>8.2f}% "
                      f"{row['mdd']:>8.2f}% "
                      f"{row['sharpe']:>8.2f}  "
                      f"{row['win_rate']:>10.2f}% "
                      f"{row['total_trades']:>8}")

        print("="*140 + "\n")


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("SMA DIVERSIFIED PARAMETER PORTFOLIO ANALYSIS")
    print("="*80)

    # 분석 실행
    analyzer = SMADiversifiedPortfolio(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )

    # 1. 데이터 로드
    analyzer.load_data()

    # 2. 최적화 결과 로드
    analyzer.load_optimization_results()

    # 3. 분산 포트폴리오 분석 (Top-1, Top-3, Top-5, Top-10)
    analyzer.run_diversified_analysis(top_n_list=[1, 3, 5, 10], metric='sharpe')

    # 4. 시각화
    saved_files, all_metrics = analyzer.plot_all_comparisons()

    # 5. 요약 출력
    analyzer.print_summary(all_metrics)

    # 6. 결과 저장
    summary_df = analyzer.save_summary(all_metrics)

    print("\n" + "="*80)
    print("DIVERSIFIED PORTFOLIO ANALYSIS COMPLETED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
