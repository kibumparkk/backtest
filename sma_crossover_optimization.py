"""
SMA 크로스오버 전략 파라미터 최적화

BTC, ETH, ADA, XRP에 대해 SMA short/long 파라미터를 최적화하고
최적 파라미터에서의 성과를 시각화합니다.

전략:
- SMA short > SMA long: 매수/보유
- SMA short < SMA long: 매도/현금 보유

파라미터 범위:
- short window: 1부터 시작
- long window: 3, 5, 10 포함하여 120까지
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from itertools import product
import json

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class SMACrossoverOptimizer:
    """SMA 크로스오버 전략 파라미터 최적화 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = {}
        self.optimization_results = {}
        self.best_params = {}

    def load_data(self):
        """모든 종목 데이터 로드"""
        print("="*80)
        print("Loading data for all symbols...")
        print("="*80)

        for symbol in self.symbols:
            file_path = f'chart_day/{symbol}.parquet'
            print(f"\nLoading {symbol} from {file_path}...")
            df = pd.read_parquet(file_path)

            # 컬럼명 변경 (소문자 -> 대문자)
            df.columns = [col.capitalize() for col in df.columns]

            # 날짜 필터링
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            self.data[symbol] = df
            print(f"  Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

        print("\n" + "="*80)
        print("Data loading completed!")
        print("="*80 + "\n")

    def strategy_sma_crossover(self, df, short_window, long_window):
        """
        SMA 크로스오버 전략

        Args:
            df: 가격 데이터프레임
            short_window: 단기 SMA 윈도우
            long_window: 장기 SMA 윈도우

        Returns:
            결과 데이터프레임 (수익률, 누적 수익률 포함)
        """
        df = df.copy()

        # SMA 계산
        df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window).mean()

        # 포지션: SMA_short > SMA_long이면 매수(1), 아니면 매도(0)
        df['position'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)

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

    def calculate_metrics(self, df):
        """성과 지표 계산"""
        returns_series = df['returns']
        cumulative = df['cumulative']

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

        return {
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': int(total_trades),
            'profit_factor': profit_factor
        }

    def optimize_parameters(self, symbol, short_range=None, long_range=None, metric='sharpe'):
        """
        파라미터 그리드 서치 최적화

        Args:
            symbol: 종목 심볼
            short_range: 단기 SMA 윈도우 범위 (list)
            long_range: 장기 SMA 윈도우 범위 (list)
            metric: 최적화 기준 지표 ('sharpe', 'cagr', 'total_return')

        Returns:
            최적화 결과 데이터프레임
        """
        if short_range is None:
            short_range = list(range(1, 51))  # 1~50
        if long_range is None:
            # 3, 5, 10 포함하여 120까지
            long_range = [3, 5, 10] + list(range(15, 121, 5))

        print(f"\n{'='*80}")
        print(f"Optimizing {symbol}...")
        print(f"Short window range: {min(short_range)} ~ {max(short_range)} ({len(short_range)} values)")
        print(f"Long window range: {min(long_range)} ~ {max(long_range)} ({len(long_range)} values)")
        print(f"Total combinations: {len(short_range) * len(long_range)}")
        print(f"Optimization metric: {metric.upper()}")
        print(f"{'='*80}")

        df = self.data[symbol]
        results = []

        total_combinations = len(short_range) * len(long_range)
        valid_combinations = 0

        # 그리드 서치
        for i, (short_win, long_win) in enumerate(product(short_range, long_range)):
            # short window는 long window보다 작아야 함
            if short_win >= long_win:
                continue

            valid_combinations += 1

            # 진행 상황 출력
            if valid_combinations % 100 == 0:
                print(f"  Progress: {valid_combinations} combinations tested...", end='\r')

            try:
                # 전략 실행
                result_df = self.strategy_sma_crossover(df, short_win, long_win)

                # 성과 지표 계산
                metrics = self.calculate_metrics(result_df)

                # 결과 저장
                results.append({
                    'short_window': short_win,
                    'long_window': long_win,
                    **metrics
                })
            except Exception as e:
                print(f"\nError with short={short_win}, long={long_win}: {e}")
                continue

        print(f"\n  Completed: {valid_combinations} valid combinations tested")

        # 결과를 데이터프레임으로 변환
        results_df = pd.DataFrame(results)

        # 최적 파라미터 찾기
        if metric == 'sharpe':
            best_idx = results_df['sharpe'].idxmax()
        elif metric == 'cagr':
            best_idx = results_df['cagr'].idxmax()
        elif metric == 'total_return':
            best_idx = results_df['total_return'].idxmax()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        best_params = results_df.loc[best_idx]

        print(f"\n{'='*80}")
        print(f"Best parameters for {symbol}:")
        print(f"  Short Window: {int(best_params['short_window'])}")
        print(f"  Long Window: {int(best_params['long_window'])}")
        print(f"  Total Return: {best_params['total_return']:.2f}%")
        print(f"  CAGR: {best_params['cagr']:.2f}%")
        print(f"  MDD: {best_params['mdd']:.2f}%")
        print(f"  Sharpe Ratio: {best_params['sharpe']:.2f}")
        print(f"  Win Rate: {best_params['win_rate']:.2f}%")
        print(f"  Total Trades: {int(best_params['total_trades'])}")
        print(f"{'='*80}\n")

        # 결과 저장
        self.optimization_results[symbol] = results_df
        self.best_params[symbol] = {
            'short_window': int(best_params['short_window']),
            'long_window': int(best_params['long_window']),
            'metrics': {
                'total_return': float(best_params['total_return']),
                'cagr': float(best_params['cagr']),
                'mdd': float(best_params['mdd']),
                'sharpe': float(best_params['sharpe']),
                'win_rate': float(best_params['win_rate']),
                'total_trades': int(best_params['total_trades']),
                'profit_factor': float(best_params['profit_factor']) if best_params['profit_factor'] != np.inf else 'inf'
            }
        }

        return results_df, best_params

    def run_optimization_all_symbols(self, short_range=None, long_range=None, metric='sharpe'):
        """모든 종목에 대해 파라미터 최적화 실행"""
        print("\n" + "="*80)
        print("Starting parameter optimization for all symbols...")
        print("="*80)

        for symbol in self.symbols:
            self.optimize_parameters(symbol, short_range, long_range, metric)

        print("\n" + "="*80)
        print("Optimization completed for all symbols!")
        print("="*80 + "\n")

    def plot_optimization_results(self, symbol, save_dir='optimization_results'):
        """
        특정 종목의 최적화 결과 시각화

        Args:
            symbol: 종목 심볼
            save_dir: 저장 디렉토리
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        results_df = self.optimization_results[symbol]
        best_params = self.best_params[symbol]

        symbol_clean = symbol.split('_')[0]

        # 최적 파라미터로 전략 실행
        df = self.data[symbol]
        result_df = self.strategy_sma_crossover(
            df,
            best_params['short_window'],
            best_params['long_window']
        )

        # 시각화
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 히트맵: Sharpe Ratio
        ax1 = fig.add_subplot(gs[0, 0])
        pivot_sharpe = results_df.pivot(index='long_window', columns='short_window', values='sharpe')
        sns.heatmap(pivot_sharpe, cmap='RdYlGn', center=0, ax=ax1, cbar_kws={'label': 'Sharpe Ratio'})
        ax1.set_title('Sharpe Ratio Heatmap', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Short Window', fontsize=11)
        ax1.set_ylabel('Long Window', fontsize=11)

        # 최적 파라미터 표시
        short_idx = pivot_sharpe.columns.get_loc(best_params['short_window'])
        long_idx = pivot_sharpe.index.get_loc(best_params['long_window'])
        ax1.plot(short_idx + 0.5, long_idx + 0.5, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)

        # 2. 히트맵: CAGR
        ax2 = fig.add_subplot(gs[0, 1])
        pivot_cagr = results_df.pivot(index='long_window', columns='short_window', values='cagr')
        sns.heatmap(pivot_cagr, cmap='RdYlGn', center=0, ax=ax2, cbar_kws={'label': 'CAGR (%)'})
        ax2.set_title('CAGR Heatmap (%)', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Short Window', fontsize=11)
        ax2.set_ylabel('Long Window', fontsize=11)

        # 최적 파라미터 표시
        ax2.plot(short_idx + 0.5, long_idx + 0.5, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)

        # 3. 히트맵: MDD
        ax3 = fig.add_subplot(gs[0, 2])
        pivot_mdd = results_df.pivot(index='long_window', columns='short_window', values='mdd')
        sns.heatmap(pivot_mdd, cmap='RdYlGn_r', center=-50, ax=ax3, cbar_kws={'label': 'MDD (%)'})
        ax3.set_title('Maximum Drawdown Heatmap (%)', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Short Window', fontsize=11)
        ax3.set_ylabel('Long Window', fontsize=11)

        # 최적 파라미터 표시
        ax3.plot(short_idx + 0.5, long_idx + 0.5, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)

        # 4. 누적 수익 곡선 (log scale)
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(result_df.index, result_df['cumulative'],
                linewidth=2.5, color='blue', label='Strategy Returns')
        ax4.set_yscale('log')
        ax4.set_title(f'Cumulative Returns (Log Scale) - Best Params: Short={best_params["short_window"]}, Long={best_params["long_window"]}',
                     fontsize=14, fontweight='bold')
        ax4.set_ylabel('Cumulative Return (Log Scale)', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.legend(loc='upper left', fontsize=11)
        ax4.grid(True, alpha=0.3)

        # Buy & Hold 비교
        buy_hold_cumulative = (1 + df['Close'].pct_change().fillna(0)).cumprod()
        ax4.plot(df.index, buy_hold_cumulative,
                linewidth=2, color='gray', alpha=0.5, linestyle='--', label='Buy & Hold')
        ax4.legend(loc='upper left', fontsize=11)

        # 5. Drawdown (-%)
        ax5 = fig.add_subplot(gs[2, :])
        cummax = result_df['cumulative'].cummax()
        drawdown = (result_df['cumulative'] - cummax) / cummax * 100
        ax5.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax5.plot(drawdown.index, drawdown, color='darkred', linewidth=2)
        ax5.set_title('Drawdown Over Time (%)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Drawdown (%)', fontsize=12)
        ax5.set_xlabel('Date', fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # MDD 표시
        mdd_value = drawdown.min()
        mdd_date = drawdown.idxmin()
        ax5.scatter([mdd_date], [mdd_value], color='red', s=200, zorder=5, marker='X')
        ax5.annotate(f'MDD: {mdd_value:.2f}%',
                    xy=(mdd_date, mdd_value),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        # 6. 월별 수익률 히트맵
        ax6 = fig.add_subplot(gs[3, :2])
        monthly_returns = result_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

        # 월별 수익률을 연도x월 형태로 피벗
        monthly_returns_df = monthly_returns.to_frame('returns')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        heatmap_data = monthly_returns_df.pivot(index='year', columns='month', values='returns')

        # 히트맵 그리기
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax6, linewidths=0.5,
                   vmin=-30, vmax=30)
        ax6.set_title('Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold')
        ax6.set_xlabel('Month', fontsize=11)
        ax6.set_ylabel('Year', fontsize=11)

        # 7. 월별 수익률 분포 (박스플롯 + 바 차트)
        ax7 = fig.add_subplot(gs[3, 2])

        # 각 월별 평균 수익률 계산
        monthly_avg = monthly_returns_df.groupby('month')['returns'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        colors = ['green' if x > 0 else 'red' for x in monthly_avg.values]
        bars = ax7.bar(range(1, 13), monthly_avg.values, color=colors, alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(1, 13))
        ax7.set_xticklabels(months, rotation=45, ha='right', fontsize=9)
        ax7.set_title('Average Monthly Returns by Month', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Average Return (%)', fontsize=10)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax7.grid(True, alpha=0.3, axis='y')

        # 값 표시
        for i, (bar, val) in enumerate(zip(bars, monthly_avg.values)):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')

        # 전체 제목
        fig.suptitle(f'{symbol_clean} - SMA Crossover Optimization Results\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=16, fontweight='bold', y=0.995)

        # 저장
        save_path = f"{save_dir}/sma_optimization_{symbol_clean}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Optimization chart saved: {save_path}")
        plt.close()

        return save_path

    def plot_all_optimization_results(self, save_dir='optimization_results'):
        """모든 종목의 최적화 결과 시각화"""
        print("\n" + "="*80)
        print("Creating optimization result charts...")
        print("="*80 + "\n")

        saved_files = []
        for symbol in self.symbols:
            print(f"Creating chart for {symbol}...")
            save_path = self.plot_optimization_results(symbol, save_dir)
            saved_files.append(save_path)

        print("\n" + "="*80)
        print(f"Optimization charts completed! {len(saved_files)} charts saved.")
        print(f"Location: {save_dir}/")
        print("="*80 + "\n")

        return saved_files

    def save_results(self, save_dir='optimization_results'):
        """최적화 결과를 CSV와 JSON으로 저장"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "="*80)
        print("Saving optimization results...")
        print("="*80 + "\n")

        # 1. 각 종목별 전체 최적화 결과 CSV 저장
        for symbol in self.symbols:
            symbol_clean = symbol.split('_')[0]
            csv_path = f"{save_dir}/sma_optimization_full_{symbol_clean}.csv"
            self.optimization_results[symbol].to_csv(csv_path, index=False)
            print(f"  Full results saved: {csv_path}")

        # 2. 최적 파라미터 요약 JSON 저장
        json_path = f"{save_dir}/sma_optimization_best_params.json"
        with open(json_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"  Best parameters saved: {json_path}")

        # 3. 최적 파라미터 요약 CSV 저장
        summary_data = []
        for symbol, params in self.best_params.items():
            symbol_clean = symbol.split('_')[0]
            summary_data.append({
                'Symbol': symbol_clean,
                'Short Window': params['short_window'],
                'Long Window': params['long_window'],
                'Total Return (%)': params['metrics']['total_return'],
                'CAGR (%)': params['metrics']['cagr'],
                'MDD (%)': params['metrics']['mdd'],
                'Sharpe Ratio': params['metrics']['sharpe'],
                'Win Rate (%)': params['metrics']['win_rate'],
                'Total Trades': params['metrics']['total_trades']
            })

        summary_df = pd.DataFrame(summary_data)
        csv_summary_path = f"{save_dir}/sma_optimization_summary.csv"
        summary_df.to_csv(csv_summary_path, index=False)
        print(f"  Summary saved: {csv_summary_path}")

        print("\n" + "="*80)
        print("Results saving completed!")
        print("="*80 + "\n")

        return summary_df

    def print_summary(self):
        """최적화 결과 요약 출력"""
        print("\n" + "="*120)
        print(f"{'SMA CROSSOVER OPTIMIZATION SUMMARY':^120}")
        print("="*120)
        print(f"\nPeriod: {self.start_date} ~ {self.end_date}")
        print(f"Symbols: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"Slippage: {self.slippage*100}%")

        print("\n" + "-"*120)
        print(f"{'Symbol':<10} {'Short':<8} {'Long':<8} {'Total Return':<15} {'CAGR':<10} {'MDD':<10} {'Sharpe':<10} {'Win Rate':<12} {'Trades':<10}")
        print("-"*120)

        for symbol, params in self.best_params.items():
            symbol_clean = symbol.split('_')[0]
            metrics = params['metrics']
            print(f"{symbol_clean:<10} "
                  f"{params['short_window']:<8} "
                  f"{params['long_window']:<8} "
                  f"{metrics['total_return']:>12.2f}%  "
                  f"{metrics['cagr']:>8.2f}% "
                  f"{metrics['mdd']:>8.2f}% "
                  f"{metrics['sharpe']:>8.2f}  "
                  f"{metrics['win_rate']:>10.2f}% "
                  f"{metrics['total_trades']:>8}")

        print("="*120 + "\n")


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("SMA CROSSOVER PARAMETER OPTIMIZATION")
    print("="*80)

    # 최적화 실행
    optimizer = SMACrossoverOptimizer(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 1. 데이터 로드
    optimizer.load_data()

    # 2. 파라미터 최적화
    # short: 1~50, long: 3,5,10,15,20,...,120
    short_range = list(range(1, 51))
    long_range = [3, 5, 10] + list(range(15, 121, 5))

    optimizer.run_optimization_all_symbols(
        short_range=short_range,
        long_range=long_range,
        metric='sharpe'  # Sharpe Ratio로 최적화
    )

    # 3. 결과 요약 출력
    optimizer.print_summary()

    # 4. 결과 저장
    summary_df = optimizer.save_results()

    # 5. 시각화
    optimizer.plot_all_optimization_results()

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
