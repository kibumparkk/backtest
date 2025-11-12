"""
SMA 크로스오버 전략 백테스트 및 파라미터 최적화

전략 로직:
- 매수 신호: 전일 종가가 짧은 이평선(SMA Short)을 상향 돌파
- 매도 신호: 전일 종가가 긴 이평선(SMA Long)을 하락 돌파

파라미터 그리드 서치:
- 다양한 SMA 조합(short_window, long_window)을 테스트
- 최적 파라미터 곡선 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class SMACrossoverBacktest:
    """SMA 크로스오버 전략 백테스트 클래스"""

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
        self.grid_results = {}

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

    def strategy_sma_crossover(self, df, short_window=20, long_window=50):
        """
        SMA 크로스오버 전략

        매수 신호: 전일 종가가 짧은 이평선(SMA Short)을 상향 돌파
        매도 신호: 전일 종가가 긴 이평선(SMA Long)을 하락 돌파

        Args:
            df: OHLCV 데이터프레임
            short_window: 짧은 이평선 기간
            long_window: 긴 이평선 기간

        Returns:
            백테스트 결과 데이터프레임
        """
        df = df.copy()

        # SMA 계산
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()

        # 전일 종가
        df['Close_Prev'] = df['Close'].shift(1)

        # 포지션 관리
        df['position'] = 0
        df['signal'] = 0  # 1: 매수, -1: 매도, 0: 유지

        for i in range(1, len(df)):
            prev_position = df.iloc[i-1]['position']
            prev_close = df.iloc[i-1]['Close']
            prev_sma_short = df.iloc[i-1]['SMA_Short']
            prev_sma_long = df.iloc[i-1]['SMA_Long']

            curr_close = df.iloc[i]['Close']
            curr_sma_short = df.iloc[i]['SMA_Short']
            curr_sma_long = df.iloc[i]['SMA_Long']

            # NaN 체크
            if pd.isna(prev_sma_short) or pd.isna(curr_sma_short) or \
               pd.isna(prev_sma_long) or pd.isna(curr_sma_long):
                df.iloc[i, df.columns.get_loc('position')] = prev_position
                continue

            # 매수 신호: 전일 종가가 짧은 이평선을 상향 돌파
            if prev_position == 0:  # 포지션 없을 때만
                if prev_close < prev_sma_short and curr_close >= curr_sma_short:
                    df.iloc[i, df.columns.get_loc('position')] = 1
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                else:
                    df.iloc[i, df.columns.get_loc('position')] = prev_position

            # 매도 신호: 전일 종가가 긴 이평선을 하락 돌파
            elif prev_position == 1:  # 포지션 있을 때
                if prev_close > prev_sma_long and curr_close <= curr_sma_long:
                    df.iloc[i, df.columns.get_loc('position')] = 0
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                else:
                    df.iloc[i, df.columns.get_loc('position')] = prev_position

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            # 매수 시그널
            if df.iloc[i]['signal'] == 1:
                # 당일 종가에 매수 (슬리피지 포함)
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['Close'] * (1 + self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = -self.slippage  # 매수 비용

            # 매도 시그널
            elif df.iloc[i]['signal'] == -1:
                # 당일 종가에 매도 (슬리피지 포함)
                buy_price = df.iloc[i-1]['buy_price']
                if pd.notna(buy_price):
                    sell_price = df.iloc[i]['Close'] * (1 - self.slippage)
                    df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)

            # 포지션 유지
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def calculate_metrics(self, returns_series):
        """성과 지표 계산"""
        # 누적 수익률
        cumulative = (1 + returns_series).cumprod()

        # 총 수익률
        total_return = (cumulative.iloc[-1] - 1) * 100 if len(cumulative) > 0 else 0

        # 연간 수익률 (CAGR)
        years = (returns_series.index[-1] - returns_series.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 and cumulative.iloc[-1] > 0 else 0

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
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades),
            'Profit Factor': profit_factor
        }

    def grid_search(self, short_windows=None, long_windows=None):
        """
        파라미터 그리드 서치

        Args:
            short_windows: 짧은 이평선 기간 리스트
            long_windows: 긴 이평선 기간 리스트
        """
        if short_windows is None:
            short_windows = [5, 10, 15, 20, 25, 30, 40, 50]

        if long_windows is None:
            long_windows = [50, 60, 70, 80, 90, 100, 120, 150, 200]

        print("\n" + "="*80)
        print("Starting Parameter Grid Search...")
        print("="*80)
        print(f"\nShort Windows: {short_windows}")
        print(f"Long Windows: {long_windows}")
        print(f"Total Combinations: {len(short_windows) * len(long_windows)}")
        print(f"Total Tests: {len(self.symbols) * len(short_windows) * len(long_windows)}")
        print("")

        total_combinations = len(short_windows) * len(long_windows)
        current_test = 0

        # 각 종목별로 그리드 서치
        for symbol in self.symbols:
            print(f"\n>>> Testing {symbol}...")
            self.grid_results[symbol] = []

            df = self.data[symbol].copy()

            for short_window, long_window in product(short_windows, long_windows):
                current_test += 1

                # short_window는 long_window보다 작아야 함
                if short_window >= long_window:
                    continue

                # 진행률 표시
                if current_test % 10 == 0:
                    progress = (current_test / (total_combinations * len(self.symbols))) * 100
                    print(f"  Progress: {progress:.1f}% - Testing Short={short_window}, Long={long_window}")

                # 백테스트 실행
                try:
                    result = self.strategy_sma_crossover(df, short_window, long_window)
                    metrics = self.calculate_metrics(result['returns'])

                    # 파라미터 및 결과 저장
                    metrics['Short_Window'] = short_window
                    metrics['Long_Window'] = long_window
                    metrics['Symbol'] = symbol.split('_')[0]

                    self.grid_results[symbol].append(metrics)

                except Exception as e:
                    print(f"  Error with Short={short_window}, Long={long_window}: {e}")
                    continue

        print("\n" + "="*80)
        print("Grid Search Completed!")
        print("="*80 + "\n")

    def get_best_parameters(self, symbol, metric='Sharpe Ratio'):
        """
        특정 종목의 최적 파라미터 찾기

        Args:
            symbol: 종목 심볼
            metric: 최적화 지표 (default: 'Sharpe Ratio')

        Returns:
            최적 파라미터 딕셔너리
        """
        results_df = pd.DataFrame(self.grid_results[symbol])

        # Profit Factor가 inf인 경우 처리
        if metric == 'Profit Factor':
            results_df = results_df[results_df['Profit Factor'] != np.inf]

        if len(results_df) == 0:
            return None

        # 최적값 찾기
        best_idx = results_df[metric].idxmax()
        best_result = results_df.loc[best_idx]

        return best_result.to_dict()

    def plot_heatmap(self, symbol, metric='Sharpe Ratio', save_path=None):
        """
        파라미터 그리드 서치 결과 히트맵

        Args:
            symbol: 종목 심볼
            metric: 시각화할 지표
            save_path: 저장 경로
        """
        results_df = pd.DataFrame(self.grid_results[symbol])

        # Profit Factor가 inf인 경우 처리
        if metric == 'Profit Factor':
            results_df = results_df[results_df['Profit Factor'] != np.inf]

        # 피벗 테이블 생성
        heatmap_data = results_df.pivot(index='Long_Window', columns='Short_Window', values=metric)

        # 히트맵 그리기
        fig, ax = plt.subplots(figsize=(14, 10))

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': metric}, ax=ax, linewidths=0.5)

        ax.set_title(f'{symbol.split("_")[0]} - SMA Crossover Strategy\n{metric} Heatmap',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Short Window (Days)', fontsize=12)
        ax.set_ylabel('Long Window (Days)', fontsize=12)

        # 최적 파라미터 표시
        best_params = self.get_best_parameters(symbol, metric)
        if best_params:
            ax.text(0.02, 0.98,
                   f'Best Parameters:\nShort={int(best_params["Short_Window"])}, Long={int(best_params["Long_Window"])}\n{metric}={best_params[metric]:.2f}',
                   transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Heatmap saved: {save_path}")

        plt.close()

    def plot_3d_surface(self, symbol, metric='Sharpe Ratio', save_path=None):
        """
        파라미터 그리드 서치 결과 3D 곡면

        Args:
            symbol: 종목 심볼
            metric: 시각화할 지표
            save_path: 저장 경로
        """
        from mpl_toolkits.mplot3d import Axes3D

        results_df = pd.DataFrame(self.grid_results[symbol])

        # Profit Factor가 inf인 경우 처리
        if metric == 'Profit Factor':
            results_df = results_df[results_df['Profit Factor'] != np.inf]

        # 피벗 테이블 생성
        heatmap_data = results_df.pivot(index='Long_Window', columns='Short_Window', values=metric)

        # 3D 플롯
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        X = heatmap_data.columns.values
        Y = heatmap_data.index.values
        X, Y = np.meshgrid(X, Y)
        Z = heatmap_data.values

        surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.8, edgecolor='none')

        ax.set_xlabel('Short Window (Days)', fontsize=12, labelpad=10)
        ax.set_ylabel('Long Window (Days)', fontsize=12, labelpad=10)
        ax.set_zlabel(metric, fontsize=12, labelpad=10)
        ax.set_title(f'{symbol.split("_")[0]} - SMA Crossover Strategy\n{metric} 3D Surface',
                    fontsize=16, fontweight='bold', pad=20)

        # 컬러바
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # 최적 파라미터 표시
        best_params = self.get_best_parameters(symbol, metric)
        if best_params:
            ax.text2D(0.02, 0.98,
                     f'Best Parameters:\nShort={int(best_params["Short_Window"])}, Long={int(best_params["Long_Window"])}\n{metric}={best_params[metric]:.2f}',
                     transform=ax.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  3D surface saved: {save_path}")

        plt.close()

    def plot_all_visualizations(self, save_dir='sma_crossover_results'):
        """
        모든 종목에 대해 시각화 생성

        Args:
            save_dir: 저장 디렉토리
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "="*80)
        print("Creating visualizations...")
        print("="*80 + "\n")

        metrics_to_plot = ['CAGR (%)', 'Sharpe Ratio', 'MDD (%)', 'Total Return (%)']

        for symbol in self.symbols:
            symbol_clean = symbol.split('_')[0]
            print(f"\n>>> Creating visualizations for {symbol_clean}...")

            for metric in metrics_to_plot:
                # 히트맵
                heatmap_path = f"{save_dir}/{symbol_clean}_{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}_heatmap.png"
                self.plot_heatmap(symbol, metric, heatmap_path)

                # 3D 곡면
                surface_path = f"{save_dir}/{symbol_clean}_{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}_3d.png"
                self.plot_3d_surface(symbol, metric, surface_path)

        print("\n" + "="*80)
        print("Visualization completed!")
        print("="*80 + "\n")

    def create_summary_report(self, save_path='sma_crossover_summary.csv'):
        """
        그리드 서치 결과 요약 리포트 생성

        Args:
            save_path: 저장 경로
        """
        summary_data = []

        for symbol in self.symbols:
            symbol_clean = symbol.split('_')[0]

            # 각 지표별 최적 파라미터
            metrics = ['CAGR (%)', 'Sharpe Ratio', 'Total Return (%)', 'Win Rate (%)']

            for metric in metrics:
                best_params = self.get_best_parameters(symbol, metric)
                if best_params:
                    summary_data.append({
                        'Symbol': symbol_clean,
                        'Optimization Metric': metric,
                        'Best Short Window': int(best_params['Short_Window']),
                        'Best Long Window': int(best_params['Long_Window']),
                        'Total Return (%)': best_params['Total Return (%)'],
                        'CAGR (%)': best_params['CAGR (%)'],
                        'MDD (%)': best_params['MDD (%)'],
                        'Sharpe Ratio': best_params['Sharpe Ratio'],
                        'Win Rate (%)': best_params['Win Rate (%)'],
                        'Total Trades': best_params['Total Trades'],
                        'Profit Factor': best_params['Profit Factor'] if best_params['Profit Factor'] != np.inf else 'N/A'
                    })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(save_path, index=False)

        print("\n" + "="*80)
        print("Summary Report")
        print("="*80)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to {save_path}")
        print("="*80 + "\n")

        return summary_df

    def plot_best_strategy_comparison(self, save_path='sma_crossover_best_comparison.png'):
        """
        각 종목의 최적 전략 비교 시각화

        Args:
            save_path: 저장 경로
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 최적 파라미터로 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])

        for symbol in self.symbols:
            symbol_clean = symbol.split('_')[0]
            best_params = self.get_best_parameters(symbol, 'Sharpe Ratio')

            if best_params:
                df = self.data[symbol].copy()
                result = self.strategy_sma_crossover(
                    df,
                    int(best_params['Short_Window']),
                    int(best_params['Long_Window'])
                )

                ax1.plot(result.index, result['cumulative'],
                        label=f"{symbol_clean} (S={int(best_params['Short_Window'])}, L={int(best_params['Long_Window'])})",
                        linewidth=2.5, alpha=0.8)

        ax1.set_title('Best SMA Crossover Strategies Comparison (Optimized by Sharpe Ratio)',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2-5. 각 종목별 지표 비교
        metrics = [
            ('CAGR (%)', gs[1, 0]),
            ('Sharpe Ratio', gs[1, 1]),
            ('MDD (%)', gs[1, 2]),
            ('Win Rate (%)', gs[2, 0])
        ]

        for metric, position in metrics:
            ax = fig.add_subplot(position)

            symbols_clean = []
            values = []

            for symbol in self.symbols:
                best_params = self.get_best_parameters(symbol, 'Sharpe Ratio')
                if best_params:
                    symbols_clean.append(symbol.split('_')[0])
                    values.append(best_params[metric])

            colors = ['green' if v > 0 else 'red' for v in values] if metric != 'MDD (%)' else ['crimson'] * len(values)
            ax.bar(symbols_clean, values, color=colors, alpha=0.7)
            ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric, fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')

            if metric == 'MDD (%)':
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 6. Total Return 비교
        ax6 = fig.add_subplot(gs[2, 1])
        symbols_clean = []
        values = []

        for symbol in self.symbols:
            best_params = self.get_best_parameters(symbol, 'Sharpe Ratio')
            if best_params:
                symbols_clean.append(symbol.split('_')[0])
                values.append(best_params['Total Return (%)'])

        colors = ['green' if v > 0 else 'red' for v in values]
        ax6.bar(symbols_clean, values, color=colors, alpha=0.7)
        ax6.set_title('Total Return (%) Comparison', fontsize=13, fontweight='bold')
        ax6.set_ylabel('Total Return (%)', fontsize=11)
        ax6.grid(True, alpha=0.3, axis='y')

        # 7. Total Trades 비교
        ax7 = fig.add_subplot(gs[2, 2])
        symbols_clean = []
        values = []

        for symbol in self.symbols:
            best_params = self.get_best_parameters(symbol, 'Sharpe Ratio')
            if best_params:
                symbols_clean.append(symbol.split('_')[0])
                values.append(best_params['Total Trades'])

        ax7.bar(symbols_clean, values, color='steelblue', alpha=0.7)
        ax7.set_title('Total Trades Comparison', fontsize=13, fontweight='bold')
        ax7.set_ylabel('Number of Trades', fontsize=11)
        ax7.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'SMA Crossover Strategy - Best Parameters Comparison\nPeriod: {self.start_date} to {self.end_date}',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nBest strategy comparison saved to {save_path}")
        plt.close()

    def run_full_analysis(self, short_windows=None, long_windows=None):
        """
        전체 분석 파이프라인 실행

        Args:
            short_windows: 짧은 이평선 기간 리스트
            long_windows: 긴 이평선 기간 리스트
        """
        # 1. 데이터 로드
        self.load_data()

        # 2. 그리드 서치
        self.grid_search(short_windows, long_windows)

        # 3. 시각화
        self.plot_all_visualizations()

        # 4. 요약 리포트
        self.create_summary_report()

        # 5. 최적 전략 비교
        self.plot_best_strategy_comparison()

        print("\n" + "="*80)
        print("Full Analysis Completed!")
        print("="*80 + "\n")


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("SMA Crossover Strategy Backtesting & Parameter Optimization")
    print("="*80)

    # 백테스트 실행
    backtest = SMACrossoverBacktest(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 파라미터 그리드 설정
    short_windows = [5, 10, 15, 20, 25, 30, 40, 50]
    long_windows = [50, 60, 70, 80, 90, 100, 120, 150, 200]

    # 전체 분석 실행
    backtest.run_full_analysis(short_windows, long_windows)

    print("\n" + "="*80)
    print("Analysis Completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
