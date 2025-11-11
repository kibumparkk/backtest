"""
BTC 적응형 SMA 전략 백테스트

SMA 윈도우 10~60을 테스트하고, 분기마다 최근 1년 성과가 좋은 윈도우로 갈아타는 전략
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class AdaptiveSMAStrategy:
    """적응형 SMA 전략 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01',
                 end_date=None, slippage=0.002,
                 sma_range=(10, 60), lookback_period=365, rebalance_freq='Q'):
        """
        Args:
            symbol: 종목 심볼 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
            sma_range: SMA 윈도우 범위 (default: 10~60)
            lookback_period: 성과 평가 기간 (일 단위, default: 365일 = 1년)
            rebalance_freq: 리밸런싱 주기 (default: 'Q' = 분기)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.sma_range = sma_range
        self.lookback_period = lookback_period
        self.rebalance_freq = rebalance_freq
        self.data = None
        self.sma_results = {}
        self.adaptive_result = None

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
        SMA 교차 전략
        - 가격이 SMA 이상일 때 매수 (보유)
        - 가격이 SMA 미만일 때 매도 후 현금 보유

        Args:
            df: 데이터프레임
            sma_period: SMA 윈도우

        Returns:
            결과 데이터프레임
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 포지션 계산
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 매수/매도 시 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def run_all_sma_strategies(self):
        """모든 SMA 윈도우에 대해 전략 실행"""
        print("="*80)
        print(f"Testing SMA windows from {self.sma_range[0]} to {self.sma_range[1]}...")
        print("="*80 + "\n")

        for sma_period in range(self.sma_range[0], self.sma_range[1] + 1):
            result = self.strategy_sma(self.data, sma_period)
            self.sma_results[sma_period] = result

            # 간략한 성과 출력
            total_return = (result['cumulative'].iloc[-1] - 1) * 100
            print(f"SMA {sma_period:2d}: Total Return = {total_return:8.2f}%")

        print("\n" + "="*80)
        print("All SMA strategies completed!")
        print("="*80 + "\n")

    def calculate_performance(self, returns_series):
        """
        주어진 수익률 시리즈의 성과 계산 (누적 수익률)

        Args:
            returns_series: 수익률 시리즈

        Returns:
            누적 수익률
        """
        if len(returns_series) == 0:
            return 0.0
        cumulative = (1 + returns_series).prod()
        return cumulative - 1

    def select_best_sma_window(self, current_date):
        """
        현재 날짜 기준 과거 lookback_period 동안 최고 성과를 낸 SMA 윈도우 선택

        Args:
            current_date: 현재 날짜

        Returns:
            최적 SMA 윈도우
        """
        lookback_start = current_date - timedelta(days=self.lookback_period)

        best_sma = None
        best_performance = -np.inf

        for sma_period, result_df in self.sma_results.items():
            # lookback 기간 데이터 추출
            mask = (result_df.index > lookback_start) & (result_df.index <= current_date)
            lookback_returns = result_df.loc[mask, 'returns']

            if len(lookback_returns) == 0:
                continue

            # 성과 계산
            performance = self.calculate_performance(lookback_returns)

            if performance > best_performance:
                best_performance = performance
                best_sma = sma_period

        return best_sma, best_performance

    def run_adaptive_strategy(self):
        """
        적응형 SMA 전략 실행
        - 분기마다 최근 1년 성과가 좋은 SMA 윈도우로 리밸런싱
        """
        print("="*80)
        print("Running Adaptive SMA Strategy...")
        print(f"Rebalancing frequency: {self.rebalance_freq}")
        print(f"Lookback period: {self.lookback_period} days")
        print("="*80 + "\n")

        # 리밸런싱 날짜 생성
        all_dates = self.data.index

        # 분기별 시작 날짜 찾기
        rebalance_dates = []

        # 첫 리밸런싱 날짜는 lookback_period 이후부터 시작
        min_date = all_dates[0] + timedelta(days=self.lookback_period)

        # 분기별 날짜 생성
        if self.rebalance_freq == 'Q':
            # 분기 시작 월: 1, 4, 7, 10
            for date in all_dates:
                if date < min_date:
                    continue
                if date.month in [1, 4, 7, 10] and date.day == 1:
                    if len(rebalance_dates) == 0 or (date - rebalance_dates[-1]).days >= 80:
                        rebalance_dates.append(date)

        # 수동으로 분기 생성 (날짜가 정확히 없을 경우 대비)
        if len(rebalance_dates) == 0:
            current_date = min_date
            end_date = all_dates[-1]
            while current_date <= end_date:
                # 가장 가까운 거래일 찾기
                closest_date = all_dates[all_dates >= current_date][0] if len(all_dates[all_dates >= current_date]) > 0 else None
                if closest_date is not None and closest_date not in rebalance_dates:
                    rebalance_dates.append(closest_date)
                # 다음 분기로 이동 (약 90일)
                current_date = current_date + timedelta(days=90)

        print(f"Number of rebalancing periods: {len(rebalance_dates)}\n")

        # 적응형 전략 결과 저장
        adaptive_returns = pd.Series(0.0, index=all_dates)
        selected_sma_windows = {}

        for i in range(len(rebalance_dates)):
            rebalance_date = rebalance_dates[i]

            # 최적 SMA 윈도우 선택
            best_sma, best_perf = self.select_best_sma_window(rebalance_date)

            if best_sma is None:
                continue

            # 다음 리밸런싱 날짜까지 기간
            if i < len(rebalance_dates) - 1:
                next_rebalance = rebalance_dates[i + 1]
            else:
                next_rebalance = all_dates[-1]

            # 해당 기간 동안 선택된 SMA 윈도우 적용
            period_mask = (all_dates > rebalance_date) & (all_dates <= next_rebalance)
            period_returns = self.sma_results[best_sma].loc[period_mask, 'returns']
            adaptive_returns[period_mask] = period_returns

            selected_sma_windows[rebalance_date] = best_sma

            print(f"[{rebalance_date.strftime('%Y-%m-%d')}] Selected SMA: {best_sma:2d} "
                  f"(Lookback Performance: {best_perf*100:6.2f}%)")

        # 누적 수익률 계산
        adaptive_cumulative = (1 + adaptive_returns).cumprod()

        # 결과 저장
        self.adaptive_result = pd.DataFrame({
            'returns': adaptive_returns,
            'cumulative': adaptive_cumulative
        }, index=all_dates)

        self.selected_sma_windows = selected_sma_windows

        print("\n" + "="*80)
        print("Adaptive strategy completed!")
        print("="*80 + "\n")

        return self.adaptive_result, selected_sma_windows

    def calculate_metrics(self, returns_series, name):
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

        return {
            'Strategy': name,
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades),
            'Profit Factor': profit_factor
        }

    def calculate_all_metrics(self):
        """모든 전략 성과 지표 계산"""
        metrics_list = []

        # 적응형 전략
        adaptive_metrics = self.calculate_metrics(
            self.adaptive_result['returns'],
            'Adaptive SMA (Quarterly Rebalance)'
        )
        metrics_list.append(adaptive_metrics)

        # 개별 SMA 윈도우 (샘플링: 10, 20, 30, 40, 50, 60)
        sample_smas = [10, 20, 30, 40, 50, 60]
        for sma_period in sample_smas:
            if sma_period in self.sma_results:
                sma_metrics = self.calculate_metrics(
                    self.sma_results[sma_period]['returns'],
                    f'SMA {sma_period}'
                )
                metrics_list.append(sma_metrics)

        return pd.DataFrame(metrics_list)

    def plot_results(self, metrics_df, save_path='btc_adaptive_sma_results.png'):
        """결과 시각화"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률 비교 (적응형 vs 고정 SMA)
        ax1 = fig.add_subplot(gs[0, :])

        # 적응형 전략
        ax1.plot(self.adaptive_result.index, self.adaptive_result['cumulative'],
                label='Adaptive SMA (Quarterly)', linewidth=3, color='red', alpha=0.9)

        # 샘플 SMA 전략들
        sample_smas = [10, 20, 30, 40, 50, 60]
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_smas)))
        for sma_period, color in zip(sample_smas, colors):
            if sma_period in self.sma_results:
                ax1.plot(self.sma_results[sma_period].index,
                        self.sma_results[sma_period]['cumulative'],
                        label=f'SMA {sma_period}', linewidth=1.5, alpha=0.6, color=color)

        ax1.set_title(f'Adaptive SMA Strategy vs Fixed SMA Strategies - {self.symbol}',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 선택된 SMA 윈도우 히스토리
        ax2 = fig.add_subplot(gs[1, :])

        dates = list(self.selected_sma_windows.keys())
        sma_values = list(self.selected_sma_windows.values())

        ax2.step(dates, sma_values, where='post', linewidth=2, color='darkblue', alpha=0.7)
        ax2.scatter(dates, sma_values, s=100, color='red', zorder=5, alpha=0.8)
        ax2.set_title('Selected SMA Window Over Time (Quarterly Rebalancing)',
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('SMA Window', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylim(self.sma_range[0] - 5, self.sma_range[1] + 5)
        ax2.grid(True, alpha=0.3)

        # 각 날짜마다 SMA 값 표시
        for date, sma in zip(dates, sma_values):
            ax2.annotate(f'{sma}', (date, sma),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=8, fontweight='bold')

        # 3. 총 수익률 비교
        ax3 = fig.add_subplot(gs[2, 0])
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True)
        colors = ['red' if x == sorted_df['Total Return (%)'].max() else
                 'green' if x > 0 else 'gray' for x in sorted_df['Total Return (%)']]
        ax3.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax3.set_xlabel('Total Return (%)', fontsize=11)
        ax3.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. CAGR 비교
        ax4 = fig.add_subplot(gs[2, 1])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        colors = ['red' if x == sorted_df['CAGR (%)'].max() else
                 'green' if x > 0 else 'gray' for x in sorted_df['CAGR (%)']]
        ax4.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax4.set_xlabel('CAGR (%)', fontsize=11)
        ax4.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. MDD 비교
        ax5 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        colors = ['red' if x == sorted_df['MDD (%)'].max() else
                 'orange' if x > sorted_df['MDD (%)'].median() else 'green'
                 for x in sorted_df['MDD (%)']]
        ax5.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color=colors, alpha=0.7)
        ax5.set_xlabel('MDD (%)', fontsize=11)
        ax5.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. 샤프 비율 비교
        ax6 = fig.add_subplot(gs[3, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['red' if x == sorted_df['Sharpe Ratio'].max() else
                 'green' if x > 0 else 'gray' for x in sorted_df['Sharpe Ratio']]
        ax6.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax6.set_xlabel('Sharpe Ratio', fontsize=11)
        ax6.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        # 7. Return vs Risk 산점도
        ax7 = fig.add_subplot(gs[3, 1])

        # 적응형 전략 강조
        adaptive_row = metrics_df[metrics_df['Strategy'].str.contains('Adaptive')].iloc[0]
        ax7.scatter(adaptive_row['MDD (%)'], adaptive_row['CAGR (%)'],
                   s=500, alpha=0.8, c='red', marker='*',
                   edgecolors='black', linewidths=2, zorder=10,
                   label='Adaptive SMA')

        # 고정 SMA 전략들
        fixed_sma = metrics_df[~metrics_df['Strategy'].str.contains('Adaptive')]
        ax7.scatter(fixed_sma['MDD (%)'], fixed_sma['CAGR (%)'],
                   s=200, alpha=0.6, c=fixed_sma['Sharpe Ratio'],
                   cmap='RdYlGn', edgecolors='black', linewidths=1)

        # 레이블 추가
        for idx, row in metrics_df.iterrows():
            label = row['Strategy'].replace('Adaptive SMA (Quarterly Rebalance)', 'Adaptive')
            ax7.annotate(label, (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=9, ha='center', va='bottom')

        ax7.set_xlabel('MDD (%)', fontsize=11)
        ax7.set_ylabel('CAGR (%)', fontsize=11)
        ax7.set_title('Return vs Risk (Adaptive highlighted)', fontsize=13, fontweight='bold')
        ax7.legend(loc='best', fontsize=10)
        ax7.grid(True, alpha=0.3)

        # 8. 드로우다운 비교
        ax8 = fig.add_subplot(gs[3, 2])

        # 적응형 전략 드로우다운
        cummax = self.adaptive_result['cumulative'].cummax()
        drawdown = (self.adaptive_result['cumulative'] - cummax) / cummax * 100
        ax8.plot(drawdown.index, drawdown, linewidth=2.5,
                color='red', alpha=0.8, label='Adaptive SMA')
        ax8.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.2)

        # 비교를 위해 SMA 30 추가
        if 30 in self.sma_results:
            cummax_30 = self.sma_results[30]['cumulative'].cummax()
            drawdown_30 = (self.sma_results[30]['cumulative'] - cummax_30) / cummax_30 * 100
            ax8.plot(drawdown_30.index, drawdown_30, linewidth=1.5,
                    color='blue', alpha=0.6, label='SMA 30', linestyle='--')

        ax8.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
        ax8.set_ylabel('Drawdown (%)', fontsize=11)
        ax8.set_xlabel('Date', fontsize=11)
        ax8.legend(loc='lower right', fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*120)
        print(f"{'BTC Adaptive SMA Strategy Performance':^120}")
        print("="*120)
        print(f"\nPeriod: {self.start_date} ~ {self.end_date}")
        print(f"Symbol: {self.symbol}")
        print(f"SMA Range: {self.sma_range[0]} ~ {self.sma_range[1]}")
        print(f"Lookback Period: {self.lookback_period} days (1 year)")
        print(f"Rebalancing Frequency: {self.rebalance_freq} (Quarterly)")
        print(f"Slippage: {self.slippage*100}%")

        print("\n" + "-"*120)
        print(f"{'Performance Metrics':^120}")
        print("-"*120)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(metrics_df.to_string(index=False))

        print("\n" + "="*120 + "\n")

    def run_full_analysis(self):
        """전체 분석 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 모든 SMA 전략 실행
        self.run_all_sma_strategies()

        # 3. 적응형 전략 실행
        self.run_adaptive_strategy()

        # 4. 성과 지표 계산
        metrics_df = self.calculate_all_metrics()

        # 5. 결과 출력
        self.print_metrics_table(metrics_df)

        # 6. 시각화
        self.plot_results(metrics_df)

        # 7. 결과 저장
        print("\nSaving results...")
        metrics_df.to_csv('btc_adaptive_sma_metrics.csv', index=False)
        print("Metrics saved to btc_adaptive_sma_metrics.csv")

        self.adaptive_result.to_csv('btc_adaptive_sma_returns.csv')
        print("Adaptive strategy returns saved to btc_adaptive_sma_returns.csv")

        # 선택된 SMA 윈도우 저장
        sma_windows_df = pd.DataFrame({
            'Date': list(self.selected_sma_windows.keys()),
            'Selected_SMA': list(self.selected_sma_windows.values())
        })
        sma_windows_df.to_csv('btc_adaptive_sma_windows.csv', index=False)
        print("Selected SMA windows saved to btc_adaptive_sma_windows.csv")

        print("\n" + "="*120)
        print("Analysis completed!")
        print("="*120 + "\n")

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("BTC Adaptive SMA Strategy Backtesting")
    print("="*80)

    # 백테스트 실행
    strategy = AdaptiveSMAStrategy(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002,  # 0.2%
        sma_range=(10, 60),
        lookback_period=365,  # 1년
        rebalance_freq='Q'  # 분기
    )

    # 분석 실행
    metrics_df = strategy.run_full_analysis()

    print("\n" + "="*80)
    print("Done!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
