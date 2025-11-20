"""
비트코인 전략 비교: RSI 최적 파라미터 vs SMA30

비교 전략:
1. RSI(5, 70) - 전체 구간 최적 (Sharpe Ratio 기준)
2. RSI(17, 55) - 전체 구간 최적 (CAGR 기준)
3. SMA(30) - 전통적인 이동평균 전략
4. Buy & Hold - 벤치마크
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


class StrategyComparison:
    """전략 비교 클래스"""

    def __init__(self, symbol='BTC_KRW',
                 start_date='2018-01-01', end_date=None,
                 slippage=0.002):
        """
        Args:
            symbol: 종목 심볼
            start_date: 시작일
            end_date: 종료일
            slippage: 슬리피지
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage

        self.data = None
        self.results = {}
        self.metrics = {}

    def load_data(self):
        """데이터 로드"""
        print("="*80)
        print(f"Loading {self.symbol} data...")
        print("="*80)

        file_path = f'chart_day/{self.symbol}.parquet'
        df = pd.read_parquet(file_path)
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df

        print(f"\nLoaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
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

    def strategy_rsi(self, df, rsi_period, rsi_threshold, name):
        """RSI 전략"""
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

        self.results[name] = df

        return df

    def strategy_sma(self, df, sma_period, name):
        """SMA 전략"""
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 시그널 생성 (가격이 SMA 위에 있으면 매수)
        df['signal'] = (df['Close'] >= df['SMA']).astype(int)
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

        self.results[name] = df

        return df

    def strategy_buy_hold(self, df, name):
        """Buy & Hold 전략"""
        df = df.copy()

        # 수익률 계산
        df['returns'] = df['Close'].pct_change().fillna(0)
        df['cumulative'] = (1 + df['returns']).cumprod()

        self.results[name] = df

        return df

    def calculate_metrics(self, df, name):
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

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = (returns.mean() / downside_std * np.sqrt(365)) if downside_std > 0 else 0

        # 승률
        total_trades = (returns != 0).sum()
        winning_trades = (returns > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit Factor
        total_profit = returns[returns > 0].sum()
        total_loss = abs(returns[returns < 0].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        # Calmar Ratio
        calmar = abs(cagr / mdd) if mdd != 0 else 0

        # 최대 연속 손실
        is_loss = (returns < 0).astype(int)
        loss_groups = (is_loss != is_loss.shift()).cumsum()
        max_consecutive_losses = is_loss.groupby(loss_groups).sum().max() if is_loss.sum() > 0 else 0

        self.metrics[name] = {
            'Strategy': name,
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades),
            'Profit Factor': profit_factor,
            'Max Consecutive Losses': int(max_consecutive_losses)
        }

        return self.metrics[name]

    def run_all_strategies(self):
        """모든 전략 실행"""
        print("\n" + "="*80)
        print("Running All Strategies...")
        print("="*80)

        # 1. RSI(5, 70) - 최적 Sharpe
        print("\n1. Running RSI(5, 70) - Best Sharpe...")
        self.strategy_rsi(self.data, 5, 70, 'RSI(5, 70)')
        self.calculate_metrics(self.results['RSI(5, 70)'], 'RSI(5, 70)')

        # 2. RSI(17, 55) - 최적 CAGR
        print("2. Running RSI(17, 55) - Best CAGR...")
        self.strategy_rsi(self.data, 17, 55, 'RSI(17, 55)')
        self.calculate_metrics(self.results['RSI(17, 55)'], 'RSI(17, 55)')

        # 3. SMA(30)
        print("3. Running SMA(30)...")
        self.strategy_sma(self.data, 30, 'SMA(30)')
        self.calculate_metrics(self.results['SMA(30)'], 'SMA(30)')

        # 4. Buy & Hold
        print("4. Running Buy & Hold...")
        self.strategy_buy_hold(self.data, 'Buy & Hold')
        self.calculate_metrics(self.results['Buy & Hold'], 'Buy & Hold')

        print("\n" + "="*80)
        print("All Strategies Completed!")
        print("="*80 + "\n")

    def print_comparison_table(self):
        """비교 테이블 출력"""
        print("\n" + "="*140)
        print(f"{'전략 성과 비교':^140}")
        print("="*140)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {self.symbol}")
        print(f"슬리피지: {self.slippage*100}%")

        # DataFrame 생성
        metrics_df = pd.DataFrame(self.metrics.values())

        print("\n" + "-"*140)
        print(f"{'전략 성과 요약':^140}")
        print("-"*140)

        # 주요 지표만 출력
        display_cols = ['Strategy', 'Total Return (%)', 'CAGR (%)', 'MDD (%)',
                       'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                       'Win Rate (%)', 'Total Trades']

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 140)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}')

        print(metrics_df[display_cols].to_string(index=False))

        print("\n" + "-"*140)

        # 순위 표시
        print("\n📊 성과 순위:")
        print("-"*140)

        ranking_metrics = ['CAGR (%)', 'Sharpe Ratio', 'MDD (%)', 'Calmar Ratio']
        for metric in ranking_metrics:
            if metric == 'MDD (%)':
                # MDD는 낮을수록 좋음 (절대값 기준)
                sorted_df = metrics_df.sort_values(metric, ascending=False)
            else:
                sorted_df = metrics_df.sort_values(metric, ascending=False)

            print(f"\n{metric} 순위:")
            for rank, (idx, row) in enumerate(sorted_df.iterrows(), 1):
                value = row[metric]
                strategy = row['Strategy']
                if rank == 1:
                    print(f"  🥇 {rank}. {strategy:<20} {value:>10.2f}")
                elif rank == 2:
                    print(f"  🥈 {rank}. {strategy:<20} {value:>10.2f}")
                elif rank == 3:
                    print(f"  🥉 {rank}. {strategy:<20} {value:>10.2f}")
                else:
                    print(f"     {rank}. {strategy:<20} {value:>10.2f}")

        print("\n" + "="*140 + "\n")

        return metrics_df

    def plot_comparison(self, save_path='strategy_comparison_rsi_vs_sma30.png'):
        """비교 시각화"""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 색상 정의
        colors = {
            'RSI(5, 70)': '#1f77b4',  # 파랑
            'RSI(17, 55)': '#ff7f0e',  # 주황
            'SMA(30)': '#2ca02c',      # 초록
            'Buy & Hold': '#d62728'    # 빨강
        }

        # 1. 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])
        for name, df in self.results.items():
            ax1.plot(df.index, df['cumulative'], label=name,
                    linewidth=2.5, alpha=0.8, color=colors[name])

        ax1.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 총 수익률 비교
        ax2 = fig.add_subplot(gs[1, 0])
        metrics_df = pd.DataFrame(self.metrics.values())
        sorted_df = metrics_df.sort_values('Total Return (%)', ascending=True)
        bar_colors = [colors[name] for name in sorted_df['Strategy']]
        ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'],
                color=bar_colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = metrics_df.sort_values('CAGR (%)', ascending=True)
        bar_colors = [colors[name] for name in sorted_df['Strategy']]
        ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'],
                color=bar_colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = metrics_df.sort_values('MDD (%)', ascending=False)
        bar_colors = [colors[name] for name in sorted_df['Strategy']]
        ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'],
                color=bar_colors, alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = metrics_df.sort_values('Sharpe Ratio', ascending=True)
        bar_colors = [colors[name] for name in sorted_df['Strategy']]
        ax5.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'],
                color=bar_colors, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Sortino Ratio 비교
        ax6 = fig.add_subplot(gs[2, 1])
        sorted_df = metrics_df.sort_values('Sortino Ratio', ascending=True)
        bar_colors = [colors[name] for name in sorted_df['Strategy']]
        ax6.barh(sorted_df['Strategy'], sorted_df['Sortino Ratio'],
                color=bar_colors, alpha=0.7)
        ax6.set_xlabel('Sortino Ratio', fontsize=11)
        ax6.set_title('Sortino Ratio Comparison', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        # 7. Calmar Ratio 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = metrics_df.sort_values('Calmar Ratio', ascending=True)
        bar_colors = [colors[name] for name in sorted_df['Strategy']]
        ax7.barh(sorted_df['Strategy'], sorted_df['Calmar Ratio'],
                color=bar_colors, alpha=0.7)
        ax7.set_xlabel('Calmar Ratio', fontsize=11)
        ax7.set_title('Calmar Ratio Comparison', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8. Drawdown 비교
        ax8 = fig.add_subplot(gs[3, :])
        for name, df in self.results.items():
            cumulative = df['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax8.plot(drawdown.index, drawdown, label=name,
                    linewidth=2, alpha=0.7, color=colors[name])

        ax8.fill_between(drawdown.index, drawdown, 0, alpha=0.1)
        ax8.set_title('Drawdown Comparison Over Time', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Drawdown (%)', fontsize=12)
        ax8.set_xlabel('Date', fontsize=12)
        ax8.legend(loc='lower right', fontsize=11)
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 전체 제목
        fig.suptitle(f'Strategy Comparison: RSI vs SMA30 vs Buy & Hold\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison chart saved to {save_path}")
        plt.close()

    def run_full_comparison(self):
        """전체 비교 프로세스 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 모든 전략 실행
        self.run_all_strategies()

        # 3. 비교 테이블 출력
        metrics_df = self.print_comparison_table()

        # 4. 시각화
        self.plot_comparison()

        # 5. 결과 저장
        print("\nSaving comparison results...")
        metrics_df.to_csv('strategy_comparison_results.csv', index=False)
        print("Results saved to strategy_comparison_results.csv")

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("전략 비교: RSI 최적 파라미터 vs SMA30")
    print("="*80)

    comparison = StrategyComparison(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )

    metrics_df = comparison.run_full_comparison()

    print("\n" + "="*80)
    print("비교 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
