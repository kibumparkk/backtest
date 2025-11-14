"""
비트코인 전일종가 > SMA30 전략 백테스트

전략:
- 전일종가 > SMA30: 매수/보유
- 전일종가 <= SMA30: 매도/현금 보유

시각화:
- 누적자산: 로그축
- 드로우다운: 마이너스 퍼센트
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


class BTCPreviousCloseSMA30Strategy:
    """비트코인 전일종가 > SMA30 전략 백테스트"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbol: 종목 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = None
        self.result = None

    def load_data(self):
        """데이터 로드"""
        print("=" * 80)
        print(f"Loading data for {self.symbol}...")
        print("=" * 80)

        file_path = f'chart_day/{self.symbol}.parquet'
        print(f"\nLoading from {file_path}...")
        df = pd.read_parquet(file_path)

        # 컬럼명 변경 (소문자 -> 대문자)
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df
        print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        print("=" * 80 + "\n")

    def strategy_previous_close_sma30(self, sma_period=30):
        """
        전일종가 > SMA30 전략
        - 전일종가 > SMA30: 매수/보유
        - 전일종가 <= SMA30: 매도/현금 보유
        """
        df = self.data.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 전일종가
        df['Previous_Close'] = df['Close'].shift(1)

        # 포지션 계산: 전일종가 > SMA30일 때 매수
        df['position'] = np.where(df['Previous_Close'] > df['SMA'], 1, 0)

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

        # Buy & Hold 전략 (비교용)
        df['bnh_returns'] = df['daily_price_return'].fillna(0)
        df['bnh_cumulative'] = (1 + df['bnh_returns']).cumprod()

        return df

    def calculate_metrics(self, returns_series, name):
        """성과 지표 계산"""
        # 누적 수익률
        cumulative = (1 + returns_series).cumprod()

        # 총 수익률
        total_return = (cumulative.iloc[-1] - 1) * 100

        # 연간 수익률 (CAGR)
        years = (returns_series.index[-1] - returns_series.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else 0

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

    def plot_results(self, save_path='btc_previous_close_sma30_results.png'):
        """결과 시각화 (로그축 누적자산, 마이너스 퍼센트 드로우다운)"""
        df = self.result

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률 비교 (로그축)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['cumulative'], label='Previous Close > SMA30 Strategy',
                 linewidth=2.5, color='blue', alpha=0.8)
        ax1.plot(df.index, df['bnh_cumulative'], label='Buy & Hold',
                 linewidth=2.5, color='gray', alpha=0.6, linestyle='--')
        ax1.set_title('BTC: Previous Close > SMA30 Strategy vs Buy & Hold (Log Scale)',
                      fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (Log Scale)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 드로우다운 (마이너스 퍼센트)
        ax2 = fig.add_subplot(gs[1, :])
        cummax = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - cummax) / cummax * 100  # 마이너스 퍼센트

        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown, color='darkred', linewidth=2)
        ax2.set_title('Drawdown Over Time (Negative Percentage)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # MDD 표시
        mdd_value = drawdown.min()
        mdd_date = drawdown.idxmin()
        ax2.scatter([mdd_date], [mdd_value], color='red', s=200, zorder=5, marker='X')
        ax2.annotate(f'MDD: {mdd_value:.2f}%',
                     xy=(mdd_date, mdd_value),
                     xytext=(10, -20), textcoords='offset points',
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

        # 3. 가격 차트 + SMA + 포지션
        ax3 = fig.add_subplot(gs[2, :2])
        ax3.plot(df.index, df['Close'], label='BTC Price', color='gray', linewidth=1.5, alpha=0.6)
        ax3.plot(df.index, df['SMA'], label='SMA 30', color='orange', linewidth=2, alpha=0.8)

        # 매수/매도 신호
        buy_signals = df[df['position_change'] == 1].index
        sell_signals = df[df['position_change'] == -1].index

        ax3.scatter(buy_signals, df.loc[buy_signals, 'Close'],
                    color='green', marker='^', s=100, label='Buy Signal', zorder=5, alpha=0.7)
        ax3.scatter(sell_signals, df.loc[sell_signals, 'Close'],
                    color='red', marker='v', s=100, label='Sell Signal', zorder=5, alpha=0.7)

        ax3.set_title('BTC Price with SMA30 and Signals', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Price (KRW, Log Scale)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.legend(loc='upper left', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # 4. 포지션 히트맵
        ax4 = fig.add_subplot(gs[2, 2])
        position_pct = df['position'].resample('M').mean() * 100
        position_pivot = position_pct.to_frame('position')
        position_pivot['year'] = position_pivot.index.year
        position_pivot['month'] = position_pivot.index.month
        heatmap_data = position_pivot.pivot(index='year', columns='month', values='position')

        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn', center=50,
                    cbar_kws={'label': 'Position (%)'}, ax=ax4, linewidths=0.5)
        ax4.set_title('Monthly Position Rate (%)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Month', fontsize=10)
        ax4.set_ylabel('Year', fontsize=10)

        # 5. 월별 수익률 히트맵
        ax5 = fig.add_subplot(gs[3, :2])
        monthly_returns = df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_returns_pivot = monthly_returns.to_frame('returns')
        monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
        heatmap_data = monthly_returns_pivot.pivot(index='year', columns='month', values='returns')

        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Return (%)'}, ax=ax5, linewidths=0.5)
        ax5.set_title('Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold')
        ax5.set_xlabel('Month', fontsize=11)
        ax5.set_ylabel('Year', fontsize=11)

        # 6. 성과 지표
        ax6 = fig.add_subplot(gs[3, 2])
        ax6.axis('off')

        # 전략 성과
        strategy_metrics = self.calculate_metrics(df['returns'], 'Strategy')
        bnh_metrics = self.calculate_metrics(df['bnh_returns'], 'Buy & Hold')

        metrics_text = "Performance Metrics\n" + "=" * 35 + "\n\n"
        metrics_text += "Previous Close > SMA30 Strategy:\n"
        metrics_text += f"  Total Return: {strategy_metrics['Total Return (%)']:.2f}%\n"
        metrics_text += f"  CAGR: {strategy_metrics['CAGR (%)']:.2f}%\n"
        metrics_text += f"  MDD: {strategy_metrics['MDD (%)']:.2f}%\n"
        metrics_text += f"  Sharpe Ratio: {strategy_metrics['Sharpe Ratio']:.2f}\n"
        metrics_text += f"  Win Rate: {strategy_metrics['Win Rate (%)']:.2f}%\n"
        metrics_text += f"  Total Trades: {strategy_metrics['Total Trades']}\n"
        if strategy_metrics['Profit Factor'] != np.inf:
            metrics_text += f"  Profit Factor: {strategy_metrics['Profit Factor']:.2f}\n"

        metrics_text += "\nBuy & Hold:\n"
        metrics_text += f"  Total Return: {bnh_metrics['Total Return (%)']:.2f}%\n"
        metrics_text += f"  CAGR: {bnh_metrics['CAGR (%)']:.2f}%\n"
        metrics_text += f"  MDD: {bnh_metrics['MDD (%)']:.2f}%\n"
        metrics_text += f"  Sharpe Ratio: {bnh_metrics['Sharpe Ratio']:.2f}\n"

        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 전체 제목
        fig.suptitle(f'BTC Previous Close > SMA30 Strategy Backtest\n'
                     f'Period: {self.start_date} to {self.end_date}',
                     fontsize=16, fontweight='bold', y=0.995)

        # 저장
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics(self):
        """성과 지표 출력"""
        df = self.result

        print("\n" + "=" * 100)
        print(f"{'BTC Previous Close > SMA30 Strategy Backtest Results':^100}")
        print("=" * 100)
        print(f"\nPeriod: {self.start_date} ~ {self.end_date}")
        print(f"Symbol: {self.symbol}")
        print(f"Slippage: {self.slippage * 100}%")

        # 전략 성과
        strategy_metrics = self.calculate_metrics(df['returns'], 'Previous Close > SMA30')
        bnh_metrics = self.calculate_metrics(df['bnh_returns'], 'Buy & Hold')

        print("\n" + "-" * 100)
        print(f"{'Strategy Performance':^100}")
        print("-" * 100)

        metrics_df = pd.DataFrame([strategy_metrics, bnh_metrics])
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 100)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}')
        print(metrics_df.to_string(index=False))

        print("\n" + "=" * 100 + "\n")

    def run_backtest(self):
        """백테스트 실행"""
        print("\n" + "=" * 80)
        print("Starting BTC Previous Close > SMA30 Strategy Backtest...")
        print("=" * 80 + "\n")

        # 1. 데이터 로드
        self.load_data()

        # 2. 전략 실행
        print("Running strategy...")
        self.result = self.strategy_previous_close_sma30(sma_period=30)
        print("Strategy execution completed!\n")

        # 3. 성과 지표 출력
        self.print_metrics()

        # 4. 시각화
        print("Creating visualization...")
        self.plot_results()

        # 5. 결과 저장
        print("\nSaving results to CSV...")
        self.result.to_csv('btc_previous_close_sma30_results.csv')
        print("Results saved to btc_previous_close_sma30_results.csv")

        print("\n" + "=" * 80)
        print("Backtest completed!")
        print("=" * 80 + "\n")


def main():
    """메인 함수"""
    # 백테스트 실행
    backtest = BTCPreviousCloseSMA30Strategy(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    backtest.run_backtest()


if __name__ == "__main__":
    main()
