"""
BTC 변동성 돌파 전략 백테스트

전략 설명:
- 래리 윌리엄스의 변동성 돌파 전략
- 목표가 = 당일 시가 + (전일 고가 - 전일 저가) * k
- 목표가 돌파 시 매수, 당일 종가에 매도
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ccxt
from datetime import datetime
import time
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class VolatilityBreakoutBacktest:
    """변동성 돌파 전략 백테스트 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01', end_date=None, k=0.5, data_source='file'):
        """
        Args:
            symbol: 티커 심볼 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            k: 변동성 계수 (default: 0.5)
            data_source: 'file' or 'api' (default: 'file')
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.k = k
        self.data = None
        self.results = None
        self.data_source = data_source

    def fetch_data(self):
        """데이터 로드"""
        if self.data_source == 'file':
            # 로컬 파일에서 데이터 읽기
            file_path = f'chart_day/{self.symbol}.parquet'
            print(f"Loading data from {file_path}...")
            df = pd.read_parquet(file_path)

            # 컬럼명 변경 (소문자 -> 대문자)
            df.columns = [col.capitalize() for col in df.columns]

            # 날짜 필터링
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            self.data = df
            print(f"Loaded {len(self.data)} data points from {df.index[0]} to {df.index[-1]}")

        else:
            # API에서 데이터 가져오기
            exchange = ccxt.binance()
            print(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}...")

            # 시작일을 밀리초로 변환
            since = exchange.parse8601(self.start_date + 'T00:00:00Z')
            end_timestamp = exchange.parse8601(self.end_date + 'T23:59:59Z')

            all_data = []
            timeframe = '1d'

            while since < end_timestamp:
                try:
                    ohlcv = exchange.fetch_ohlcv(self.symbol, timeframe, since, limit=1000)
                    if len(ohlcv) == 0:
                        break

                    all_data.extend(ohlcv)
                    since = ohlcv[-1][0] + 86400000
                    time.sleep(0.1)

                    if len(ohlcv) < 1000:
                        break
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    break

            df = pd.DataFrame(all_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]

            self.data = df
            print(f"Downloaded {len(self.data)} data points")

        return self.data

    def calculate_signals(self, slippage=0.002):
        """
        변동성 돌파 전략 신호 계산

        Args:
            slippage: 슬리피지 (default: 0.2%)
        """
        df = self.data.copy()

        # 전일 변동성 계산
        df['prev_high'] = df['High'].shift(1)
        df['prev_low'] = df['Low'].shift(1)
        df['prev_close'] = df['Close'].shift(1)
        df['volatility'] = df['prev_high'] - df['prev_low']

        # 목표가 계산
        df['target_price'] = df['Open'] + df['volatility'] * self.k

        # 매수 신호: 고가가 목표가를 돌파
        df['buy_signal'] = df['High'] >= df['target_price']

        # 매수가: 목표가 돌파 시 목표가로 매수 (실제로는 돌파 직후)
        df['buy_price'] = np.where(df['buy_signal'], df['target_price'], np.nan)

        # 매도가: 당일 종가에 매도
        df['sell_price'] = np.where(df['buy_signal'], df['Close'], np.nan)

        # 일일 수익률 계산 (슬리피지 반영)
        df['daily_return'] = np.where(
            df['buy_signal'],
            (df['sell_price'] / df['buy_price'] - 1) - slippage,
            0
        )

        # 누적 수익률
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()

        # Buy & Hold 수익률
        df['buy_hold_return'] = df['Close'] / df['Close'].iloc[0]

        self.results = df
        return df

    def calculate_performance_metrics(self):
        """성과 지표 계산"""
        df = self.results

        # 거래 통계
        total_trades = df['buy_signal'].sum()
        winning_trades = (df['daily_return'] > 0).sum()
        losing_trades = (df['daily_return'] < 0).sum()
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # 수익률 통계
        total_return = (df['cumulative_return'].iloc[-1] - 1) * 100
        buy_hold_return = (df['buy_hold_return'].iloc[-1] - 1) * 100

        # 연간 수익률 (CAGR)
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['cumulative_return'].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
        buy_hold_cagr = (df['buy_hold_return'].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # 최대 낙폭 (MDD)
        cummax = df['cumulative_return'].cummax()
        drawdown = (df['cumulative_return'] - cummax) / cummax
        mdd = drawdown.min() * 100

        # Buy & Hold MDD
        cummax_bh = df['buy_hold_return'].cummax()
        drawdown_bh = (df['buy_hold_return'] - cummax_bh) / cummax_bh
        mdd_bh = drawdown_bh.min() * 100

        # 샤프 비율 (일간 수익률 기준, 연율화)
        daily_returns = df['daily_return']
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0

        # 승리/패배 평균
        avg_win = df[df['daily_return'] > 0]['daily_return'].mean() * 100 if winning_trades > 0 else 0
        avg_loss = df[df['daily_return'] < 0]['daily_return'].mean() * 100 if losing_trades > 0 else 0

        # Profit Factor
        total_profit = df[df['daily_return'] > 0]['daily_return'].sum()
        total_loss = abs(df[df['daily_return'] < 0]['daily_return'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        metrics = {
            '전략': {
                '총 수익률 (%)': f'{total_return:.2f}',
                'CAGR (%)': f'{cagr:.2f}',
                'MDD (%)': f'{mdd:.2f}',
                '샤프 비율': f'{sharpe:.2f}',
                '총 거래 횟수': int(total_trades),
                '승률 (%)': f'{win_rate:.2f}',
                '평균 수익 (%)': f'{avg_win:.2f}',
                '평균 손실 (%)': f'{avg_loss:.2f}',
                'Profit Factor': f'{profit_factor:.2f}',
            },
            'Buy & Hold': {
                '총 수익률 (%)': f'{buy_hold_return:.2f}',
                'CAGR (%)': f'{buy_hold_cagr:.2f}',
                'MDD (%)': f'{mdd_bh:.2f}',
            }
        }

        return metrics

    def plot_results(self, save_path='backtest_results.png'):
        """결과 시각화"""
        df = self.results

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # 1. 가격 차트 + 매매 신호
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['Close'], label='BTC Price', linewidth=1, alpha=0.7)

        # 매수 신호 표시
        buy_signals = df[df['buy_signal'] == True]
        ax1.scatter(buy_signals.index, buy_signals['buy_price'],
                   color='green', marker='^', s=50, alpha=0.6, label='Buy Signal', zorder=5)

        ax1.set_title(f'{self.symbol} - Volatility Breakout Strategy (k={self.k})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=11)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 누적 수익률 비교
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(df.index, df['cumulative_return'], label='Volatility Breakout Strategy', linewidth=2)
        ax2.plot(df.index, df['buy_hold_return'], label='Buy & Hold', linewidth=2, alpha=0.7)
        ax2.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cumulative Return', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 3. 드로우다운
        ax3 = fig.add_subplot(gs[2, :])
        cummax = df['cumulative_return'].cummax()
        drawdown = (df['cumulative_return'] - cummax) / cummax * 100

        cummax_bh = df['buy_hold_return'].cummax()
        drawdown_bh = (df['buy_hold_return'] - cummax_bh) / cummax_bh * 100

        ax3.fill_between(df.index, drawdown, 0, alpha=0.3, color='red', label='Strategy Drawdown')
        ax3.plot(df.index, drawdown_bh, color='blue', alpha=0.5, label='Buy & Hold Drawdown')
        ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 월별 수익률 히트맵
        ax4 = fig.add_subplot(gs[3, 0])
        df_monthly = df.copy()
        df_monthly['year'] = df_monthly.index.year
        df_monthly['month'] = df_monthly.index.month

        # 월별 수익률 계산
        monthly_returns = df_monthly.groupby(['year', 'month'])['daily_return'].sum().reset_index()
        monthly_returns['monthly_return'] = monthly_returns['daily_return'] * 100

        pivot_table = monthly_returns.pivot_table(
            values='monthly_return',
            index='year',
            columns='month',
            aggfunc='first'
        )

        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax4, linewidths=0.5)
        ax4.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Month', fontsize=10)
        ax4.set_ylabel('Year', fontsize=10)

        # 5. 수익률 분포
        ax5 = fig.add_subplot(gs[3, 1])
        returns = df[df['daily_return'] != 0]['daily_return'] * 100
        ax5.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax5.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax5.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax5.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Return (%)', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.show()

    def print_metrics(self, metrics):
        """성과 지표 출력"""
        print("\n" + "="*80)
        print(f"{'변동성 돌파 전략 백테스트 결과':^80}")
        print("="*80)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {self.symbol}")
        print(f"변동성 계수 (k): {self.k}")
        print("\n" + "-"*80)

        # 전략 성과
        print(f"\n{'[전략 성과]':^80}")
        print("-"*80)
        for key, value in metrics['전략'].items():
            print(f"{key:<25} : {value:>15}")

        # Buy & Hold 비교
        print(f"\n{'[Buy & Hold 비교]':^80}")
        print("-"*80)
        for key, value in metrics['Buy & Hold'].items():
            print(f"{key:<25} : {value:>15}")

        print("\n" + "="*80 + "\n")

    def run(self, slippage=0.002, save_path='backtest_results.png'):
        """백테스트 실행"""
        # 데이터 다운로드
        self.fetch_data()

        # 신호 계산
        self.calculate_signals(slippage=slippage)

        # 성과 지표 계산
        metrics = self.calculate_performance_metrics()

        # 결과 출력
        self.print_metrics(metrics)

        # 시각화
        self.plot_results(save_path=save_path)

        return self.results, metrics


def main():
    """메인 함수"""
    # 백테스트 실행
    backtest = VolatilityBreakoutBacktest(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,  # 현재까지
        k=0.5,  # 변동성 계수
        data_source='file'  # 로컬 파일 사용
    )

    results, metrics = backtest.run(slippage=0.002)

    # 결과 저장
    print("\nSaving results to CSV...")
    results.to_csv('backtest_results.csv')
    print("Results saved to backtest_results.csv")

    # k 값 변화에 따른 성과 비교
    print("\n" + "="*80)
    print(f"{'k 값 변화에 따른 성과 비교':^80}")
    print("="*80)

    k_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    comparison_results = []

    for k in k_values:
        bt = VolatilityBreakoutBacktest(symbol='BTC_KRW', start_date='2018-01-01', k=k, data_source='file')
        bt.fetch_data()
        bt.calculate_signals(slippage=0.002)
        metrics = bt.calculate_performance_metrics()

        comparison_results.append({
            'k': k,
            '총 수익률': metrics['전략']['총 수익률 (%)'],
            'CAGR': metrics['전략']['CAGR (%)'],
            'MDD': metrics['전략']['MDD (%)'],
            '샤프 비율': metrics['전략']['샤프 비율'],
            '승률': metrics['전략']['승률 (%)'],
            '총 거래': metrics['전략']['총 거래 횟수'],
            'Profit Factor': metrics['전략']['Profit Factor']
        })

    df_comparison = pd.DataFrame(comparison_results)
    print("\n", df_comparison.to_string(index=False))
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
