"""
BTC SMA 30 교차 전략 백테스트

전략 설명:
- 가격이 SMA 30 이상일 때 매수 (보유)
- 가격이 SMA 30 미만일 때 매도 후 현금 보유
- 슬리피지 0.2% 반영
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


class SMABacktest:
    """SMA 30 교차 전략 백테스트 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01', end_date=None, sma_period=30, data_source='file'):
        """
        Args:
            symbol: 티커 심볼 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            sma_period: SMA 기간 (default: 30)
            data_source: 'file' or 'api' (default: 'file')
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.sma_period = sma_period
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
        SMA 30 교차 전략 신호 계산

        Args:
            slippage: 슬리피지 (default: 0.2%)
        """
        self.slippage = slippage  # 슬리피지를 저장
        df = self.data.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=self.sma_period).mean()

        # 포지션 계산: 가격이 SMA 이상이면 1 (매수), 미만이면 0 (매도/현금)
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 매수 신호: 0에서 1로 변경 (포지션 진입)
        df['buy_signal'] = df['position_change'] == 1

        # 매도 신호: 1에서 0으로 변경 (포지션 청산)
        df['sell_signal'] = df['position_change'] == -1

        # 매수가/매도가 (종가 기준)
        df['buy_price'] = np.where(df['buy_signal'], df['Close'], np.nan)
        df['sell_price'] = np.where(df['sell_signal'], df['Close'], np.nan)

        # 일일 수익률 계산
        # 포지션을 보유 중일 때는 가격 변동률, 현금 보유 중일 때는 0
        df['daily_price_return'] = df['Close'].pct_change()
        df['daily_return'] = df['position'].shift(1) * df['daily_price_return']

        # 매수/매도 시 슬리피지 적용 (매수 시 -슬리피지, 매도 시 -슬리피지)
        # 매수 다음날부터 포지션 진입 효과가 반영되므로, 매수 당일에 슬리피지 차감
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['buy_signal']] = -slippage
        slippage_cost[df['sell_signal']] = -slippage

        df['daily_return'] = df['daily_return'] + slippage_cost

        # 누적 수익률
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()

        # Buy & Hold 수익률
        df['buy_hold_return'] = df['Close'] / df['Close'].iloc[0]

        self.results = df
        return df

    def calculate_performance_metrics(self):
        """성과 지표 계산"""
        df = self.results.copy()
        # NaN 값이 있는 초기 기간 제거 (SMA 계산을 위한 기간)
        df = df[df['SMA'].notna() & df['cumulative_return'].notna()]

        if len(df) == 0:
            raise ValueError("No valid data after removing NaN values")

        # 거래 통계
        total_buy_trades = df['buy_signal'].sum()
        total_sell_trades = df['sell_signal'].sum()

        # 보유 기간 통계
        holding_days = (df['position'] == 1).sum()
        cash_days = (df['position'] == 0).sum()

        # 수익 거래 계산 (매도 시점의 누적 수익 기준)
        winning_trades = 0
        losing_trades = 0

        # 매도 시점에서 이전 매수 대비 수익 계산
        for idx in df[df['sell_signal']].index:
            # 이전 매수 시점 찾기
            buy_dates = df[(df['buy_signal']) & (df.index < idx)].index
            if len(buy_dates) > 0:
                last_buy_date = buy_dates[-1]
                buy_price = df.loc[last_buy_date, 'Close']
                sell_price = df.loc[idx, 'Close']
                # 슬리피지를 고려한 수익률 계산
                trade_return = (sell_price / buy_price - 1) - (2 * self.slippage)
                if trade_return > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

        total_completed_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_completed_trades * 100 if total_completed_trades > 0 else 0

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

        # 평균 수익/손실
        avg_win = df[df['daily_return'] > 0]['daily_return'].mean() * 100 if (df['daily_return'] > 0).sum() > 0 else 0
        avg_loss = df[df['daily_return'] < 0]['daily_return'].mean() * 100 if (df['daily_return'] < 0).sum() > 0 else 0

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
                '총 매수 횟수': int(total_buy_trades),
                '총 매도 횟수': int(total_sell_trades),
                '완료된 거래': int(total_completed_trades),
                '승률 (%)': f'{win_rate:.2f}',
                '평균 수익 (%)': f'{avg_win:.2f}',
                '평균 손실 (%)': f'{avg_loss:.2f}',
                'Profit Factor': f'{profit_factor:.2f}',
                '보유 기간 (일)': int(holding_days),
                '현금 보유 (일)': int(cash_days),
            },
            'Buy & Hold': {
                '총 수익률 (%)': f'{buy_hold_return:.2f}',
                'CAGR (%)': f'{buy_hold_cagr:.2f}',
                'MDD (%)': f'{mdd_bh:.2f}',
            }
        }

        return metrics

    def plot_results(self, save_path='sma_backtest_results.png'):
        """결과 시각화"""
        df = self.results.copy()
        # NaN 값이 있는 초기 기간 제거 (SMA 계산을 위한 기간)
        df = df[df['SMA'].notna() & df['cumulative_return'].notna()]

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # 1. 가격 차트 + SMA + 매매 신호
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['Close'], label='BTC Price', linewidth=1.5, alpha=0.8)
        ax1.plot(df.index, df['SMA'], label=f'SMA {self.sma_period}', linewidth=1.5, alpha=0.8, color='orange')

        # 매수 신호 표시
        buy_signals = df[df['buy_signal'] == True]
        ax1.scatter(buy_signals.index, buy_signals['Close'],
                   color='green', marker='^', s=100, alpha=0.8, label='Buy Signal', zorder=5)

        # 매도 신호 표시
        sell_signals = df[df['sell_signal'] == True]
        ax1.scatter(sell_signals.index, sell_signals['Close'],
                   color='red', marker='v', s=100, alpha=0.8, label='Sell Signal', zorder=5)

        ax1.set_title(f'{self.symbol} - SMA {self.sma_period} Crossover Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (KRW)', fontsize=11)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 누적 수익률 비교
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(df.index, df['cumulative_return'], label='SMA Strategy', linewidth=2)
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
        print(f"{'SMA {self.sma_period} 교차 전략 백테스트 결과':^80}")
        print("="*80)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {self.symbol}")
        print(f"SMA 기간: {self.sma_period}")
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

    def run(self, slippage=0.002, save_path='sma_backtest_results.png'):
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
    backtest = SMABacktest(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,  # 현재까지
        sma_period=30,  # SMA 30
        data_source='file'  # 로컬 파일 사용
    )

    results, metrics = backtest.run(slippage=0.002)  # 슬리피지 0.2%

    # 결과 저장
    print("\nSaving results to CSV...")
    results.to_csv('sma_backtest_results.csv')
    print("Results saved to sma_backtest_results.csv")

    # SMA 기간 변화에 따른 성과 비교
    print("\n" + "="*80)
    print(f"{'SMA 기간 변화에 따른 성과 비교':^80}")
    print("="*80)

    sma_periods = [10, 20, 30, 50, 100]
    comparison_results = []

    for period in sma_periods:
        bt = SMABacktest(symbol='BTC_KRW', start_date='2018-01-01', sma_period=period, data_source='file')
        bt.fetch_data()
        bt.calculate_signals(slippage=0.002)
        metrics = bt.calculate_performance_metrics()

        comparison_results.append({
            'SMA 기간': period,
            '총 수익률': metrics['전략']['총 수익률 (%)'],
            'CAGR': metrics['전략']['CAGR (%)'],
            'MDD': metrics['전략']['MDD (%)'],
            '샤프 비율': metrics['전략']['샤프 비율'],
            '승률': metrics['전략']['승률 (%)'],
            '완료된 거래': metrics['전략']['완료된 거래'],
            'Profit Factor': metrics['전략']['Profit Factor']
        })

    df_comparison = pd.DataFrame(comparison_results)
    print("\n", df_comparison.to_string(index=False))
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
