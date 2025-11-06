"""
BTC 이격도 돌파 전략 백테스트

전략 설명:
- 이격도(Disparity Index) = (현재가 / 이동평균) * 100
- 이격도가 기준선을 돌파하면 매수
- 이격도가 기준선 아래로 내려가면 매도 후 현금 보유
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


class DisparityIndexBacktest:
    """이격도 돌파 전략 백테스트 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01', end_date=None,
                 ma_period=20, disparity_threshold=100, data_source='file'):
        """
        Args:
            symbol: 티커 심볼 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            ma_period: 이동평균 기간 (default: 20일)
            disparity_threshold: 이격도 돌파 기준선 (default: 100)
            data_source: 'file' or 'api' (default: 'file')
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.ma_period = ma_period
        self.disparity_threshold = disparity_threshold
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
        이격도 돌파 전략 신호 계산

        Args:
            slippage: 슬리피지 (default: 0.2%)
        """
        df = self.data.copy()

        # 이동평균 계산
        df['MA'] = df['Close'].rolling(window=self.ma_period).mean()

        # 이격도 계산: (현재가 / 이동평균) * 100
        df['Disparity'] = (df['Close'] / df['MA']) * 100

        # 포지션 설정
        # 이격도가 기준선보다 높으면 매수, 낮으면 현금 보유
        df['position'] = np.where(df['Disparity'] > self.disparity_threshold, 1, 0)

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 매수 신호: 포지션이 0에서 1로 변경
        df['buy_signal'] = (df['position_change'] == 1).astype(int)

        # 매도 신호: 포지션이 1에서 0으로 변경
        df['sell_signal'] = (df['position_change'] == -1).astype(int)

        # 실제 매매 시뮬레이션
        position = 0  # 0: 현금 보유, 1: 비트코인 보유
        buy_price = 0
        trades = []

        for idx, row in df.iterrows():
            if row['buy_signal'] == 1 and position == 0:
                # 매수 실행 (슬리피지 적용)
                buy_price = row['Close'] * (1 + slippage)
                position = 1
                trades.append({
                    'date': idx,
                    'type': 'buy',
                    'price': buy_price,
                    'disparity': row['Disparity']
                })
            elif row['sell_signal'] == 1 and position == 1:
                # 매도 실행 (슬리피지 적용)
                sell_price = row['Close'] * (1 - slippage)
                position = 0
                trade_return = (sell_price / buy_price - 1)
                trades.append({
                    'date': idx,
                    'type': 'sell',
                    'price': sell_price,
                    'return': trade_return,
                    'disparity': row['Disparity']
                })

        # 일별 수익률 계산
        df['daily_return'] = 0.0
        df['cumulative_return'] = 1.0

        current_position = 0
        current_buy_price = 0
        cumulative = 1.0

        for idx, row in df.iterrows():
            daily_return = 0.0

            if row['buy_signal'] == 1 and current_position == 0:
                # 매수
                current_buy_price = row['Close'] * (1 + slippage)
                current_position = 1
            elif row['sell_signal'] == 1 and current_position == 1:
                # 매도
                sell_price = row['Close'] * (1 - slippage)
                daily_return = (sell_price / current_buy_price - 1)
                cumulative *= (1 + daily_return)
                current_position = 0
                current_buy_price = 0

            df.at[idx, 'daily_return'] = daily_return
            df.at[idx, 'cumulative_return'] = cumulative

        # Buy & Hold 수익률
        df['buy_hold_return'] = df['Close'] / df['Close'].iloc[0]

        self.results = df
        self.trades = trades
        return df

    def calculate_performance_metrics(self):
        """성과 지표 계산"""
        df = self.results
        trades = self.trades

        # 거래 통계
        sell_trades = [t for t in trades if t['type'] == 'sell']
        total_trades = len(sell_trades)
        winning_trades = len([t for t in sell_trades if t['return'] > 0])
        losing_trades = len([t for t in sell_trades if t['return'] < 0])
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
        returns = [t['return'] for t in sell_trades]
        if len(returns) > 0 and np.std(returns) > 0:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            # 거래 기반 샤프 비율 근사치
            sharpe = (avg_return / std_return) * np.sqrt(252 / len(df) * total_trades)
        else:
            sharpe = 0

        # 승리/패배 평균
        avg_win = np.mean([t['return'] * 100 for t in sell_trades if t['return'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['return'] * 100 for t in sell_trades if t['return'] < 0]) if losing_trades > 0 else 0

        # Profit Factor
        total_profit = sum([t['return'] for t in sell_trades if t['return'] > 0])
        total_loss = abs(sum([t['return'] for t in sell_trades if t['return'] < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        # 보유 비율 계산
        holding_days = df['position'].sum()
        total_days = len(df)
        holding_ratio = holding_days / total_days * 100

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
                '보유 비율 (%)': f'{holding_ratio:.2f}',
            },
            'Buy & Hold': {
                '총 수익률 (%)': f'{buy_hold_return:.2f}',
                'CAGR (%)': f'{buy_hold_cagr:.2f}',
                'MDD (%)': f'{mdd_bh:.2f}',
            }
        }

        return metrics

    def plot_results(self, save_path='disparity_backtest_results.png'):
        """결과 시각화"""
        df = self.results

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # 1. 가격 차트 + 매매 신호
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['Close'], label='BTC Price', linewidth=1, alpha=0.7)

        # 매수/매도 신호 표시
        buy_signals = df[df['buy_signal'] == 1]
        sell_signals = df[df['sell_signal'] == 1]

        ax1.scatter(buy_signals.index, buy_signals['Close'],
                   color='green', marker='^', s=100, alpha=0.7, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['Close'],
                   color='red', marker='v', s=100, alpha=0.7, label='Sell Signal', zorder=5)

        ax1.set_title(f'{self.symbol} - Disparity Index Strategy (MA={self.ma_period}, Threshold={self.disparity_threshold})',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (KRW)', fontsize=11)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 이격도 차트
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(df.index, df['Disparity'], label='Disparity Index', linewidth=1, color='purple')
        ax2.axhline(y=self.disparity_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold ({self.disparity_threshold})')

        # 매수/매도 구간 표시
        ax2.fill_between(df.index, df['Disparity'], self.disparity_threshold,
                        where=(df['position'] == 1), alpha=0.3, color='green', label='Long Position')

        ax2.set_title('Disparity Index', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Disparity Index', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 누적 수익률 비교
        ax3 = fig.add_subplot(gs[2, :])
        ax3.plot(df.index, df['cumulative_return'], label='Disparity Index Strategy', linewidth=2)
        ax3.plot(df.index, df['buy_hold_return'], label='Buy & Hold', linewidth=2, alpha=0.7)
        ax3.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Cumulative Return', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # 4. 드로우다운
        ax4 = fig.add_subplot(gs[3, 0])
        cummax = df['cumulative_return'].cummax()
        drawdown = (df['cumulative_return'] - cummax) / cummax * 100

        cummax_bh = df['buy_hold_return'].cummax()
        drawdown_bh = (df['buy_hold_return'] - cummax_bh) / cummax_bh * 100

        ax4.fill_between(df.index, drawdown, 0, alpha=0.3, color='red', label='Strategy Drawdown')
        ax4.plot(df.index, drawdown_bh, color='blue', alpha=0.5, label='Buy & Hold Drawdown')
        ax4.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Drawdown (%)', fontsize=11)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 수익률 분포
        ax5 = fig.add_subplot(gs[3, 1])
        sell_trades = [t for t in self.trades if t['type'] == 'sell']
        if len(sell_trades) > 0:
            returns = [t['return'] * 100 for t in sell_trades]
            ax5.hist(returns, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax5.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(returns):.2f}%')
            ax5.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax5.set_title('Trade Returns Distribution', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Return (%)', fontsize=10)
            ax5.set_ylabel('Frequency', fontsize=10)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics(self, metrics):
        """성과 지표 출력"""
        print("\n" + "="*80)
        print(f"{'이격도 돌파 전략 백테스트 결과':^80}")
        print("="*80)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {self.symbol}")
        print(f"이동평균 기간: {self.ma_period}일")
        print(f"이격도 임계값: {self.disparity_threshold}")
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

    def run(self, slippage=0.002, save_path='disparity_backtest_results.png'):
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
    backtest = DisparityIndexBacktest(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,  # 현재까지
        ma_period=20,  # 20일 이동평균
        disparity_threshold=100,  # 이격도 기준선 100
        data_source='file'  # 로컬 파일 사용
    )

    results, metrics = backtest.run(slippage=0.002)  # 슬리피지 0.2%

    # 결과 저장
    print("\nSaving results to CSV...")
    results.to_csv('disparity_backtest_results.csv')
    print("Results saved to disparity_backtest_results.csv")

    # 이동평균 기간에 따른 성과 비교
    print("\n" + "="*80)
    print(f"{'이동평균 기간에 따른 성과 비교':^80}")
    print("="*80)

    ma_periods = [10, 20, 30, 40, 50]
    comparison_results = []

    for ma_period in ma_periods:
        bt = DisparityIndexBacktest(
            symbol='BTC_KRW',
            start_date='2018-01-01',
            ma_period=ma_period,
            disparity_threshold=100,
            data_source='file'
        )
        bt.fetch_data()
        bt.calculate_signals(slippage=0.002)
        metrics = bt.calculate_performance_metrics()

        comparison_results.append({
            'MA 기간': ma_period,
            '총 수익률': metrics['전략']['총 수익률 (%)'],
            'CAGR': metrics['전략']['CAGR (%)'],
            'MDD': metrics['전략']['MDD (%)'],
            '샤프 비율': metrics['전략']['샤프 비율'],
            '승률': metrics['전략']['승률 (%)'],
            '총 거래': metrics['전략']['총 거래 횟수'],
            '보유 비율': metrics['전략']['보유 비율 (%)'],
            'Profit Factor': metrics['전략']['Profit Factor']
        })

    df_comparison = pd.DataFrame(comparison_results)
    print("\n", df_comparison.to_string(index=False))
    print("\n" + "="*80 + "\n")

    # 이격도 임계값에 따른 성과 비교
    print("\n" + "="*80)
    print(f"{'이격도 임계값에 따른 성과 비교':^80}")
    print("="*80)

    thresholds = [95, 100, 105, 110, 115]
    comparison_results2 = []

    for threshold in thresholds:
        bt = DisparityIndexBacktest(
            symbol='BTC_KRW',
            start_date='2018-01-01',
            ma_period=20,
            disparity_threshold=threshold,
            data_source='file'
        )
        bt.fetch_data()
        bt.calculate_signals(slippage=0.002)
        metrics = bt.calculate_performance_metrics()

        comparison_results2.append({
            '임계값': threshold,
            '총 수익률': metrics['전략']['총 수익률 (%)'],
            'CAGR': metrics['전략']['CAGR (%)'],
            'MDD': metrics['전략']['MDD (%)'],
            '샤프 비율': metrics['전략']['샤프 비율'],
            '승률': metrics['전략']['승률 (%)'],
            '총 거래': metrics['전략']['총 거래 횟수'],
            '보유 비율': metrics['전략']['보유 비율 (%)'],
            'Profit Factor': metrics['전략']['Profit Factor']
        })

    df_comparison2 = pd.DataFrame(comparison_results2)
    print("\n", df_comparison2.to_string(index=False))
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
