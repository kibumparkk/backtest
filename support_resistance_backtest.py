"""
BTC 지지선/저항선 트레이딩 전략 백테스트

전략 설명:
- 지지선(Support): 일정 기간 동안의 최저가를 기준으로 계산
- 저항선(Resistance): 일정 기간 동안의 최고가를 기준으로 계산
- 매수 신호: 가격이 지지선 근처(지지선 ± threshold%)에 도달할 때
- 매도 신호: 가격이 저항선 근처(저항선 ± threshold%)에 도달할 때
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


class SupportResistanceBacktest:
    """지지선/저항선 트레이딩 전략 백테스트 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01', end_date=None,
                 lookback_period=20, threshold=0.02, data_source='file'):
        """
        Args:
            symbol: 티커 심볼 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            lookback_period: 지지/저항선 계산을 위한 과거 기간 (default: 20일)
            threshold: 지지/저항선 근처 판단 임계값 (default: 2% = 0.02)
            data_source: 'file' or 'api' (default: 'file')
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.lookback_period = lookback_period
        self.threshold = threshold
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

    def calculate_support_resistance(self, df):
        """
        지지선과 저항선 계산

        Args:
            df: OHLCV 데이터프레임

        Returns:
            지지선과 저항선이 추가된 데이터프레임
        """
        # 지지선: lookback_period 기간 동안의 최저가
        df['support'] = df['Low'].rolling(window=self.lookback_period, min_periods=1).min()

        # 저항선: lookback_period 기간 동안의 최고가
        df['resistance'] = df['High'].rolling(window=self.lookback_period, min_periods=1).max()

        # 지지선/저항선 범위 계산
        df['support_upper'] = df['support'] * (1 + self.threshold)
        df['support_lower'] = df['support'] * (1 - self.threshold)
        df['resistance_upper'] = df['resistance'] * (1 + self.threshold)
        df['resistance_lower'] = df['resistance'] * (1 - self.threshold)

        return df

    def calculate_signals(self, slippage=0.002):
        """
        지지선/저항선 전략 신호 계산

        Args:
            slippage: 슬리피지 (default: 0.2%)
        """
        df = self.data.copy()

        # 지지선/저항선 계산
        df = self.calculate_support_resistance(df)

        # 매수/매도 신호 초기화
        df['signal'] = 0  # 0: 현금, 1: 보유
        df['buy_signal'] = False
        df['sell_signal'] = False

        # 포지션 추적
        position = 0  # 0: 현금, 1: 보유

        for i in range(1, len(df)):
            current_close = df['Close'].iloc[i]
            prev_close = df['Close'].iloc[i-1]

            # 현재 포지션이 없을 때 (현금 보유)
            if position == 0:
                # 매수 조건: 가격이 지지선 범위 내에 있을 때
                if df['support_lower'].iloc[i] <= current_close <= df['support_upper'].iloc[i]:
                    df.iloc[i, df.columns.get_loc('buy_signal')] = True
                    position = 1

            # 현재 포지션이 있을 때 (코인 보유)
            elif position == 1:
                # 매도 조건: 가격이 저항선 범위 내에 있을 때
                if df['resistance_lower'].iloc[i] <= current_close <= df['resistance_upper'].iloc[i]:
                    df.iloc[i, df.columns.get_loc('sell_signal')] = True
                    position = 0

            # 포지션 상태 기록
            df.iloc[i, df.columns.get_loc('signal')] = position

        # 슬리피지를 반영한 실제 거래가
        df['trade_price'] = df['Close'].copy()
        df.loc[df['buy_signal'], 'trade_price'] = df.loc[df['buy_signal'], 'Close'] * (1 + slippage)
        df.loc[df['sell_signal'], 'trade_price'] = df.loc[df['sell_signal'], 'Close'] * (1 - slippage)

        # 일일 수익률 계산
        df['daily_return'] = 0.0
        df['portfolio_value'] = 1.0  # 초기 자본 1.0

        portfolio = 1.0
        last_buy_price = 0

        for i in range(1, len(df)):
            if df['buy_signal'].iloc[i]:
                # 매수: 슬리피지 비용 반영
                last_buy_price = df['trade_price'].iloc[i]
                df.iloc[i, df.columns.get_loc('daily_return')] = -slippage
                portfolio = portfolio * (1 - slippage)

            elif df['sell_signal'].iloc[i]:
                # 매도: 수익 실현
                if last_buy_price > 0:
                    sell_price = df['trade_price'].iloc[i]
                    trade_return = (sell_price / last_buy_price) - 1
                    df.iloc[i, df.columns.get_loc('daily_return')] = trade_return
                    portfolio = portfolio * (1 + trade_return)
                    last_buy_price = 0

            elif df['signal'].iloc[i] == 1 and last_buy_price > 0:
                # 보유 중: 미실현 손익은 계산하지 않음 (실제 거래가 아님)
                df.iloc[i, df.columns.get_loc('daily_return')] = 0
            else:
                # 현금 보유: 수익률 0
                df.iloc[i, df.columns.get_loc('daily_return')] = 0

            df.iloc[i, df.columns.get_loc('portfolio_value')] = portfolio

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
        buy_trades = df['buy_signal'].sum()
        sell_trades = df['sell_signal'].sum()
        total_trades = buy_trades + sell_trades

        # 보유 기간 통계
        holding_days = df['signal'].sum()
        total_days = len(df)
        holding_ratio = holding_days / total_days * 100

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

        # 거래별 수익률 계산
        trade_returns = []
        for i in range(len(df)):
            if df['sell_signal'].iloc[i] and df['daily_return'].iloc[i] != 0:
                trade_returns.append(df['daily_return'].iloc[i])

        # 승패 통계
        if len(trade_returns) > 0:
            winning_trades = sum(1 for r in trade_returns if r > 0)
            losing_trades = sum(1 for r in trade_returns if r < 0)
            win_rate = winning_trades / len(trade_returns) * 100 if len(trade_returns) > 0 else 0

            avg_win = np.mean([r for r in trade_returns if r > 0]) * 100 if winning_trades > 0 else 0
            avg_loss = np.mean([r for r in trade_returns if r < 0]) * 100 if losing_trades > 0 else 0

            total_profit = sum([r for r in trade_returns if r > 0])
            total_loss = abs(sum([r for r in trade_returns if r < 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        metrics = {
            '전략': {
                '총 수익률 (%)': f'{total_return:.2f}',
                'CAGR (%)': f'{cagr:.2f}',
                'MDD (%)': f'{mdd:.2f}',
                '샤프 비율': f'{sharpe:.2f}',
                '총 거래 횟수': int(total_trades),
                '매수 횟수': int(buy_trades),
                '매도 횟수': int(sell_trades),
                '보유 비율 (%)': f'{holding_ratio:.2f}',
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

    def plot_results(self, save_path='support_resistance_backtest_results.png'):
        """결과 시각화"""
        df = self.results

        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

        # 1. 가격 차트 + 지지선/저항선
        ax1 = fig.add_subplot(gs[0:2, :])
        ax1.plot(df.index, df['Close'], label='BTC Price', linewidth=1.5, alpha=0.8, color='black')

        # 지지선/저항선 그리기
        ax1.plot(df.index, df['support'], label='Support Line', linewidth=1, alpha=0.6, color='green', linestyle='--')
        ax1.plot(df.index, df['resistance'], label='Resistance Line', linewidth=1, alpha=0.6, color='red', linestyle='--')

        # 지지선/저항선 범위 표시
        ax1.fill_between(df.index, df['support_lower'], df['support_upper'],
                         alpha=0.2, color='green', label='Support Zone')
        ax1.fill_between(df.index, df['resistance_lower'], df['resistance_upper'],
                         alpha=0.2, color='red', label='Resistance Zone')

        # 매수/매도 신호 표시
        buy_signals = df[df['buy_signal'] == True]
        sell_signals = df[df['sell_signal'] == True]

        ax1.scatter(buy_signals.index, buy_signals['Close'],
                   color='blue', marker='^', s=100, alpha=0.9, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['Close'],
                   color='orange', marker='v', s=100, alpha=0.9, label='Sell Signal', zorder=5)

        ax1.set_title(f'{self.symbol} - Support/Resistance Trading Strategy\n(Lookback: {self.lookback_period} days, Threshold: {self.threshold*100:.1f}%)',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (KRW)', fontsize=11)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 누적 수익률 비교
        ax2 = fig.add_subplot(gs[2, :])
        ax2.plot(df.index, df['cumulative_return'], label='Support/Resistance Strategy', linewidth=2, color='blue')
        ax2.plot(df.index, df['buy_hold_return'], label='Buy & Hold', linewidth=2, alpha=0.7, color='gray')
        ax2.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cumulative Return', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 3. 드로우다운
        ax3 = fig.add_subplot(gs[3, :])
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
        ax4 = fig.add_subplot(gs[4, 0])
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

        # 5. 거래별 수익률 분포
        ax5 = fig.add_subplot(gs[4, 1])
        trade_returns = []
        for i in range(len(df)):
            if df['sell_signal'].iloc[i] and df['daily_return'].iloc[i] != 0:
                trade_returns.append(df['daily_return'].iloc[i] * 100)

        if len(trade_returns) > 0:
            ax5.hist(trade_returns, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax5.axvline(np.mean(trade_returns), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(trade_returns):.2f}%')
            ax5.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax5.set_title('Trade Returns Distribution', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Return per Trade (%)', fontsize=10)
            ax5.set_ylabel('Frequency', fontsize=10)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics(self, metrics):
        """성과 지표 출력"""
        print("\n" + "="*80)
        print(f"{'지지선/저항선 트레이딩 전략 백테스트 결과':^80}")
        print("="*80)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {self.symbol}")
        print(f"Lookback 기간: {self.lookback_period}일")
        print(f"임계값(Threshold): {self.threshold*100:.1f}%")
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

    def run(self, slippage=0.002, save_path='support_resistance_backtest_results.png'):
        """백테스트 실행"""
        # 데이터 로드
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
    backtest = SupportResistanceBacktest(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,  # 현재까지
        lookback_period=20,  # 20일 기준 지지/저항선
        threshold=0.02,  # 2% 임계값
        data_source='file'  # 로컬 파일 사용
    )

    # 슬리피지 0.2% 반영
    results, metrics = backtest.run(slippage=0.002)

    # 결과 저장
    print("\nSaving results to CSV...")
    results.to_csv('support_resistance_backtest_results.csv')
    print("Results saved to support_resistance_backtest_results.csv")

    # 파라미터 최적화: Lookback 기간과 임계값 변화에 따른 성과 비교
    print("\n" + "="*80)
    print(f"{'파라미터 최적화: Lookback 기간별 비교':^80}")
    print("="*80)

    lookback_periods = [10, 15, 20, 30, 40]
    thresholds = [0.01, 0.02, 0.03]

    comparison_results = []

    for lookback in lookback_periods:
        for threshold in thresholds:
            print(f"\nTesting: Lookback={lookback}, Threshold={threshold*100:.1f}%")
            bt = SupportResistanceBacktest(
                symbol='BTC_KRW',
                start_date='2018-01-01',
                lookback_period=lookback,
                threshold=threshold,
                data_source='file'
            )
            bt.fetch_data()
            bt.calculate_signals(slippage=0.002)
            metrics = bt.calculate_performance_metrics()

            comparison_results.append({
                'Lookback': lookback,
                'Threshold (%)': f'{threshold*100:.1f}',
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
    df_comparison.to_csv('support_resistance_parameter_comparison.csv', index=False)
    print("\nParameter comparison saved to support_resistance_parameter_comparison.csv")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
