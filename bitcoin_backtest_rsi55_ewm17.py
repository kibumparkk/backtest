"""
비트코인 백테스트: RSI 55 이상 매수, 17일 EWM 전략

전략 상세:
- RSI를 17일 기간의 EWM(지수이동평균)으로 계산
- RSI >= 55: 매수/보유
- RSI < 55: 매도 후 현금 보유
- 슬리피지 0.2% 적용
- Look-ahead bias 방지: shift(1) 사용
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


class BitcoinRSI_EWM_Backtest:
    """비트코인 RSI 55 + 17일 EWM 백테스트"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01', end_date=None,
                 rsi_period=17, rsi_threshold=55, slippage=0.002):
        """
        Args:
            symbol: 종목 심볼 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            rsi_period: RSI 계산 기간 (default: 17일)
            rsi_threshold: RSI 매수 기준선 (default: 55)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.slippage = slippage
        self.data = None
        self.result = None

    def load_data(self):
        """데이터 로드"""
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

        return df

    def calculate_rsi_ewm(self, prices, period=17):
        """
        RSI 계산 (EWM 사용)

        표준 RSI는 EWM(Exponential Weighted Moving)을 사용합니다.
        Wilder's smoothing을 사용하여 gain과 loss를 계산합니다.

        Args:
            prices: 가격 시리즈
            period: RSI 기간

        Returns:
            RSI 시리즈
        """
        delta = prices.diff()

        # gain과 loss 분리
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Wilder's smoothing (alpha = 1/period)
        # pandas의 ewm에서 alpha = 1/period는 adjust=False일 때 Wilder's smoothing과 동일
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        # RS와 RSI 계산
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def run_backtest(self):
        """백테스트 실행"""
        print("\n" + "="*80)
        print("Running backtest...")
        print("="*80)

        df = self.data.copy()

        # RSI 계산 (17일 EWM)
        print(f"\nCalculating RSI with {self.rsi_period}-day EWM...")
        df['RSI'] = self.calculate_rsi_ewm(df['Close'], self.rsi_period)

        # 매수/매도 신호 생성
        # ✅ Look-ahead bias 방지: RSI는 당일 종가로 계산되므로 다음날 시그널로 사용
        df['signal'] = (df['RSI'] >= self.rsi_threshold).astype(int)

        # 포지션 (다음날 적용)
        df['position'] = df['signal'].shift(1)

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 가격 수익률
        df['daily_price_return'] = df['Close'].pct_change()

        # 전략 수익률 (position이 1일 때만 수익)
        df['strategy_return'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용 (매수/매도 시점)
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage  # 매수
        slippage_cost[df['position_change'] == -1] = -self.slippage  # 매도

        df['returns'] = df['strategy_return'] + slippage_cost

        # NaN 처리
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        # Buy & Hold 수익률 (비교용)
        df['bnh_returns'] = df['daily_price_return'].fillna(0)
        df['bnh_cumulative'] = (1 + df['bnh_returns']).cumprod()

        self.result = df

        print("\nBacktest completed!")
        print("="*80 + "\n")

        return df

    def calculate_metrics(self):
        """성과 지표 계산"""
        df = self.result

        # 전략 메트릭
        strategy_returns = df['returns']
        strategy_cumulative = df['cumulative']

        # Buy & Hold 메트릭
        bnh_returns = df['bnh_returns']
        bnh_cumulative = df['bnh_cumulative']

        # 기간
        years = (df.index[-1] - df.index[0]).days / 365.25

        # 전략 성과
        strategy_total_return = (strategy_cumulative.iloc[-1] - 1) * 100
        strategy_cagr = (strategy_cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # MDD
        cummax = strategy_cumulative.cummax()
        drawdown = (strategy_cumulative - cummax) / cummax
        strategy_mdd = drawdown.min() * 100

        # 샤프 비율
        strategy_sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(365)) if strategy_returns.std() > 0 else 0

        # 승률
        total_trades = (strategy_returns != 0).sum()
        winning_trades = (strategy_returns > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit Factor
        total_profit = strategy_returns[strategy_returns > 0].sum()
        total_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        # Buy & Hold 성과
        bnh_total_return = (bnh_cumulative.iloc[-1] - 1) * 100
        bnh_cagr = (bnh_cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # BnH MDD
        bnh_cummax = bnh_cumulative.cummax()
        bnh_drawdown = (bnh_cumulative - bnh_cummax) / bnh_cummax
        bnh_mdd = bnh_drawdown.min() * 100

        # BnH 샤프
        bnh_sharpe = (bnh_returns.mean() / bnh_returns.std() * np.sqrt(365)) if bnh_returns.std() > 0 else 0

        metrics = {
            'Strategy': {
                'Total Return (%)': strategy_total_return,
                'CAGR (%)': strategy_cagr,
                'MDD (%)': strategy_mdd,
                'Sharpe Ratio': strategy_sharpe,
                'Win Rate (%)': win_rate,
                'Total Trades': int(total_trades),
                'Profit Factor': profit_factor
            },
            'Buy & Hold': {
                'Total Return (%)': bnh_total_return,
                'CAGR (%)': bnh_cagr,
                'MDD (%)': bnh_mdd,
                'Sharpe Ratio': bnh_sharpe,
                'Win Rate (%)': np.nan,
                'Total Trades': 1,
                'Profit Factor': np.nan
            }
        }

        return metrics

    def print_results(self):
        """결과 출력"""
        metrics = self.calculate_metrics()

        print("\n" + "="*100)
        print(f"{'비트코인 백테스트 결과':^100}")
        print("="*100)
        print(f"\n전략: RSI {self.rsi_threshold} 이상 매수 ({self.rsi_period}일 EWM)")
        print(f"종목: {self.symbol}")
        print(f"기간: {self.start_date} ~ {self.end_date}")
        print(f"슬리피지: {self.slippage*100}%")
        print(f"\n✅ Look-ahead Bias 방지: shift(1) 적용")
        print(f"✅ 슬리피지 적용: 매수/매도 시 {self.slippage*100}%")

        print("\n" + "-"*100)
        print(f"{'성과 비교':^100}")
        print("-"*100)
        print(f"\n{'Metric':<25} {'Strategy':>20} {'Buy & Hold':>20} {'Difference':>20}")
        print("-"*100)

        for key in metrics['Strategy'].keys():
            strategy_val = metrics['Strategy'][key]
            bnh_val = metrics['Buy & Hold'][key]

            if pd.isna(bnh_val):
                print(f"{key:<25} {strategy_val:>20.2f} {'N/A':>20} {'N/A':>20}")
            elif key == 'Total Trades':
                print(f"{key:<25} {strategy_val:>20.0f} {bnh_val:>20.0f} {strategy_val - bnh_val:>20.0f}")
            elif key == 'Profit Factor' and strategy_val == np.inf:
                print(f"{key:<25} {'INF':>20} {'N/A':>20} {'N/A':>20}")
            else:
                diff = strategy_val - bnh_val
                print(f"{key:<25} {strategy_val:>20.2f} {bnh_val:>20.2f} {diff:>+20.2f}")

        print("\n" + "="*100 + "\n")

        # 체크리스트 검증
        print("\n" + "="*100)
        print(f"{'백테스팅 체크리스트 검증':^100}")
        print("="*100)

        sharpe = metrics['Strategy']['Sharpe Ratio']
        win_rate = metrics['Strategy']['Win Rate (%)']
        mdd = metrics['Strategy']['MDD (%)']

        checks = []
        checks.append(("Look-ahead Bias", "PASS", "shift(1) 사용"))
        checks.append(("Slippage", "PASS", f"{self.slippage*100}% 적용"))

        if sharpe > 3.0:
            checks.append(("Sharpe Ratio", "WARNING", f"{sharpe:.2f} > 3.0 (의심)"))
        else:
            checks.append(("Sharpe Ratio", "PASS", f"{sharpe:.2f}"))

        if win_rate > 70:
            checks.append(("Win Rate", "WARNING", f"{win_rate:.1f}% > 70% (의심)"))
        else:
            checks.append(("Win Rate", "PASS", f"{win_rate:.1f}%"))

        if mdd > -10:
            checks.append(("MDD", "WARNING", f"{mdd:.1f}% > -10% (의심)"))
        else:
            checks.append(("MDD", "PASS", f"{mdd:.1f}%"))

        print(f"\n{'Check Item':<30} {'Status':^15} {'Value':<50}")
        print("-"*100)
        for check, status, value in checks:
            status_symbol = "✅" if status == "PASS" else "⚠️"
            print(f"{check:<30} {status_symbol} {status:^13} {value:<50}")

        print("\n" + "="*100 + "\n")

    def plot_results(self, save_path='bitcoin_rsi55_ewm17_backtest.png'):
        """결과 시각화"""
        df = self.result
        metrics = self.calculate_metrics()

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 가격 차트 + RSI
        ax1 = fig.add_subplot(gs[0, :2])
        ax1_twin = ax1.twinx()

        # 가격 (왼쪽 축)
        ax1.plot(df.index, df['Close'], label='BTC Price', color='gray', linewidth=1.5, alpha=0.6)
        ax1.set_ylabel('Price (KRW)', fontsize=11, color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')
        ax1.set_yscale('log')

        # RSI (오른쪽 축)
        ax1_twin.plot(df.index, df['RSI'], label=f'RSI ({self.rsi_period}d EWM)',
                     color='blue', linewidth=2)
        ax1_twin.axhline(y=self.rsi_threshold, color='red', linestyle='--',
                        linewidth=2, label=f'Threshold {self.rsi_threshold}')
        ax1_twin.axhline(y=70, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax1_twin.axhline(y=30, color='green', linestyle=':', linewidth=1, alpha=0.5)
        ax1_twin.set_ylabel('RSI', fontsize=11, color='blue')
        ax1_twin.tick_params(axis='y', labelcolor='blue')
        ax1_twin.set_ylim(0, 100)

        ax1.set_title(f'Bitcoin Price & RSI ({self.rsi_period}-day EWM)',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1_twin.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. 누적 수익률 비교
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(df.index, df['cumulative'], label='Strategy', color='blue', linewidth=2.5)
        ax2.plot(df.index, df['bnh_cumulative'], label='Buy & Hold',
                color='orange', linewidth=2.5, alpha=0.7)
        ax2.set_title('Cumulative Returns', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Cumulative Return', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # 3. 매수/매도 신호
        ax3 = fig.add_subplot(gs[1, :])
        position_changes = df['position'].diff()
        buy_signals = df[position_changes == 1].index
        sell_signals = df[position_changes == -1].index

        ax3.plot(df.index, df['Close'], color='gray', linewidth=1, alpha=0.5)
        ax3.scatter(buy_signals, df.loc[buy_signals, 'Close'],
                   color='green', marker='^', s=100, label='Buy', zorder=5, alpha=0.7)
        ax3.scatter(sell_signals, df.loc[sell_signals, 'Close'],
                   color='red', marker='v', s=100, label='Sell', zorder=5, alpha=0.7)
        ax3.set_title('Entry/Exit Signals', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Price (KRW)', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # 4. Drawdown 비교
        ax4 = fig.add_subplot(gs[2, :])

        # Strategy DD
        cummax = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - cummax) / cummax * 100
        ax4.plot(drawdown.index, drawdown, label='Strategy',
                color='blue', linewidth=2, alpha=0.7)
        ax4.fill_between(drawdown.index, drawdown, 0, color='blue', alpha=0.2)

        # Buy & Hold DD
        bnh_cummax = df['bnh_cumulative'].cummax()
        bnh_drawdown = (df['bnh_cumulative'] - bnh_cummax) / bnh_cummax * 100
        ax4.plot(bnh_drawdown.index, bnh_drawdown, label='Buy & Hold',
                color='orange', linewidth=2, alpha=0.7)
        ax4.fill_between(bnh_drawdown.index, bnh_drawdown, 0, color='orange', alpha=0.2)

        ax4.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Drawdown (%)', fontsize=11)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # MDD 표시
        mdd_value = drawdown.min()
        mdd_date = drawdown.idxmin()
        ax4.scatter([mdd_date], [mdd_value], color='red', s=200, zorder=5, marker='X')
        ax4.annotate(f'Strategy MDD: {mdd_value:.2f}%',
                    xy=(mdd_date, mdd_value),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

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

        # 6. 성과 지표 요약
        ax6 = fig.add_subplot(gs[3, 2])
        ax6.axis('off')

        metrics_text = f"Performance Metrics\n{'='*40}\n\n"
        metrics_text += "STRATEGY vs BUY & HOLD\n"
        metrics_text += "-"*40 + "\n\n"

        for key in metrics['Strategy'].keys():
            strategy_val = metrics['Strategy'][key]
            bnh_val = metrics['Buy & Hold'][key]

            if pd.isna(bnh_val):
                metrics_text += f"{key}:\n  Strategy: {strategy_val:.2f}\n\n"
            elif key == 'Total Trades':
                metrics_text += f"{key}:\n  Strategy: {strategy_val:.0f}\n  BnH: {bnh_val:.0f}\n\n"
            elif key == 'Profit Factor' and strategy_val == np.inf:
                metrics_text += f"{key}:\n  Strategy: INF\n\n"
            else:
                metrics_text += f"{key}:\n  Strategy: {strategy_val:.2f}\n  BnH: {bnh_val:.2f}\n\n"

        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 전체 제목
        fig.suptitle(f'Bitcoin Backtest: RSI {self.rsi_threshold} Strategy ({self.rsi_period}-day EWM)\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=16, fontweight='bold', y=0.995)

        # 저장
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

        return save_path

    def run_full_analysis(self):
        """전체 분석 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 백테스트 실행
        self.run_backtest()

        # 3. 결과 출력
        self.print_results()

        # 4. 시각화
        self.plot_results()

        # 5. 결과 저장
        print("\nSaving results to CSV...")
        self.result.to_csv('bitcoin_rsi55_ewm17_result.csv')
        print("Results saved to bitcoin_rsi55_ewm17_result.csv")

        return self.result


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("비트코인 백테스트 시작: RSI 55 이상 매수 (17일 EWM)")
    print("="*80)

    # 백테스트 실행
    backtest = BitcoinRSI_EWM_Backtest(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None,
        rsi_period=17,  # 17일 EWM
        rsi_threshold=55,  # RSI 55 이상 매수
        slippage=0.002  # 0.2% 슬리피지
    )

    # 전체 분석 실행
    result = backtest.run_full_analysis()

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
