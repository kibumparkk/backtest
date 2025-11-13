"""
BTC SMA30 백테스트 전략
전략: 전일 종가 > SMA30일 때 매수/보유, 그 외 매도/현금 보유
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class BTCSMABacktest:
    def __init__(self, data_path, slippage=0.002):
        """
        Parameters:
        -----------
        data_path : str
            BTC 데이터 파일 경로
        slippage : float
            슬리피지 (기본 0.2%)
        """
        self.data_path = data_path
        self.slippage = slippage
        self.df = None
        self.results = None

    def load_data(self):
        """BTC 데이터 로드"""
        print("="*80)
        print("BTC 데이터 로딩 중...")
        print("="*80)

        self.df = pd.read_parquet(self.data_path)
        self.df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        print(f"데이터 로드 완료: {len(self.df)}개 데이터 포인트")
        print(f"기간: {self.df.index[0].date()} ~ {self.df.index[-1].date()}")
        print(f"초기 가격: {self.df['Close'].iloc[0]:,.0f}원")
        print(f"최종 가격: {self.df['Close'].iloc[-1]:,.0f}원")
        print()

    def run_strategy(self, sma_period=30):
        """
        SMA30 전략 실행

        전략 로직:
        - 전일 종가 > 전일 SMA30 → 오늘 매수/보유
        - 전일 종가 <= 전일 SMA30 → 오늘 매도/현금

        Parameters:
        -----------
        sma_period : int
            SMA 기간 (기본 30일)
        """
        print("="*80)
        print(f"SMA{sma_period} 전략 백테스트 실행 중...")
        print("="*80)

        df = self.df.copy()

        # SMA 계산
        df['SMA30'] = df['Close'].rolling(window=sma_period).mean()

        # 매매 신호 생성 (전일 종가 > 전일 SMA30)
        df['signal'] = np.where(df['Close'] > df['SMA30'], 1, 0)

        # 포지션 결정 (전일 신호를 오늘 적용 - Look-ahead bias 방지)
        df['position'] = df['signal'].shift(1)

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()

        # 전략 수익률 (전일 포지션 × 오늘 수익률)
        df['strategy_returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용 (포지션 변화 시)
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage  # 매수 시
        slippage_cost[df['position_change'] == -1] = -self.slippage  # 매도 시

        df['strategy_returns'] = df['strategy_returns'] + slippage_cost

        # 누적 수익률
        df['strategy_cumulative'] = (1 + df['strategy_returns']).cumprod()
        df['buy_hold_cumulative'] = (1 + df['daily_price_return']).cumprod()

        # NaN 제거 후 저장
        self.results = df.dropna()

        print(f"전략 실행 완료!")
        print(f"유효 데이터: {len(self.results)}개")
        print()

    def calculate_metrics(self):
        """성과 지표 계산"""
        df = self.results

        # 기본 통계
        total_return = (df['strategy_cumulative'].iloc[-1] - 1) * 100
        buy_hold_return = (df['buy_hold_cumulative'].iloc[-1] - 1) * 100

        # 연간 수익률 (CAGR)
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['strategy_cumulative'].iloc[-1] ** (1/years) - 1) * 100
        buy_hold_cagr = (df['buy_hold_cumulative'].iloc[-1] ** (1/years) - 1) * 100

        # MDD (Maximum Drawdown)
        running_max = df['strategy_cumulative'].cummax()
        drawdown = (df['strategy_cumulative'] - running_max) / running_max
        mdd = drawdown.min() * 100

        buy_hold_running_max = df['buy_hold_cumulative'].cummax()
        buy_hold_drawdown = (df['buy_hold_cumulative'] - buy_hold_running_max) / buy_hold_running_max
        buy_hold_mdd = buy_hold_drawdown.min() * 100

        # Sharpe Ratio (연율화, 무위험 수익률 0 가정)
        sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
        buy_hold_sharpe = df['daily_price_return'].mean() / df['daily_price_return'].std() * np.sqrt(252)

        # 거래 통계
        trades = df[df['position_change'] != 0]
        total_trades = len(trades)

        # 승률 계산 (포지션 보유 기간의 수익률로 계산)
        position_changes = df[df['position_change'].abs() == 1].index
        wins = 0
        total_positions = 0

        for i in range(len(position_changes) - 1):
            start = position_changes[i]
            end = position_changes[i + 1]
            position_return = df.loc[start:end, 'strategy_returns'].sum()
            if position_return > 0:
                wins += 1
            total_positions += 1

        win_rate = (wins / total_positions * 100) if total_positions > 0 else 0

        # 결과 딕셔너리
        metrics = {
            'strategy': {
                'total_return': total_return,
                'cagr': cagr,
                'mdd': mdd,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'total_trades': total_trades
            },
            'buy_hold': {
                'total_return': buy_hold_return,
                'cagr': buy_hold_cagr,
                'mdd': buy_hold_mdd,
                'sharpe': buy_hold_sharpe
            }
        }

        return metrics

    def print_metrics(self, metrics):
        """성과 지표 출력"""
        print("="*80)
        print("백테스트 성과 지표")
        print("="*80)
        print()

        print("【 SMA30 전략 성과 】")
        print(f"  총 수익률:        {metrics['strategy']['total_return']:>10.2f}%")
        print(f"  연평균 수익률:    {metrics['strategy']['cagr']:>10.2f}%")
        print(f"  최대 낙폭(MDD):   {metrics['strategy']['mdd']:>10.2f}%")
        print(f"  샤프 비율:        {metrics['strategy']['sharpe']:>10.2f}")
        print(f"  승률:             {metrics['strategy']['win_rate']:>10.2f}%")
        print(f"  총 거래 횟수:     {metrics['strategy']['total_trades']:>10d}회")
        print()

        print("【 매수후보유(Buy&Hold) 성과 】")
        print(f"  총 수익률:        {metrics['buy_hold']['total_return']:>10.2f}%")
        print(f"  연평균 수익률:    {metrics['buy_hold']['cagr']:>10.2f}%")
        print(f"  최대 낙폭(MDD):   {metrics['buy_hold']['mdd']:>10.2f}%")
        print(f"  샤프 비율:        {metrics['buy_hold']['sharpe']:>10.2f}")
        print()

        print("【 초과 성과 】")
        print(f"  수익률 차이:      {metrics['strategy']['total_return'] - metrics['buy_hold']['total_return']:>10.2f}%p")
        print(f"  CAGR 차이:        {metrics['strategy']['cagr'] - metrics['buy_hold']['cagr']:>10.2f}%p")
        print(f"  MDD 차이:         {metrics['strategy']['mdd'] - metrics['buy_hold']['mdd']:>10.2f}%p")
        print()

    def plot_results(self, save_path='btc_sma30_backtest_result.png'):
        """결과 시각화"""
        print("="*80)
        print("결과 시각화 중...")
        print("="*80)

        df = self.results

        # 플롯 스타일 설정
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(16, 12))

        # 1. 누적 수익률 비교
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df.index, df['strategy_cumulative'], label='SMA30 Strategy', linewidth=2, color='#2E86AB')
        ax1.plot(df.index, df['buy_hold_cumulative'], label='Buy & Hold', linewidth=2, color='#A23B72', alpha=0.7)
        ax1.set_title('BTC SMA30 Strategy vs Buy&Hold - Cumulative Returns', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 2. 가격과 SMA30
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(df.index, df['Close'], label='BTC Price', linewidth=1.5, color='#F18F01', alpha=0.8)
        ax2.plot(df.index, df['SMA30'], label='SMA30', linewidth=2, color='#C73E1D', linestyle='--')

        # 매수/매도 신호 표시
        buy_signals = df[df['position_change'] == 1]
        sell_signals = df[df['position_change'] == -1]
        ax2.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy', zorder=5, alpha=0.7)
        ax2.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell', zorder=5, alpha=0.7)

        ax2.set_title('BTC Price & SMA30 with Trading Signals', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Price (KRW)', fontsize=12)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)

        # 3. 포지션과 일일 수익률
        ax3 = plt.subplot(3, 1, 3)

        # 포지션 표시 (배경색)
        for i in range(len(df) - 1):
            if df['position'].iloc[i] == 1:
                ax3.axvspan(df.index[i], df.index[i+1], alpha=0.1, color='green')

        ax3.bar(df.index, df['strategy_returns'] * 100, label='Daily Returns (%)', color='#2E86AB', alpha=0.6, width=1)
        ax3.set_title('Daily Returns (Green background = Position holding)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Daily Return (%)', fontsize=12)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.legend(loc='best', fontsize=11)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"차트 저장 완료: {save_path}")
        print()

    def save_results(self, csv_path='btc_sma30_backtest_results.csv'):
        """결과를 CSV로 저장"""
        print("="*80)
        print("결과 저장 중...")
        print("="*80)

        # 필요한 컬럼만 선택하여 저장
        output_df = self.results[[
            'Close', 'SMA30', 'signal', 'position', 'position_change',
            'daily_price_return', 'strategy_returns',
            'strategy_cumulative', 'buy_hold_cumulative'
        ]].copy()

        output_df.to_csv(csv_path)
        print(f"결과 저장 완료: {csv_path}")
        print(f"저장된 데이터: {len(output_df)}행")
        print()


def main():
    """메인 실행 함수"""
    # 데이터 경로
    data_path = '/home/user/backtest/chart_day/BTC_KRW.parquet'

    # 백테스트 객체 생성
    backtest = BTCSMABacktest(data_path=data_path, slippage=0.002)

    # 1. 데이터 로드
    backtest.load_data()

    # 2. 전략 실행
    backtest.run_strategy(sma_period=30)

    # 3. 성과 지표 계산 및 출력
    metrics = backtest.calculate_metrics()
    backtest.print_metrics(metrics)

    # 4. 결과 시각화
    backtest.plot_results('btc_sma30_backtest_result.png')

    # 5. 결과 저장
    backtest.save_results('btc_sma30_backtest_results.csv')

    print("="*80)
    print("백테스트 완료!")
    print("="*80)


if __name__ == '__main__':
    main()
