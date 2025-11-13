"""
BTC SMA 파라미터 최적화 백테스트
다양한 SMA 윈도우에 대해 전략 성과 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BTCSMAOptimization:
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
        self.optimization_results = []

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

    def run_single_strategy(self, sma_period):
        """
        단일 SMA 윈도우에 대한 백테스트 실행

        Parameters:
        -----------
        sma_period : int
            SMA 기간

        Returns:
        --------
        dict : 성과 지표
        """
        df = self.df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 매매 신호 생성 (전일 종가 > 전일 SMA)
        df['signal'] = np.where(df['Close'] > df['SMA'], 1, 0)

        # 포지션 결정 (전일 신호를 오늘 적용)
        df['position'] = df['signal'].shift(1)

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()

        # 전략 수익률
        df['strategy_returns'] = df['position'] * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['strategy_returns'] = df['strategy_returns'] + slippage_cost

        # 누적 수익률
        df['strategy_cumulative'] = (1 + df['strategy_returns']).cumprod()

        # NaN 제거
        df = df.dropna()

        # 성과 지표 계산
        metrics = self._calculate_metrics(df, sma_period)

        return metrics, df

    def _calculate_metrics(self, df, sma_period):
        """성과 지표 계산"""
        # 기본 통계
        total_return = (df['strategy_cumulative'].iloc[-1] - 1) * 100

        # CAGR
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['strategy_cumulative'].iloc[-1] ** (1/years) - 1) * 100

        # MDD
        running_max = df['strategy_cumulative'].cummax()
        drawdown = (df['strategy_cumulative'] - running_max) / running_max
        mdd = drawdown.min() * 100

        # Sharpe Ratio
        if df['strategy_returns'].std() != 0:
            sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
        else:
            sharpe = 0

        # 거래 통계
        trades = df[df['position_change'] != 0]
        total_trades = len(trades)

        # 승률
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

        # 최종 자산
        final_value = df['strategy_cumulative'].iloc[-1]

        return {
            'sma_period': sma_period,
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_value': final_value
        }

    def optimize(self, sma_range):
        """
        여러 SMA 윈도우에 대해 최적화 수행

        Parameters:
        -----------
        sma_range : list or range
            테스트할 SMA 윈도우 리스트
        """
        print("="*80)
        print(f"SMA 파라미터 최적화 시작")
        print(f"테스트 범위: SMA {min(sma_range)} ~ {max(sma_range)} ({len(sma_range)}개)")
        print("="*80)

        self.optimization_results = []

        for i, sma_period in enumerate(sma_range, 1):
            metrics, _ = self.run_single_strategy(sma_period)
            self.optimization_results.append(metrics)

            if i % 10 == 0:
                print(f"진행: {i}/{len(sma_range)} - SMA{sma_period}: "
                      f"CAGR {metrics['cagr']:.2f}%, MDD {metrics['mdd']:.2f}%")

        print()
        print("최적화 완료!")
        print()

        # DataFrame으로 변환
        self.results_df = pd.DataFrame(self.optimization_results)

        return self.results_df

    def print_top_results(self, top_n=10):
        """상위 N개 결과 출력"""
        print("="*80)
        print(f"상위 {top_n}개 SMA 윈도우 (CAGR 기준)")
        print("="*80)
        print()

        top_results = self.results_df.nlargest(top_n, 'cagr')

        print(f"{'순위':<4} {'SMA':<6} {'CAGR':<10} {'MDD':<10} {'Sharpe':<8} {'승률':<8} {'거래횟수':<10}")
        print("-"*80)

        for idx, row in enumerate(top_results.itertuples(), 1):
            print(f"{idx:<4} {row.sma_period:<6} {row.cagr:>8.2f}% {row.mdd:>8.2f}% "
                  f"{row.sharpe:>6.2f}   {row.win_rate:>6.2f}%  {row.total_trades:>8d}회")

        print()

    def plot_comprehensive_analysis(self, save_path='btc_sma_optimization_analysis.png'):
        """종합 분석 시각화"""
        print("="*80)
        print("종합 분석 시각화 중...")
        print("="*80)

        # 플롯 스타일 설정
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(20, 12))

        # 1. CAGR vs SMA Window
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(self.results_df['sma_period'], self.results_df['cagr'],
                 linewidth=2, color='#2E86AB', marker='o', markersize=3)
        best_cagr_idx = self.results_df['cagr'].idxmax()
        best_cagr = self.results_df.loc[best_cagr_idx]
        ax1.scatter(best_cagr['sma_period'], best_cagr['cagr'],
                   color='red', s=200, marker='*', zorder=5, label=f'Best: SMA{int(best_cagr["sma_period"])}')
        ax1.set_xlabel('SMA Window', fontsize=12)
        ax1.set_ylabel('CAGR (%)', fontsize=12)
        ax1.set_title('CAGR vs SMA Window', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. MDD vs SMA Window
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(self.results_df['sma_period'], self.results_df['mdd'],
                 linewidth=2, color='#C73E1D', marker='o', markersize=3)
        best_mdd_idx = self.results_df['mdd'].idxmax()  # MDD는 음수이므로 최대값이 최소 손실
        best_mdd = self.results_df.loc[best_mdd_idx]
        ax2.scatter(best_mdd['sma_period'], best_mdd['mdd'],
                   color='green', s=200, marker='*', zorder=5, label=f'Best: SMA{int(best_mdd["sma_period"])}')
        ax2.set_xlabel('SMA Window', fontsize=12)
        ax2.set_ylabel('MDD (%)', fontsize=12)
        ax2.set_title('Maximum Drawdown vs SMA Window', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. CAGR vs MDD Scatter
        ax3 = plt.subplot(2, 3, 3)
        scatter = ax3.scatter(self.results_df['mdd'], self.results_df['cagr'],
                             c=self.results_df['sharpe'], cmap='viridis',
                             s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        # 상위 3개 표시
        top3 = self.results_df.nlargest(3, 'cagr')
        for _, row in top3.iterrows():
            ax3.scatter(row['mdd'], row['cagr'], color='red', s=300, marker='*', zorder=5)
            ax3.annotate(f"SMA{int(row['sma_period'])}",
                        xy=(row['mdd'], row['cagr']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Sharpe Ratio', fontsize=10)
        ax3.set_xlabel('MDD (%)', fontsize=12)
        ax3.set_ylabel('CAGR (%)', fontsize=12)
        ax3.set_title('CAGR vs MDD (colored by Sharpe)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Sharpe Ratio vs SMA Window
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(self.results_df['sma_period'], self.results_df['sharpe'],
                 linewidth=2, color='#A23B72', marker='o', markersize=3)
        best_sharpe_idx = self.results_df['sharpe'].idxmax()
        best_sharpe = self.results_df.loc[best_sharpe_idx]
        ax4.scatter(best_sharpe['sma_period'], best_sharpe['sharpe'],
                   color='red', s=200, marker='*', zorder=5, label=f'Best: SMA{int(best_sharpe["sma_period"])}')
        ax4.set_xlabel('SMA Window', fontsize=12)
        ax4.set_ylabel('Sharpe Ratio', fontsize=12)
        ax4.set_title('Sharpe Ratio vs SMA Window', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. Win Rate vs SMA Window
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(self.results_df['sma_period'], self.results_df['win_rate'],
                 linewidth=2, color='#F18F01', marker='o', markersize=3)
        ax5.set_xlabel('SMA Window', fontsize=12)
        ax5.set_ylabel('Win Rate (%)', fontsize=12)
        ax5.set_title('Win Rate vs SMA Window', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. Total Trades vs SMA Window
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(self.results_df['sma_period'], self.results_df['total_trades'],
                 linewidth=2, color='#6A994E', marker='o', markersize=3)
        ax6.set_xlabel('SMA Window', fontsize=12)
        ax6.set_ylabel('Total Trades', fontsize=12)
        ax6.set_title('Total Trades vs SMA Window', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"차트 저장 완료: {save_path}")
        print()

    def plot_cumulative_returns(self, selected_smas=None, save_path='btc_sma_cumulative_returns.png'):
        """
        선택된 SMA들의 누적 수익률 비교 (로그 스케일)

        Parameters:
        -----------
        selected_smas : list
            시각화할 SMA 윈도우 리스트 (None이면 자동 선택)
        """
        print("="*80)
        print("누적 수익률 비교 차트 생성 중...")
        print("="*80)

        if selected_smas is None:
            # 자동 선택: CAGR 상위 5개 + Buy&Hold
            top5 = self.results_df.nlargest(5, 'cagr')
            selected_smas = top5['sma_period'].tolist()

        # 플롯 스타일 설정
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'DejaVu Sans'

        fig, ax = plt.subplots(figsize=(16, 10))

        # 각 SMA에 대해 누적 수익률 계산 및 플롯
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_smas)))

        for i, sma_period in enumerate(selected_smas):
            _, df = self.run_single_strategy(sma_period)
            metrics = self.results_df[self.results_df['sma_period'] == sma_period].iloc[0]

            label = f"SMA{int(sma_period)} (CAGR: {metrics['cagr']:.1f}%, MDD: {metrics['mdd']:.1f}%)"
            ax.plot(df.index, df['strategy_cumulative'],
                   label=label, linewidth=2, color=colors[i], alpha=0.8)

        # Buy & Hold 추가
        df_base = self.df.copy()
        df_base['daily_return'] = df_base['Close'].pct_change()
        df_base['buy_hold_cumulative'] = (1 + df_base['daily_return']).cumprod()
        df_base = df_base.dropna()

        # Buy & Hold 성과 계산
        years = (df_base.index[-1] - df_base.index[0]).days / 365.25
        bh_cagr = (df_base['buy_hold_cumulative'].iloc[-1] ** (1/years) - 1) * 100
        bh_running_max = df_base['buy_hold_cumulative'].cummax()
        bh_drawdown = (df_base['buy_hold_cumulative'] - bh_running_max) / bh_running_max
        bh_mdd = bh_drawdown.min() * 100

        ax.plot(df_base.index, df_base['buy_hold_cumulative'],
               label=f"Buy&Hold (CAGR: {bh_cagr:.1f}%, MDD: {bh_mdd:.1f}%)",
               linewidth=2.5, color='black', linestyle='--', alpha=0.7)

        ax.set_yscale('log')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Return (Log Scale)', fontsize=14)
        ax.set_title('BTC SMA Strategies - Cumulative Returns Comparison (Log Scale)',
                    fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"누적 수익률 차트 저장 완료: {save_path}")
        print()

    def save_results(self, csv_path='btc_sma_optimization_results.csv'):
        """최적화 결과를 CSV로 저장"""
        print("="*80)
        print("최적화 결과 저장 중...")
        print("="*80)

        self.results_df.to_csv(csv_path, index=False)
        print(f"결과 저장 완료: {csv_path}")
        print(f"저장된 데이터: {len(self.results_df)}행")
        print()


def main():
    """메인 실행 함수"""
    # 데이터 경로
    data_path = '/home/user/backtest/chart_day/BTC_KRW.parquet'

    # 최적화 객체 생성
    optimizer = BTCSMAOptimization(data_path=data_path, slippage=0.002)

    # 1. 데이터 로드
    optimizer.load_data()

    # 2. SMA 파라미터 최적화 (5~100, 1단위)
    sma_range = range(5, 101, 1)  # 5, 6, 7, ..., 100
    results_df = optimizer.optimize(sma_range)

    # 3. 상위 결과 출력
    optimizer.print_top_results(top_n=15)

    # 4. 종합 분석 시각화
    optimizer.plot_comprehensive_analysis('btc_sma_optimization_analysis.png')

    # 5. 누적 수익률 비교 (로그 스케일)
    # 상위 10개 + 특정 SMA (20, 30, 50, 100) 선택
    top10_smas = results_df.nlargest(10, 'cagr')['sma_period'].tolist()
    important_smas = [20, 30, 50, 100]
    selected_smas = list(set(top10_smas[:5] + important_smas))  # 중복 제거
    selected_smas.sort()

    optimizer.plot_cumulative_returns(selected_smas=selected_smas,
                                     save_path='btc_sma_cumulative_returns.png')

    # 6. 결과 저장
    optimizer.save_results('btc_sma_optimization_results.csv')

    # 7. 최적 파라미터 출력
    print("="*80)
    print("최적 파라미터 요약")
    print("="*80)
    print()

    best_cagr = results_df.loc[results_df['cagr'].idxmax()]
    best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
    best_mdd = results_df.loc[results_df['mdd'].idxmax()]

    print(f"【 최고 CAGR 】")
    print(f"  SMA 윈도우:       {int(best_cagr['sma_period'])}일")
    print(f"  CAGR:             {best_cagr['cagr']:.2f}%")
    print(f"  MDD:              {best_cagr['mdd']:.2f}%")
    print(f"  Sharpe:           {best_cagr['sharpe']:.2f}")
    print(f"  승률:             {best_cagr['win_rate']:.2f}%")
    print()

    print(f"【 최고 Sharpe Ratio 】")
    print(f"  SMA 윈도우:       {int(best_sharpe['sma_period'])}일")
    print(f"  CAGR:             {best_sharpe['cagr']:.2f}%")
    print(f"  MDD:              {best_sharpe['mdd']:.2f}%")
    print(f"  Sharpe:           {best_sharpe['sharpe']:.2f}")
    print(f"  승률:             {best_sharpe['win_rate']:.2f}%")
    print()

    print(f"【 최소 MDD (최고 위험조정수익) 】")
    print(f"  SMA 윈도우:       {int(best_mdd['sma_period'])}일")
    print(f"  CAGR:             {best_mdd['cagr']:.2f}%")
    print(f"  MDD:              {best_mdd['mdd']:.2f}%")
    print(f"  Sharpe:           {best_mdd['sharpe']:.2f}")
    print(f"  승률:             {best_mdd['win_rate']:.2f}%")
    print()

    print("="*80)
    print("최적화 완료!")
    print("="*80)


if __name__ == '__main__':
    main()
