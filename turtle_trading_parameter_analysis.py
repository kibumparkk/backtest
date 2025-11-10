#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
터틀 트레이딩 파라미터 최적화 분석
=====================================

목적:
- entry_period와 exit_period 파라미터 조합별 성과 분석
- 최적 파라미터 조합 찾기
- 파라미터 민감도 분석

파라미터:
- entry_period: 진입 채널 기간 (N일 최고가)
- exit_period: 청산 채널 기간 (M일 최저가)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class TurtleTradingParameterAnalyzer:
    """터틀 트레이딩 파라미터 분석 클래스"""

    def __init__(self, data_path='chart_day', slippage=0.002):
        """
        초기화

        Parameters:
        -----------
        data_path : str
            데이터 경로
        slippage : float
            슬리피지 (기본 0.2%)
        """
        self.data_path = Path(data_path)
        self.slippage = slippage
        self.data = {}
        self.results = []

    def load_data(self, symbols, start_date=None):
        """데이터 로드"""
        print("\n" + "="*80)
        print("Loading data...")
        print("="*80)

        for symbol in symbols:
            file_path = self.data_path / f"{symbol}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df.index = pd.to_datetime(df.index)
                if start_date:
                    df = df[df.index >= start_date]
                self.data[symbol] = df
                print(f"  - Loaded {symbol}: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
            else:
                print(f"  - WARNING: {symbol} file not found!")

        print(f"\nTotal symbols loaded: {len(self.data)}")

    def strategy_turtle_trading(self, df, entry_period=20, exit_period=10):
        """
        터틀 트레이딩 전략 (매일 미실현 손익 반영)

        Parameters:
        -----------
        df : DataFrame
            OHLCV 데이터
        entry_period : int
            진입 채널 기간 (N일 최고가)
        exit_period : int
            청산 채널 기간 (M일 최저가)

        Returns:
        --------
        DataFrame : 전략 결과 (position, returns, cumulative 컬럼 추가)

        수정사항:
        - 포지션 보유 기간 동안 매일 일일 수익률 계산
        - MDD가 정확히 반영되도록 개선
        """
        df = df.copy()

        # 터틀 채널
        df['entry_high'] = df['high'].rolling(window=entry_period).max().shift(1)
        df['exit_low'] = df['low'].rolling(window=exit_period).min().shift(1)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 최고가 돌파 시 매수
            if df.iloc[i]['high'] > df.iloc[i]['entry_high'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 최저가 하향 돌파 시 매도
            elif df.iloc[i]['low'] < df.iloc[i]['exit_low'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산 (매일 업데이트)
        df['daily_price_return'] = df['close'].pct_change()
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
        return df

    def calculate_metrics(self, returns_series):
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

        # Calmar Ratio (CAGR / |MDD|)
        calmar = cagr / abs(mdd) if mdd != 0 else 0

        return {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Calmar Ratio': calmar,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades,
            'Profit Factor': profit_factor
        }

    def run_parameter_grid_search(self, symbols, entry_periods, exit_periods):
        """
        파라미터 그리드 서치 수행

        Parameters:
        -----------
        symbols : list
            종목 리스트
        entry_periods : list
            테스트할 진입 기간 리스트
        exit_periods : list
            테스트할 청산 기간 리스트
        """
        print("\n" + "="*80)
        print("Running Parameter Grid Search...")
        print("="*80)
        print(f"\nEntry Periods: {entry_periods}")
        print(f"Exit Periods: {exit_periods}")
        print(f"Total Combinations: {len(entry_periods) * len(exit_periods)}")
        print(f"Symbols: {symbols}\n")

        total_combinations = len(entry_periods) * len(exit_periods)
        count = 0

        for entry_period in entry_periods:
            for exit_period in exit_periods:
                count += 1
                print(f"\n[{count}/{total_combinations}] Testing entry={entry_period}, exit={exit_period}")

                # 각 종목별로 전략 실행
                symbol_results = {}
                for symbol in symbols:
                    df = self.data[symbol].copy()
                    result = self.strategy_turtle_trading(df, entry_period, exit_period)
                    symbol_results[symbol] = result

                # 포트폴리오 수익률 계산 (동일 비중)
                weight = 1.0 / len(symbols)

                # 모든 종목의 공통 날짜 인덱스 찾기
                all_indices = [symbol_results[symbol].index for symbol in symbols]
                common_index = all_indices[0]
                for idx in all_indices[1:]:
                    common_index = common_index.intersection(idx)

                # 포트폴리오 수익률
                portfolio_returns = pd.Series(0.0, index=common_index)
                for symbol in symbols:
                    symbol_returns = symbol_results[symbol].loc[common_index, 'returns']
                    portfolio_returns += symbol_returns * weight

                # 성과 지표 계산
                metrics = self.calculate_metrics(portfolio_returns)

                # 결과 저장
                result_row = {
                    'entry_period': entry_period,
                    'exit_period': exit_period,
                    **metrics
                }
                self.results.append(result_row)

                # 진행상황 출력
                print(f"  - CAGR: {metrics['CAGR (%)']:.2f}%, MDD: {metrics['MDD (%)']:.2f}%, "
                      f"Sharpe: {metrics['Sharpe Ratio']:.2f}, Calmar: {metrics['Calmar Ratio']:.2f}")

        # 결과를 DataFrame으로 변환
        self.results_df = pd.DataFrame(self.results)

        print("\n" + "="*80)
        print("Parameter Grid Search Completed!")
        print("="*80)

    def save_results(self, output_file='turtle_trading_parameter_results.csv'):
        """결과 저장"""
        self.results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_file}")

    def plot_heatmaps(self, output_file='turtle_trading_parameter_heatmaps.png'):
        """파라미터 히트맵 생성"""
        print("\n" + "="*80)
        print("Creating parameter heatmaps...")
        print("="*80)

        # 주요 지표들
        metrics = ['CAGR (%)', 'Sharpe Ratio', 'Calmar Ratio', 'MDD (%)', 'Win Rate (%)', 'Total Trades']

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # 피벗 테이블 생성
            pivot_table = self.results_df.pivot_table(
                values=metric,
                index='exit_period',
                columns='entry_period',
                aggfunc='first'
            )

            # 히트맵 색상 설정
            if metric == 'MDD (%)':
                cmap = 'RdYlGn'  # MDD는 낮을수록 좋음 (빨강->초록)
            else:
                cmap = 'RdYlGn_r'  # 나머지는 높을수록 좋음 (초록->빨강)

            # 히트맵 그리기
            sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap=cmap,
                       ax=ax, cbar_kws={'label': metric}, linewidths=0.5)

            ax.set_title(f'{metric} by Parameters', fontsize=14, fontweight='bold')
            ax.set_xlabel('Entry Period (days)', fontsize=12)
            ax.set_ylabel('Exit Period (days)', fontsize=12)

            # 최적값 표시
            if metric == 'MDD (%)':
                best_val = pivot_table.max().max()  # MDD는 최댓값(0에 가까운)
            else:
                best_val = pivot_table.max().max()

            ax.text(0.02, 0.98, f'Best: {best_val:.2f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmaps saved to: {output_file}")
        plt.close()

    def plot_top_performers(self, top_n=10, output_file='turtle_trading_top_performers.png'):
        """상위 성과 파라미터 시각화"""
        print("\n" + "="*80)
        print(f"Plotting top {top_n} performers...")
        print("="*80)

        # CAGR 기준 상위 N개
        top_by_cagr = self.results_df.nlargest(top_n, 'CAGR (%)')

        # Sharpe 기준 상위 N개
        top_by_sharpe = self.results_df.nlargest(top_n, 'Sharpe Ratio')

        # Calmar 기준 상위 N개
        top_by_calmar = self.results_df.nlargest(top_n, 'Calmar Ratio')

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # 1. CAGR 상위 10개
        ax = axes[0, 0]
        top_by_cagr['param_label'] = top_by_cagr.apply(
            lambda x: f"E{int(x['entry_period'])}/X{int(x['exit_period'])}", axis=1
        )
        ax.barh(range(len(top_by_cagr)), top_by_cagr['CAGR (%)'], color='skyblue')
        ax.set_yticks(range(len(top_by_cagr)))
        ax.set_yticklabels(top_by_cagr['param_label'])
        ax.set_xlabel('CAGR (%)', fontsize=12)
        ax.set_title(f'Top {top_n} by CAGR', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        # 2. Sharpe Ratio 상위 10개
        ax = axes[0, 1]
        top_by_sharpe['param_label'] = top_by_sharpe.apply(
            lambda x: f"E{int(x['entry_period'])}/X{int(x['exit_period'])}", axis=1
        )
        ax.barh(range(len(top_by_sharpe)), top_by_sharpe['Sharpe Ratio'], color='lightgreen')
        ax.set_yticks(range(len(top_by_sharpe)))
        ax.set_yticklabels(top_by_sharpe['param_label'])
        ax.set_xlabel('Sharpe Ratio', fontsize=12)
        ax.set_title(f'Top {top_n} by Sharpe Ratio', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        # 3. Calmar Ratio 상위 10개
        ax = axes[1, 0]
        top_by_calmar['param_label'] = top_by_calmar.apply(
            lambda x: f"E{int(x['entry_period'])}/X{int(x['exit_period'])}", axis=1
        )
        ax.barh(range(len(top_by_calmar)), top_by_calmar['Calmar Ratio'], color='lightcoral')
        ax.set_yticks(range(len(top_by_calmar)))
        ax.set_yticklabels(top_by_calmar['param_label'])
        ax.set_xlabel('Calmar Ratio', fontsize=12)
        ax.set_title(f'Top {top_n} by Calmar Ratio', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        # 4. 산점도: CAGR vs Sharpe Ratio
        ax = axes[1, 1]
        scatter = ax.scatter(self.results_df['Sharpe Ratio'],
                           self.results_df['CAGR (%)'],
                           c=self.results_df['MDD (%)'],
                           cmap='RdYlGn',
                           s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

        # 상위 5개 라벨링
        top_5 = self.results_df.nlargest(5, 'CAGR (%)')
        for _, row in top_5.iterrows():
            ax.annotate(f"E{int(row['entry_period'])}/X{int(row['exit_period'])}",
                       (row['Sharpe Ratio'], row['CAGR (%)']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        ax.set_xlabel('Sharpe Ratio', fontsize=12)
        ax.set_ylabel('CAGR (%)', fontsize=12)
        ax.set_title('CAGR vs Sharpe Ratio (colored by MDD)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('MDD (%)', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Top performers chart saved to: {output_file}")
        plt.close()

    def print_summary(self):
        """요약 통계 출력"""
        print("\n" + "="*80)
        print("PARAMETER ANALYSIS SUMMARY")
        print("="*80)

        # 최고 CAGR
        best_cagr = self.results_df.loc[self.results_df['CAGR (%)'].idxmax()]
        print(f"\n[Best CAGR]")
        print(f"  Entry Period: {int(best_cagr['entry_period'])} days")
        print(f"  Exit Period: {int(best_cagr['exit_period'])} days")
        print(f"  CAGR: {best_cagr['CAGR (%)']:.2f}%")
        print(f"  MDD: {best_cagr['MDD (%)']:.2f}%")
        print(f"  Sharpe: {best_cagr['Sharpe Ratio']:.2f}")
        print(f"  Calmar: {best_cagr['Calmar Ratio']:.2f}")

        # 최고 Sharpe Ratio
        best_sharpe = self.results_df.loc[self.results_df['Sharpe Ratio'].idxmax()]
        print(f"\n[Best Sharpe Ratio]")
        print(f"  Entry Period: {int(best_sharpe['entry_period'])} days")
        print(f"  Exit Period: {int(best_sharpe['exit_period'])} days")
        print(f"  CAGR: {best_sharpe['CAGR (%)']:.2f}%")
        print(f"  MDD: {best_sharpe['MDD (%)']:.2f}%")
        print(f"  Sharpe: {best_sharpe['Sharpe Ratio']:.2f}")
        print(f"  Calmar: {best_sharpe['Calmar Ratio']:.2f}")

        # 최고 Calmar Ratio
        best_calmar = self.results_df.loc[self.results_df['Calmar Ratio'].idxmax()]
        print(f"\n[Best Calmar Ratio]")
        print(f"  Entry Period: {int(best_calmar['entry_period'])} days")
        print(f"  Exit Period: {int(best_calmar['exit_period'])} days")
        print(f"  CAGR: {best_calmar['CAGR (%)']:.2f}%")
        print(f"  MDD: {best_calmar['MDD (%)']:.2f}%")
        print(f"  Sharpe: {best_calmar['Sharpe Ratio']:.2f}")
        print(f"  Calmar: {best_calmar['Calmar Ratio']:.2f}")

        # 최소 MDD (절댓값이 가장 작은)
        best_mdd = self.results_df.loc[self.results_df['MDD (%)'].idxmax()]  # MDD는 음수이므로 max가 최소 낙폭
        print(f"\n[Minimum MDD]")
        print(f"  Entry Period: {int(best_mdd['entry_period'])} days")
        print(f"  Exit Period: {int(best_mdd['exit_period'])} days")
        print(f"  CAGR: {best_mdd['CAGR (%)']:.2f}%")
        print(f"  MDD: {best_mdd['MDD (%)']:.2f}%")
        print(f"  Sharpe: {best_mdd['Sharpe Ratio']:.2f}")
        print(f"  Calmar: {best_mdd['Calmar Ratio']:.2f}")

        # 통계 요약
        print(f"\n[Overall Statistics]")
        print(f"  Total Combinations Tested: {len(self.results_df)}")
        print(f"  Average CAGR: {self.results_df['CAGR (%)'].mean():.2f}%")
        print(f"  Average Sharpe: {self.results_df['Sharpe Ratio'].mean():.2f}")
        print(f"  Average MDD: {self.results_df['MDD (%)'].mean():.2f}%")
        print(f"  Average Win Rate: {self.results_df['Win Rate (%)'].mean():.2f}%")

        print("\n" + "="*80)


def main():
    """메인 실행 함수"""

    # 파라미터 설정
    SYMBOLS = ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW']
    START_DATE = None  # None = 전체 데이터 구간 사용
    SLIPPAGE = 0.002

    # 테스트할 파라미터 범위
    # 터틀 트레이딩 전통적 파라미터: 20일/55일 시스템
    ENTRY_PERIODS = [10, 15, 20, 25, 30, 40, 55]  # 진입 채널 기간
    EXIT_PERIODS = [5, 10, 15, 20]  # 청산 채널 기간

    print("\n" + "="*80)
    print("TURTLE TRADING PARAMETER OPTIMIZATION ANALYSIS")
    print("="*80)
    print(f"\nSymbols: {SYMBOLS}")
    print(f"Start Date: {START_DATE if START_DATE else 'Full Data Range (earliest available)'}")
    print(f"Slippage: {SLIPPAGE*100:.1f}%")
    print(f"\nEntry Periods to test: {ENTRY_PERIODS}")
    print(f"Exit Periods to test: {EXIT_PERIODS}")
    print(f"Total combinations: {len(ENTRY_PERIODS) * len(EXIT_PERIODS)}")

    # 분석 객체 생성
    analyzer = TurtleTradingParameterAnalyzer(slippage=SLIPPAGE)

    # 데이터 로드
    analyzer.load_data(SYMBOLS, START_DATE)

    # 파라미터 그리드 서치 실행
    analyzer.run_parameter_grid_search(SYMBOLS, ENTRY_PERIODS, EXIT_PERIODS)

    # 결과 저장
    analyzer.save_results('turtle_trading_parameter_results.csv')

    # 요약 출력
    analyzer.print_summary()

    # 시각화
    analyzer.plot_heatmaps('turtle_trading_parameter_heatmaps.png')
    analyzer.plot_top_performers(top_n=10, output_file='turtle_trading_top_performers.png')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. turtle_trading_parameter_results.csv - 전체 파라미터 조합 결과")
    print("  2. turtle_trading_parameter_heatmaps.png - 파라미터 히트맵")
    print("  3. turtle_trading_top_performers.png - 상위 성과 분석")
    print("\n")


if __name__ == '__main__':
    main()
