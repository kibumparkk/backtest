"""
터틀 트레이딩 전략 파라미터 최적화
BTC에 적용하여 최적의 파라미터 조합 탐색

전략 설명:
- Richard Dennis의 터틀 트레이딩 전략
- Donchian Channel 기반 브레이크아웃
- 진입: N일 최고가 돌파
- 청산: M일 최저가 돌파
- ATR 기반 리스크 관리 (선택)

최적화 파라미터:
1. entry_period: 진입 채널 기간 (10, 20, 30, 40, 55일)
2. exit_period: 청산 채널 기간 (5, 10, 15, 20일)
3. slippage: 슬리피지 (0.1%, 0.2%, 0.3%)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from itertools import product
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class TurtleTradingOptimization:
    """터틀 트레이딩 전략 최적화 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01', end_date=None):
        """
        Args:
            symbol: 티커 심볼 (default: 'BTC_KRW')
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.optimization_results = None

    def load_data(self):
        """데이터 로드"""
        file_path = f'chart_day/{self.symbol}.parquet'
        print(f"Loading data from {file_path}...")
        df = pd.read_parquet(file_path)

        # 컬럼명 변경 (소문자 -> 대문자)
        df.columns = [col.capitalize() for col in df.columns]

        # 날짜 필터링
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df
        print(f"Loaded {len(self.data)} data points from {df.index[0]} to {df.index[-1]}")
        return self.data

    def turtle_trading_strategy(self, entry_period=20, exit_period=10, slippage=0.002):
        """
        터틀 트레이딩 전략 백테스트

        Args:
            entry_period: 진입 채널 기간 (N일 최고가)
            exit_period: 청산 채널 기간 (M일 최저가)
            slippage: 슬리피지

        Returns:
            df: 백테스트 결과 DataFrame
        """
        df = self.data.copy()

        # 터틀 채널 계산
        df['entry_high'] = df['High'].rolling(window=entry_period).max().shift(1)
        df['exit_low'] = df['Low'].rolling(window=exit_period).min().shift(1)

        # ATR 계산 (리스크 관리용)
        df['prev_close'] = df['Close'].shift(1)
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['prev_close'])
        df['tr3'] = abs(df['Low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=20).mean()

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 최고가 돌파 시 매수 (포지션이 없을 때)
            if df.iloc[i]['High'] > df.iloc[i]['entry_high'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 최저가 하향 돌파 시 매도
            elif df.iloc[i]['Low'] < df.iloc[i]['exit_low'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan
        df['sell_price'] = np.nan

        for i in range(1, len(df)):
            # 매수 진입
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['entry_high']

            # 매도 청산
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                df.iloc[i, df.columns.get_loc('sell_price')] = df.iloc[i]['exit_low']
                df.iloc[i, df.columns.get_loc('returns')] = (df.iloc[i]['sell_price'] / buy_price - 1) - slippage

            # 포지션 유지
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def calculate_metrics(self, df):
        """성과 지표 계산"""
        # 거래 통계
        total_trades = (df['returns'] != 0).sum()
        winning_trades = (df['returns'] > 0).sum()
        losing_trades = (df['returns'] < 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # 수익률
        total_return = (df['cumulative'].iloc[-1] - 1) * 100

        # 연간 수익률 (CAGR)
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['cumulative'].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # MDD (Maximum Drawdown)
        cummax = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - cummax) / cummax
        mdd = drawdown.min() * 100

        # 샤프 비율 (연율화)
        returns = df['returns']
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0

        # 소르티노 비율 (하방 리스크만 고려)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = (returns.mean() / downside_std * np.sqrt(365)) if downside_std > 0 else 0

        # Calmar 비율 (CAGR / abs(MDD))
        calmar = cagr / abs(mdd) if mdd != 0 else 0

        # Profit Factor
        total_profit = df[df['returns'] > 0]['returns'].sum()
        total_loss = abs(df[df['returns'] < 0]['returns'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        # 평균 수익/손실
        avg_win = df[df['returns'] > 0]['returns'].mean() * 100 if winning_trades > 0 else 0
        avg_loss = df[df['returns'] < 0]['returns'].mean() * 100 if losing_trades > 0 else 0

        # 최대 연속 승/패
        df['win_streak'] = (df['returns'] > 0).astype(int)
        df['loss_streak'] = (df['returns'] < 0).astype(int)

        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0

        for i in range(len(df)):
            if df.iloc[i]['returns'] > 0:
                current_win += 1
                current_loss = 0
                max_win_streak = max(max_win_streak, current_win)
            elif df.iloc[i]['returns'] < 0:
                current_loss += 1
                current_win = 0
                max_loss_streak = max(max_loss_streak, current_loss)
            else:
                current_win = 0
                current_loss = 0

        # 평균 보유 기간
        hold_periods = []
        hold_days = 0
        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1:
                hold_days += 1
                if df.iloc[i+1]['position'] == 0 if i+1 < len(df) else True:
                    hold_periods.append(hold_days)
                    hold_days = 0

        avg_holding_period = np.mean(hold_periods) if hold_periods else 0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'avg_holding_period': avg_holding_period
        }

    def optimize_parameters(self,
                          entry_periods=[10, 20, 30, 40, 55],
                          exit_periods=[5, 10, 15, 20],
                          slippages=[0.001, 0.002, 0.003]):
        """
        그리드 서치를 통한 파라미터 최적화

        Args:
            entry_periods: 진입 채널 기간 리스트
            exit_periods: 청산 채널 기간 리스트
            slippages: 슬리피지 리스트

        Returns:
            results_df: 최적화 결과 DataFrame
        """
        print("\n" + "="*80)
        print("터틀 트레이딩 파라미터 최적화 시작...")
        print("="*80)
        print(f"진입 채널: {entry_periods}")
        print(f"청산 채널: {exit_periods}")
        print(f"슬리피지: {[f'{s*100:.1f}%' for s in slippages]}")
        print(f"총 조합 수: {len(entry_periods) * len(exit_periods) * len(slippages)}")
        print("="*80 + "\n")

        results = []
        total_combinations = len(list(product(entry_periods, exit_periods, slippages)))

        with tqdm(total=total_combinations, desc="Optimizing") as pbar:
            for entry_p, exit_p, slip in product(entry_periods, exit_periods, slippages):
                # 청산 기간이 진입 기간보다 크면 스킵
                if exit_p > entry_p:
                    pbar.update(1)
                    continue

                # 백테스트 실행
                df_result = self.turtle_trading_strategy(
                    entry_period=entry_p,
                    exit_period=exit_p,
                    slippage=slip
                )

                # 성과 지표 계산
                metrics = self.calculate_metrics(df_result)

                # 결과 저장
                results.append({
                    'entry_period': entry_p,
                    'exit_period': exit_p,
                    'slippage': slip * 100,
                    **metrics
                })

                pbar.update(1)

        self.optimization_results = pd.DataFrame(results)
        return self.optimization_results

    def print_top_results(self, top_n=10, sort_by='sharpe'):
        """
        상위 결과 출력

        Args:
            top_n: 출력할 결과 수
            sort_by: 정렬 기준 ('sharpe', 'cagr', 'calmar', 'total_return')
        """
        if self.optimization_results is None:
            print("최적화를 먼저 실행해주세요.")
            return

        print("\n" + "="*150)
        print(f"터틀 트레이딩 최적화 결과 (상위 {top_n}개, 정렬: {sort_by.upper()})".center(150))
        print("="*150)

        # 정렬
        sorted_results = self.optimization_results.sort_values(by=sort_by, ascending=False).head(top_n)

        # 출력 포맷 설정
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}')

        # 컬럼 순서 재정렬
        columns_order = [
            'entry_period', 'exit_period', 'slippage',
            'cagr', 'total_return', 'mdd', 'sharpe', 'sortino', 'calmar',
            'win_rate', 'total_trades', 'profit_factor',
            'avg_win', 'avg_loss', 'avg_holding_period'
        ]

        print(sorted_results[columns_order].to_string(index=False))
        print("\n" + "="*150 + "\n")

    def get_best_parameters(self, metric='sharpe'):
        """
        최적 파라미터 반환

        Args:
            metric: 최적화 기준 ('sharpe', 'cagr', 'calmar', 'total_return')

        Returns:
            best_params: 최적 파라미터 딕셔너리
        """
        if self.optimization_results is None:
            print("최적화를 먼저 실행해주세요.")
            return None

        best_row = self.optimization_results.loc[self.optimization_results[metric].idxmax()]

        best_params = {
            'entry_period': int(best_row['entry_period']),
            'exit_period': int(best_row['exit_period']),
            'slippage': best_row['slippage'] / 100,
            'metrics': {
                'CAGR': best_row['cagr'],
                'Total Return': best_row['total_return'],
                'MDD': best_row['mdd'],
                'Sharpe': best_row['sharpe'],
                'Sortino': best_row['sortino'],
                'Calmar': best_row['calmar'],
                'Win Rate': best_row['win_rate'],
                'Total Trades': int(best_row['total_trades']),
                'Profit Factor': best_row['profit_factor']
            }
        }

        return best_params

    def plot_optimization_results(self, save_path='turtle_trading_optimization.png'):
        """최적화 결과 시각화"""
        if self.optimization_results is None:
            print("최적화를 먼저 실행해주세요.")
            return

        df = self.optimization_results

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # 1. Entry Period별 성과 (박스플롯)
        ax1 = fig.add_subplot(gs[0, 0])
        entry_data = [df[df['entry_period'] == ep]['cagr'].values for ep in sorted(df['entry_period'].unique())]
        ax1.boxplot(entry_data, labels=sorted(df['entry_period'].unique()))
        ax1.set_xlabel('Entry Period (days)', fontsize=11)
        ax1.set_ylabel('CAGR (%)', fontsize=11)
        ax1.set_title('CAGR by Entry Period', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Exit Period별 성과 (박스플롯)
        ax2 = fig.add_subplot(gs[0, 1])
        exit_data = [df[df['exit_period'] == ep]['cagr'].values for ep in sorted(df['exit_period'].unique())]
        ax2.boxplot(exit_data, labels=sorted(df['exit_period'].unique()))
        ax2.set_xlabel('Exit Period (days)', fontsize=11)
        ax2.set_ylabel('CAGR (%)', fontsize=11)
        ax2.set_title('CAGR by Exit Period', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Slippage별 성과 (박스플롯)
        ax3 = fig.add_subplot(gs[0, 2])
        slip_data = [df[df['slippage'] == s]['cagr'].values for s in sorted(df['slippage'].unique())]
        ax3.boxplot(slip_data, labels=[f'{s:.1f}%' for s in sorted(df['slippage'].unique())])
        ax3.set_xlabel('Slippage', fontsize=11)
        ax3.set_ylabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR by Slippage', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Entry vs Exit Period 히트맵 (CAGR)
        ax4 = fig.add_subplot(gs[1, 0])
        pivot_cagr = df.groupby(['entry_period', 'exit_period'])['cagr'].mean().unstack()
        sns.heatmap(pivot_cagr, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'CAGR (%)'}, ax=ax4, linewidths=0.5)
        ax4.set_title('CAGR Heatmap (Entry vs Exit)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Exit Period', fontsize=11)
        ax4.set_ylabel('Entry Period', fontsize=11)

        # 5. Entry vs Exit Period 히트맵 (Sharpe)
        ax5 = fig.add_subplot(gs[1, 1])
        pivot_sharpe = df.groupby(['entry_period', 'exit_period'])['sharpe'].mean().unstack()
        sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Sharpe Ratio'}, ax=ax5, linewidths=0.5)
        ax5.set_title('Sharpe Ratio Heatmap (Entry vs Exit)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Exit Period', fontsize=11)
        ax5.set_ylabel('Entry Period', fontsize=11)

        # 6. Entry vs Exit Period 히트맵 (MDD)
        ax6 = fig.add_subplot(gs[1, 2])
        pivot_mdd = df.groupby(['entry_period', 'exit_period'])['mdd'].mean().unstack()
        sns.heatmap(pivot_mdd, annot=True, fmt='.1f', cmap='RdYlGn_r', center=-50,
                   cbar_kws={'label': 'MDD (%)'}, ax=ax6, linewidths=0.5)
        ax6.set_title('MDD Heatmap (Entry vs Exit)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Exit Period', fontsize=11)
        ax6.set_ylabel('Entry Period', fontsize=11)

        # 7. CAGR vs MDD 산점도
        ax7 = fig.add_subplot(gs[2, 0])
        scatter = ax7.scatter(df['mdd'], df['cagr'],
                            c=df['sharpe'], s=100, alpha=0.6, cmap='RdYlGn')
        ax7.set_xlabel('MDD (%)', fontsize=11)
        ax7.set_ylabel('CAGR (%)', fontsize=11)
        ax7.set_title('Risk-Return Profile (CAGR vs MDD)', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.colorbar(scatter, ax=ax7, label='Sharpe Ratio')

        # 8. Sharpe vs Win Rate 산점도
        ax8 = fig.add_subplot(gs[2, 1])
        scatter2 = ax8.scatter(df['win_rate'], df['sharpe'],
                             c=df['cagr'], s=100, alpha=0.6, cmap='RdYlGn')
        ax8.set_xlabel('Win Rate (%)', fontsize=11)
        ax8.set_ylabel('Sharpe Ratio', fontsize=11)
        ax8.set_title('Sharpe vs Win Rate', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.colorbar(scatter2, ax=ax8, label='CAGR (%)')

        # 9. Total Trades 분포
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.hist(df['total_trades'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax9.axvline(df['total_trades'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {df["total_trades"].mean():.0f}')
        ax9.set_xlabel('Total Trades', fontsize=11)
        ax9.set_ylabel('Frequency', fontsize=11)
        ax9.set_title('Total Trades Distribution', fontsize=12, fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')

        # 10. Sharpe Ratio 분포
        ax10 = fig.add_subplot(gs[3, 0])
        ax10.hist(df['sharpe'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax10.axvline(df['sharpe'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {df["sharpe"].mean():.2f}')
        ax10.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax10.set_xlabel('Sharpe Ratio', fontsize=11)
        ax10.set_ylabel('Frequency', fontsize=11)
        ax10.set_title('Sharpe Ratio Distribution', fontsize=12, fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3, axis='y')

        # 11. Calmar Ratio 분포
        ax11 = fig.add_subplot(gs[3, 1])
        # Calmar가 극단값일 수 있으므로 outlier 제거
        calmar_filtered = df['calmar'][(df['calmar'] > -10) & (df['calmar'] < 10)]
        ax11.hist(calmar_filtered, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax11.axvline(calmar_filtered.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {calmar_filtered.mean():.2f}')
        ax11.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax11.set_xlabel('Calmar Ratio', fontsize=11)
        ax11.set_ylabel('Frequency', fontsize=11)
        ax11.set_title('Calmar Ratio Distribution', fontsize=12, fontweight='bold')
        ax11.legend()
        ax11.grid(True, alpha=0.3, axis='y')

        # 12. Profit Factor vs Win Rate
        ax12 = fig.add_subplot(gs[3, 2])
        # Profit Factor가 무한대가 아닌 경우만
        df_filtered = df[df['profit_factor'] != np.inf].copy()
        if len(df_filtered) > 0:
            scatter3 = ax12.scatter(df_filtered['win_rate'], df_filtered['profit_factor'],
                                  c=df_filtered['sharpe'], s=100, alpha=0.6, cmap='RdYlGn')
            ax12.set_xlabel('Win Rate (%)', fontsize=11)
            ax12.set_ylabel('Profit Factor', fontsize=11)
            ax12.set_title('Profit Factor vs Win Rate', fontsize=12, fontweight='bold')
            ax12.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax12.grid(True, alpha=0.3)
            plt.colorbar(scatter3, ax=ax12, label='Sharpe Ratio')

        # 13. 상위 10개 파라미터 조합 (Sharpe 기준)
        ax13 = fig.add_subplot(gs[4, :])
        top10 = df.nlargest(10, 'sharpe').copy()
        top10['params'] = top10.apply(
            lambda x: f"E{int(x['entry_period'])}/X{int(x['exit_period'])}\n({x['slippage']:.1f}%)",
            axis=1
        )

        x_pos = np.arange(len(top10))
        bars = ax13.bar(x_pos, top10['sharpe'], alpha=0.7, color='steelblue', edgecolor='black')

        # 각 바 위에 CAGR 표시
        for i, (idx, row) in enumerate(top10.iterrows()):
            ax13.text(i, row['sharpe'] + 0.05, f"{row['cagr']:.1f}%",
                     ha='center', va='bottom', fontsize=9)

        ax13.set_xticks(x_pos)
        ax13.set_xticklabels(top10['params'], fontsize=9)
        ax13.set_ylabel('Sharpe Ratio', fontsize=11)
        ax13.set_title('Top 10 Parameter Combinations (by Sharpe Ratio, CAGR shown on bars)',
                      fontsize=12, fontweight='bold')
        ax13.grid(True, alpha=0.3, axis='y')
        ax13.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nOptimization chart saved to {save_path}")
        plt.show()

    def backtest_with_best_params(self, metric='sharpe', save_path='turtle_trading_best_backtest.png'):
        """최적 파라미터로 백테스트 실행 및 시각화"""
        best_params = self.get_best_parameters(metric=metric)

        if best_params is None:
            return

        print("\n" + "="*80)
        print(f"최적 파라미터 백테스트 (기준: {metric.upper()})".center(80))
        print("="*80)
        print(f"\n진입 채널: {best_params['entry_period']}일")
        print(f"청산 채널: {best_params['exit_period']}일")
        print(f"슬리피지: {best_params['slippage']*100:.1f}%")
        print("\n성과 지표:")
        for key, value in best_params['metrics'].items():
            print(f"  {key:<20}: {value:>12.2f}")
        print("="*80 + "\n")

        # 백테스트 실행
        df_result = self.turtle_trading_strategy(
            entry_period=best_params['entry_period'],
            exit_period=best_params['exit_period'],
            slippage=best_params['slippage']
        )

        # Buy & Hold 비교
        df_result['buy_hold'] = df_result['Close'] / df_result['Close'].iloc[0]

        # 시각화
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # 1. 가격 차트 + 매매 신호
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df_result.index, df_result['Close'], label='BTC Price', linewidth=1.5, alpha=0.7, color='blue')

        # 진입/청산 채널
        ax1.plot(df_result.index, df_result['entry_high'], label=f'Entry High ({best_params["entry_period"]}d)',
                linewidth=1, alpha=0.5, linestyle='--', color='green')
        ax1.plot(df_result.index, df_result['exit_low'], label=f'Exit Low ({best_params["exit_period"]}d)',
                linewidth=1, alpha=0.5, linestyle='--', color='red')

        # 매수/매도 신호
        buy_signals = df_result[df_result['buy_price'].notna()]
        sell_signals = df_result[df_result['sell_price'].notna()]

        ax1.scatter(buy_signals.index, buy_signals['buy_price'],
                   color='green', marker='^', s=100, alpha=0.8, label='Buy', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['sell_price'],
                   color='red', marker='v', s=100, alpha=0.8, label='Sell', zorder=5)

        ax1.set_title(f'Turtle Trading - Best Parameters (Entry: {best_params["entry_period"]}d, Exit: {best_params["exit_period"]}d)',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=11)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 누적 수익률
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(df_result.index, df_result['cumulative'], label='Turtle Trading', linewidth=2, color='green')
        ax2.plot(df_result.index, df_result['buy_hold'], label='Buy & Hold', linewidth=2, alpha=0.7, color='blue')
        ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cumulative Return', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 3. 드로우다운
        ax3 = fig.add_subplot(gs[2, :])
        cummax = df_result['cumulative'].cummax()
        drawdown = (df_result['cumulative'] - cummax) / cummax * 100

        cummax_bh = df_result['buy_hold'].cummax()
        drawdown_bh = (df_result['buy_hold'] - cummax_bh) / cummax_bh * 100

        ax3.fill_between(df_result.index, drawdown, 0, alpha=0.5, color='red', label='Strategy Drawdown')
        ax3.plot(df_result.index, drawdown_bh, color='blue', alpha=0.5, label='Buy & Hold Drawdown', linewidth=1.5)
        ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 포지션 히스토리
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.fill_between(df_result.index, 0, df_result['position'],
                        alpha=0.3, color='green', label='Position')
        ax4.set_title('Position History', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Position (0=Cash, 1=Long)', fontsize=11)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylim(-0.1, 1.1)
        ax4.grid(True, alpha=0.3)

        # 5. 거래별 수익률
        ax5 = fig.add_subplot(gs[3, 1])
        trade_returns = df_result[df_result['returns'] != 0]['returns'] * 100
        colors = ['green' if r > 0 else 'red' for r in trade_returns]
        ax5.bar(range(len(trade_returns)), trade_returns, color=colors, alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax5.set_title('Individual Trade Returns', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Trade Number', fontsize=11)
        ax5.set_ylabel('Return (%)', fontsize=11)
        ax5.grid(True, alpha=0.3, axis='y')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Backtest chart saved to {save_path}")
        plt.show()

        return df_result


def main():
    """메인 함수"""
    # 최적화 객체 생성
    optimizer = TurtleTradingOptimization(
        symbol='BTC_KRW',
        start_date='2018-01-01',
        end_date=None
    )

    # 데이터 로드
    optimizer.load_data()

    # 파라미터 최적화 실행
    results = optimizer.optimize_parameters(
        entry_periods=[10, 20, 30, 40, 55],
        exit_periods=[5, 10, 15, 20],
        slippages=[0.001, 0.002, 0.003]
    )

    # 결과 저장
    print("\nSaving optimization results to CSV...")
    results.to_csv('turtle_trading_optimization_results.csv', index=False)
    print("Results saved to turtle_trading_optimization_results.csv")

    # 상위 결과 출력 (여러 기준으로)
    print("\n" + "="*80)
    print("다양한 기준별 상위 결과")
    print("="*80)

    for metric in ['sharpe', 'cagr', 'calmar', 'sortino']:
        optimizer.print_top_results(top_n=10, sort_by=metric)

    # 최적화 결과 시각화
    optimizer.plot_optimization_results(save_path='turtle_trading_optimization.png')

    # 각 기준별 최적 파라미터 출력
    print("\n" + "="*80)
    print("각 기준별 최적 파라미터")
    print("="*80)

    for metric in ['sharpe', 'cagr', 'calmar', 'sortino']:
        best = optimizer.get_best_parameters(metric=metric)
        print(f"\n[{metric.upper()} 기준 최적 파라미터]")
        print(f"  진입 채널: {best['entry_period']}일")
        print(f"  청산 채널: {best['exit_period']}일")
        print(f"  슬리피지: {best['slippage']*100:.1f}%")
        print(f"  성과: CAGR={best['metrics']['CAGR']:.2f}%, Sharpe={best['metrics']['Sharpe']:.2f}")

    # Sharpe 기준 최적 파라미터로 상세 백테스트
    optimizer.backtest_with_best_params(
        metric='sharpe',
        save_path='turtle_trading_best_backtest.png'
    )

    print("\n" + "="*80)
    print("터틀 트레이딩 최적화 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
