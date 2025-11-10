#!/usr/bin/env python3
"""
종합 전략 비교: 기존 전략 + SMA 윈도우 최적화

기존 전략:
- Turtle Trading (Fixed)
- RSI 55

SMA 윈도우 최적화:
- SMA 5, 10, 20, 30, 50, 100, 200

모든 전략의 포트폴리오 성과를 비교합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ComprehensiveStrategyComparison:
    """종합 전략 비교"""

    def __init__(self, symbols, start_date, end_date, sma_windows, slippage=0.002):
        """
        Parameters:
        -----------
        symbols : list
            백테스트할 심볼 리스트
        start_date : str
            시작일 (YYYY-MM-DD)
        end_date : str
            종료일 (YYYY-MM-DD)
        sma_windows : list
            테스트할 SMA 윈도우 기간 리스트
        slippage : float
            슬리피지 (기본값: 0.2%)
        """
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.sma_windows = sorted(sma_windows)
        self.slippage = slippage

        self.data = {}
        self.strategy_results = {}  # {strategy_name: {symbol: DataFrame}}
        self.portfolio_results = {}  # {strategy_name: DataFrame}
        self.metrics = {}  # {strategy_name: {metric: value}}

    def load_data(self):
        """데이터 로드"""
        print("=" * 80)
        print("데이터 로딩 중...")
        print("=" * 80)

        data_dir = Path('chart_day')

        for symbol in self.symbols:
            file_path = data_dir / f"{symbol}.parquet"

            if not file_path.exists():
                print(f"경고: {symbol} 데이터 파일을 찾을 수 없습니다: {file_path}")
                continue

            df = pd.read_parquet(file_path)
            df.index = pd.to_datetime(df.index)

            # 날짜 필터링
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            # 컬럼명 표준화
            df.columns = [col.capitalize() for col in df.columns]

            self.data[symbol] = df
            print(f"✓ {symbol}: {len(df):,} 일 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")

        print(f"\n총 {len(self.data)}개 심볼 로드 완료\n")

    def strategy_turtle_trading(self, df):
        """
        Turtle Trading 전략 (Fixed)

        - Entry: 20-day high breakout
        - Exit: 10-day low breakdown
        - Execution: Close price + slippage
        """
        df = df.copy()

        # 20일 최고가, 10일 최저가
        df['high_20'] = df['High'].rolling(window=20).max()
        df['low_10'] = df['Low'].rolling(window=10).min()

        # 포지션 초기화
        df['position'] = 0
        position = 0

        for i in range(1, len(df)):
            # 전날 포지션 유지
            position = df.iloc[i - 1]['position']

            # Entry: 고점 돌파
            if position == 0 and df.iloc[i]['High'] >= df.iloc[i - 1]['high_20']:
                position = 1

            # Exit: 저점 이탈
            elif position == 1 and df.iloc[i]['Low'] <= df.iloc[i - 1]['low_10']:
                position = 0

            df.iloc[i, df.columns.get_loc('position')] = position

        # 포지션 변경 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage   # Buy
        slippage_cost[df['position_change'] == -1] = -self.slippage  # Sell

        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_rsi_55(self, df, rsi_period=14, rsi_threshold=55):
        """
        RSI 55 전략

        - Entry: RSI >= 55
        - Exit: RSI < 55
        """
        df = df.copy()

        # RSI 계산
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 포지션
        df['position'] = np.where(df['RSI'] >= rsi_threshold, 1, 0)

        # 포지션 변경 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage   # Buy
        slippage_cost[df['position_change'] == -1] = -self.slippage  # Sell

        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_sma(self, df, sma_period):
        """
        SMA 크로스오버 전략

        - Price >= SMA: Buy/Hold
        - Price < SMA: Sell/Cash
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 포지션
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)

        # 포지션 변경 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage   # Buy
        slippage_cost[df['position_change'] == -1] = -self.slippage  # Sell

        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def run_all_strategies(self):
        """모든 전략 실행"""
        print("=" * 80)
        print("모든 전략 백테스트 실행 중...")
        print("=" * 80)

        strategies = ['Turtle Trading', 'RSI 55'] + [f'SMA {w}' for w in self.sma_windows]
        total_combinations = len(strategies) * len(self.symbols)
        current = 0

        # Turtle Trading
        self.strategy_results['Turtle Trading'] = {}
        for symbol in self.symbols:
            current += 1
            print(f"[{current}/{total_combinations}] Turtle Trading - {symbol} 처리 중...")
            df = self.data[symbol].copy()
            result = self.strategy_turtle_trading(df)
            self.strategy_results['Turtle Trading'][symbol] = result

        # RSI 55
        self.strategy_results['RSI 55'] = {}
        for symbol in self.symbols:
            current += 1
            print(f"[{current}/{total_combinations}] RSI 55 - {symbol} 처리 중...")
            df = self.data[symbol].copy()
            result = self.strategy_rsi_55(df)
            self.strategy_results['RSI 55'][symbol] = result

        # SMA 윈도우들
        for window in self.sma_windows:
            strategy_name = f'SMA {window}'
            self.strategy_results[strategy_name] = {}

            for symbol in self.symbols:
                current += 1
                print(f"[{current}/{total_combinations}] SMA {window} - {symbol} 처리 중...")
                df = self.data[symbol].copy()
                result = self.strategy_sma(df, sma_period=window)
                self.strategy_results[strategy_name][symbol] = result

        print(f"\n총 {total_combinations}개 조합 백테스트 완료\n")

    def create_portfolios(self):
        """각 전략별 포트폴리오 생성 (equal-weight)"""
        print("=" * 80)
        print("포트폴리오 생성 중...")
        print("=" * 80)

        weight = 1.0 / len(self.symbols)

        for strategy_name in self.strategy_results.keys():
            # 모든 심볼의 공통 날짜 찾기
            all_indices = [self.strategy_results[strategy_name][symbol].index
                          for symbol in self.symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            # Equal-weight 포트폴리오 수익률 계산
            portfolio_returns = pd.Series(0.0, index=common_index)

            for symbol in self.symbols:
                symbol_returns = self.strategy_results[strategy_name][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight

            # 포트폴리오 결과 저장
            portfolio_df = pd.DataFrame({
                'returns': portfolio_returns,
                'cumulative': (1 + portfolio_returns).cumprod()
            })

            self.portfolio_results[strategy_name] = portfolio_df

            print(f"✓ {strategy_name:20s}: {len(portfolio_df):,} 일")

        print(f"\n총 {len(self.portfolio_results)}개 포트폴리오 생성 완료\n")

    def calculate_metrics(self, returns_series):
        """성과 지표 계산"""
        returns = returns_series.dropna()

        if len(returns) == 0:
            return {
                'Total Return (%)': 0,
                'CAGR (%)': 0,
                'MDD (%)': 0,
                'Sharpe Ratio': 0,
                'Total Trades': 0
            }

        cumulative = (1 + returns).cumprod()
        total_return = (cumulative.iloc[-1] - 1) * 100

        years = len(returns) / 252
        cagr = ((cumulative.iloc[-1] ** (1 / years)) - 1) * 100 if years > 0 else 0

        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        mdd = drawdown.min()

        excess_returns = returns
        sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0

        total_trades = (returns != 0).sum()

        return {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Total Trades': total_trades
        }

    def calculate_all_metrics(self):
        """모든 포트폴리오의 성과 지표 계산"""
        print("=" * 80)
        print("성과 지표 계산 중...")
        print("=" * 80)

        for strategy_name in self.portfolio_results.keys():
            portfolio_returns = self.portfolio_results[strategy_name]['returns']
            self.metrics[strategy_name] = self.calculate_metrics(portfolio_returns)

            print(f"{strategy_name:20s}: "
                  f"Return={self.metrics[strategy_name]['Total Return (%)']:>8,.1f}%, "
                  f"CAGR={self.metrics[strategy_name]['CAGR (%)']:>6,.1f}%, "
                  f"Sharpe={self.metrics[strategy_name]['Sharpe Ratio']:>5.2f}")

        print()

    def print_metrics_table(self):
        """성과 지표 테이블 출력"""
        print("=" * 80)
        print("종합 전략 비교 - 포트폴리오 성과")
        print("=" * 80)

        # 데이터프레임 생성
        metrics_data = []
        for strategy_name in self.portfolio_results.keys():
            metrics_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': self.metrics[strategy_name]['Total Return (%)'],
                'CAGR (%)': self.metrics[strategy_name]['CAGR (%)'],
                'MDD (%)': self.metrics[strategy_name]['MDD (%)'],
                'Sharpe Ratio': self.metrics[strategy_name]['Sharpe Ratio'],
                'Total Trades': self.metrics[strategy_name]['Total Trades']
            })

        df = pd.DataFrame(metrics_data)

        # Total Return으로 정렬
        df = df.sort_values('Total Return (%)', ascending=False)

        # 포맷팅하여 출력
        print("\n" + df.to_string(index=False))
        print()

        # 최적 전략 찾기
        best_return_idx = df['Total Return (%)'].idxmax()
        best_sharpe_idx = df['Sharpe Ratio'].idxmax()
        best_cagr_idx = df['CAGR (%)'].idxmax()

        print("=" * 80)
        print("최적 전략")
        print("=" * 80)
        print(f"최고 수익률: {df.loc[best_return_idx, 'Strategy']} "
              f"({df.loc[best_return_idx, 'Total Return (%)']:,.1f}%)")
        print(f"최고 샤프비율: {df.loc[best_sharpe_idx, 'Strategy']} "
              f"({df.loc[best_sharpe_idx, 'Sharpe Ratio']:.2f})")
        print(f"최고 CAGR: {df.loc[best_cagr_idx, 'Strategy']} "
              f"({df.loc[best_cagr_idx, 'CAGR (%)']:.1f}%)")
        print()

        # CSV 저장
        csv_path = 'comprehensive_strategy_comparison_metrics.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"성과 지표를 {csv_path}에 저장했습니다.\n")

        return df

    def plot_comprehensive_comparison(self):
        """종합 전략 비교 시각화"""
        print("=" * 80)
        print("종합 비교 차트 생성 중...")
        print("=" * 80)

        fig = plt.figure(figsize=(24, 16))

        # 색상 팔레트
        all_strategies = list(self.portfolio_results.keys())
        colors = sns.color_palette("husl", len(all_strategies))
        color_map = dict(zip(all_strategies, colors))

        # 1. 포트폴리오 누적 수익률 비교 (전체)
        ax1 = plt.subplot(3, 4, 1)
        for strategy_name in all_strategies:
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            ax1.plot(cumulative.index, cumulative.values,
                    label=strategy_name, linewidth=1.5,
                    color=color_map[strategy_name], alpha=0.8)
        ax1.set_yscale('log')
        ax1.set_title('All Strategies - Cumulative Returns (Log)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='best', fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        # 2. SMA 윈도우만 비교
        ax2 = plt.subplot(3, 4, 2)
        sma_strategies = [f'SMA {w}' for w in self.sma_windows]
        for strategy_name in sma_strategies:
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            ax2.plot(cumulative.index, cumulative.values,
                    label=strategy_name, linewidth=2,
                    color=color_map[strategy_name])
        ax2.set_yscale('log')
        ax2.set_title('SMA Windows Comparison (Log)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 총 수익률 비교 (Top 10)
        ax3 = plt.subplot(3, 4, 3)
        total_returns = [self.metrics[s]['Total Return (%)'] for s in all_strategies]
        sorted_indices = np.argsort(total_returns)[::-1][:10]
        top_strategies = [all_strategies[i] for i in sorted_indices]
        top_returns = [total_returns[i] for i in sorted_indices]
        top_colors = [color_map[all_strategies[i]] for i in sorted_indices]

        bars = ax3.barh(range(len(top_strategies)), top_returns, color=top_colors)
        ax3.set_yticks(range(len(top_strategies)))
        ax3.set_yticklabels(top_strategies, fontsize=9)
        ax3.set_title('Top 10 Total Returns', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Total Return (%)')
        ax3.grid(True, alpha=0.3, axis='x')
        # 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:,.0f}%', ha='left', va='center', fontsize=8)

        # 4. 샤프 비율 비교 (Top 10)
        ax4 = plt.subplot(3, 4, 4)
        sharpes = [self.metrics[s]['Sharpe Ratio'] for s in all_strategies]
        sorted_indices = np.argsort(sharpes)[::-1][:10]
        top_strategies = [all_strategies[i] for i in sorted_indices]
        top_sharpes = [sharpes[i] for i in sorted_indices]
        top_colors = [color_map[all_strategies[i]] for i in sorted_indices]

        bars = ax4.barh(range(len(top_strategies)), top_sharpes, color=top_colors)
        ax4.set_yticks(range(len(top_strategies)))
        ax4.set_yticklabels(top_strategies, fontsize=9)
        ax4.set_title('Top 10 Sharpe Ratios', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Sharpe Ratio')
        ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='x')
        # 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center', fontsize=8)

        # 5. CAGR 비교 (SMA 윈도우)
        ax5 = plt.subplot(3, 4, 5)
        sma_cagrs = [self.metrics[s]['CAGR (%)'] for s in sma_strategies]
        bars = ax5.bar(range(len(sma_strategies)), sma_cagrs,
                      color=[color_map[s] for s in sma_strategies])
        ax5.set_xticks(range(len(sma_strategies)))
        ax5.set_xticklabels(sma_strategies, rotation=45, fontsize=9)
        ax5.set_title('CAGR - SMA Windows', fontsize=11, fontweight='bold')
        ax5.set_ylabel('CAGR (%)')
        ax5.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        # 6. MDD 비교 (SMA 윈도우)
        ax6 = plt.subplot(3, 4, 6)
        sma_mdds = [self.metrics[s]['MDD (%)'] for s in sma_strategies]
        bars = ax6.bar(range(len(sma_strategies)), sma_mdds,
                      color=[color_map[s] for s in sma_strategies])
        ax6.set_xticks(range(len(sma_strategies)))
        ax6.set_xticklabels(sma_strategies, rotation=45, fontsize=9)
        ax6.set_title('MDD - SMA Windows (Higher is Better)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('MDD (%)')
        ax6.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        # 7. Return vs Risk (전체)
        ax7 = plt.subplot(3, 4, 7)
        returns = [self.metrics[s]['CAGR (%)'] for s in all_strategies]
        risks = [abs(self.metrics[s]['MDD (%)']) for s in all_strategies]
        sharpes_all = [self.metrics[s]['Sharpe Ratio'] for s in all_strategies]

        scatter = ax7.scatter(risks, returns, c=sharpes_all, cmap='RdYlGn',
                             s=150, alpha=0.7, edgecolors='black', linewidth=1)
        for i, strategy in enumerate(all_strategies):
            if strategy in ['Turtle Trading', 'RSI 55'] or strategy in [f'SMA {w}' for w in [5, 20, 50, 100, 200]]:
                ax7.annotate(strategy, (risks[i], returns[i]),
                           fontsize=8, ha='center', va='bottom')
        ax7.set_title('Return vs Risk (All Strategies)', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Risk (|MDD| %)')
        ax7.set_ylabel('Return (CAGR %)')
        plt.colorbar(scatter, ax=ax7, label='Sharpe Ratio')
        ax7.grid(True, alpha=0.3)

        # 8. Drawdown 비교 (주요 전략)
        ax8 = plt.subplot(3, 4, 8)
        main_strategies = ['Turtle Trading', 'RSI 55', 'SMA 20', 'SMA 30', 'SMA 50']
        for strategy_name in main_strategies:
            if strategy_name in self.portfolio_results:
                cumulative = self.portfolio_results[strategy_name]['cumulative']
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max * 100
                ax8.plot(drawdown.index, drawdown.values,
                        label=strategy_name, linewidth=1.5,
                        color=color_map[strategy_name])
        ax8.set_title('Drawdown - Main Strategies', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Drawdown (%)')
        ax8.legend(loc='best', fontsize=8)
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 9. 거래 횟수 비교 (주요 전략)
        ax9 = plt.subplot(3, 4, 9)
        main_trades = [self.metrics[s]['Total Trades'] for s in main_strategies if s in self.portfolio_results]
        main_labels = [s for s in main_strategies if s in self.portfolio_results]
        bars = ax9.bar(range(len(main_labels)), main_trades,
                      color=[color_map[s] for s in main_labels])
        ax9.set_xticks(range(len(main_labels)))
        ax9.set_xticklabels(main_labels, rotation=45, fontsize=9)
        ax9.set_title('Total Trades - Main Strategies', fontsize=11, fontweight='bold')
        ax9.set_ylabel('Total Trades')
        ax9.grid(True, alpha=0.3, axis='y')
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=8)

        # 10. 기존 전략 vs 최고 SMA
        ax10 = plt.subplot(3, 4, 10)
        # 최고 성과 SMA 찾기
        best_sma = max(sma_strategies, key=lambda s: self.metrics[s]['Total Return (%)'])
        comparison_strategies = ['Turtle Trading', 'RSI 55', best_sma]

        for strategy_name in comparison_strategies:
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            ax10.plot(cumulative.index, cumulative.values,
                    label=strategy_name, linewidth=2.5,
                    color=color_map[strategy_name])
        ax10.set_yscale('log')
        ax10.set_title(f'Best Performers Comparison (Log)', fontsize=11, fontweight='bold')
        ax10.set_ylabel('Cumulative Return')
        ax10.legend(loc='best', fontsize=9)
        ax10.grid(True, alpha=0.3)

        # 11. 성과 지표 요약 (레이더 차트)
        ax11 = plt.subplot(3, 4, 11, projection='polar')

        # 정규화된 지표 계산
        def normalize_metric(values, reverse=False):
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [0.5] * len(values)
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
            return [1 - n for n in normalized] if reverse else normalized

        categories = ['Return', 'CAGR', 'Sharpe', 'MDD']
        num_vars = len(categories)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        for strategy_name in comparison_strategies:
            values = [
                self.metrics[strategy_name]['Total Return (%)'],
                self.metrics[strategy_name]['CAGR (%)'],
                self.metrics[strategy_name]['Sharpe Ratio'] * 100,
                self.metrics[strategy_name]['MDD (%)']
            ]

            # 정규화
            all_returns = [self.metrics[s]['Total Return (%)'] for s in comparison_strategies]
            all_cagrs = [self.metrics[s]['CAGR (%)'] for s in comparison_strategies]
            all_sharpes = [self.metrics[s]['Sharpe Ratio'] * 100 for s in comparison_strategies]
            all_mdds = [self.metrics[s]['MDD (%)'] for s in comparison_strategies]

            normalized = [
                normalize_metric(all_returns)[comparison_strategies.index(strategy_name)],
                normalize_metric(all_cagrs)[comparison_strategies.index(strategy_name)],
                normalize_metric(all_sharpes)[comparison_strategies.index(strategy_name)],
                normalize_metric(all_mdds, reverse=True)[comparison_strategies.index(strategy_name)]
            ]
            normalized += normalized[:1]

            ax11.plot(angles, normalized, 'o-', linewidth=2,
                     label=strategy_name, color=color_map[strategy_name])
            ax11.fill(angles, normalized, alpha=0.15, color=color_map[strategy_name])

        ax11.set_xticks(angles[:-1])
        ax11.set_xticklabels(categories, fontsize=9)
        ax11.set_ylim(0, 1)
        ax11.set_title('Performance Radar (Normalized)', fontsize=11, fontweight='bold', pad=20)
        ax11.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax11.grid(True)

        # 12. 요약 테이블
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')

        # 최고 성과 찾기
        best_return_strategy = max(all_strategies, key=lambda s: self.metrics[s]['Total Return (%)'])
        best_sharpe_strategy = max(all_strategies, key=lambda s: self.metrics[s]['Sharpe Ratio'])
        best_cagr_strategy = max(all_strategies, key=lambda s: self.metrics[s]['CAGR (%)'])

        summary_text = f"""
COMPREHENSIVE STRATEGY COMPARISON
{'=' * 50}

BEST PERFORMERS:

Total Return:
  {best_return_strategy}
  {self.metrics[best_return_strategy]['Total Return (%)']:,.1f}%

Sharpe Ratio:
  {best_sharpe_strategy}
  {self.metrics[best_sharpe_strategy]['Sharpe Ratio']:.2f}

CAGR:
  {best_cagr_strategy}
  {self.metrics[best_cagr_strategy]['CAGR (%)']:.1f}%

PORTFOLIO DETAILS:
  Equal-weight: {', '.join(self.symbols)}
  Period: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}
  Slippage: {self.slippage * 100:.2f}%
  Strategies Tested: {len(all_strategies)}
"""

        ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Comprehensive Strategy Comparison: All Strategies vs SMA Window Optimization',
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # 저장
        output_path = 'comprehensive_strategy_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 종합 비교 차트를 {output_path}에 저장했습니다.")
        plt.close()

    def run_analysis(self):
        """전체 분석 실행"""
        print("\n" + "=" * 80)
        print("종합 전략 비교 분석 시작")
        print("=" * 80)
        print(f"심볼: {', '.join(self.symbols)}")
        print(f"기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        print(f"전략: Turtle Trading, RSI 55, SMA Windows {self.sma_windows}")
        print(f"슬리피지: {self.slippage * 100:.2f}%")
        print()

        # 1. 데이터 로드
        self.load_data()

        # 2. 모든 전략 실행
        self.run_all_strategies()

        # 3. 포트폴리오 생성
        self.create_portfolios()

        # 4. 성과 지표 계산
        self.calculate_all_metrics()

        # 5. 결과 출력
        self.print_metrics_table()

        # 6. 종합 비교 차트
        self.plot_comprehensive_comparison()

        print("=" * 80)
        print("종합 분석 완료!")
        print("=" * 80)


def main():
    """메인 함수"""
    # 설정
    symbols = ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW']
    start_date = '2018-01-01'
    end_date = '2025-11-07'
    sma_windows = [5, 10, 20, 30, 50, 100, 200]
    slippage = 0.002  # 0.2%

    # 분석 실행
    comparison = ComprehensiveStrategyComparison(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        sma_windows=sma_windows,
        slippage=slippage
    )

    comparison.run_analysis()


if __name__ == '__main__':
    main()
