"""
알트코인 역추세 전략 백테스팅

5개 알트코인에 역추세(Contrarian/Mean Reversion) 전략 적용:
선택된 알트코인: STEEM, ANKR, CHZ, MANA, ZIL

전략: 볼린저 밴드 역추세 전략
- 가격이 볼린저 밴드 하단을 하회하면 매수 (과매도 영역)
- 가격이 볼린저 밴드 상단을 상회하면 매도 (과매수 영역)
- 중간선(MA) 돌파 시 청산
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


class AltcoinContrarianStrategy:
    """알트코인 역추세 전략 클래스"""

    def __init__(self, symbols=['STEEM_KRW', 'ANKR_KRW', 'CHZ_KRW', 'MANA_KRW', 'ZIL_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일 (None이면 오늘까지)
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = {}
        self.strategy_results = {}
        self.portfolio_results = {}

    def load_data(self):
        """모든 종목 데이터 로드"""
        print("="*80)
        print("Loading data for all symbols...")
        print("="*80)

        for symbol in self.symbols:
            file_path = f'chart_day/{symbol}.parquet'
            print(f"\nLoading {symbol} from {file_path}...")
            df = pd.read_parquet(file_path)

            # 컬럼명 변경 (소문자 -> 대문자)
            df.columns = [col.capitalize() for col in df.columns]

            # 날짜 필터링
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            self.data[symbol] = df
            print(f"  Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

        print("\n" + "="*80)
        print("Data loading completed!")
        print("="*80 + "\n")

    # ==================== 전략 1: 볼린저 밴드 역추세 ====================
    def strategy_bollinger_contrarian(self, df, bb_period=20, bb_std=2):
        """
        볼린저 밴드 역추세 전략
        - 가격이 하단 밴드 하회 시 매수 (과매도)
        - 가격이 상단 밴드 상회 시 매도 (과매수)
        - 중간선(MA) 터치 시 청산

        Args:
            df: 가격 데이터프레임
            bb_period: 볼린저 밴드 기간 (default: 20)
            bb_std: 표준편차 배수 (default: 2)
        """
        df = df.copy()

        # 볼린저 밴드 계산
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)

        # %B 계산 (가격이 밴드 내에서 어디에 위치하는지)
        df['Percent_B'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # 포지션 관리
        df['position'] = 0  # 0: 현금, 1: 롱, -1: 숏 (공매도는 실제로는 하지 않음, 단지 매도 신호)
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            # 이전 포지션 유지
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 현재 포지션이 없을 때
            if df.iloc[i-1]['position'] == 0:
                # 하단 밴드 하회 시 매수 (역추세 - 반등 기대)
                if df.iloc[i]['Close'] < df.iloc[i]['BB_Lower']:
                    df.iloc[i, df.columns.get_loc('position')] = 1

            # 롱 포지션 보유 중일 때
            elif df.iloc[i-1]['position'] == 1:
                # 상단 밴드 상회 또는 중간선 돌파 시 매도
                if (df.iloc[i]['Close'] > df.iloc[i]['BB_Upper'] or
                    df.iloc[i]['Close'] > df.iloc[i]['BB_Middle']):
                    df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0

        for i in range(1, len(df)):
            # 매수 진입
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['Close'] * (1 + self.slippage)

            # 매도 청산
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                sell_price = df.iloc[i]['Close'] * (1 - self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)

            # 포지션 유지
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 2: RSI 역추세 ====================
    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def strategy_rsi_contrarian(self, df, rsi_period=14, oversold=30, overbought=70):
        """
        RSI 역추세 전략
        - RSI < 30: 과매도 → 매수
        - RSI > 70: 과매수 → 매도
        - 중간선(50) 돌파 시 청산

        Args:
            df: 가격 데이터프레임
            rsi_period: RSI 계산 기간 (default: 14)
            oversold: 과매도 기준 (default: 30)
            overbought: 과매수 기준 (default: 70)
        """
        df = df.copy()

        # RSI 계산
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)

        # 포지션 관리
        df['position'] = 0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 현재 포지션이 없을 때
            if df.iloc[i-1]['position'] == 0:
                # 과매도 영역에서 매수
                if df.iloc[i]['RSI'] < oversold:
                    df.iloc[i, df.columns.get_loc('position')] = 1

            # 롱 포지션 보유 중일 때
            elif df.iloc[i-1]['position'] == 1:
                # 과매수 영역 또는 중간선 돌파 시 매도
                if df.iloc[i]['RSI'] > overbought or df.iloc[i]['RSI'] > 50:
                    df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['Close'] * (1 + self.slippage)
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                sell_price = df.iloc[i]['Close'] * (1 - self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 3: Z-Score 역추세 ====================
    def strategy_zscore_contrarian(self, df, lookback=20, entry_threshold=2.0, exit_threshold=0.5):
        """
        Z-Score 역추세 전략
        - Z-Score < -threshold: 과매도 → 매수
        - Z-Score > threshold: 과매수 → 매도
        - |Z-Score| < exit_threshold: 청산

        Args:
            df: 가격 데이터프레임
            lookback: Z-Score 계산 기간 (default: 20)
            entry_threshold: 진입 임계값 (default: 2.0)
            exit_threshold: 청산 임계값 (default: 0.5)
        """
        df = df.copy()

        # Z-Score 계산
        df['Price_Mean'] = df['Close'].rolling(window=lookback).mean()
        df['Price_Std'] = df['Close'].rolling(window=lookback).std()
        df['Z_Score'] = (df['Close'] - df['Price_Mean']) / df['Price_Std']

        # 포지션 관리
        df['position'] = 0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 현재 포지션이 없을 때
            if df.iloc[i-1]['position'] == 0:
                # 과매도 (Z-Score가 음수로 큰 값)
                if df.iloc[i]['Z_Score'] < -entry_threshold:
                    df.iloc[i, df.columns.get_loc('position')] = 1

            # 롱 포지션 보유 중일 때
            elif df.iloc[i-1]['position'] == 1:
                # 평균 회귀 또는 과매수
                if abs(df.iloc[i]['Z_Score']) < exit_threshold or df.iloc[i]['Z_Score'] > entry_threshold:
                    df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['Close'] * (1 + self.slippage)
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                sell_price = df.iloc[i]['Close'] * (1 - self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)
            elif df.iloc[i]['position'] == 1:
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 실행 ====================
    def run_all_strategies(self):
        """모든 전략을 모든 종목에 대해 실행"""
        strategies = {
            'Bollinger Contrarian': lambda df: self.strategy_bollinger_contrarian(df, bb_period=20, bb_std=2),
            'RSI Contrarian': lambda df: self.strategy_rsi_contrarian(df, rsi_period=14, oversold=30, overbought=70),
            'Z-Score Contrarian': lambda df: self.strategy_zscore_contrarian(df, lookback=20, entry_threshold=2.0, exit_threshold=0.5)
        }

        print("\n" + "="*80)
        print("Running all contrarian strategies for all altcoins...")
        print("="*80 + "\n")

        for strategy_name, strategy_func in strategies.items():
            print(f"\n>>> Running {strategy_name} strategy...")
            self.strategy_results[strategy_name] = {}

            for symbol in self.symbols:
                print(f"  - {symbol}...")
                df = self.data[symbol].copy()
                result = strategy_func(df)
                self.strategy_results[strategy_name][symbol] = result

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80 + "\n")

    # ==================== 포트폴리오 구성 ====================
    def create_portfolios(self):
        """각 전략별로 동일 비중 포트폴리오 생성"""
        print("\n" + "="*80)
        print("Creating equal-weight portfolios...")
        print("="*80 + "\n")

        weight = 1.0 / len(self.symbols)  # 동일 비중 (20% each for 5 coins)

        for strategy_name in self.strategy_results.keys():
            print(f"\n>>> Creating portfolio for {strategy_name}...")

            # 모든 종목의 공통 날짜 인덱스 찾기
            all_indices = [self.strategy_results[strategy_name][symbol].index
                          for symbol in self.symbols]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            # 포트폴리오 수익률 계산
            portfolio_returns = pd.Series(0.0, index=common_index)

            for symbol in self.symbols:
                symbol_returns = self.strategy_results[strategy_name][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight
                print(f"  - Added {symbol} with weight {weight:.2%}")

            # 포트폴리오 누적 수익률
            portfolio_cumulative = (1 + portfolio_returns).cumprod()

            # 결과 저장
            self.portfolio_results[strategy_name] = pd.DataFrame({
                'returns': portfolio_returns,
                'cumulative': portfolio_cumulative
            }, index=common_index)

        print("\n" + "="*80)
        print("Portfolio creation completed!")
        print("="*80 + "\n")

    # ==================== 성과 지표 계산 ====================
    def calculate_metrics(self, returns_series, name):
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

        return {
            'Strategy': name,
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades),
            'Profit Factor': profit_factor
        }

    def calculate_all_metrics(self):
        """모든 전략 및 종목별 성과 지표 계산"""
        metrics_list = []

        # 각 전략의 포트폴리오 성과
        for strategy_name in self.portfolio_results.keys():
            returns = self.portfolio_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, f"{strategy_name} Portfolio")
            metrics_list.append(metrics)

        # 개별 종목별 성과
        for strategy_name in self.strategy_results.keys():
            for symbol in self.symbols:
                returns = self.strategy_results[strategy_name][symbol]['returns']
                metrics = self.calculate_metrics(returns, f"{strategy_name} - {symbol.split('_')[0]}")
                metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================
    def plot_comparison(self, metrics_df, save_path='altcoin_contrarian_comparison.png'):
        """포트폴리오 비교 시각화"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # 1. 포트폴리오 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            ax1.plot(cumulative.index, cumulative, label=f'{strategy_name} Portfolio',
                    linewidth=2.5, alpha=0.8)

        ax1.set_title('Altcoin Contrarian Strategy Portfolio Comparison - Equal-Weight: STEEM, ANKR, CHZ, MANA, ZIL',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 포트폴리오만 필터링
        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()

        # 2. 총 수익률 비교
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = portfolio_metrics.sort_values('Total Return (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
        ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = portfolio_metrics.sort_values('CAGR (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['CAGR (%)']]
        ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = portfolio_metrics.sort_values('MDD (%)', ascending=False)
        ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color='crimson', alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = portfolio_metrics.sort_values('Sharpe Ratio', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Sharpe Ratio']]
        ax5.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Return vs Risk 산점도
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.scatter(portfolio_metrics['MDD (%)'], portfolio_metrics['CAGR (%)'],
                   s=300, alpha=0.6, c=portfolio_metrics['Sharpe Ratio'], cmap='RdYlGn')
        for idx, row in portfolio_metrics.iterrows():
            ax6.annotate(row['Strategy'].replace(' Portfolio', ''),
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=10, ha='center', va='bottom')
        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 7. Profit Factor 비교
        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = portfolio_metrics.copy()
        sorted_df = sorted_df[sorted_df['Profit Factor'] != np.inf]
        if len(sorted_df) > 0:
            sorted_df = sorted_df.sort_values('Profit Factor', ascending=True)
            colors = ['green' if x > 1 else 'red' for x in sorted_df['Profit Factor']]
            ax7.barh(sorted_df['Strategy'], sorted_df['Profit Factor'], color=colors, alpha=0.7)
        ax7.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax7.set_xlabel('Profit Factor', fontsize=11)
        ax7.set_title('Profit Factor Comparison', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8-10. 각 전략별 종목 비교
        for idx, strategy_name in enumerate(self.strategy_results.keys()):
            strategy_metrics = metrics_df[metrics_df['Strategy'].str.contains(strategy_name)].copy()
            strategy_metrics = strategy_metrics[~strategy_metrics['Strategy'].str.contains('Portfolio')]

            if len(strategy_metrics) > 0:
                ax = fig.add_subplot(gs[3, idx])
                sorted_df = strategy_metrics.sort_values('Total Return (%)', ascending=True)
                colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
                strategy_labels = [s.replace(f'{strategy_name} - ', '') for s in sorted_df['Strategy']]
                ax.barh(strategy_labels, sorted_df['Total Return (%)'], color=colors, alpha=0.7)
                ax.set_xlabel('Total Return (%)', fontsize=10)
                ax.set_title(f'{strategy_name} - By Asset', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')

        # 11. 드로우다운 비교
        ax11 = fig.add_subplot(gs[4, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax11.plot(drawdown.index, drawdown, label=f'{strategy_name}', linewidth=2, alpha=0.7)

        ax11.fill_between(drawdown.index, drawdown, 0, alpha=0.2)
        ax11.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax11.set_ylabel('Drawdown (%)', fontsize=12)
        ax11.set_xlabel('Date', fontsize=12)
        ax11.legend(loc='lower right', fontsize=11)
        ax11.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'알트코인 역추세 전략 성과 비교':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"포트폴리오 구성: 각 종목 동일 비중 (20%)")
        print(f"슬리피지: {self.slippage*100}%")
        print(f"\n전략 설명:")
        print(f"  1. Bollinger Contrarian: 볼린저 밴드 하단 하회 시 매수, 상단 상회 또는 중간선 돌파 시 매도")
        print(f"  2. RSI Contrarian: RSI < 30 시 매수, RSI > 70 또는 50 돌파 시 매도")
        print(f"  3. Z-Score Contrarian: Z-Score < -2 시 매수, |Z-Score| < 0.5 또는 > 2 시 매도")

        # 포트폴리오 성과
        print("\n" + "-"*150)
        print(f"{'포트폴리오 성과 비교 (동일 비중)':^150}")
        print("-"*150)
        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(portfolio_metrics.to_string(index=False))

        # 종목별 성과
        print("\n" + "-"*150)
        print(f"{'종목별 전략 성과':^150}")
        print("-"*150)
        asset_metrics = metrics_df[~metrics_df['Strategy'].str.contains('Portfolio')].copy()
        print(asset_metrics.to_string(index=False))

        print("\n" + "="*150 + "\n")

    def run_analysis(self):
        """전체 분석 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 모든 전략 실행
        self.run_all_strategies()

        # 3. 포트폴리오 생성
        self.create_portfolios()

        # 4. 성과 지표 계산
        metrics_df = self.calculate_all_metrics()

        # 5. 결과 출력
        self.print_metrics_table(metrics_df)

        # 6. 포트폴리오 비교 시각화
        self.plot_comparison(metrics_df)

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("알트코인 역추세 전략 백테스팅 시작")
    print("="*80)

    # 백테스트 실행
    contrarian = AltcoinContrarianStrategy(
        symbols=['STEEM_KRW', 'ANKR_KRW', 'CHZ_KRW', 'MANA_KRW', 'ZIL_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = contrarian.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('altcoin_contrarian_metrics.csv', index=False)
    print("Metrics saved to altcoin_contrarian_metrics.csv")

    # 각 포트폴리오 상세 결과 저장
    for strategy_name in contrarian.portfolio_results.keys():
        filename = f"portfolio_{strategy_name.replace(' ', '_').lower()}.csv"
        contrarian.portfolio_results[strategy_name].to_csv(filename)
        print(f"Portfolio details saved to {filename}")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
