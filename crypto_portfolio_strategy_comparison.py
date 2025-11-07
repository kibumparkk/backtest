"""
암호화폐 포트폴리오 전략 비교 분석

세 가지 전략을 BTC, ETH, ADA, XRP에 적용하여 동일 비중 포트폴리오 성과 비교:
1. Turtle Trading (터틀트레이딩)
2. RSI 55 전략
3. SMA 30 전략

각 전략은 4개 종목에 25%씩 동일 비중 투자
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


class CryptoPortfolioComparison:
    """암호화폐 포트폴리오 전략 비교 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트 (default: ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'])
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

    # ==================== 전략 1: Turtle Trading ====================
    def strategy_turtle_trading(self, df, entry_period=20, exit_period=10):
        """
        터틀 트레이딩 전략
        - N일 최고가 돌파 시 매수
        - M일 최저가 하향 돌파 시 매도
        """
        df = df.copy()

        # 터틀 채널
        df['entry_high'] = df['High'].rolling(window=entry_period).max().shift(1)
        df['exit_low'] = df['Low'].rolling(window=exit_period).min().shift(1)

        # 포지션 관리
        df['position'] = 0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 최고가 돌파 시 매수
            if df.iloc[i]['High'] > df.iloc[i]['entry_high'] and df.iloc[i-1]['position'] == 0:
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 최저가 하향 돌파 시 매도
            elif df.iloc[i]['Low'] < df.iloc[i]['exit_low'] and df.iloc[i-1]['position'] == 1:
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                # 매수 진입
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['entry_high']
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                # 매도 청산
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                df.iloc[i, df.columns.get_loc('returns')] = (df.iloc[i]['exit_low'] / buy_price - 1) - self.slippage
            elif df.iloc[i]['position'] == 1:
                # 포지션 유지
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 2: RSI 55 ====================
    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def strategy_rsi_55(self, df, rsi_period=14, rsi_threshold=55):
        """
        RSI 55 전략
        - RSI >= 55: 매수/보유
        - RSI < 55: 매도 후 현금 보유
        """
        df = df.copy()

        # RSI 계산
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)

        # 매수/매도 신호 생성
        df['signal'] = (df['RSI'] >= rsi_threshold).astype(int)

        # 포지션 변화 감지
        df['position_change'] = df['signal'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['signal'].shift(1) * df['daily_price_return']

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

    # ==================== 전략 3: SMA 30 ====================
    def strategy_sma_30(self, df, sma_period=30):
        """
        SMA 30 교차 전략
        - 가격이 SMA 30 이상일 때 매수 (보유)
        - 가격이 SMA 30 미만일 때 매도 후 현금 보유
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # 포지션 계산
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)

        # 포지션 변화 감지
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 매수/매도 시 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['returns'] + slippage_cost

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 실행 ====================
    def run_all_strategies(self):
        """모든 전략을 모든 종목에 대해 실행"""
        strategies = {
            'Turtle Trading': lambda df: self.strategy_turtle_trading(df, entry_period=20, exit_period=10),
            'RSI 55': lambda df: self.strategy_rsi_55(df, rsi_period=14, rsi_threshold=55),
            'SMA 30': lambda df: self.strategy_sma_30(df, sma_period=30)
        }

        print("\n" + "="*80)
        print("Running all strategies for all symbols...")
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

        weight = 1.0 / len(self.symbols)  # 동일 비중 (25% each)

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

        # 개별 종목별 성과 (참고용)
        for strategy_name in self.strategy_results.keys():
            for symbol in self.symbols:
                returns = self.strategy_results[strategy_name][symbol]['returns']
                metrics = self.calculate_metrics(returns, f"{strategy_name} - {symbol.split('_')[0]}")
                metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ==================== 시각화 ====================
    def plot_comparison(self, metrics_df, save_path='crypto_portfolio_comparison.png'):
        """포트폴리오 비교 시각화"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # 1. 포트폴리오 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            ax1.plot(cumulative.index, cumulative, label=f'{strategy_name} Portfolio',
                    linewidth=2.5, alpha=0.8)

        ax1.set_title('Portfolio Cumulative Returns Comparison (Equal-Weight: BTC, ETH, ADA, XRP)',
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

        # 8-10. 각 전략별 종목 비교 (Turtle Trading)
        ax8 = fig.add_subplot(gs[3, 0])
        turtle_metrics = metrics_df[metrics_df['Strategy'].str.contains('Turtle Trading')].copy()
        sorted_df = turtle_metrics.sort_values('Total Return (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
        strategy_labels = [s.replace('Turtle Trading - ', '') for s in sorted_df['Strategy']]
        ax8.barh(strategy_labels, sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax8.set_xlabel('Total Return (%)', fontsize=10)
        ax8.set_title('Turtle Trading - By Asset', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')

        # 9. RSI 55
        ax9 = fig.add_subplot(gs[3, 1])
        rsi_metrics = metrics_df[metrics_df['Strategy'].str.contains('RSI 55')].copy()
        sorted_df = rsi_metrics.sort_values('Total Return (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
        strategy_labels = [s.replace('RSI 55 - ', '') for s in sorted_df['Strategy']]
        ax9.barh(strategy_labels, sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax9.set_xlabel('Total Return (%)', fontsize=10)
        ax9.set_title('RSI 55 - By Asset', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='x')

        # 10. SMA 30
        ax10 = fig.add_subplot(gs[3, 2])
        sma_metrics = metrics_df[metrics_df['Strategy'].str.contains('SMA 30')].copy()
        sorted_df = sma_metrics.sort_values('Total Return (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
        strategy_labels = [s.replace('SMA 30 - ', '') for s in sorted_df['Strategy']]
        ax10.barh(strategy_labels, sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax10.set_xlabel('Total Return (%)', fontsize=10)
        ax10.set_title('SMA 30 - By Asset', fontsize=12, fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='x')

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
        print(f"{'암호화폐 포트폴리오 전략 성과 비교':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"포트폴리오 구성: 각 종목 동일 비중 (25%)")
        print(f"슬리피지: {self.slippage*100}%")

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

        # 6. 시각화
        self.plot_comparison(metrics_df)

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("암호화폐 포트폴리오 전략 비교 분석 시작")
    print("="*80)

    # 백테스트 실행
    comparison = CryptoPortfolioComparison(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    metrics_df = comparison.run_analysis()

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('crypto_portfolio_metrics.csv', index=False)
    print("Metrics saved to crypto_portfolio_metrics.csv")

    # 각 포트폴리오 상세 결과 저장
    for strategy_name in comparison.portfolio_results.keys():
        filename = f"portfolio_{strategy_name.replace(' ', '_').lower()}.csv"
        comparison.portfolio_results[strategy_name].to_csv(filename)
        print(f"Portfolio details saved to {filename}")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
