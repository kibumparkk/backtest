"""
20개 알트코인 추세 추종 전략 백테스팅

20개 알트코인에 추세 추종 전략 적용 (다양한 섹터 포함)

선택된 20개 알트코인:
- Layer 1: ATOM, ALGO, HBAR, NEAR, EOS, QTUM, ZIL, TRX
- DeFi: AAVE, KAVA, LINK
- Gaming/Metaverse: AXS, SAND, MANA, CHZ
- Infrastructure: ANKR, VET
- Social: STEEM
- Meme: DOGE
- Payment: XLM

전략: 볼린저 밴드, RSI, Z-Score 추세 추종
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


class Altcoin20TrendFollowingStrategy:
    """20개 알트코인 추세 추종 전략 클래스"""

    def __init__(self, symbols=None, start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트 (None이면 기본 20개 사용)
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            slippage: 슬리피지 (default: 0.2%)
        """
        if symbols is None:
            # 20개 알트코인 (다양한 섹터)
            self.symbols = [
                # Layer 1 블록체인
                'ATOM_KRW', 'ALGO_KRW', 'HBAR_KRW', 'NEAR_KRW', 'EOS_KRW',
                'QTUM_KRW', 'ZIL_KRW', 'TRX_KRW',
                # DeFi
                'AAVE_KRW', 'KAVA_KRW', 'LINK_KRW',
                # Gaming & Metaverse
                'AXS_KRW', 'SAND_KRW', 'MANA_KRW', 'CHZ_KRW',
                # Infrastructure
                'ANKR_KRW', 'VET_KRW',
                # Social Media
                'STEEM_KRW',
                # Meme
                'DOGE_KRW',
                # Payment
                'XLM_KRW'
            ]
        else:
            self.symbols = symbols

        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = {}
        self.strategy_results = {}
        self.portfolio_results = {}

        # 섹터 분류
        self.sectors = {
            'Layer 1': ['ATOM_KRW', 'ALGO_KRW', 'HBAR_KRW', 'NEAR_KRW', 'EOS_KRW', 'QTUM_KRW', 'ZIL_KRW', 'TRX_KRW'],
            'DeFi': ['AAVE_KRW', 'KAVA_KRW', 'LINK_KRW'],
            'Gaming/Metaverse': ['AXS_KRW', 'SAND_KRW', 'MANA_KRW', 'CHZ_KRW'],
            'Infrastructure': ['ANKR_KRW', 'VET_KRW'],
            'Social': ['STEEM_KRW'],
            'Meme': ['DOGE_KRW'],
            'Payment': ['XLM_KRW']
        }

    def load_data(self):
        """모든 종목 데이터 로드"""
        print("="*80)
        print("Loading data for 20 altcoins...")
        print("="*80)

        for symbol in self.symbols:
            file_path = f'chart_day/{symbol}.parquet'
            try:
                print(f"\nLoading {symbol} from {file_path}...")
                df = pd.read_parquet(file_path)
                df.columns = [col.capitalize() for col in df.columns]
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                self.data[symbol] = df
                print(f"  Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
            except Exception as e:
                print(f"  ERROR loading {symbol}: {e}")

        print(f"\n" + "="*80)
        print(f"Data loading completed! Successfully loaded {len(self.data)} coins.")
        print("="*80 + "\n")

    # ==================== 전략 함수들 ====================
    def strategy_bollinger_trendfollowing(self, df, bb_period=20, bb_std=2):
        """볼린저 밴드 추세 추종 전략"""
        df = df.copy()
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
        df['position'] = 0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]
            if df.iloc[i-1]['position'] == 0:
                if df.iloc[i]['Close'] > df.iloc[i]['BB_Upper']:
                    df.iloc[i, df.columns.get_loc('position')] = 1
            elif df.iloc[i-1]['position'] == 1:
                if (df.iloc[i]['Close'] < df.iloc[i]['BB_Lower'] or
                    df.iloc[i]['Close'] < df.iloc[i]['BB_Middle']):
                    df.iloc[i, df.columns.get_loc('position')] = 0

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

    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def strategy_rsi_trendfollowing(self, df, rsi_period=14, overbought=70, oversold=30):
        """RSI 추세 추종 전략"""
        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], rsi_period)
        df['position'] = 0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]
            if df.iloc[i-1]['position'] == 0:
                if df.iloc[i]['RSI'] > overbought:
                    df.iloc[i, df.columns.get_loc('position')] = 1
            elif df.iloc[i-1]['position'] == 1:
                if df.iloc[i]['RSI'] < oversold or df.iloc[i]['RSI'] < 50:
                    df.iloc[i, df.columns.get_loc('position')] = 0

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

    def strategy_zscore_trendfollowing(self, df, lookback=20, entry_threshold=2.0, exit_threshold=0.5):
        """Z-Score 추세 추종 전략"""
        df = df.copy()
        df['Price_Mean'] = df['Close'].rolling(window=lookback).mean()
        df['Price_Std'] = df['Close'].rolling(window=lookback).std()
        df['Z_Score'] = (df['Close'] - df['Price_Mean']) / df['Price_Std']
        df['position'] = 0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]
            if df.iloc[i-1]['position'] == 0:
                if df.iloc[i]['Z_Score'] > entry_threshold:
                    df.iloc[i, df.columns.get_loc('position')] = 1
            elif df.iloc[i-1]['position'] == 1:
                if abs(df.iloc[i]['Z_Score']) < exit_threshold or df.iloc[i]['Z_Score'] < -entry_threshold:
                    df.iloc[i, df.columns.get_loc('position')] = 0

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

    def run_all_strategies(self):
        """모든 전략 실행"""
        strategies = {
            'Bollinger Trend-Following': lambda df: self.strategy_bollinger_trendfollowing(df),
            'RSI Trend-Following': lambda df: self.strategy_rsi_trendfollowing(df),
            'Z-Score Trend-Following': lambda df: self.strategy_zscore_trendfollowing(df)
        }

        print("\n" + "="*80)
        print("Running all trend-following strategies for 20 altcoins...")
        print("="*80 + "\n")

        for strategy_name, strategy_func in strategies.items():
            print(f"\n>>> Running {strategy_name} strategy...")
            self.strategy_results[strategy_name] = {}

            for symbol in self.data.keys():
                print(f"  - {symbol}...")
                df = self.data[symbol].copy()
                result = strategy_func(df)
                self.strategy_results[strategy_name][symbol] = result

        print("\n" + "="*80)
        print("All strategies completed!")
        print("="*80 + "\n")

    def create_portfolios(self):
        """포트폴리오 생성"""
        print("\n" + "="*80)
        print("Creating equal-weight portfolios...")
        print("="*80 + "\n")

        weight = 1.0 / len(self.data)

        for strategy_name in self.strategy_results.keys():
            print(f"\n>>> Creating portfolio for {strategy_name}...")

            all_indices = [self.strategy_results[strategy_name][symbol].index
                          for symbol in self.data.keys()]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)

            portfolio_returns = pd.Series(0.0, index=common_index)

            for symbol in self.data.keys():
                symbol_returns = self.strategy_results[strategy_name][symbol].loc[common_index, 'returns']
                portfolio_returns += symbol_returns * weight

            portfolio_cumulative = (1 + portfolio_returns).cumprod()

            self.portfolio_results[strategy_name] = pd.DataFrame({
                'returns': portfolio_returns,
                'cumulative': portfolio_cumulative
            }, index=common_index)

            print(f"  Portfolio created with {len(self.data)} coins at {weight:.2%} weight each")

        print("\n" + "="*80)
        print("Portfolio creation completed!")
        print("="*80 + "\n")

    def calculate_metrics(self, returns_series, name):
        """성과 지표 계산"""
        cumulative = (1 + returns_series).cumprod()
        total_return = (cumulative.iloc[-1] - 1) * 100
        years = (returns_series.index[-1] - returns_series.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100
        sharpe = (returns_series.mean() / returns_series.std() * np.sqrt(365)) if returns_series.std() > 0 else 0
        total_trades = (returns_series != 0).sum()
        winning_trades = (returns_series > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
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
        """모든 성과 지표 계산"""
        metrics_list = []

        for strategy_name in self.portfolio_results.keys():
            returns = self.portfolio_results[strategy_name]['returns']
            metrics = self.calculate_metrics(returns, f"{strategy_name} Portfolio")
            metrics_list.append(metrics)

        for strategy_name in self.strategy_results.keys():
            for symbol in self.data.keys():
                returns = self.strategy_results[strategy_name][symbol]['returns']
                metrics = self.calculate_metrics(returns, f"{strategy_name} - {symbol.split('_')[0]}")
                metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    def plot_comparison(self, metrics_df, save_path='altcoin20_trendfollowing_comparison.png'):
        """시각화"""
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # 1. 포트폴리오 누적 수익률
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            ax1.plot(cumulative.index, cumulative, label=f'{strategy_name}',
                    linewidth=3, alpha=0.8)

        ax1.set_title('20 Altcoins Trend-Following Strategy Portfolio Comparison (Equal-Weight 5% each)',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (log scale)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()

        # 2-4. 성과 지표
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = portfolio_metrics.sort_values('Total Return (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Total Return (%)']]
        bars = ax2.barh(sorted_df['Strategy'], sorted_df['Total Return (%)'], color=colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.0f}%' if abs(width) < 10000 else f'{width/1000:.0f}k%',
                    ha='left' if width > 0 else 'right', va='center', fontsize=10)

        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = portfolio_metrics.sort_values('CAGR (%)', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['CAGR (%)']]
        bars = ax3.barh(sorted_df['Strategy'], sorted_df['CAGR (%)'], color=colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        for bar in bars:
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.0f}%',
                    ha='left' if width > 0 else 'right', va='center', fontsize=10)

        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = portfolio_metrics.sort_values('MDD (%)', ascending=False)
        bars = ax4.barh(sorted_df['Strategy'], sorted_df['MDD (%)'], color='crimson', alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        for bar in bars:
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%',
                    ha='left' if width > 0 else 'right', va='center', fontsize=10)

        # 5-7. 추가 지표
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = portfolio_metrics.sort_values('Sharpe Ratio', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['Sharpe Ratio']]
        bars = ax5.barh(sorted_df['Strategy'], sorted_df['Sharpe Ratio'], color=colors, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.scatter(portfolio_metrics['MDD (%)'], portfolio_metrics['CAGR (%)'],
                   s=300, alpha=0.6, c=portfolio_metrics['Sharpe Ratio'], cmap='RdYlGn')
        for idx, row in portfolio_metrics.iterrows():
            ax6.annotate(row['Strategy'].replace(' Portfolio', '').replace(' Trend-Following', ''),
                        (row['MDD (%)'], row['CAGR (%)']),
                        fontsize=10, ha='center', va='bottom')
        ax6.set_xlabel('MDD (%)', fontsize=11)
        ax6.set_ylabel('CAGR (%)', fontsize=11)
        ax6.set_title('Return vs Risk', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        ax7 = fig.add_subplot(gs[2, 2])
        sorted_df = portfolio_metrics.copy()
        sorted_df = sorted_df[sorted_df['Profit Factor'] != np.inf]
        if len(sorted_df) > 0:
            sorted_df = sorted_df.sort_values('Profit Factor', ascending=True)
            colors = ['green' if x > 1 else 'red' for x in sorted_df['Profit Factor']]
            bars = ax7.barh(sorted_df['Strategy'], sorted_df['Profit Factor'], color=colors, alpha=0.7)
        ax7.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax7.set_xlabel('Profit Factor', fontsize=11)
        ax7.set_title('Profit Factor', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')

        # 8. 섹터별 성과 (Bollinger 전략)
        ax8 = fig.add_subplot(gs[3, :])
        strategy_name = 'Bollinger Trend-Following'
        sector_returns = {}
        for sector, coins in self.sectors.items():
            returns = []
            for coin in coins:
                if coin in self.data.keys():
                    coin_metrics = metrics_df[metrics_df['Strategy'] == f'{strategy_name} - {coin.split("_")[0]}']
                    if len(coin_metrics) > 0:
                        returns.append(coin_metrics.iloc[0]['Total Return (%)'])
            if returns:
                sector_returns[sector] = np.mean(returns)

        if sector_returns:
            sectors_sorted = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
            sectors_names = [s[0] for s in sectors_sorted]
            sectors_values = [s[1] for s in sectors_sorted]
            colors = ['green' if x > 0 else 'red' for x in sectors_values]
            bars = ax8.bar(sectors_names, sectors_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax8.set_ylabel('Average Total Return (%)', fontsize=11)
            ax8.set_title('Sector Performance (Bollinger Strategy)', fontsize=13, fontweight='bold')
            ax8.grid(True, alpha=0.3, axis='y')
            ax8.tick_params(axis='x', rotation=30)
            for bar in bars:
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%',
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

        # 9. 드로우다운
        ax9 = fig.add_subplot(gs[4, :])
        for strategy_name in self.portfolio_results.keys():
            cumulative = self.portfolio_results[strategy_name]['cumulative']
            cummax = cumulative.cummax()
            drawdown = (cumulative - cummax) / cummax * 100
            ax9.plot(drawdown.index, drawdown, label=strategy_name, linewidth=2.5, alpha=0.7)

        ax9.fill_between(drawdown.index, drawdown, 0, alpha=0.2)
        ax9.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax9.set_ylabel('Drawdown (%)', fontsize=12)
        ax9.set_xlabel('Date', fontsize=12)
        ax9.legend(loc='lower right', fontsize=11)
        ax9.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to {save_path}")
        plt.close()

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'20개 알트코인 추세 추종 전략 성과 비교':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목 수: {len(self.data)}개")
        print(f"포트폴리오 구성: 동일 비중 ({100/len(self.data):.2f}% each)")
        print(f"슬리피지: {self.slippage*100}%\n")

        print("\n" + "-"*150)
        print(f"{'포트폴리오 성과':^150}")
        print("-"*150)
        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(portfolio_metrics.to_string(index=False))

        print("\n" + "-"*150)
        print(f"{'개별 코인 Top 10 (Bollinger Strategy)':^150}")
        print("-"*150)
        bollinger_metrics = metrics_df[metrics_df['Strategy'].str.contains('Bollinger')].copy()
        bollinger_metrics = bollinger_metrics[~bollinger_metrics['Strategy'].str.contains('Portfolio')]
        top10 = bollinger_metrics.nlargest(10, 'Total Return (%)')
        print(top10.to_string(index=False))

        print("\n" + "="*150 + "\n")

    def run_analysis(self):
        """전체 분석 실행"""
        self.load_data()
        self.run_all_strategies()
        self.create_portfolios()
        metrics_df = self.calculate_all_metrics()
        self.print_metrics_table(metrics_df)
        self.plot_comparison(metrics_df)
        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("20개 알트코인 추세 추종 전략 백테스팅 시작")
    print("="*80)

    strategy = Altcoin20TrendFollowingStrategy(
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )

    metrics_df = strategy.run_analysis()

    print("\nSaving results...")
    metrics_df.to_csv('altcoin20_trendfollowing_metrics.csv', index=False)
    print("Metrics saved to altcoin20_trendfollowing_metrics.csv")

    for strategy_name in strategy.portfolio_results.keys():
        filename = f"portfolio20_{strategy_name.replace(' ', '_').replace('-', '_').lower()}.csv"
        strategy.portfolio_results[strategy_name].to_csv(filename)
        print(f"Portfolio details saved to {filename}")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
