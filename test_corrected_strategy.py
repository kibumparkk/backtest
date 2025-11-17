"""
수정된 SMA 30 전략과 기존 전략 비교 테스트
"""

import pandas as pd
import numpy as np

class StrategyTester:
    def __init__(self, slippage=0.002):
        self.slippage = slippage

    def load_data(self, symbol='BTC_KRW'):
        """데이터 로드"""
        df = pd.read_parquet(f'chart_day/{symbol}.parquet')
        df.columns = [col.capitalize() for col in df.columns]
        df = df[(df.index >= '2018-01-01') & (df.index <= '2024-11-10')]
        return df

    def strategy_sma_30_original(self, df, sma_period=30):
        """기존 방식 (슬리피지 타이밍 오류 포함)"""
        df = df.copy()

        df['SMA'] = df['Close'].rolling(window=sma_period).mean()
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)
        df['position_change'] = df['position'].diff()

        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용 (타이밍 오류)
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def strategy_sma_30_corrected(self, df, sma_period=30):
        """수정된 방식 (슬리피지 타이밍 수정)"""
        df = df.copy()

        df['SMA'] = df['Close'].rolling(window=sma_period).mean()
        df['position'] = np.where(df['Close'] >= df['SMA'], 1, 0)

        # 포지션을 shift하여 사용
        df['position_shifted'] = df['position'].shift(1)
        df['position_change_shifted'] = df['position_shifted'].diff()

        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position_shifted'] * df['daily_price_return']

        # 슬리피지 적용 (타이밍 수정)
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change_shifted'] == 1] = -self.slippage
        slippage_cost[df['position_change_shifted'] == -1] = -self.slippage

        df['returns'] = df['returns'] + slippage_cost
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def calculate_metrics(self, df, name):
        """성과 지표 계산"""
        returns = df['returns'].fillna(0)
        cumulative = (1 + returns).cumprod()

        total_return = (cumulative.iloc[-1] - 1) * 100
        years = (returns.index[-1] - returns.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100

        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0

        total_trades = (returns != 0).sum()
        winning_trades = (returns > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'Strategy': name,
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades)
        }

# 테스트 실행
print("="*80)
print("슬리피지 타이밍 오류 영향 분석")
print("="*80)

tester = StrategyTester(slippage=0.002)

symbols = ['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW']
results = []

for symbol in symbols:
    print(f"\n{symbol} 분석 중...")
    df = tester.load_data(symbol)

    # 기존 방식
    df_original = tester.strategy_sma_30_original(df)
    metrics_original = tester.calculate_metrics(df_original, f'{symbol.split("_")[0]} - Original')
    results.append(metrics_original)

    # 수정된 방식
    df_corrected = tester.strategy_sma_30_corrected(df)
    metrics_corrected = tester.calculate_metrics(df_corrected, f'{symbol.split("_")[0]} - Corrected')
    results.append(metrics_corrected)

# 결과 출력
print("\n" + "="*80)
print("비교 결과")
print("="*80)

results_df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
print(results_df.to_string(index=False))

# 차이 분석
print("\n" + "="*80)
print("차이 분석 (Corrected - Original)")
print("="*80)

for i in range(0, len(results), 2):
    original = results[i]
    corrected = results[i+1]
    symbol_name = original['Strategy'].split(' - ')[0]

    print(f"\n{symbol_name}:")
    print(f"  Total Return 차이: {corrected['Total Return (%)'] - original['Total Return (%)']:+.2f}%p")
    print(f"  CAGR 차이: {corrected['CAGR (%)'] - original['CAGR (%)']:+.2f}%p")
    print(f"  Sharpe 차이: {corrected['Sharpe Ratio'] - original['Sharpe Ratio']:+.4f}")
    print(f"  상대적 차이: {(corrected['Total Return (%)'] / original['Total Return (%)'] - 1) * 100:+.4f}%")
