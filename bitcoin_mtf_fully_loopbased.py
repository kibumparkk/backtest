"""
MTF 전략 - 완전한 Loop-based 검증
주봉을 미리 계산하지 않고, 매일 루프에서 "오늘까지의 데이터"로만 주봉 계산
이렇게 하면 lookahead bias가 100% 불가능
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MTFFullyLoopBased:
    """완전한 Loop-based MTF 전략 백테스터 (교차 검증용)"""

    def __init__(self, slippage=0.002):
        self.slippage = slippage
        self.daily_data = None
        self.results = {}

    def load_data(self):
        """데이터 로드"""
        print("="*80)
        print("Loading Bitcoin data (Fully Loop-based - Cross Validation)...")
        print("="*80)

        df_daily = pd.read_parquet('chart_day/BTC_KRW.parquet')
        df_daily.columns = [col.capitalize() for col in df_daily.columns]
        df_daily = df_daily[df_daily.index >= '2018-01-01']
        self.daily_data = df_daily

        print(f"\nDaily data: {len(df_daily)} bars")
        print("\n⚠️ 방법: 매일 루프에서 '오늘까지의 데이터'로만 주봉 계산")
        print("   → Lookahead bias 100% 불가능")
        print("="*80 + "\n")

    def calculate_weekly_signal_at_date(self, date, data_until_date, indicator_type, period):
        """
        특정 날짜에서 사용 가능한 주봉 신호 계산

        Args:
            date: 현재 날짜
            data_until_date: 현재 날짜까지의 일봉 데이터
            indicator_type: 'SMA', 'EMA', 'Donchian' 등
            period: 지표 기간 (예: 10, 20, 50)

        Returns:
            weekly_signal: 0 또는 1
        """
        # 오늘까지의 데이터로 주봉 생성
        weekly = data_until_date.resample('W-MON', label='left', closed='left').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if len(weekly) < period + 1:  # 최소 데이터 필요
            return 0

        # 가장 최근 완료된 주봉 찾기
        # 현재 날짜가 속한 주는 아직 완료되지 않았으므로 제외
        current_week_start = date - pd.Timedelta(days=date.dayofweek)  # 이번 주 월요일

        # 완료된 주봉만 필터링 (현재 주 제외)
        completed_weeks = weekly[weekly.index < current_week_start]

        if len(completed_weeks) < period:
            return 0

        # 가장 최근 완료된 주봉
        latest_week = completed_weeks.iloc[-1]

        # 지표 계산
        if indicator_type == 'SMA':
            sma = completed_weeks['Close'].rolling(period).mean()
            if pd.isna(sma.iloc[-1]):
                return 0
            signal = 1 if latest_week['Close'] > sma.iloc[-1] else 0

        elif indicator_type == 'EMA':
            ema = completed_weeks['Close'].ewm(span=period, adjust=False).mean()
            if pd.isna(ema.iloc[-1]):
                return 0
            signal = 1 if latest_week['Close'] > ema.iloc[-1] else 0

        elif indicator_type == 'Donchian':
            high_20 = completed_weeks['High'].rolling(period).max()
            if pd.isna(high_20.iloc[-1]):
                return 0
            signal = 1 if latest_week['Close'] > high_20.iloc[-1] * 0.95 else 0

        else:
            signal = 0

        return signal

    def backtest_strategy(self, indicator_type, period, name):
        """
        완전한 Loop-based 백테스트
        매일 루프에서 주봉 신호를 새로 계산
        """
        df = self.daily_data.copy()

        capital = 1.0
        position = 0
        equity_curve = []
        trades = []

        # SMA30 데이터 미리 계산 (일봉은 lookahead 없음)
        df['SMA30'] = df['Close'].rolling(30).mean()

        print(f"Processing {name}...")
        print_interval = len(df) // 10  # 10% 단위로 진행상황 출력

        for i in range(len(df)):
            date = df.index[i]
            close = df.iloc[i]['Close']

            # 진행상황 출력
            if i % print_interval == 0:
                pct = (i / len(df)) * 100
                print(f"  Progress: {pct:.0f}% ({date.strftime('%Y-%m-%d')})")

            # 일봉 신호 (오늘 종가 기준)
            daily_signal = 1 if close > df.iloc[i]['SMA30'] else 0
            if pd.isna(df.iloc[i]['SMA30']):
                daily_signal = 0

            # 주봉 신호 (오늘까지의 데이터로만 계산)
            data_until_today = df.iloc[:i+1]  # 오늘까지만
            weekly_signal = self.calculate_weekly_signal_at_date(
                date, data_until_today, indicator_type, period
            )

            # 복합 신호
            final_signal = 1 if (daily_signal == 1 and weekly_signal == 1) else 0

            # 수익률 계산 및 자본 업데이트
            if i > 0:
                prev_close = df.iloc[i-1]['Close']
                daily_return = (close - prev_close) / prev_close

                # 어제 포지션에 따라 수익 실현
                if position == 1:
                    capital = capital * (1 + daily_return)

                # 포지션 변경 시 슬리피지
                if position == 0 and final_signal == 1:
                    capital = capital * (1 - self.slippage)
                    trades.append({'date': date, 'action': 'BUY', 'price': close})
                elif position == 1 and final_signal == 0:
                    capital = capital * (1 - self.slippage)
                    trades.append({'date': date, 'action': 'SELL', 'price': close})

            position = final_signal
            equity_curve.append({
                'date': date,
                'capital': capital,
                'position': position,
                'daily_signal': daily_signal,
                'weekly_signal': weekly_signal
            })

        print(f"  Completed! Final capital: {capital:.2f}")

        # 성과 지표 계산
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        returns = equity_df['capital'].pct_change().fillna(0)

        total_return = (capital - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (capital ** (1/years) - 1) * 100

        cummax = equity_df['capital'].cummax()
        drawdown = (equity_df['capital'] - cummax) / cummax
        mdd = drawdown.min() * 100

        sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0

        position_changes = equity_df['position'].diff().fillna(0)
        total_trades = (position_changes != 0).sum()

        self.results[name] = {
            'metrics': {
                'Strategy': name,
                'Total Return (%)': total_return,
                'CAGR (%)': cagr,
                'MDD (%)': mdd,
                'Sharpe Ratio': sharpe,
                'Total Trades': int(total_trades)
            }
        }

        return self.results[name]['metrics']

    def benchmark_daily_sma30(self):
        """벤치마크: Close > SMA30 (일봉만)"""
        df = self.daily_data.copy()
        df['SMA30'] = df['Close'].rolling(30).mean()
        df['signal'] = (df['Close'] > df['SMA30']).astype(int)
        df['pos_change'] = df['signal'].diff()
        df['daily_ret'] = df['Close'].pct_change()
        df['strat_ret'] = df['signal'].shift(1) * df['daily_ret']

        slip_cost = pd.Series(0.0, index=df.index)
        slip_cost[df['pos_change'] == 1] = -self.slippage
        slip_cost[df['pos_change'] == -1] = -self.slippage
        df['strat_ret'] = df['strat_ret'] + slip_cost
        df['strat_ret'] = df['strat_ret'].fillna(0)

        df['cumulative'] = (1 + df['strat_ret']).cumprod()

        total_return = (df['cumulative'].iloc[-1] - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (df['cumulative'].iloc[-1] ** (1/years) - 1) * 100

        cummax = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - cummax) / cummax
        mdd = drawdown.min() * 100

        sharpe = df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(365) if df['strat_ret'].std() > 0 else 0

        total_trades = (df['strat_ret'] != 0).sum()

        self.results['0_BENCHMARK'] = {
            'metrics': {
                'Strategy': '0_BENCHMARK_Daily_SMA30',
                'Total Return (%)': total_return,
                'CAGR (%)': cagr,
                'MDD (%)': mdd,
                'Sharpe Ratio': sharpe,
                'Total Trades': int(total_trades)
            }
        }

        return self.results['0_BENCHMARK']['metrics']

    def run_strategies(self):
        """전략 실행"""
        print("="*80)
        print("Running Fully Loop-based MTF Strategies (Cross Validation)")
        print("="*80 + "\n")

        strategies = [
            ('BENCHMARK', None, None),
            ('Weekly_Donchian20+Daily_SMA30', 'Donchian', 20),
            ('Weekly_EMA20+Daily_SMA30', 'EMA', 20),
            ('Weekly_SMA10+Daily_SMA30', 'SMA', 10),
            ('Weekly_SMA20+Daily_SMA30', 'SMA', 20),
            ('Weekly_SMA50+Daily_SMA30', 'SMA', 50),
        ]

        metrics_list = []

        for name, indicator, period in strategies:
            try:
                if name == 'BENCHMARK':
                    metrics = self.benchmark_daily_sma30()
                else:
                    metrics = self.backtest_strategy(indicator, period, name)
                metrics_list.append(metrics)
                print(f"✓ {name}: Sharpe {metrics['Sharpe Ratio']:.4f}\n")
            except Exception as e:
                print(f"✗ {name} Error: {e}\n")
                import traceback
                traceback.print_exc()

        return pd.DataFrame(metrics_list)


def main():
    """메인 실행"""
    print("\n" + "="*80)
    print("Bitcoin MTF Strategy - Fully Loop-based Cross Validation")
    print("Method: Calculate weekly signals INSIDE daily loop with data up to today")
    print("="*80)

    analyzer = MTFFullyLoopBased(slippage=0.002)
    analyzer.load_data()

    # 전략 실행
    metrics_df = analyzer.run_strategies()

    # 결과 출력
    print("\n" + "="*120)
    print("CROSS VALIDATION RESULTS (Fully Loop-based)")
    print("="*120)

    for idx, row in metrics_df.iterrows():
        print(f"\n{row['Strategy']}")
        print(f"  Sharpe: {row['Sharpe Ratio']:.4f}")
        print(f"  Total Return: {row['Total Return (%)']:.2f}%")
        print(f"  CAGR: {row['CAGR (%)']:.2f}%")
        print(f"  MDD: {row['MDD (%)']:.2f}%")
        print(f"  Trades: {row['Total Trades']}")

    # 이전 결과와 비교
    print("\n" + "="*120)
    print("COMPARISON WITH PREVIOUS LOOP-BASED RESULTS")
    print("="*120)

    # 이전 결과 로드
    try:
        prev_results = pd.read_csv('bitcoin_mtf_loopbased_results.csv')

        comparison = []
        for idx, row in metrics_df.iterrows():
            strategy = row['Strategy']

            # 이전 결과 찾기
            prev_row = prev_results[prev_results['Strategy'].str.contains(
                strategy.replace('_', '').replace('0BENCHMARK', 'BENCHMARK'),
                case=False,
                regex=False
            )]

            if len(prev_row) > 0:
                prev_sharpe = prev_row.iloc[0]['Sharpe Ratio']
                curr_sharpe = row['Sharpe Ratio']
                diff = abs(curr_sharpe - prev_sharpe)
                diff_pct = (diff / prev_sharpe * 100) if prev_sharpe != 0 else 0

                comparison.append({
                    'Strategy': strategy,
                    'Previous Sharpe': prev_sharpe,
                    'Current Sharpe': curr_sharpe,
                    'Difference': diff,
                    'Diff %': diff_pct
                })

        if comparison:
            comp_df = pd.DataFrame(comparison)
            print(comp_df.to_string(index=False))

            max_diff = comp_df['Diff %'].max()
            print(f"\n최대 차이: {max_diff:.2f}%")

            if max_diff < 1.0:
                print("\n✅ VALIDATION PASSED: 두 구현 방식의 결과가 일치합니다!")
                print("   → Lookahead bias가 없음을 확인")
            else:
                print(f"\n⚠️ WARNING: {max_diff:.2f}% 차이 발생")
                print("   → 두 구현을 재검토 필요")

    except FileNotFoundError:
        print("이전 결과 파일이 없습니다.")

    # 결과 저장
    metrics_df.to_csv('bitcoin_mtf_fully_loopbased_results.csv', index=False)
    print(f"\n✓ Results saved to bitcoin_mtf_fully_loopbased_results.csv")
    print("="*120)

    return metrics_df


if __name__ == "__main__":
    main()
