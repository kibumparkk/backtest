"""
암호화폐 전략 자동 탐색 스크립트

전일종가 > SMA30 전략보다 좋은 전략 5개를 자동으로 찾습니다.
반복문을 사용하여 다양한 전략과 파라미터를 테스트합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class StrategyLoopFinder:
    """여러 전략을 반복문으로 테스트하여 최적 전략을 찾는 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
                 start_date='2018-01-01', end_date=None, slippage=0.002):
        """
        Args:
            symbols: 종목 리스트
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            slippage: 슬리피지 (default: 0.2%)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = {}
        self.all_results = []
        self.portfolio_returns_dict = {}  # 전략별 포트폴리오 수익률 시계열 저장

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

    # ==================== 기술적 지표 계산 ====================
    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    # ==================== 전략 1: SMA 교차 전략 ====================
    def strategy_sma(self, df, period):
        """
        SMA 교차 전략
        - 전일 종가 > SMA: 매수
        - 전일 종가 < SMA: 매도
        """
        df = df.copy()

        # SMA 계산
        df['SMA'] = df['Close'].rolling(window=period).mean()

        # 전일 종가를 사용
        df['prev_close'] = df['Close'].shift(1)

        # 포지션 계산
        df['position'] = np.where(df['prev_close'] >= df['SMA'], 1, 0)

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

    # ==================== 전략 2: EMA 교차 전략 ====================
    def strategy_ema(self, df, period):
        """EMA 교차 전략"""
        df = df.copy()

        # EMA 계산
        df['EMA'] = df['Close'].ewm(span=period, adjust=False).mean()

        # 전일 종가를 사용
        df['prev_close'] = df['Close'].shift(1)

        # 포지션 계산
        df['position'] = np.where(df['prev_close'] >= df['EMA'], 1, 0)

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

    # ==================== 전략 3: RSI 전략 ====================
    def strategy_rsi(self, df, rsi_period, rsi_threshold):
        """RSI 전략"""
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
        slippage_cost[df['position_change'] == 1] = -self.slippage
        slippage_cost[df['position_change'] == -1] = -self.slippage

        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()
        return df

    # ==================== 전략 4: 이중 이동평균 교차 전략 ====================
    def strategy_dual_sma(self, df, fast_period, slow_period):
        """이중 이동평균 교차 전략 (골든크로스/데드크로스)"""
        df = df.copy()

        # 단기 및 장기 이동평균 계산
        df['SMA_fast'] = df['Close'].rolling(window=fast_period).mean()
        df['SMA_slow'] = df['Close'].rolling(window=slow_period).mean()

        # 포지션 계산: 단기 이동평균 > 장기 이동평균
        df['position'] = np.where(df['SMA_fast'] > df['SMA_slow'], 1, 0)

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

    # ==================== 전략 5: Bollinger Band 전략 ====================
    def strategy_bollinger(self, df, period, std_dev):
        """
        Bollinger Band 전략
        - 가격이 하단 밴드 아래로 떨어지면 매수
        - 가격이 상단 밴드 위로 올라가면 매도
        """
        df = df.copy()

        # Bollinger Band 계산
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(df['Close'], period, std_dev)
        df['BB_upper'] = upper_band
        df['BB_middle'] = middle_band
        df['BB_lower'] = lower_band

        # 포지션 계산: 중간선 기준으로 매수/매도
        df['position'] = np.where(df['Close'] >= df['BB_middle'], 1, 0)

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

    # ==================== 전략 6: MACD 전략 ====================
    def strategy_macd(self, df, fast, slow, signal):
        """MACD 전략"""
        df = df.copy()

        # MACD 계산
        macd, signal_line = self.calculate_macd(df['Close'], fast, slow, signal)
        df['MACD'] = macd
        df['MACD_signal'] = signal_line

        # 포지션 계산: MACD > Signal Line
        df['position'] = np.where(df['MACD'] > df['MACD_signal'], 1, 0)

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

    # ==================== 전략 7: 모멘텀 전략 ====================
    def strategy_momentum(self, df, period):
        """
        모멘텀 전략
        - N일 수익률이 양수이면 매수
        """
        df = df.copy()

        # 모멘텀 계산
        df['momentum'] = df['Close'].pct_change(periods=period)

        # 포지션 계산: 모멘텀 > 0
        df['position'] = np.where(df['momentum'] > 0, 1, 0)

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

    # ==================== 성과 지표 계산 ====================
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

        return {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'MDD (%)': mdd,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate,
            'Total Trades': int(total_trades),
            'Profit Factor': profit_factor
        }

    # ==================== 포트폴리오 생성 및 평가 ====================
    def create_portfolio(self, strategy_results):
        """동일 비중 포트폴리오 생성"""
        weight = 1.0 / len(self.symbols)

        # 모든 종목의 공통 날짜 인덱스 찾기
        all_indices = [strategy_results[symbol].index for symbol in self.symbols]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)

        # 포트폴리오 수익률 계산
        portfolio_returns = pd.Series(0.0, index=common_index)

        for symbol in self.symbols:
            symbol_returns = strategy_results[symbol].loc[common_index, 'returns']
            portfolio_returns += symbol_returns * weight

        return portfolio_returns

    # ==================== 모든 전략 반복 실행 ====================
    def loop_all_strategies(self):
        """모든 전략과 파라미터를 반복문으로 테스트"""
        print("\n" + "="*80)
        print("Starting strategy loop testing...")
        print("="*80 + "\n")

        total_strategies = 0

        # 1. SMA 전략 (다양한 기간)
        print("Testing SMA strategies...")
        for period in [5, 10, 15, 20, 25, 30, 40, 50, 60, 100, 120, 150, 200]:
            total_strategies += 1
            strategy_name = f"SMA_{period}"
            strategy_results = {}

            for symbol in self.symbols:
                df = self.data[symbol].copy()
                result = self.strategy_sma(df, period)
                strategy_results[symbol] = result

            portfolio_returns = self.create_portfolio(strategy_results)
            self.portfolio_returns_dict[strategy_name] = portfolio_returns  # 저장
            metrics = self.calculate_metrics(portfolio_returns)
            metrics['Strategy'] = strategy_name
            metrics['Type'] = 'SMA'
            metrics['Parameters'] = f"period={period}"
            self.all_results.append(metrics)

            print(f"  {strategy_name}: CAGR={metrics['CAGR (%)']:.2f}%, Sharpe={metrics['Sharpe Ratio']:.2f}")

        # 2. EMA 전략
        print("\nTesting EMA strategies...")
        for period in [5, 10, 15, 20, 25, 30, 40, 50, 60, 100, 120, 150, 200]:
            total_strategies += 1
            strategy_name = f"EMA_{period}"
            strategy_results = {}

            for symbol in self.symbols:
                df = self.data[symbol].copy()
                result = self.strategy_ema(df, period)
                strategy_results[symbol] = result

            portfolio_returns = self.create_portfolio(strategy_results)
            metrics = self.calculate_metrics(portfolio_returns)
            metrics['Strategy'] = strategy_name
            metrics['Type'] = 'EMA'
            metrics['Parameters'] = f"period={period}"
            self.all_results.append(metrics)

            print(f"  {strategy_name}: CAGR={metrics['CAGR (%)']:.2f}%, Sharpe={metrics['Sharpe Ratio']:.2f}")

        # 3. RSI 전략
        print("\nTesting RSI strategies...")
        for rsi_period in [10, 14, 20]:
            for rsi_threshold in [40, 45, 50, 55, 60, 65, 70]:
                total_strategies += 1
                strategy_name = f"RSI_{rsi_period}_{rsi_threshold}"
                strategy_results = {}

                for symbol in self.symbols:
                    df = self.data[symbol].copy()
                    result = self.strategy_rsi(df, rsi_period, rsi_threshold)
                    strategy_results[symbol] = result

                portfolio_returns = self.create_portfolio(strategy_results)
                metrics = self.calculate_metrics(portfolio_returns)
                metrics['Strategy'] = strategy_name
                metrics['Type'] = 'RSI'
                metrics['Parameters'] = f"period={rsi_period}, threshold={rsi_threshold}"
                self.all_results.append(metrics)

                print(f"  {strategy_name}: CAGR={metrics['CAGR (%)']:.2f}%, Sharpe={metrics['Sharpe Ratio']:.2f}")

        # 4. 이중 이동평균 교차 전략
        print("\nTesting Dual SMA strategies...")
        dual_sma_pairs = [(5, 20), (10, 30), (10, 50), (20, 50), (20, 100), (50, 200)]
        for fast, slow in dual_sma_pairs:
            total_strategies += 1
            strategy_name = f"Dual_SMA_{fast}_{slow}"
            strategy_results = {}

            for symbol in self.symbols:
                df = self.data[symbol].copy()
                result = self.strategy_dual_sma(df, fast, slow)
                strategy_results[symbol] = result

            portfolio_returns = self.create_portfolio(strategy_results)
            metrics = self.calculate_metrics(portfolio_returns)
            metrics['Strategy'] = strategy_name
            metrics['Type'] = 'Dual_SMA'
            metrics['Parameters'] = f"fast={fast}, slow={slow}"
            self.all_results.append(metrics)

            print(f"  {strategy_name}: CAGR={metrics['CAGR (%)']:.2f}%, Sharpe={metrics['Sharpe Ratio']:.2f}")

        # 5. Bollinger Band 전략
        print("\nTesting Bollinger Band strategies...")
        for period in [10, 20, 30]:
            for std_dev in [1.5, 2.0, 2.5]:
                total_strategies += 1
                strategy_name = f"BB_{period}_{std_dev}"
                strategy_results = {}

                for symbol in self.symbols:
                    df = self.data[symbol].copy()
                    result = self.strategy_bollinger(df, period, std_dev)
                    strategy_results[symbol] = result

                portfolio_returns = self.create_portfolio(strategy_results)
                metrics = self.calculate_metrics(portfolio_returns)
                metrics['Strategy'] = strategy_name
                metrics['Type'] = 'Bollinger_Band'
                metrics['Parameters'] = f"period={period}, std_dev={std_dev}"
                self.all_results.append(metrics)

                print(f"  {strategy_name}: CAGR={metrics['CAGR (%)']:.2f}%, Sharpe={metrics['Sharpe Ratio']:.2f}")

        # 6. MACD 전략
        print("\nTesting MACD strategies...")
        macd_params = [(12, 26, 9), (8, 17, 9), (5, 35, 5)]
        for fast, slow, signal in macd_params:
            total_strategies += 1
            strategy_name = f"MACD_{fast}_{slow}_{signal}"
            strategy_results = {}

            for symbol in self.symbols:
                df = self.data[symbol].copy()
                result = self.strategy_macd(df, fast, slow, signal)
                strategy_results[symbol] = result

            portfolio_returns = self.create_portfolio(strategy_results)
            metrics = self.calculate_metrics(portfolio_returns)
            metrics['Strategy'] = strategy_name
            metrics['Type'] = 'MACD'
            metrics['Parameters'] = f"fast={fast}, slow={slow}, signal={signal}"
            self.all_results.append(metrics)

            print(f"  {strategy_name}: CAGR={metrics['CAGR (%)']:.2f}%, Sharpe={metrics['Sharpe Ratio']:.2f}")

        # 7. 모멘텀 전략
        print("\nTesting Momentum strategies...")
        for period in [5, 10, 15, 20, 30, 60]:
            total_strategies += 1
            strategy_name = f"Momentum_{period}"
            strategy_results = {}

            for symbol in self.symbols:
                df = self.data[symbol].copy()
                result = self.strategy_momentum(df, period)
                strategy_results[symbol] = result

            portfolio_returns = self.create_portfolio(strategy_results)
            metrics = self.calculate_metrics(portfolio_returns)
            metrics['Strategy'] = strategy_name
            metrics['Type'] = 'Momentum'
            metrics['Parameters'] = f"period={period}"
            self.all_results.append(metrics)

            print(f"  {strategy_name}: CAGR={metrics['CAGR (%)']:.2f}%, Sharpe={metrics['Sharpe Ratio']:.2f}")

        print("\n" + "="*80)
        print(f"Tested total {total_strategies} strategies!")
        print("="*80 + "\n")

    # ==================== 결과 분석 ====================
    def find_top_strategies(self, benchmark_strategy='SMA_30', top_n=5):
        """벤치마크 전략보다 좋은 상위 N개 전략 찾기"""
        # 결과 데이터프레임 생성
        results_df = pd.DataFrame(self.all_results)

        # 벤치마크 전략 찾기
        benchmark = results_df[results_df['Strategy'] == benchmark_strategy]
        if benchmark.empty:
            print(f"Warning: Benchmark strategy '{benchmark_strategy}' not found!")
            return None

        benchmark_cagr = benchmark['CAGR (%)'].values[0]
        benchmark_sharpe = benchmark['Sharpe Ratio'].values[0]
        benchmark_mdd = benchmark['MDD (%)'].values[0]

        print("="*100)
        print(f"Benchmark Strategy: {benchmark_strategy}")
        print("="*100)
        print(f"CAGR: {benchmark_cagr:.2f}%")
        print(f"Sharpe Ratio: {benchmark_sharpe:.2f}")
        print(f"MDD: {benchmark_mdd:.2f}%")
        print("="*100 + "\n")

        # 벤치마크보다 좋은 전략 필터링
        # CAGR이 더 높고, 샤프 비율이 더 높은 전략
        better_strategies = results_df[
            (results_df['CAGR (%)'] > benchmark_cagr) &
            (results_df['Sharpe Ratio'] > benchmark_sharpe)
        ].copy()

        # 복합 점수 계산 (CAGR과 Sharpe Ratio 고려)
        better_strategies['Score'] = (
            better_strategies['CAGR (%)'] * 0.6 +
            better_strategies['Sharpe Ratio'] * 10 * 0.4
        )

        # 점수 기준으로 정렬
        better_strategies = better_strategies.sort_values('Score', ascending=False)

        # 상위 N개 선택
        top_strategies = better_strategies.head(top_n)

        print("="*100)
        print(f"Top {top_n} Strategies Better than {benchmark_strategy}")
        print("="*100)

        if len(top_strategies) == 0:
            print(f"\nNo strategies found that are better than {benchmark_strategy}")
            print("\nShowing top 5 strategies by Score instead:")
            results_df['Score'] = (
                results_df['CAGR (%)'] * 0.6 +
                results_df['Sharpe Ratio'] * 10 * 0.4
            )
            top_strategies = results_df.sort_values('Score', ascending=False).head(5)

        # 결과 출력
        for idx, row in top_strategies.iterrows():
            print(f"\n{row.name + 1}. {row['Strategy']} ({row['Type']})")
            print(f"   Parameters: {row['Parameters']}")
            print(f"   CAGR: {row['CAGR (%)']:.2f}% (vs {benchmark_cagr:.2f}%, +{row['CAGR (%)'] - benchmark_cagr:.2f}%)")
            print(f"   Sharpe Ratio: {row['Sharpe Ratio']:.2f} (vs {benchmark_sharpe:.2f}, +{row['Sharpe Ratio'] - benchmark_sharpe:.2f})")
            print(f"   MDD: {row['MDD (%)']:.2f}% (vs {benchmark_mdd:.2f}%)")
            print(f"   Total Return: {row['Total Return (%)']:.2f}%")
            print(f"   Win Rate: {row['Win Rate (%)']:.2f}%")
            print(f"   Score: {row['Score']:.2f}")

        print("\n" + "="*100 + "\n")

        return top_strategies, results_df

    def run(self):
        """전체 실행"""
        # 1. 데이터 로드
        self.load_data()

        # 2. 모든 전략 반복 실행
        self.loop_all_strategies()

        # 3. 상위 전략 찾기
        top_strategies, all_results = self.find_top_strategies(benchmark_strategy='SMA_30', top_n=5)

        # 4. 결과 저장
        if all_results is not None:
            all_results.to_csv('loop_strategy_results_all.csv', index=False)
            print("All results saved to: loop_strategy_results_all.csv")

        if top_strategies is not None:
            top_strategies.to_csv('loop_strategy_results_top5.csv', index=False)
            print("Top 5 results saved to: loop_strategy_results_top5.csv\n")

        return top_strategies, all_results


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("암호화폐 전략 자동 탐색 시작")
    print("목표: 전일종가 > SMA30 전략보다 좋은 전략 5개 찾기")
    print("="*80)

    # 전략 탐색 실행
    finder = StrategyLoopFinder(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002
    )

    top_strategies, all_results = finder.run()

    print("\n" + "="*80)
    print("전략 탐색 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
