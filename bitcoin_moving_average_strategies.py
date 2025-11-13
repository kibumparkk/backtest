"""
비트코인 다양한 이동평균선 전략 비교 및 파라미터 최적화

10종의 이동평균선 전략:
1. SMA (Simple Moving Average)
2. EMA (Exponential Moving Average)
3. Hull MA (Hull Moving Average)
4. EKF (Extended Kalman Filter)
5. WMA (Weighted Moving Average)
6. DEMA (Double Exponential Moving Average)
7. TEMA (Triple Exponential Moving Average)
8. KAMA (Kaufman Adaptive Moving Average)
9. VWMA (Volume Weighted Moving Average)
10. ZLEMA (Zero Lag Exponential Moving Average)

각 전략에 대해 30개 파라미터로 그리드 서치를 수행하고 성과 분석 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from typing import Dict, List, Tuple
from scipy.signal import butter, filtfilt
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class MovingAverageIndicators:
    """다양한 이동평균선 지표 계산 클래스"""

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def hull_ma(prices: pd.Series, period: int) -> pd.Series:
        """Hull Moving Average
        HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
        """
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))

        wma_half = MovingAverageIndicators.wma(prices, half_period)
        wma_full = MovingAverageIndicators.wma(prices, period)

        raw_hma = 2 * wma_half - wma_full
        hma = MovingAverageIndicators.wma(raw_hma, sqrt_period)

        return hma

    @staticmethod
    def wma(prices: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)

        def weighted_mean(x):
            if len(x) < period:
                return np.nan
            return np.dot(x[-period:], weights) / weights.sum()

        return prices.rolling(window=period).apply(weighted_mean, raw=True)

    @staticmethod
    def dema(prices: pd.Series, period: int) -> pd.Series:
        """Double Exponential Moving Average
        DEMA = 2 * EMA - EMA(EMA)
        """
        ema1 = prices.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()

        return 2 * ema1 - ema2

    @staticmethod
    def tema(prices: pd.Series, period: int) -> pd.Series:
        """Triple Exponential Moving Average
        TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
        """
        ema1 = prices.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()

        return 3 * ema1 - 3 * ema2 + ema3

    @staticmethod
    def kama(prices: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """Kaufman Adaptive Moving Average"""
        change = abs(prices - prices.shift(period))
        volatility = prices.diff().abs().rolling(window=period).sum()

        er = change / volatility  # Efficiency Ratio

        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)

        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2  # Smoothing Constant

        kama_values = np.zeros(len(prices))
        kama_values[period-1] = prices.iloc[period-1]

        for i in range(period, len(prices)):
            if pd.notna(sc.iloc[i]):
                kama_values[i] = kama_values[i-1] + sc.iloc[i] * (prices.iloc[i] - kama_values[i-1])
            else:
                kama_values[i] = kama_values[i-1]

        return pd.Series(kama_values, index=prices.index)

    @staticmethod
    def vwma(prices: pd.Series, volumes: pd.Series, period: int) -> pd.Series:
        """Volume Weighted Moving Average"""
        return (prices * volumes).rolling(window=period).sum() / volumes.rolling(window=period).sum()

    @staticmethod
    def zlema(prices: pd.Series, period: int) -> pd.Series:
        """Zero Lag Exponential Moving Average
        ZLEMA = EMA(Price + (Price - Price[lag]))
        """
        lag = int((period - 1) / 2)
        data = prices + (prices - prices.shift(lag))
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def ekf(prices: pd.Series, period: int, process_noise: float = 0.01,
            measurement_noise: float = 1.0) -> pd.Series:
        """Extended Kalman Filter for price estimation

        Args:
            prices: Price series
            period: Window period (affects initial covariance)
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
        """
        n = len(prices)

        # Initialize
        x = np.zeros(n)  # State estimate
        P = np.zeros(n)  # Error covariance

        x[0] = prices.iloc[0]
        P[0] = 1.0

        Q = process_noise  # Process noise covariance
        R = measurement_noise  # Measurement noise covariance

        for i in range(1, n):
            # Prediction
            x_pred = x[i-1]
            P_pred = P[i-1] + Q

            # Update
            K = P_pred / (P_pred + R)  # Kalman gain
            x[i] = x_pred + K * (prices.iloc[i] - x_pred)
            P[i] = (1 - K) * P_pred

        return pd.Series(x, index=prices.index)


class BitcoinMAStrategies:
    """비트코인 이동평균선 전략 백테스팅 클래스"""

    def __init__(self, symbol='BTC_KRW', start_date='2018-01-01',
                 end_date=None, slippage=0.002):
        """
        Args:
            symbol: 종목 심볼
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            slippage: 슬리피지
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.slippage = slippage
        self.data = None
        self.ma_indicators = MovingAverageIndicators()

    def load_data(self):
        """데이터 로드"""
        file_path = f'chart_day/{self.symbol}.parquet'
        print(f"\nLoading {self.symbol} from {file_path}...")

        df = pd.read_parquet(file_path)
        df.columns = [col.capitalize() for col in df.columns]
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

        self.data = df
        print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

        return df

    def calculate_ma(self, ma_type: str, period: int) -> pd.Series:
        """이동평균선 계산

        Args:
            ma_type: MA 타입 (sma, ema, hull, ekf, wma, dema, tema, kama, vwma, zlema)
            period: 기간
        """
        prices = self.data['Close']

        if ma_type == 'sma':
            return self.ma_indicators.sma(prices, period)
        elif ma_type == 'ema':
            return self.ma_indicators.ema(prices, period)
        elif ma_type == 'hull':
            return self.ma_indicators.hull_ma(prices, period)
        elif ma_type == 'ekf':
            return self.ma_indicators.ekf(prices, period)
        elif ma_type == 'wma':
            return self.ma_indicators.wma(prices, period)
        elif ma_type == 'dema':
            return self.ma_indicators.dema(prices, period)
        elif ma_type == 'tema':
            return self.ma_indicators.tema(prices, period)
        elif ma_type == 'kama':
            return self.ma_indicators.kama(prices, period)
        elif ma_type == 'vwma':
            volumes = self.data['Volume']
            return self.ma_indicators.vwma(prices, volumes, period)
        elif ma_type == 'zlema':
            return self.ma_indicators.zlema(prices, period)
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")

    def backtest_ma_strategy(self, ma_type: str, period: int) -> Dict:
        """이동평균선 크로스오버 전략 백테스트

        전략: 가격이 MA 위에 있으면 매수, 아래면 매도

        Args:
            ma_type: MA 타입
            period: MA 기간

        Returns:
            성과 지표 딕셔너리
        """
        df = self.data.copy()

        # MA 계산
        df['MA'] = self.calculate_ma(ma_type, period)

        # 포지션 계산
        df['position'] = np.where(df['Close'] >= df['MA'], 1, 0)
        df['position_change'] = df['position'].diff()

        # 일일 수익률 계산
        df['daily_price_return'] = df['Close'].pct_change()
        df['returns'] = df['position'].shift(1) * df['daily_price_return']

        # 슬리피지 적용
        slippage_cost = pd.Series(0.0, index=df.index)
        slippage_cost[df['position_change'] == 1] = -self.slippage  # 매수
        slippage_cost[df['position_change'] == -1] = -self.slippage  # 매도

        df['returns'] = df['returns'] + slippage_cost
        df['returns'] = df['returns'].fillna(0)

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        # 성과 지표 계산
        metrics = self.calculate_metrics(df['returns'], df['cumulative'])
        metrics['ma_type'] = ma_type
        metrics['period'] = period

        return {
            'metrics': metrics,
            'df': df
        }

    def calculate_metrics(self, returns: pd.Series, cumulative: pd.Series) -> Dict:
        """성과 지표 계산"""
        # 총 수익률
        total_return = (cumulative.iloc[-1] - 1) * 100

        # 연간 수익률 (CAGR)
        years = (returns.index[-1] - returns.index[0]).days / 365.25
        cagr = (cumulative.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0

        # MDD
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax) / cummax
        mdd = drawdown.min() * 100

        # 샤프 비율
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0

        # 승률
        total_trades = (returns != 0).sum()
        winning_trades = (returns > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit Factor
        total_profit = returns[returns > 0].sum()
        total_loss = abs(returns[returns < 0].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        # Calmar Ratio
        calmar = cagr / abs(mdd) if mdd != 0 else 0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': int(total_trades),
            'profit_factor': profit_factor,
            'calmar': calmar
        }

    def grid_search(self, ma_type: str, param_range: List[int]) -> pd.DataFrame:
        """파라미터 그리드 서치

        Args:
            ma_type: MA 타입
            param_range: 테스트할 파라미터 범위

        Returns:
            결과 DataFrame
        """
        results = []

        print(f"\n>>> Grid search for {ma_type.upper()} strategy...")
        print(f"    Parameter range: {param_range[0]} to {param_range[-1]} ({len(param_range)} values)")

        for i, period in enumerate(param_range):
            try:
                result = self.backtest_ma_strategy(ma_type, period)
                results.append(result['metrics'])

                if (i + 1) % 5 == 0:
                    print(f"    Progress: {i+1}/{len(param_range)} completed")

            except Exception as e:
                print(f"    Error with period {period}: {str(e)}")
                continue

        print(f"    ✓ Completed!")

        return pd.DataFrame(results)

    def run_all_strategies_grid_search(self, param_range: List[int]) -> Dict[str, pd.DataFrame]:
        """모든 전략에 대해 그리드 서치 실행

        Args:
            param_range: 테스트할 파라미터 범위

        Returns:
            전략별 결과 딕셔너리
        """
        ma_types = ['sma', 'ema', 'hull', 'ekf', 'wma', 'dema', 'tema', 'kama', 'vwma', 'zlema']

        all_results = {}

        print("\n" + "="*80)
        print("Starting Grid Search for All Moving Average Strategies")
        print("="*80)

        for ma_type in ma_types:
            results_df = self.grid_search(ma_type, param_range)
            all_results[ma_type] = results_df

        print("\n" + "="*80)
        print("Grid Search Completed!")
        print("="*80)

        return all_results

    def find_optimal_parameters(self, results: Dict[str, pd.DataFrame],
                               metric: str = 'sharpe') -> pd.DataFrame:
        """최적 파라미터 찾기

        Args:
            results: 그리드 서치 결과
            metric: 최적화할 지표 (sharpe, cagr, calmar, etc.)

        Returns:
            최적 파라미터 DataFrame
        """
        optimal_params = []

        for ma_type, df in results.items():
            if len(df) == 0:
                continue

            # metric 기준으로 최적 파라미터 선택
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_params = df.loc[best_idx].to_dict()
                best_params['ma_type'] = ma_type
                optimal_params.append(best_params)

        return pd.DataFrame(optimal_params)

    def plot_parameter_curves(self, results: Dict[str, pd.DataFrame],
                             save_path: str = 'ma_parameter_curves.png'):
        """파라미터별 성과 곡선 시각화

        Args:
            results: 그리드 서치 결과
            save_path: 저장 경로
        """
        fig = plt.figure(figsize=(25, 24))
        gs = fig.add_gridspec(8, 5, hspace=0.4, wspace=0.3)

        ma_types = list(results.keys())
        metrics = ['total_return', 'cagr', 'sharpe', 'mdd', 'calmar', 'win_rate']

        # 1. 각 전략별 Total Return vs Period
        for i, ma_type in enumerate(ma_types):
            row = (i // 5) * 2
            col = i % 5

            ax = fig.add_subplot(gs[row, col])
            df = results[ma_type]

            if len(df) > 0:
                ax.plot(df['period'], df['total_return'], linewidth=2, color='blue', alpha=0.7)

                # 최적값 표시
                best_idx = df['total_return'].idxmax()
                best_period = df.loc[best_idx, 'period']
                best_value = df.loc[best_idx, 'total_return']

                ax.scatter([best_period], [best_value], color='red', s=200,
                          zorder=5, marker='*', edgecolor='black', linewidth=1.5)
                ax.annotate(f'Best: {best_period}\n{best_value:.1f}%',
                           xy=(best_period, best_value),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

                ax.set_title(f'{ma_type.upper()} - Total Return', fontsize=11, fontweight='bold')
                ax.set_xlabel('Period', fontsize=9)
                ax.set_ylabel('Total Return (%)', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # 2. 각 전략별 Sharpe Ratio vs Period
        for i, ma_type in enumerate(ma_types):
            row = (i // 5) * 2 + 1
            col = i % 5

            ax = fig.add_subplot(gs[row, col])
            df = results[ma_type]

            if len(df) > 0:
                ax.plot(df['period'], df['sharpe'], linewidth=2, color='green', alpha=0.7)

                # 최적값 표시
                best_idx = df['sharpe'].idxmax()
                best_period = df.loc[best_idx, 'period']
                best_value = df.loc[best_idx, 'sharpe']

                ax.scatter([best_period], [best_value], color='red', s=200,
                          zorder=5, marker='*', edgecolor='black', linewidth=1.5)
                ax.annotate(f'Best: {best_period}\n{best_value:.2f}',
                           xy=(best_period, best_value),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

                ax.set_title(f'{ma_type.upper()} - Sharpe Ratio', fontsize=11, fontweight='bold')
                ax.set_xlabel('Period', fontsize=9)
                ax.set_ylabel('Sharpe Ratio', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axhline(y=1, color='orange', linestyle='--', linewidth=0.5, alpha=0.3)

        plt.suptitle(f'Bitcoin Moving Average Strategies - Parameter Optimization\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nParameter curves saved to {save_path}")
        plt.close()

    def plot_heatmaps(self, results: Dict[str, pd.DataFrame],
                     save_path: str = 'ma_heatmaps.png'):
        """성과 지표 히트맵 시각화

        Args:
            results: 그리드 서치 결과
            save_path: 저장 경로
        """
        ma_types = list(results.keys())
        metrics = ['total_return', 'cagr', 'sharpe', 'mdd', 'calmar', 'win_rate']

        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        for idx, metric in enumerate(metrics):
            row = idx // 3
            col = idx % 3

            ax = fig.add_subplot(gs[row, col])

            # 각 전략별 성과를 행렬로 구성
            heatmap_data = []
            for ma_type in ma_types:
                df = results[ma_type]
                if len(df) > 0 and metric in df.columns:
                    # period를 인덱스로, metric 값을 사용
                    values = df.set_index('period')[metric].to_dict()
                    heatmap_data.append(values)
                else:
                    heatmap_data.append({})

            # DataFrame으로 변환
            heatmap_df = pd.DataFrame(heatmap_data, index=[ma.upper() for ma in ma_types])
            heatmap_df = heatmap_df.fillna(0)

            # 히트맵 그리기
            if metric == 'mdd':
                cmap = 'RdYlGn_r'  # MDD는 낮을수록 좋음
            else:
                cmap = 'RdYlGn'

            sns.heatmap(heatmap_df, annot=False, fmt='.1f', cmap=cmap, center=0,
                       cbar_kws={'label': metric}, ax=ax, linewidths=0)

            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Period', fontsize=11)
            ax.set_ylabel('Strategy', fontsize=11)

        plt.suptitle(f'Bitcoin Moving Average Strategies - Performance Heatmaps\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmaps saved to {save_path}")
        plt.close()

    def plot_optimal_comparison(self, optimal_params: pd.DataFrame,
                               save_path: str = 'ma_optimal_comparison.png'):
        """최적 파라미터 전략 비교 시각화

        Args:
            optimal_params: 최적 파라미터 DataFrame
            save_path: 저장 경로
        """
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. 누적 수익률 비교
        ax1 = fig.add_subplot(gs[0, :])

        for _, row in optimal_params.iterrows():
            ma_type = row['ma_type']
            period = int(row['period'])

            result = self.backtest_ma_strategy(ma_type, period)
            cumulative = result['df']['cumulative']

            ax1.plot(cumulative.index, cumulative,
                    label=f"{ma_type.upper()} ({period})", linewidth=2, alpha=0.8)

        ax1.set_title('Cumulative Returns - Optimal Parameters', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 총 수익률 비교
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_df = optimal_params.sort_values('total_return', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['total_return']]
        labels = [f"{row['ma_type'].upper()}\n({int(row['period'])})"
                 for _, row in sorted_df.iterrows()]

        ax2.barh(labels, sorted_df['total_return'], color=colors, alpha=0.7)
        ax2.set_xlabel('Total Return (%)', fontsize=11)
        ax2.set_title('Total Return Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # 3. CAGR 비교
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_df = optimal_params.sort_values('cagr', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['cagr']]
        labels = [f"{row['ma_type'].upper()}\n({int(row['period'])})"
                 for _, row in sorted_df.iterrows()]

        ax3.barh(labels, sorted_df['cagr'], color=colors, alpha=0.7)
        ax3.set_xlabel('CAGR (%)', fontsize=11)
        ax3.set_title('CAGR Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # 4. MDD 비교
        ax4 = fig.add_subplot(gs[1, 2])
        sorted_df = optimal_params.sort_values('mdd', ascending=False)
        labels = [f"{row['ma_type'].upper()}\n({int(row['period'])})"
                 for _, row in sorted_df.iterrows()]

        ax4.barh(labels, sorted_df['mdd'], color='crimson', alpha=0.7)
        ax4.set_xlabel('MDD (%)', fontsize=11)
        ax4.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. 샤프 비율 비교
        ax5 = fig.add_subplot(gs[2, 0])
        sorted_df = optimal_params.sort_values('sharpe', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['sharpe']]
        labels = [f"{row['ma_type'].upper()}\n({int(row['period'])})"
                 for _, row in sorted_df.iterrows()]

        ax5.barh(labels, sorted_df['sharpe'], color=colors, alpha=0.7)
        ax5.set_xlabel('Sharpe Ratio', fontsize=11)
        ax5.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # 6. Calmar 비율 비교
        ax6 = fig.add_subplot(gs[2, 1])
        sorted_df = optimal_params.sort_values('calmar', ascending=True)
        colors = ['green' if x > 0 else 'red' for x in sorted_df['calmar']]
        labels = [f"{row['ma_type'].upper()}\n({int(row['period'])})"
                 for _, row in sorted_df.iterrows()]

        ax6.barh(labels, sorted_df['calmar'], color=colors, alpha=0.7)
        ax6.set_xlabel('Calmar Ratio', fontsize=11)
        ax6.set_title('Calmar Ratio Comparison', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # 7. Return vs Risk 산점도
        ax7 = fig.add_subplot(gs[2, 2])
        scatter = ax7.scatter(optimal_params['mdd'], optimal_params['cagr'],
                   s=400, alpha=0.6, c=optimal_params['sharpe'],
                   cmap='RdYlGn', edgecolor='black', linewidth=1.5)

        for _, row in optimal_params.iterrows():
            ax7.annotate(f"{row['ma_type'].upper()}\n({int(row['period'])})",
                        (row['mdd'], row['cagr']),
                        fontsize=8, ha='center', va='center', fontweight='bold')

        plt.colorbar(scatter, ax=ax7, label='Sharpe Ratio')
        ax7.set_xlabel('MDD (%)', fontsize=11)
        ax7.set_ylabel('CAGR (%)', fontsize=11)
        ax7.set_title('Return vs Risk (colored by Sharpe)', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 8. 승률 비교
        ax8 = fig.add_subplot(gs[3, 0])
        sorted_df = optimal_params.sort_values('win_rate', ascending=True)
        labels = [f"{row['ma_type'].upper()}\n({int(row['period'])})"
                 for _, row in sorted_df.iterrows()]

        bars = ax8.barh(labels, sorted_df['win_rate'], color='steelblue', alpha=0.7)
        ax8.set_xlabel('Win Rate (%)', fontsize=11)
        ax8.set_title('Win Rate Comparison', fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')
        ax8.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5)

        # 9. Profit Factor 비교
        ax9 = fig.add_subplot(gs[3, 1])
        pf_df = optimal_params[optimal_params['profit_factor'] != np.inf].copy()
        if len(pf_df) > 0:
            sorted_df = pf_df.sort_values('profit_factor', ascending=True)
            colors = ['green' if x > 1 else 'red' for x in sorted_df['profit_factor']]
            labels = [f"{row['ma_type'].upper()}\n({int(row['period'])})"
                     for _, row in sorted_df.iterrows()]

            ax9.barh(labels, sorted_df['profit_factor'], color=colors, alpha=0.7)

        ax9.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax9.set_xlabel('Profit Factor', fontsize=11)
        ax9.set_title('Profit Factor Comparison', fontsize=13, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='x')

        # 10. 최적 파라미터 분포
        ax10 = fig.add_subplot(gs[3, 2])
        periods = optimal_params['period'].values
        ma_types = [row['ma_type'].upper() for _, row in optimal_params.iterrows()]

        colors_list = plt.cm.Set3(np.linspace(0, 1, len(ma_types)))
        bars = ax10.bar(range(len(periods)), periods, color=colors_list, alpha=0.7, edgecolor='black')

        ax10.set_xticks(range(len(ma_types)))
        ax10.set_xticklabels(ma_types, rotation=45, ha='right', fontsize=9)
        ax10.set_ylabel('Optimal Period', fontsize=11)
        ax10.set_title('Optimal Parameter Distribution', fontsize=13, fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='y')

        # 값 표시
        for i, (bar, period) in enumerate(zip(bars, periods)):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(period)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle(f'Bitcoin Moving Average Strategies - Optimal Parameters Comparison\n'
                    f'Period: {self.start_date} to {self.end_date}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimal comparison saved to {save_path}")
        plt.close()


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("Bitcoin Moving Average Strategies - Parameter Optimization")
    print("="*80)

    # 백테스터 초기화
    bt = BitcoinMAStrategies(
        symbol='BTC_KRW',
        start_date='2017-01-01',
        end_date='2026-12-31',
        slippage=0.002
    )

    # 데이터 로드
    bt.load_data()

    # 파라미터 범위 설정 (30개)
    param_range = list(range(5, 155, 5))  # 5, 10, 15, ..., 150
    print(f"\nParameter range: {param_range}")
    print(f"Total parameters to test: {len(param_range)}")

    # 그리드 서치 실행
    all_results = bt.run_all_strategies_grid_search(param_range)

    # 최적 파라미터 찾기 (Sharpe 기준)
    print("\n" + "="*80)
    print("Finding Optimal Parameters (based on Sharpe Ratio)...")
    print("="*80)

    optimal_params = bt.find_optimal_parameters(all_results, metric='sharpe')

    print("\nOptimal Parameters:")
    print("-" * 80)
    print(optimal_params[['ma_type', 'period', 'total_return', 'cagr', 'mdd',
                          'sharpe', 'calmar', 'win_rate']].to_string(index=False))

    # 결과 저장
    print("\n" + "="*80)
    print("Saving Results...")
    print("="*80)

    # CSV 저장
    for ma_type, df in all_results.items():
        filename = f'ma_grid_results_{ma_type}.csv'
        df.to_csv(filename, index=False)
        print(f"✓ {filename}")

    optimal_params.to_csv('ma_optimal_parameters.csv', index=False)
    print(f"✓ ma_optimal_parameters.csv")

    # 시각화
    print("\n" + "="*80)
    print("Creating Visualizations...")
    print("="*80)

    bt.plot_parameter_curves(all_results, 'ma_parameter_curves.png')
    bt.plot_heatmaps(all_results, 'ma_heatmaps.png')
    bt.plot_optimal_comparison(optimal_params, 'ma_optimal_comparison.png')

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)

    return all_results, optimal_params


if __name__ == "__main__":
    all_results, optimal_params = main()
