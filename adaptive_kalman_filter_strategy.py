"""
적응형 칼만 필터 돌파 전략

최근 5일간의 ATR을 프로세스 노이즈로 사용하여 적응형 칼만 필터 구현:
- ATR이 크면 → 프로세스 노이즈 증가 → 칼만 게인 감소 → 스무딩 강화 (보수적)
- ATR이 작으면 → 프로세스 노이즈 감소 → 칼만 게인 증가 → 스무딩 약화 (민감)

돌파 전략:
- 가격이 칼만 필터 상단 밴드를 돌파하면 매수
- 가격이 칼만 필터 하단 밴드를 하향 돌파하면 매도
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


class AdaptiveKalmanFilterStrategy:
    """적응형 칼만 필터 돌파 전략 클래스"""

    def __init__(self, symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
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

    def calculate_atr(self, df, period=5):
        """
        ATR (Average True Range) 계산

        Args:
            df: OHLC 데이터프레임
            period: ATR 계산 기간 (default: 5일)

        Returns:
            ATR 시리즈
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        # True Range 계산
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = TR의 이동평균
        atr = tr.rolling(window=period).mean()

        return atr

    def adaptive_kalman_filter(self, df, atr_period=5, measurement_noise_ratio=0.01,
                                band_multiplier=1.5):
        """
        적응형 칼만 필터 구현 (개선 버전)

        Args:
            df: OHLC 데이터프레임
            atr_period: ATR 계산 기간 (default: 5일)
            measurement_noise_ratio: 측정 노이즈 비율 (default: 0.01 = 1%)
            band_multiplier: 밴드 배수 (default: 1.5)

        Returns:
            칼만 필터 값, 상단 밴드, 하단 밴드
        """
        df = df.copy()

        # ATR 계산 (프로세스 노이즈로 사용)
        df['ATR'] = self.calculate_atr(df, period=atr_period)

        # ATR을 가격 대비 비율로 정규화 (ATR%)
        df['ATR_Ratio'] = df['ATR'] / df['Close']

        # 칼만 필터 초기화
        n = len(df)
        kalman_estimate = np.zeros(n)
        kalman_gain = np.zeros(n)
        estimate_error = np.zeros(n)

        # 초기값 설정
        kalman_estimate[0] = df['Close'].iloc[0]
        initial_atr_ratio = df['ATR_Ratio'].iloc[0] if pd.notna(df['ATR_Ratio'].iloc[0]) else 0.01
        estimate_error[0] = initial_atr_ratio

        # 칼만 필터 순환
        for i in range(1, n):
            # ATR 비율을 프로세스 노이즈로 사용 (가격 수준과 무관하게 일관성 유지)
            current_price = df['Close'].iloc[i]
            atr_ratio = df['ATR_Ratio'].iloc[i] if pd.notna(df['ATR_Ratio'].iloc[i]) else estimate_error[i-1]

            # 프로세스 노이즈 = ATR 비율의 제곱 (분산)
            process_noise_variance = atr_ratio ** 2

            # 측정 노이즈 = 측정 노이즈 비율의 제곱
            measurement_noise_variance = measurement_noise_ratio ** 2

            # 예측 단계 (Prediction)
            prediction = kalman_estimate[i-1]
            prediction_error_variance = estimate_error[i-1] + process_noise_variance

            # 업데이트 단계 (Update)
            # 칼만 게인 계산
            # ATR이 크면 → process_noise 크다 → prediction_error 크다 → K 크다 → 관측값에 더 반응 (빠른 반응)
            # ATR이 작으면 → process_noise 작다 → prediction_error 작다 → K 작다 → 관측값에 덜 반응 (스무딩 강화)
            kalman_gain[i] = prediction_error_variance / (prediction_error_variance + measurement_noise_variance)

            # 상태 업데이트
            observation = current_price
            kalman_estimate[i] = prediction + kalman_gain[i] * (observation - prediction)

            # 추정 오차 업데이트 (분산)
            estimate_error[i] = (1 - kalman_gain[i]) * prediction_error_variance

        # 결과 저장
        df['Kalman'] = kalman_estimate
        df['Kalman_Gain'] = kalman_gain
        df['Estimate_Error'] = estimate_error

        # 밴드 계산 (ATR 직접 사용)
        df['Upper_Band'] = df['Kalman'] + band_multiplier * df['ATR']
        df['Lower_Band'] = df['Kalman'] - band_multiplier * df['ATR']

        return df

    def strategy_adaptive_kalman_breakout(self, df, atr_period=5,
                                          measurement_noise_ratio=0.01,
                                          band_multiplier=1.5):
        """
        적응형 칼만 필터 돌파 전략

        매수 신호: 가격이 칼만 필터 라인을 상향 돌파
        매도 신호: 가격이 칼만 필터 라인을 하향 돌파

        밴드는 시각화 및 참고용으로만 사용

        Args:
            df: OHLC 데이터프레임
            atr_period: ATR 계산 기간
            measurement_noise_ratio: 측정 노이즈 비율
            band_multiplier: 밴드 배수 (시각화용)

        Returns:
            전략 결과 데이터프레임
        """
        # 칼만 필터 적용
        df = self.adaptive_kalman_filter(df, atr_period, measurement_noise_ratio, band_multiplier)

        # 포지션 관리 (칼만 필터 라인 돌파 전략)
        df['position'] = 0

        for i in range(1, len(df)):
            # 이전 포지션 유지
            df.iloc[i, df.columns.get_loc('position')] = df.iloc[i-1, df.columns.get_loc('position')]

            # 가격이 칼만 필터를 상향 돌파 시 매수
            # (이전 가격이 칼만 필터 아래, 현재 가격이 칼만 필터 위)
            if (df.iloc[i]['Close'] > df.iloc[i]['Kalman'] and
                df.iloc[i-1]['Close'] <= df.iloc[i-1]['Kalman'] and
                df.iloc[i-1]['position'] == 0 and
                pd.notna(df.iloc[i]['Kalman'])):
                df.iloc[i, df.columns.get_loc('position')] = 1

            # 가격이 칼만 필터를 하향 돌파 시 매도
            # (이전 가격이 칼만 필터 위, 현재 가격이 칼만 필터 아래)
            elif (df.iloc[i]['Close'] < df.iloc[i]['Kalman'] and
                  df.iloc[i-1]['Close'] >= df.iloc[i-1]['Kalman'] and
                  df.iloc[i-1]['position'] == 1 and
                  pd.notna(df.iloc[i]['Kalman'])):
                df.iloc[i, df.columns.get_loc('position')] = 0

        # 수익률 계산
        df['returns'] = 0.0
        df['buy_price'] = np.nan

        for i in range(1, len(df)):
            if df.iloc[i]['position'] == 1 and df.iloc[i-1]['position'] == 0:
                # 매수: 당일 종가에 체결 (슬리피지 포함)
                df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i]['Close'] * (1 + self.slippage)
            elif df.iloc[i]['position'] == 0 and df.iloc[i-1]['position'] == 1:
                # 매도: 당일 종가에 체결 (슬리피지 포함)
                buy_price = df.iloc[i-1]['buy_price'] if pd.notna(df.iloc[i-1]['buy_price']) else df.iloc[i-1]['Close']
                sell_price = df.iloc[i]['Close'] * (1 - self.slippage)
                df.iloc[i, df.columns.get_loc('returns')] = (sell_price / buy_price - 1)
            elif df.iloc[i]['position'] == 1:
                # 포지션 유지
                if pd.notna(df.iloc[i-1]['buy_price']):
                    df.iloc[i, df.columns.get_loc('buy_price')] = df.iloc[i-1]['buy_price']

        # 누적 수익률
        df['cumulative'] = (1 + df['returns']).cumprod()

        return df

    def run_strategy(self, atr_period=5, measurement_noise_ratio=0.01, band_multiplier=1.5):
        """모든 종목에 대해 전략 실행"""
        print("\n" + "="*80)
        print("Running Adaptive Kalman Filter Breakout Strategy...")
        print("="*80)
        print(f"Parameters:")
        print(f"  - ATR Period: {atr_period} days")
        print(f"  - Measurement Noise Ratio: {measurement_noise_ratio*100:.1f}%")
        print(f"  - Band Multiplier: {band_multiplier}")
        print("\n")

        strategy_name = 'Adaptive Kalman Filter'
        self.strategy_results[strategy_name] = {}

        for symbol in self.symbols:
            print(f"  Processing {symbol}...")
            df = self.data[symbol].copy()
            result = self.strategy_adaptive_kalman_breakout(
                df, atr_period, measurement_noise_ratio, band_multiplier
            )
            self.strategy_results[strategy_name][symbol] = result

        print("\n" + "="*80)
        print("Strategy execution completed!")
        print("="*80 + "\n")

    def create_portfolio(self):
        """동일 비중 포트폴리오 생성"""
        print("\n" + "="*80)
        print("Creating equal-weight portfolio...")
        print("="*80 + "\n")

        weight = 1.0 / len(self.symbols)
        strategy_name = 'Adaptive Kalman Filter'

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
            print(f"  Added {symbol} with weight {weight:.2%}")

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
        strategy_name = 'Adaptive Kalman Filter'

        # 포트폴리오 성과
        returns = self.portfolio_results[strategy_name]['returns']
        metrics = self.calculate_metrics(returns, f"{strategy_name} Portfolio")
        metrics_list.append(metrics)

        # 개별 종목별 성과
        for symbol in self.symbols:
            returns = self.strategy_results[strategy_name][symbol]['returns']
            metrics = self.calculate_metrics(returns, f"{strategy_name} - {symbol.split('_')[0]}")
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    def plot_strategy_details(self, symbol, save_path=None):
        """전략 상세 시각화 (칼만 필터, 밴드, 시그널 등)"""
        strategy_name = 'Adaptive Kalman Filter'
        df = self.strategy_results[strategy_name][symbol].copy()

        if save_path is None:
            symbol_clean = symbol.split('_')[0]
            save_path = f'adaptive_kalman_{symbol_clean}_details.png'

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

        # 1. 가격 + 칼만 필터 + 밴드
        ax1 = fig.add_subplot(gs[0:2, :])
        ax1.plot(df.index, df['Close'], label='Price', color='gray', linewidth=1.5, alpha=0.7)
        ax1.plot(df.index, df['Kalman'], label='Kalman Filter', color='blue', linewidth=2)
        ax1.plot(df.index, df['Upper_Band'], label='Upper Band', color='red',
                linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.plot(df.index, df['Lower_Band'], label='Lower Band', color='green',
                linestyle='--', linewidth=1.5, alpha=0.7)

        # 매수/매도 신호
        position_changes = df['position'].diff()
        buy_signals = df[position_changes == 1].index
        sell_signals = df[position_changes == -1].index

        ax1.scatter(buy_signals, df.loc[buy_signals, 'Close'],
                   color='green', marker='^', s=150, label='Buy Signal', zorder=5, alpha=0.8)
        ax1.scatter(sell_signals, df.loc[sell_signals, 'Close'],
                   color='red', marker='v', s=150, label='Sell Signal', zorder=5, alpha=0.8)

        ax1.set_title(f'{symbol.split("_")[0]} - Adaptive Kalman Filter with ATR-based Process Noise',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (KRW)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. ATR (프로세스 노이즈)
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.plot(df.index, df['ATR'], label='ATR (Process Noise)', color='purple', linewidth=2)
        ax2.fill_between(df.index, 0, df['ATR'], alpha=0.3, color='purple')
        ax2.set_title('ATR (Process Noise) - 5 Days', fontsize=13, fontweight='bold')
        ax2.set_ylabel('ATR', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. 칼만 게인
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.plot(df.index, df['Kalman_Gain'], label='Kalman Gain', color='orange', linewidth=2)
        ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='K=0.5')
        ax3.fill_between(df.index, 0, df['Kalman_Gain'], alpha=0.3, color='orange')
        ax3.set_title('Kalman Gain (Adaptiveness)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Kalman Gain', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 텍스트 추가
        info_text = "High ATR → Low Gain → Strong Smoothing (Conservative)\n"
        info_text += "Low ATR → High Gain → Weak Smoothing (Sensitive)"
        ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # 4. 누적 수익률
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(df.index, df['cumulative'], label='Cumulative Return',
                color='blue', linewidth=2.5)
        ax4.set_title('Cumulative Returns', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Cumulative Return', fontsize=11)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        ax4.axhline(y=1, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        # 5. Drawdown
        ax5 = fig.add_subplot(gs[3, 1])
        cummax = df['cumulative'].cummax()
        drawdown = (df['cumulative'] - cummax) / cummax * 100
        ax5.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax5.plot(drawdown.index, drawdown, color='darkred', linewidth=2)
        ax5.set_title('Drawdown', fontsize=13, fontweight='bold')
        ax5.set_ylabel('Drawdown (%)', fontsize=11)
        ax5.set_xlabel('Date', fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Chart saved: {save_path}")
        plt.close()

        return save_path

    def plot_all_details(self):
        """모든 종목에 대해 상세 차트 생성"""
        print("\n" + "="*80)
        print("Creating detailed charts for all symbols...")
        print("="*80 + "\n")

        saved_files = []
        strategy_name = 'Adaptive Kalman Filter'

        for symbol in self.symbols:
            print(f"  Processing {symbol}...")
            symbol_clean = symbol.split('_')[0]
            save_path = f'adaptive_kalman_{symbol_clean}_details.png'
            self.plot_strategy_details(symbol, save_path)
            saved_files.append(save_path)

        print("\n" + "="*80)
        print(f"Detailed charts completed! {len(saved_files)} charts saved.")
        print("="*80 + "\n")

        return saved_files

    def print_metrics_table(self, metrics_df):
        """성과 지표 테이블 출력"""
        print("\n" + "="*150)
        print(f"{'Adaptive Kalman Filter Strategy - Performance Report':^150}")
        print("="*150)
        print(f"\n기간: {self.start_date} ~ {self.end_date}")
        print(f"종목: {', '.join([s.split('_')[0] for s in self.symbols])}")
        print(f"포트폴리오 구성: 각 종목 동일 비중 ({100/len(self.symbols):.1f}%)")
        print(f"슬리피지: {self.slippage*100}%")
        print(f"\n전략 특징:")
        print(f"  - ATR 기반 적응형 칼만 필터")
        print(f"  - ATR이 크면 → 스무딩 강화 (변동성 큰 시기에 보수적)")
        print(f"  - ATR이 작으면 → 스무딩 약화 (변동성 작은 시기에 민감)")

        # 포트폴리오 성과
        print("\n" + "-"*150)
        print(f"{'포트폴리오 성과':^150}")
        print("-"*150)
        portfolio_metrics = metrics_df[metrics_df['Strategy'].str.contains('Portfolio')].copy()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.float_format', lambda x: f'{x:.2f}' if abs(x) < 1000 else f'{x:.0f}')
        print(portfolio_metrics.to_string(index=False))

        # 종목별 성과
        print("\n" + "-"*150)
        print(f"{'종목별 성과':^150}")
        print("-"*150)
        asset_metrics = metrics_df[~metrics_df['Strategy'].str.contains('Portfolio')].copy()
        print(asset_metrics.to_string(index=False))

        print("\n" + "="*150 + "\n")

    def run_analysis(self, atr_period=5, measurement_noise_ratio=0.01, band_multiplier=1.5,
                    create_detail_charts=True):
        """
        전체 분석 실행

        Args:
            atr_period: ATR 계산 기간 (default: 5일)
            measurement_noise_ratio: 측정 노이즈 비율 (default: 0.01 = 1%)
            band_multiplier: 밴드 배수 (default: 1.5)
            create_detail_charts: 상세 차트 생성 여부
        """
        # 1. 데이터 로드
        self.load_data()

        # 2. 전략 실행
        self.run_strategy(atr_period, measurement_noise_ratio, band_multiplier)

        # 3. 포트폴리오 생성
        self.create_portfolio()

        # 4. 성과 지표 계산
        metrics_df = self.calculate_all_metrics()

        # 5. 결과 출력
        self.print_metrics_table(metrics_df)

        # 6. 상세 차트 생성
        if create_detail_charts:
            self.plot_all_details()

        return metrics_df


def main():
    """메인 함수"""
    print("\n" + "="*80)
    print("Adaptive Kalman Filter Breakout Strategy - Backtest")
    print("="*80)

    # 백테스트 실행
    strategy = AdaptiveKalmanFilterStrategy(
        symbols=['BTC_KRW', 'ETH_KRW', 'ADA_KRW', 'XRP_KRW'],
        start_date='2018-01-01',
        end_date=None,
        slippage=0.002  # 0.2%
    )

    # 분석 실행
    # atr_period=5: 최근 5일 ATR 사용
    # measurement_noise_ratio=0.01: 측정 노이즈 1%
    # band_multiplier=1.5: 밴드 배수
    metrics_df = strategy.run_analysis(
        atr_period=5,
        measurement_noise_ratio=0.01,
        band_multiplier=1.5,
        create_detail_charts=True
    )

    # 결과 저장
    print("\nSaving results to CSV...")
    metrics_df.to_csv('adaptive_kalman_filter_metrics.csv', index=False)
    print("Metrics saved to adaptive_kalman_filter_metrics.csv")

    # 포트폴리오 상세 결과 저장
    strategy_name = 'Adaptive Kalman Filter'
    filename = f"portfolio_adaptive_kalman_filter.csv"
    strategy.portfolio_results[strategy_name].to_csv(filename)
    print(f"Portfolio details saved to {filename}")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
