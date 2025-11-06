import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import os


output_folder = 'chart_day'
os.makedirs(output_folder, exist_ok=True)

update_data = 0

# 모든 코인의 데이터를 가져오는 함수
def fetch_all_data():
    # 모든 코인 심볼을 조회
    symbols = [market['symbol'] for market in upbit.fetch_markets() if '/KRW' in market['symbol']]
    timeframe = '1d'  # 원하는 시간 단위 설정 (예: 하루 단위)
    symbols = ['BTC/KRW']

    for symbol in symbols:
        try:
            print(f"Fetching data for {symbol}")
            df = fetch_data(symbol, timeframe)
            file_path = os.path.join(output_folder, f"{symbol.replace('/', '_')}.parquet")
            df.to_parquet(file_path)
            print(f"Data for {symbol} saved to {file_path}")
        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {e}")

# 특정 코인의 데이터 가져오는 함수
def fetch_data(symbol, timeframe='1d'):
    since = upbit.milliseconds()  # 현재 시점 (밀리초 단위)
    all_data = []
    while True:
        data = upbit.fetch_ohlcv(symbol, timeframe=timeframe, since=since - 86400000 * 200, limit=200)
        if len(data) == 0:
            break
        all_data = data + all_data  # 새로운 데이터를 앞에 추가
        since = data[0][0]  # 가장 과거 데이터의 타임스탬프로 업데이트
        time.sleep(0.1)  # API 요청 수 조절을 위해 0.1초 대기
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# 모든 코인 데이터 가져와서 저장
if update_data == 1:
    import ccxt
    upbit = ccxt.upbit()
    fetch_all_data()
    
#%%

fnames = os.listdir(output_folder)

sym = 'BTC'
symbol = f'{sym}_KRW'
fname = f'{symbol}.parquet'
# fname = fnames[90]

df = pd.read_parquet(f'{output_folder}/{fname}')

def mean_reversion_volatility_breakout(df, ma_period=20, slippage=0.002): #그냥 심플 ma전략임
    df['ma'] = df['open'].rolling(window=ma_period).mean()

    # 매수/매도 조건 반전 설정
    buy_condition = (df['open'] > df['ma'].shift(1)*1.05)  # 캘리브레이션 (권장 ma 30, k =0, ma 20, k = 1.05)
    sell_condition = ~buy_condition
    df['pct_change'] = df['open'].pct_change()*100
    
    # 포지션 설정
    df['position'] = np.where(sell_condition, -1, np.where(buy_condition, 1, 0))
    sell_signals = []
    changes = []
    buy_prices = []
    on_holds = []
    on_hold = 0
    buy_price = 0
    for row in df.itertuples(index=False):
        # print(row.A, row.B)
        position = row.position
        open_price = row.open
        sell_signal = 0
        change = 0
        
        if (position == 1) & (on_hold == 0):
            on_hold = 1
            buy_price = open_price
        elif (position == -1) & (on_hold == 1):
            sell_signal = 1
            on_hold = 0
            change = open_price / buy_price - 1 - slippage
            
        
        sell_signals.append(sell_signal)
        changes.append(change)
        buy_prices.append(buy_price)
        on_holds.append(on_hold)


    df['buy_condition'] = buy_condition
    df['sell_condition'] = sell_condition
    df['sell_position'] = sell_signals
    df['changes'] = changes
    df['buy_prices'] = buy_prices
    df['on_holds'] = on_holds
    
    
    
    # 슬리피지 반영한 수익률 계산
    # df['return'] = df['sell_position'].shift(1) * (df['changes'] - slippage)
    df['return'] = (df['changes'])
    # df['return'] = df['position'].shift(1) * (df['close'].pct_change() - slippage) * np.where(df['position'].shift(1) == -1, -1, 1)

    df['cumulative_return'] = (1 + df['return']).cumprod()
    return df





# 비트코인 홀딩 성과
df['btc_hold'] = df['close'] / df['close'].iloc[0]

# 평균 회귀 + 변동성 돌파 전략 실행
df_mrvb = mean_reversion_volatility_breakout(df.copy())

# 누적 수익 계산 (이미 구한 df_mrvb['cumulative_return'] 사용)
# drawdown 계산
df_mrvb['cumulative_max_strategy'] = df_mrvb['cumulative_return'].cummax()  # 전략의 누적 최대 수익
df_mrvb['drawdown_strategy'] = (df_mrvb['cumulative_return'] - df_mrvb['cumulative_max_strategy']) / df_mrvb['cumulative_max_strategy']  # 전략의 drawdown 계산

# BTC 홀딩의 누적 최대 수익과 drawdown 계산
df['cumulative_max_btc'] = df['btc_hold'].cummax()
df['drawdown_btc'] = (df['btc_hold'] - df['cumulative_max_btc']) / df['cumulative_max_btc']

# 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,7), sharex=True, dpi = 150)

# 누적 수익률 (1에서 시작, LOG Y축)
ax1.plot(df.index, df['btc_hold'], label='BTC Hold', linewidth=2)
ax1.plot(df_mrvb.index, df_mrvb['cumulative_return'], label='Moving Average Breakout Strategy', linewidth=2)
ax1.set_ylabel('Cumulative Return (Starting from 1)')
ax1.set_title('Moving Average Breakout Strategy vs. BTC Hold')
ax1.legend()
ax1.set_yscale('log')

# 드로우다운 (% 표시)
ax2.plot(df.index, df['drawdown_btc'] * 100, label='BTC Hold Drawdown', linewidth=2)
ax2.plot(df_mrvb.index, df_mrvb['drawdown_strategy'] * 100, label='Strategy Drawdown', linewidth=2) #매수/매도 시점의 Drawdown이라서 실제 고점대비 drawdown대비 과소평가될수있음
ax2.set_ylabel('Drawdown (%)')
ax2.set_xlabel('Date')
ax2.legend()
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

ax1.grid(True)
ax2.grid(True)

plt.suptitle(fname)
plt.show()
