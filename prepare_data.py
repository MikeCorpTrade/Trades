# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import ta
import MetaTrader5 as mt5
import plotly.express as px
from datetime import datetime, time
import xlsxwriter
import cufflinks as cf
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# Import Data
# --------------------------------------------------------------------------------------
def import_csv_data(file_path: str) -> pd.DataFrame:
	return pd.read_csv(file_path, parse_dates=["time"], index_col="time")

def import_mt5_data(ticker, timeFrame, startDate, endDate=datetime.now()):
	mt5.initialize()
	bars = mt5.copy_rates_range(ticker, timeFrame, startDate, endDate)
	data = pd.DataFrame(bars)
	data["time"] = pd.to_datetime(data["time"], unit="s")
	data = data.set_index("time")
	data = data.drop(labels=["tick_volume", "spread", "real_volume"], axis=1)
	return data

def shift_data(data: pd.DataFrame, columns, number_shift) -> pd.DataFrame:
  for column in columns:
    if column not in data.columns:
      return f"column {column} not found in data"
    elif number_shift == 1:
      data[f"prev_{column}"] = data[f"{column}"].shift(number_shift)
    else:
      for i in range(1, number_shift+1):
        data[f"prev_{column}_{i}"] = data[f"{column}"].shift(i)
  return data

# Plot the data
def plot_data(data, y_axis=["close"]):
  cf.set_config_file(offline=True)
  return data.iplot(kind="candle")

# Plot the trade results
def plot_trades_result(data, result, columns=['close', "EMA"]):
  fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'])])

  fig.update_layout(xaxis_rangeslider_visible=False)

  # adding trades to plots
  for i, position in result.iterrows():
      if position.status == 'closed':
          fig.add_shape(type="line",
              x0=position.open_datetime, y0=position.open_price, x1=position.close_datetime, y1=position.close_price,
              line=dict(
                  color="green" if position.profit >= 0 else "red",
                  width=3)
              )
  return fig

def clean_data(data):
  clean_columns = ["open", "high", "low", "close"]
  for column in data.columns:
    if column not in clean_columns:
      data = data.drop(labels=column, axis=1)
  return data

# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------

ema = 50
rsi_sma = 14
xtreme_high_bb = 0.8
xtreme_low_bb = 1-xtreme_high_bb
stochRSIUp = 80
stochRSILow = 100 - stochRSIUp
# --------------------------------------------------------------------------------------
#  Indicators
# --------------------------------------------------------------------------------------

# find ADX > 25%
def isADX(adx):
  if adx >= 25:
    return True
  return False

# Stochastic calculation
def calculate_stochastic_k(close, low_14, high_14):
  return (close - low_14)*100 / (high_14 - low_14)

def calculate_stochastic_d(stochastic_k, smooth=3):
  return stochastic_k.rolling(3).mean(smooth)

#  Stochastic crossover
def find_stochastic_crossover(stochastic_k, stochastic_d, prev_stochastic_k, prev_stochastic_d):
  if stochastic_k > stochastic_d and prev_stochastic_k < prev_stochastic_d and prev_stochastic_k < stochRSILow:
    return "bullish crossover"
  if stochastic_k < stochastic_d and prev_stochastic_k > prev_stochastic_d and prev_stochastic_k > stochRSIUp:
    return "bearish crossover"
  return None

def find_buy_stoch_ema_signal(close, ema, stoch_crossover):
  if close > ema and stoch_crossover == "bullish crossover" :
    return True
  return False

def find_sell_stoch_ema_signal(close, ema, stoch_crossover):
  if close < ema and stoch_crossover == "bearish crossover" :
    return True
  return False


# Bollinger Band 80% 20%
def find_overbought_BB(bb_1, bb_2, bb_3):
  if bb_1 >= xtreme_high_bb or bb_2 >= xtreme_high_bb or bb_3 >= xtreme_high_bb:
    return True
  return False

def find_oversold_BB(bb_1, bb_2, bb_3):
  if bb_1 <= xtreme_low_bb or bb_2 <= xtreme_low_bb or bb_3 <= xtreme_low_bb:
    return True
  return False

# --------------------------------------------------------------------------------------
# BackTesting
# --------------------------------------------------------------------------------------
# RSI
def get_sell_signal_rsi(rsi_crossover):
  if rsi_crossover == "bearish crossover":
    return True
  return False

def get_buy_signal_rsi(rsi_crossover):
  if rsi_crossover == "bullish crossover":
    return True
  return False

#  Stochastic
def find_bullish_stochastic_crossover(prev_buy_stoch_ema_signal_1, prev_buy_stoch_ema_signal_2, prev_buy_stoch_ema_signal_3, prev_buy_stoch_ema_signal_4):
  return prev_buy_stoch_ema_signal_1 or prev_buy_stoch_ema_signal_2 or prev_buy_stoch_ema_signal_3 or prev_buy_stoch_ema_signal_4

def find_bearish_stochastic_crossover(prev_sell_stoch_ema_signal_1, prev_sell_stoch_ema_signal_2, prev_sell_stoch_ema_signal_3, prev_sell_stoch_ema_signal_4):
  return prev_sell_stoch_ema_signal_1 or prev_sell_stoch_ema_signal_2 or prev_sell_stoch_ema_signal_3 or prev_sell_stoch_ema_signal_4

#  EMA
def above_ema(close, ema):
  if close > ema:
    return True
  return False

def below_ema(close, ema):
  if close < ema:
    return True
  return False

# Stochastic RSI
def exit_oversold_area(stochRSI, prev_stochRSI):
  return stochRSI > stochRSILow and prev_stochRSI < stochRSILow

def exit_overbought_area(stochRSI, prev_stochRSI):
  return stochRSI < stochRSIUp and prev_stochRSI > stochRSIUp

# Alligator
def cross_up_leeps(close, leeps):
  return close > leeps

def cross_down_leeps(close, leeps):
  return close < leeps

# if the current time lies between say 09:00 AM and 11:00 PM
def isTradingTimePeriod(startTime=time(9,00), endTime=time(23,00), nowTime=datetime.now().time()): 
    if startTime < endTime: 
        return nowTime >= startTime and nowTime <= endTime
    return False

# # find previous swing low
# def previous_swing_low(low, previous):
#   previous_low = low
#   for i in range(1, previous):
#     candidate = low.shift(i)
#     if candidate < previous_low:
#       previous_low = candidate
#   return previous_low

# # find previous swing high
# def previous_swing_high(high, previous):
#   previous_high = high
#   for i in range(1, previous):
#     candidate = high.shift(i)
#     if candidate > previous_high:
#       previous_high = candidate
#   return previous_high

 # RSI Crossover
def find_rsi_crossover(rsi, prev_rsi, rsi_sma, prev_rsi_sma):
	if rsi > rsi_sma and prev_rsi < prev_rsi_sma:
		return "bullish crossover"
	if rsi < rsi_sma and prev_rsi > prev_rsi_sma:
		return "bearish crossover"
	return None

 #  Stochastic crossover
def find_stochastic_crossover(stochastic_k, stochastic_d, prev_stochastic_k, prev_stochastic_d):
  if stochastic_k > stochastic_d and prev_stochastic_k < prev_stochastic_d and prev_stochastic_k < stochRSILow:
    return "bullish crossover"
  if stochastic_k < stochastic_d and prev_stochastic_k > prev_stochastic_d and prev_stochastic_k > stochRSIUp:
    return "bearish crossover"
  return None


def find_buy_stoch_ema_signal(close, ema, stoch_crossover, time):
  if close > ema and stoch_crossover == "bullish crossover":
    return time
  return None

def find_sell_stoch_ema_signal(close, ema, stoch_crossover, time):
  if close < ema and stoch_crossover == "bearish crossover":
    return time
  return None

# Avoid two consecutive signals
def in_the_buy_trade_time(prev_buy_stoch_ema_signal_1, prev_buy_stoch_ema_signal_2, prev_buy_stoch_ema_signal_3, prev_buy_stoch_ema_signal_4, trade_time):
  if prev_buy_stoch_ema_signal_1 in trade_time:
    return prev_buy_stoch_ema_signal_1
  elif prev_buy_stoch_ema_signal_2 in trade_time:
    return prev_buy_stoch_ema_signal_2
  elif prev_buy_stoch_ema_signal_3 in trade_time:
    return prev_buy_stoch_ema_signal_3
  elif prev_buy_stoch_ema_signal_4 in trade_time:
    return prev_buy_stoch_ema_signal_4
  else:
    return False

def in_the_sell_trade_time(prev_sell_stoch_ema_signal_1, prev_sell_stoch_ema_signal_2, prev_sell_stoch_ema_signal_3, prev_sell_stoch_ema_signal_4, trade_time):
  if prev_sell_stoch_ema_signal_1 in trade_time:
    return prev_sell_stoch_ema_signal_1
  elif prev_sell_stoch_ema_signal_2 in trade_time:
    return prev_sell_stoch_ema_signal_2
  elif prev_sell_stoch_ema_signal_3 in trade_time:
    return prev_sell_stoch_ema_signal_3
  elif prev_sell_stoch_ema_signal_4 in trade_time:
    return prev_sell_stoch_ema_signal_4
  else:
    return False


# Strategy rsi stoch ema50 signal
def find_final_buy_signal(stoch_crossover, prev_buy_stoch_ema_signal_1, prev_buy_stoch_ema_signal_2, prev_buy_stoch_ema_signal_3, prev_buy_stoch_ema_signal_4) -> bool:
  if stoch_crossover == "bullish crossover" or prev_buy_stoch_ema_signal_1 or prev_buy_stoch_ema_signal_2 or prev_buy_stoch_ema_signal_3 or prev_buy_stoch_ema_signal_4:
    return True
  return False 

def find_final_sell_signal(stoch_crossover, prev_sell_stoch_ema_signal_1, prev_sell_stoch_ema_signal_2, prev_sell_stoch_ema_signal_3, prev_sell_stoch_ema_signal_4) -> bool:
  if stoch_crossover == "bearish crossover" or prev_sell_stoch_ema_signal_1 or prev_sell_stoch_ema_signal_2 or prev_sell_stoch_ema_signal_3 or prev_sell_stoch_ema_signal_4:
    return True
  return False 


# --------------------------------------------------------------------------------------
# Writing in Excel
# --------------------------------------------------------------------------------------
# Create an new Excel file and add a worksheet.
def write_excel_file(result, file_name="demo.xlsx", profit="profit"):
  workbook = xlsxwriter.Workbook(file_name)
  worksheet = workbook.add_worksheet()
  # Write some simple text.
  for index, row in result.iterrows():
    worksheet.write(f'A{index + 1}', row[profit])
  return workbook.close()

# --------------------------------------------------------------------------------------
# TODO find previous swing high/low
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Prepare DATA
# --------------------------------------------------------------------------------------

def prepare_data(data: pd.DataFrame, ema: int=ema) -> pd.DataFrame:
  rsi_sma = 14
  # --------------------------------------------------------------------------------------
  # Calculate EMA
  # --------------------------------------------------------------------------------------
  data["EMA"] = ta.trend.ema_indicator(data["close"], window=ema)
  # --------------------------------------------------------------------------------------
  # Bollinger Bands
  # --------------------------------------------------------------------------------------
  data["Lower"] = ta.volatility.bollinger_lband(data["close"])
  data["Upper"] = ta.volatility.bollinger_hband(data["close"])
  # --------------------------------------------------------------------------------------
  # RSI
  # --------------------------------------------------------------------------------------
  data["RSI"] = ta.momentum.rsi(data['close'])
  # --------------------------------------------------------------------------------------
  #  RSI SMA
  # --------------------------------------------------------------------------------------
  data["RSI_SMA"] = ta.trend.sma_indicator(data["RSI"], rsi_sma)
  # --------------------------------------------------------------------------------------
  #  prev RSI, RSI SMA
  # --------------------------------------------------------------------------------------
  data = shift_data(data, ["RSI", "RSI_SMA"], 1)
  # --------------------------------------------------------------------------------------
  # ADX
  # --------------------------------------------------------------------------------------
  data["ADX"] = ta.trend.adx(high=data["high"], low=data["low"], close=data["close"])
  # --------------------------------------------------------------------------------------
  # %BB
  # --------------------------------------------------------------------------------------
  data["%_BB"] = ta.volatility.bollinger_pband(data["close"])
  data = shift_data(data, ["%_BB"], 3)
  # --------------------------------------------------------------------------------------
  #  Stoch RSI
  # --------------------------------------------------------------------------------------
  data["StochRSI_k"] = ta.momentum.stochrsi_k(data["close"]) * 100
  data["StochRSI_d"] = ta.momentum.stochrsi_d(data["close"]) * 100
  # --------------------------------------------------------------------------------------
  #  Stochastic
  # --------------------------------------------------------------------------------------
  data["low_14"] = data["low"].rolling(14).min()
  data["high_14"] = data["high"].rolling(14).max()
  data["Stoch_%K"] = calculate_stochastic_k(data["close"], data["low_14"], data["high_14"])
  data["Stoch_%D"] = data["Stoch_%K"].rolling(3).mean()
  data = shift_data(data, ["Stoch_%K", "Stoch_%D"], 1)

  # --------------------------------------------------------------------------------------
  # ATR
  # --------------------------------------------------------------------------------------
  data["ATR"] = ta.volatility.average_true_range(data["high"], data["low"], data["close"])
  # --------------------------------------------------------------------------------------
  #  Strategy
  # --------------------------------------------------------------------------------------
  data.dropna(inplace=True)
  # --------------------------------------------------------------------------------------
  # RSI Crossover
  # --------------------------------------------------------------------------------------
  data["rsi_crossover"] = np.vectorize(find_rsi_crossover)(data["RSI"], data["prev_RSI"], data["RSI_SMA"], data["prev_RSI_SMA"])
  data = shift_data(data, ["rsi_crossover"], 4)
  # --------------------------------------------------------------------------------------
  #  Stochastic crossover
  # --------------------------------------------------------------------------------------
  data["stochastic_crossover"] = np.vectorize(find_stochastic_crossover)(data["Stoch_%K"], data["Stoch_%D"], data["prev_Stoch_%K"], data["prev_Stoch_%D"])
  data["buy_stoch_ema_signal"] = np.vectorize(find_buy_stoch_ema_signal)(data["close"], data["EMA"], data["stochastic_crossover"], data.index)
  data["sell_stoch_ema_signal"] = np.vectorize(find_sell_stoch_ema_signal)(data["close"], data["EMA"], data["stochastic_crossover"], data.index)
  data = shift_data(data, ["buy_stoch_ema_signal", "sell_stoch_ema_signal"], 4)
  data["sell_strategy_signal"] = np.vectorize(find_final_sell_signal)(data["stochastic_crossover"], data["prev_sell_stoch_ema_signal_1"], data["prev_sell_stoch_ema_signal_2"], data["prev_sell_stoch_ema_signal_3"], data["prev_sell_stoch_ema_signal_4"])
  data["buy_strategy_signal"] = np.vectorize(find_final_buy_signal)(data["stochastic_crossover"], data["prev_buy_stoch_ema_signal_1"], data["prev_buy_stoch_ema_signal_2"], data["prev_buy_stoch_ema_signal_3"], data["prev_buy_stoch_ema_signal_4"])
  return data