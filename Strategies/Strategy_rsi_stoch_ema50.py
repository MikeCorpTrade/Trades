import pandas as pd
from prepare_data import *
from Position.Position import Position

class Strategy_rsi_stoch_ema50_1H:
    def __init__(self, df, starting_balance, volume, spread, ratio=1.5, security_sl=41):
        self.starting_balance = starting_balance
        self.volume = volume
        self.positions = []
        self.data = df
        self.security_sl = security_sl
        self.ratio = ratio
        self.spread = spread
        self.buy_trade_time = {}
        self.sell_trade_time = {}
        
        
    def get_positions_df(self):
        df = pd.DataFrame([position._asdict() for position in self.positions])
        df['pnl'] = df['profit'].cumsum() + self.starting_balance
        return df
        
    def add_position(self, position):
        self.positions.append(position)
        
    def trading_allowed(self):
        for pos in self.positions:
            if pos.status == 'open':
                return False
        
        return True

    def strategy_rsi_stoch_ema50(self, index, row):
        # BUY
        if row["RSI"] > 50 and row["buy_strategy_signal"] and above_ema(row["close"], row["EMA"]) and not in_the_buy_trade_time(row["prev_buy_stoch_ema_signal_1"], row["prev_buy_stoch_ema_signal_2"], row["prev_buy_stoch_ema_signal_3"], row["prev_buy_stoch_ema_signal_4"], self.buy_trade_time) and self.trading_allowed():
            sl = row["low"] - (row["ATR"] + self.security_sl)
            # if row["ATR"] < 47.3:
            #     sl =row["low"] - (row["ATR"] + self.security_sl_1)
            # if 47.3 < row["ATR"] < 62.5:
            #     sl = row["low"] - (row["ATR"] + self.security_sl_2)
            tp = row["close"] + self.ratio*(row["close"] - sl )
            buy_stoch_ema_signal_time = in_the_buy_trade_time(row["prev_buy_stoch_ema_signal_1"], row["prev_buy_stoch_ema_signal_2"], row["prev_buy_stoch_ema_signal_3"], row["prev_buy_stoch_ema_signal_4"], self.buy_trade_time)
            self.buy_trade_time[buy_stoch_ema_signal_time] = True
            self.add_position(Position(index, row["close"], "buy", self.volume, sl, tp, self.spread))

        # SELL
        if row["RSI"] < 50 and row["sell_strategy_signal"] and not above_ema(row["close"], row["EMA"]) and not in_the_sell_trade_time(row["prev_sell_stoch_ema_signal_1"], row["prev_sell_stoch_ema_signal_2"], row["prev_sell_stoch_ema_signal_3"], row["prev_sell_stoch_ema_signal_4"], self.sell_trade_time) and self.trading_allowed():
            sl = row["high"] + (row["ATR"] + self.security_sl)
            # if row["ATR"] < 47.3:
            #     sl = row["high"] + (row["ATR"] + self.security_sl_1)
            # if 47.3 < row["ATR"] < 62.5:
            #     sl = row["high"] + (row["ATR"] + self.security_sl_2)
            tp = row["close"] - self.ratio*(sl - row["close"])
            sell_stoch_ema_signal_time = in_the_sell_trade_time(row["prev_sell_stoch_ema_signal_1"], row["prev_sell_stoch_ema_signal_2"], row["prev_sell_stoch_ema_signal_3"], row["prev_sell_stoch_ema_signal_4"], self.sell_trade_time)
            self.sell_trade_time[sell_stoch_ema_signal_time] = True
            self.add_position(Position(index, row["close"], "sell", self.volume, sl, tp, self.spread))
    
    def control_position(self, pos, index, row):
        if pos.status == 'open':
            if (pos.sl >= row["low"] and pos.order_type == 'buy'):
                pos.close_position(index, pos.sl)
            elif (pos.sl <= row["high"] and pos.order_type == 'sell'):
                pos.close_position(index, pos.sl)
            elif (pos.tp <= row["high"] and pos.order_type == 'buy'):
                pos.close_position(index, pos.tp)
            elif (pos.tp >= row["low"] and pos.order_type == 'sell'):
                pos.close_position(index, pos.tp)

        
    def run(self):
        for index, row in self.data.iterrows():
            self.strategy_rsi_stoch_ema50(index, row)
                
            for pos in self.positions:
                self.control_position(pos, index, row)
                        
        return self.get_positions_df()