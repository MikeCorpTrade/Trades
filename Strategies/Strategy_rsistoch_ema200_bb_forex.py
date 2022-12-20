import pandas as pd
from prepare_data import *
from Position.Position import Position

class Strategy_rsi_stoch_ema50:
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
        self.current_latest_high = 0
        self.iteration = 0
        
        
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

    def strategy_bb_adx_rsi_ema200(self, index, row):
        # BUY
        if buy_stochrsi_signal(row["StochRSI_k"], row["prev_StochRSI_k"]) and find_oversold_BB(row["%_BB"], row["prev_%_BB_1"], row["prev_%_BB_2"], row["prev_%_BB_3"]) and above_ema(row["close"], row["EMA200"]) and self.trading_allowed():
            sl = row["close"] - row["ATR"]
            tp = row["close"] + self.ratio*row["ATR"]
            self.add_position(Position(index, row["close"], "buy", self.volume, sl, tp, self.spread))
        
        #  SELL
        if sell_stochrsi_signal(row["StochRSI_k"], row["prev_StochRSI_k"]) and find_overbought_BB(row["%_BB"], row["prev_%_BB_1"], row["prev_%_BB_2"], row["prev_%_BB_3"]) and below_ema(row["close"], row["EMA200"]) and self.trading_allowed():
            sl = row["close"] + row["ATR"]
            tp = row["close"] - self.ratio*row["ATR"]
            self.add_position(Position(index, row["close"], "buy", self.volume, sl, tp, self.spread))
        
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