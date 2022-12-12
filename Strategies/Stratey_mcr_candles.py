import pandas as pd
from prepare_data import *
from Position.Position import Position

class Strategy_mcr_candle:
    def __init__(self, df, starting_balance, volume, security_sl, ratio, spread):
        self.starting_balance = starting_balance
        self.volume = volume
        self.positions = []
        self.data = df
        self.security_sl = security_sl
        self.ratio = ratio
        self.spread=spread
        
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

    def strategy_mcs_candle(self, index, row):
        # BUY
        if  row["prev_MCR"] is not None:
            buy_stop = row["MCR"][1]
            if row["MCR"][0] == "bullish" and row["low"] < buy_stop < row["high"] and self.trading_allowed():
                sl = row["close"] - row["ATR"]
                tp = row["close"] + self.ratio*(row["close"] - sl )
                self.add_position(Position(index, buy_stop, "buy", self.volume, sl, tp, self.spread))
                
        # SELL
        if  row["prev_MCR"] is not None:
            sell_stop = row["MCR"][1]
            if row["MCR"][0] == "bearish" and row["low"] < sell_stop < row["high"] and self.trading_allowed():
                sl = row["close"] + row["ATR"]
                tp = row["close"] - self.ratio*(row["close"] - sl )
                self.add_position(Position(index, sell_stop, "sell", self.volume, sl, tp, self.spread))
    
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
            self.strategy_mcr_candle(index, row)
                
            for pos in self.positions:
                self.control_position(pos, index, row)
                        
        return self.get_positions_df()