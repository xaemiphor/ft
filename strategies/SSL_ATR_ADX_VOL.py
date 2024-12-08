



from cmath import nan
from functools import reduce
from math import sqrt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Union
from pandas import DataFrame, Series


from freqtrade.persistence import Trade

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, stoploss_from_open, DecimalParameter,
                                IntParameter, IStrategy, informative, merge_informative_pair)


import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib




















class SSL_ATR_ADX_VOL(IStrategy):


    INTERFACE_VERSION = 3

    timeframe = '15m'

    can_short = False


    minimal_roi = {





    "0": 0.184,
    "416": 0.14,
    "933": 0.073,
    "1982": 0

    }


    stoploss = -0.20

    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 30

    leverage_optimize = True
    leverage_num = IntParameter(low=1, high=5, default=5, space='buy', optimize=leverage_optimize)

    parameters_yes = True
    parameters_no = False

    volume_check = IntParameter(10, 45, default=38, space="buy", optimize= parameters_yes)
    volume_check_s = IntParameter(15, 45, default=20, space="buy", optimize= parameters_yes)

    df24h_val =  IntParameter(1, 245, default=20, space="buy", optimize= parameters_yes)
    df36h_val =  IntParameter(1, 245, default=29, space="buy", optimize= parameters_yes)

    atr_long_mul = DecimalParameter(1.1, 6.0, default=3.5, decimals = 1, space="sell", optimize = parameters_yes)
    atr_short_mul = DecimalParameter(1.1, 6.0, default=4.5, decimals = 1, space="sell", optimize = parameters_yes)

    ema_period_l_exit = IntParameter(22, 200, default=91, space="sell", optimize= parameters_yes)
    ema_period_s_exit = IntParameter(22, 200, default=147, space="sell", optimize= parameters_yes)

    volume_check_exit = IntParameter(10, 45, default=19, space="sell", optimize= parameters_yes)
    volume_check_exit_s = IntParameter(15, 45, default=41, space="sell", optimize= parameters_yes)

    protect_optimize = True

    max_drawdown_lookback = IntParameter(1, 50, default=2, space="protection", optimize=protect_optimize)
    max_drawdown_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    max_drawdown_stop_duration = IntParameter(1, 50, default=4, space="protection", optimize=protect_optimize)
    max_allowed_drawdown = DecimalParameter(0.05, 0.30, default=0.10, decimals=2, space="protection",
                                            optimize=protect_optimize)
    stoploss_guard_lookback = IntParameter(1, 50, default=8, space="protection", optimize=protect_optimize)
    stoploss_guard_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    stoploss_guard_stop_duration = IntParameter(1, 50, default=4, space="protection", optimize=protect_optimize)

    @property
    def protections(self):
        return [




            {
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,
                "trade_limit": self.max_drawdown_trade_limit.value,
                "stop_duration_candles": self.max_drawdown_stop_duration.value,
                "max_allowed_drawdown": self.max_allowed_drawdown.value
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,
                "trade_limit": self.stoploss_guard_trade_limit.value,
                "stop_duration_candles": self.stoploss_guard_stop_duration.value,
                "only_per_pair": False
            }
        ]




    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    @property
    def plot_config(self):
        return {

            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {

                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                }
            }
        }

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df36h = dataframe.copy().shift(self.df36h_val.value) # 432 mins , 86 for 5m, 29 for 15m
        df24h = dataframe.copy().shift(self.df24h_val.value) # 288 mins, 58 for 5m , 20 for 15m
        dataframe['volume_mean_short'] = dataframe['volume'].rolling(1).mean() # 4 rolling candles
        dataframe['volume_mean_long'] = df24h['volume'].rolling(5).mean() # 48 rolling candles, 10 for 5m
        dataframe['volume_mean_base'] = df36h['volume'].rolling(48).mean() # 238 rolling candles, 48 for 15m


        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0),
                                                -1, 0)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:

            return dataframe

        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)
        dataframe['candle-dn-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() <= 2, 1, 0)

        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['cmf'] = chaikin_mf(dataframe, periods=20)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=20)

        dataframe['ema_l'] = ta.EMA(dataframe['close'], timeperiod=self.ema_period_l_exit.value)
        dataframe['ema_s'] = ta.EMA(dataframe['close'], timeperiod=self.ema_period_s_exit.value)

        ssldown, sslup = SSLChannels_ATR(dataframe, length=21)
        dataframe['ssl-dir'] = np.where(sslup < ssldown, 'down', 'up')

        dataframe['volume_mean'] = dataframe['volume'].rolling(self.volume_check.value).mean().shift(1)
        dataframe['volume_mean_exit'] = dataframe['volume'].rolling(self.volume_check_exit.value).mean().shift(1)

        dataframe['volume_mean_s'] = dataframe['volume'].rolling(self.volume_check_s.value).mean().shift(1)
        dataframe['volume_mean_exit_s'] = dataframe['volume'].rolling(self.volume_check_exit_s.value).mean().shift(1)

        dataframe = self.pump_dump_protection(dataframe, metadata)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'entry_tag'] = ''

        buy_1 = (
                        (dataframe['candle-up-trend'] == 1) &
                        (dataframe['adx'] > 25) &
                        (dataframe['adx'] < 50) &
                        (dataframe['volume'] > dataframe['volume_mean']) &
                        (dataframe['volume'] > 0)
                        & (dataframe['ssl-dir'] == 'up')
                        & (dataframe['cmf'] > 0)
        )

        buy_2 = (
                        (dataframe['candle-dn-trend'] == 1) &
                        (dataframe['adx'] > 25) &
                        (dataframe['adx'] < 50) &
                        (dataframe['volume'] > dataframe['volume_mean_s']) & # volume weighted indicator
                        (dataframe['volume'] > 0)
                        & (dataframe['ssl-dir'] == 'down')
                        & (dataframe['cmf'] < 0)
        )

        conditions_long.append(buy_1)
        dataframe.loc[buy_1, 'entry_tag'] += 'long: buy_1'

        conditions_short.append(buy_2)
        dataframe.loc[buy_2, 'entry_tag'] += 'short: buy_2'

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                ['enter_long', 'enter_tag']] = (1, 'long')

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_short),
                ['enter_short', 'enter_tag']] = (1, 'short')


        dont_buy_conditions = []

        dont_buy_conditions.append((dataframe['pnd_volume_warn'] < 0.0))

        if conditions_long:

            combined_conditions = [condition for condition in conditions_long]
            final_condition = reduce(lambda x, y: x | y, combined_conditions)
            dataframe.loc[final_condition, ['enter_long', 'enter_tag']] = (1, 'long')
        elif conditions_short:

            combined_conditions = [condition for condition in conditions_short]
            final_condition = reduce(lambda x, y: x | y, combined_conditions)
            dataframe.loc[final_condition, ['enter_short', 'enter_tag']] = (1, 'short')
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'enter_long'] = 0
                dataframe.loc[condition, 'enter_short'] = 0


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'exit_tag'] = ''

        exit_long = (
                (dataframe['close'] < (dataframe['ema_l'] - (self.atr_long_mul.value * dataframe['atr']))) &
                (dataframe['volume'] > dataframe['volume_mean_exit'])
        )

        exit_short = (
                (dataframe['close'] > (dataframe['ema_s'] + (self.atr_short_mul.value * dataframe['atr']))) &
                (dataframe['volume'] > dataframe['volume_mean_exit_s'])
        )

        conditions_short.append(exit_short)
        dataframe.loc[exit_short, 'exit_tag'] += 'exit_short'


        conditions_long.append(exit_long)
        dataframe.loc[exit_long, 'exit_tag'] += 'exit_long'


        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'exit_long'] = 1

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_short),
                'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        open_trades = Trade.get_open_trades()

        num_shorts, num_longs = 0, 0
        for trade in open_trades:
            if "short" in trade.enter_tag:
                num_shorts += 1
            elif "long" in trade.enter_tag:
                num_longs += 1

        if side == "long" and num_longs >= 5:
            return False

        if side == "short" and num_shorts >= 5:
            return False

        return True

def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

def chaikin_mf(df, periods=20):
    close = df["close"]
    low = df["low"]
    high = df["high"]
    volume = df["volume"]
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name="cmf")

def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]
