# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib
from functools import reduce

class IndicatorsOverlap(IStrategy):

    INTERFACE_VERSION = 3

    can_short: bool = False

    minimal_roi = {
        # "120": 0.0,  # exit after 120 minutes at break even
        # "60": 0.01,
        # "30": 0.02,
        # "0": 0.04,
    }

    stoploss = -0.10

    trailing_stop = False

    timeframe = "1m" # price movement timeframe
    informative_timeframe = '5m' # Signal timeframe

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    buy_cci = IntParameter(low=-120, high=-90, default=-100, space="buy", optimize=True, load=True)
    sell_cci = IntParameter(low=90, high=120, default=100, space="sell", optimize=True, load=True)
    buy_stoch = IntParameter(low=1, high=50, default=20, space="buy", optimize=True, load=True)
    sell_stoch = IntParameter(low=50, high=100, default=80, space="sell", optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tema": {"color":"white"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "white"},
                "rsi_buy": {"color": "red"},
                "rsi_sell": {"color": "green"},
            },
            "CCI": {
                "cci": {"color": "white"},
                "cci_buy": {"color": "red"},
                "cci_sell": {"color": "green"},
            },
            "stochastic": {
                "fastd": {"color": "white"},
                "fastk": {"color": "yellow"},
                "stoch_buy": {"color": "red"},
                "stoch_sell": {"color": "green"},
            },
        },
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def do_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        market = self.dp.market(metadata['pair'])
        dataframe["close_fee"] = (dataframe["close"] * market['maker'])
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)
        # rsi
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["rsi_buy"] = self.buy_rsi.value
        dataframe["rsi_sell"] = self.sell_rsi.value
        # cci
        dataframe["cci"] = ta.CCI(dataframe)
        dataframe["cci_buy"] = self.buy_cci.value
        dataframe["cci_sell"] = self.sell_cci.value
        # stochasticfast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]
        dataframe["stoch_buy"] = self.buy_stoch.value
        dataframe["stoch_sell"] = self.sell_stoch.value

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value in ('backtest', 'hyperopt'):
            assert (timeframe_to_minutes(self.timeframe) <= 5), "Backtest this strategy in 5m or 1m timeframe."

        if self.timeframe == self.informative_timeframe:
            dataframe = self.do_indicators(dataframe, metadata)
        else:
            if not self.dp:
                return dataframe

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

            informative = self.do_indicators(informative.copy(), metadata)

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)
            skip_columns = [(s + "_" + self.informative_timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s, inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = {
            'rsi': [],
            'cci': [],
            'stochfast': [],
        }
        conditions['rsi'].append(qtpylib.crossed_below(dataframe["rsi"], self.buy_rsi.value))
        conditions['cci'].append(qtpylib.crossed_below(dataframe["cci"], self.buy_cci.value))
        conditions['stochfast'].append(dataframe["fastk"] < self.buy_stoch.value)
        conditions['stochfast'].append(dataframe["fastd"] < self.buy_stoch.value)
        conditions['stochfast'].append(qtpylib.crossed_above(dataframe["fastk"], dataframe["fastd"]))

        for key, value in conditions:
            for x in range(10):
                conditions[key].append(dataframe["volume"].shift(x) > 0)

        dataframe.loc[
            reduce(lambda x, y: x & y, (reduce(lambda a, b: a & b, conditions[key]) for key in conditions)),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = {
            'rsi': [],
            'cci': [],
            'stochfast': [],
        }
        conditions['rsi'].append(qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value))
        conditions['cci'].append(qtpylib.crossed_above(dataframe["cci"], self.sell_cci.value))
        conditions['stochfast'].append(dataframe["fastk"] > self.sell_stoch.value)
        conditions['stochfast'].append(dataframe["fastd"] > self.sell_stoch.value)
        conditions['stochfast'].append(qtpylib.crossed_below(dataframe["fastk"], dataframe["fastd"]))

        for key, value in conditions:
            for x in range(10):
                conditions[key].append(dataframe["volume"].shift(x) > 0)

        dataframe.loc[
            reduce(lambda x, y: x & y, (reduce(lambda a, b: a & b, conditions[key]) for key in conditions)),
            'exit_long'] = 1
        return dataframe
