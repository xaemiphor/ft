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

ma_types = ['SMA','EMA']

class SMAOffset(IStrategy):

    INTERFACE_VERSION = 3

    can_short: bool = False

    minimal_roi = {
        # "120": 0.0,  # exit after 120 minutes at break even
        # "60": 0.01,
        # "30": 0.02,
        # "0": 0.04,
    }

    stoploss = -0.50

    trailing_stop = False

    timeframe = "1m" # price movement timeframe
    informative_timeframe = '5m' # Signal timeframe

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    base_nb_candles_buy = IntParameter(5, 80, default=30, space='buy')
    base_nb_candles_sell = IntParameter(5, 80, default=30, space='sell')
    low_offset = DecimalParameter(0.8, 0.99, default=0.950, space='buy')
    high_offset = DecimalParameter(0.8, 1.1, default=1.010, space='sell')
    buy_trigger = CategoricalParameter(ma_types, default='SMA', space='buy')
    sell_trigger = CategoricalParameter(ma_types, default='EMA', space='sell')

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
            "close": {"color":"blue"},
            "ma_offset_buy": {"color":"red"},
            "ma_offset_sell": {"color":"green"},
        },
        "subplots": {
        },
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def do_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)
        buy_type = getattr(ta,self.buy_trigger.value)
        sell_type = getattr(ta,self.sell_trigger.value)
        dataframe['ma_offset_buy'] = buy_type(dataframe, int(self.base_nb_candles_buy.value)) * self.low_offset.value
        dataframe['ma_offset_sell'] = sell_type(dataframe, int(self.base_nb_candles_sell.value)) * self.high_offset.value

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
        conditions = []
        for x in range(10):
            conditions.append(dataframe["volume"].shift(x) > 0)
        conditions.append(dataframe['close'] < dataframe['ma_offset_buy'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        for x in range(10):
            conditions.append(dataframe["volume"].shift(x) > 0)
        conditions.append(dataframe['close'] > dataframe['ma_offset_sell'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        return dataframe
