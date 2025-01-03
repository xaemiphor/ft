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

class SimpleKCInner(IStrategy):

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

    timeframe = "5m"

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

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
            "kc_upperband": {"color": "#9f9c03","type": "line","fill_to": "kc_lowerband"},
            "kc_lowerband": {"color": "#9f9c03","type": "line"},
            "kc_middleband": {"color": "#9f9c03","type": "line"},
        },
        "subplots": {
        },
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"] 
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_stoploss"] = dataframe["kc_lowerband"] - ((dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / 2)  
        dataframe["kc_percent"] = (
            (dataframe["close"] - dataframe["kc_lowerband"]) /
            (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        ) 
        dataframe["kc_width"] = (
            (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe["volume"] > 0)
        conditions.append(qtpylib.crossed_above(dataframe["tema"], dataframe["kc_lowerband"]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe["volume"] > 0)
        conditions.append(qtpylib.crossed_below(dataframe["tema"], dataframe["kc_upperband"]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        return dataframe
