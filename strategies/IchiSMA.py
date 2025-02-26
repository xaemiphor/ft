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
import technical.indicators as ftt

timeperiods = [1,3,6,12,24,48,72,96]
ma_types = ['SMA','EMA']

class IchiSMA(IStrategy):

    INTERFACE_VERSION = 3

    can_short: bool = False

    minimal_roi = {
        # "120": 0.0,  # exit after 120 minutes at break even
        # "60": 0.01,
        # "30": 0.02,
        # "0": 0.04,
    }

    stoploss = -0.275

    trailing_stop = False

    timeframe = "1m" # price movement timeframe
    informative_timeframe = '5m' # Signal timeframe

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    # ichi
    buy_trend_above_senkou_level = IntParameter(low=1, high=8, default=1, space="buy", optimize=True, load=True)
    buy_trend_bullish_level = IntParameter(low=1, high=8, default=6, space="buy", optimize=True, load=True)
    buy_fan_magnitude_shift_value = IntParameter(low=1, high=10, default=3, space="buy", optimize=True, load=True)
    buy_min_fan_magnitude_gain = DecimalParameter(low=1.000, high=1.010, decimals=3, default=1.002, space="buy", optimize=True, load=True)
    sell_trend_indicator = CategoricalParameter(
            timeperiods,
            default="24",
            space="sell",
            optimize=True,
            load=True,
            )
    # smaoffset
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
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(255,76,46,0.2)',
            },
            'senkou_b': {},
            'trend_close_1': {'color': '#FF5733'},
            'trend_close_3': {'color': '#FF8333'},
            'trend_close_6': {'color': '#FFB533'},
            'trend_close_12': {'color': '#FFE633'},
            'trend_close_24': {'color': '#E3FF33'},
            'trend_close_48': {'color': '#C4FF33'},
            'trend_close_72': {'color': '#61FF33'},
            'trend_close_96': {'color': '#33FF7D'},
            "ma_offset_buy": {"color":"red"},
            "ma_offset_sell": {"color":"green"},
        },
        'subplots': {
            'fan_magnitude': {
                'fan_magnitude': {}
            },
            'fan_magnitude_gain': {
                'fan_magnitude_gain': {}
            }
        }
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def do_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        market = self.dp.market(metadata['pair'])
        dataframe["close_fee"] = (dataframe["close"] * market['maker'])

        # ichi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        #dataframe['close'] = heikinashi['close']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        for timeperiod in timeperiods:
            if timeperiod == 1:
                dataframe[f'trend_close_{timeperiod}'] = dataframe['close']
                dataframe[f'trend_open_{timeperiod}'] = dataframe['open']
            else:
                dataframe[f'trend_close_{timeperiod}'] = ta.EMA(dataframe['close'], timeperiod=timeperiod)
                dataframe[f'trend_open_{timeperiod}'] = ta.EMA(dataframe['open'], timeperiod=timeperiod)

        dataframe['fan_magnitude'] = (dataframe['trend_close_12'] / dataframe['trend_close_96'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['chikou_span'] = ichimoku['chikou_span']
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green']
        dataframe['cloud_red'] = ichimoku['cloud_red']

        dataframe['atr'] = ta.ATR(dataframe)

        # smaoffset
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
        sma = []
        ichi = []

        for x in range(10):
            sma.append(dataframe["volume"].shift(x) > 0)
            ichi.append(dataframe["volume"].shift(x) > 0)

        # smaoffset
        if self.config['runmode'].value in ('hyperopt'):
            sma.append(dataframe['close'] < dataframe['ma_offset_buy'])
        else:
            sma.append(qtpylib.crossed_below(dataframe['close'],dataframe['ma_offset_buy']))

        # ichi
        # Trending market
        for idx, timeperiod in enumerate(timeperiods):
            if self.buy_trend_above_senkou_level.value >= idx:
                ichi.append(dataframe[f'trend_close_{timeperiod}'] > dataframe['senkou_a'])
                ichi.append(dataframe[f'trend_close_{timeperiod}'] > dataframe['senkou_b'])

        # Trends bullish
        for idx, timeperiod in enumerate(timeperiods):
            if self.buy_trend_bullish_level.value >= idx:
                ichi.append(dataframe[f'trend_close_{timeperiod}'] > dataframe[f'trend_open_{timeperiod}'])

        # Trends magnitude
        ichi.append(dataframe['fan_magnitude_gain'] >= self.buy_min_fan_magnitude_gain.value)
        ichi.append(dataframe['fan_magnitude'] > 1)

        for x in range(self.buy_fan_magnitude_shift_value.value):
            ichi.append(dataframe['fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])

        if sma:
            dataframe.loc[
                reduce(lambda x, y: x & y, sma),
                'enter_long_sma'] = 1
        if ichi:
            dataframe.loc[
                reduce(lambda x, y: x & y, ichi),
                'enter_long_ichi'] = 1
        if sma or ichi:
            dataframe.loc[
                reduce(lambda x, y: x & y, sma) | reduce(lambda x, y: x & y, ichi),
                'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sma = []
        ichi = []
        for x in range(10):
            sma.append(dataframe["volume"].shift(x) > 0)
            ichi.append(dataframe["volume"].shift(x) > 0)
        # smaoffset
        if self.config['runmode'].value in ('hyperopt'):
            sma.append(dataframe['close'] > dataframe['ma_offset_sell'])
        else:
            sma.append(qtpylib.crossed_above(dataframe['close'],dataframe['ma_offset_sell']))

        # ichi
        ichi.append(qtpylib.crossed_below(dataframe['trend_close_1'], dataframe[f'trend_close_{self.sell_trend_indicator.value}']))

        if sma:
            dataframe.loc[
                reduce(lambda x, y: x & y, sma),
                'exit_long_sma'] = 1
        if ichi:
            dataframe.loc[
                reduce(lambda x, y: x & y, ichi),
                'exit_long_ichi'] = 1
        if sma or ichi:
            dataframe.loc[
                reduce(lambda x, y: x & y, sma) | reduce(lambda x, y: x & y, ichi),
                'exit_long'] = 1
        return dataframe
