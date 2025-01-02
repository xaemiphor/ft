
# --- Do not remove these libs ---
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from datetime import datetime
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class madrid_ribbon(IStrategy):
    """
    Madrid Ribbon 001
    author@: Hessebo
    This strategy aims to follow emas.
    How to use it?

    """
    INTERFACE_VERSION: int = 3
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }
    can_short = True
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.10

    # Optimal timeframe for the strategy
    timeframe = '15m'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    use_custom_stoploss = True


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # Calculate as `-desired_stop_from_open + current_profit` to get the distance between current_profit and initial price
        if (current_profit > 0.3):
            return 0.01
        elif (current_profit > 0.1):
            return 0.015
        elif (current_profit > 0.06):
            return 0.01
        elif (current_profit > 0.02):
            return 0.05
        elif (current_profit > 0.01):
            return 0.003

        return 0.15


    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    use_custom_stoploss = False


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # Calculate as `-desired_stop_from_open + current_profit` to get the distance between current_profit and initial price
        if (current_profit > 0.3):
            return 0.01
        elif (current_profit > 0.1):
            return 0.015
        elif (current_profit > 0.06):
            return 0.01
        elif (current_profit > 0.02):
            return 0.05
        elif (current_profit > 0.01):
            return 0.003

        return 0.15



    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        stoch = ta.STOCH(
            dataframe,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )
        dataframe["slowd"] = stoch["slowd"]
        dataframe["slowk"] = stoch["slowk"]
        # MACD
        # https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html
        macd = ta.MACD(
            dataframe,
            fastperiod=12,
            fastmatype=0,
            slowperiod=26,
            slowmatype=0,
            signalperiod=9,
            signalmatype=0,
        )
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]
        # print(metadata)
        # print(dataframe.tail(20))
        dataframe['ema_madrid'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema15'] = ta.EMA(dataframe, timeperiod=15)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema25'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['ema30'] = ta.EMA(dataframe, timeperiod=30)
        dataframe['ema35'] = ta.EMA(dataframe, timeperiod=35)
        dataframe['ema40'] = ta.EMA(dataframe, timeperiod=40)
        dataframe['ema45'] = ta.EMA(dataframe, timeperiod=45)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema55'] = ta.EMA(dataframe, timeperiod=55)
        dataframe['ema60'] = ta.EMA(dataframe, timeperiod=60)
        dataframe['ema65'] = ta.EMA(dataframe, timeperiod=65)
        dataframe['ema70'] = ta.EMA(dataframe, timeperiod=70)
        dataframe['ema75'] = ta.EMA(dataframe, timeperiod=75)
        dataframe['ema80'] = ta.EMA(dataframe, timeperiod=80)
        dataframe['ema85'] = ta.EMA(dataframe, timeperiod=85)
        dataframe['ema90'] = ta.EMA(dataframe, timeperiod=90)
        dataframe['ema95'] = ta.EMA(dataframe, timeperiod=95)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
#               qtpylib.crossed_above(dataframe['ema5'], dataframe['ema20'].shift(1)) &
                (dataframe["rsi"] > 51) &
#                (dataframe['ema5'] > dataframe['ema_madrid']) &
                (dataframe['ema5'] > dataframe['ema10']) &
                (dataframe['ema10'] > dataframe['ema15']) &
                (dataframe['ema15'] > dataframe['ema20']) &
                (dataframe['ema20'] > dataframe['ema25']) &
                (dataframe['ema25'] > dataframe['ema30']) &
                (dataframe['ema30'] > dataframe['ema35']) &
                (dataframe['ema35'] > dataframe['ema40']) &
                (dataframe['ema40'] > dataframe['ema45']) &
                (dataframe['ema45'] > dataframe['ema50']) &
                (dataframe['ema50'] > dataframe['ema55']) &
                (dataframe['ema55'] > dataframe['ema60']) &
                (dataframe['ema60'] > dataframe['ema65']) &
                (dataframe['ema65'] > dataframe['ema70']) &
                (dataframe['ema70'] > dataframe['ema75']) &
                (dataframe['ema75'] > dataframe['ema80']) &
                (dataframe['ema80'] > dataframe['ema85']) &
                (dataframe['ema85'] > dataframe['ema90']) &
                (dataframe['ema90'] > dataframe['ema95']) &
                (dataframe['ema95'] > dataframe['ema100']) &
                (dataframe['ema100'] > dataframe['ema200']) &
                (dataframe['close'] > dataframe['bb_middleband'])
 #               (dataframe['ha_open'] < dataframe['ha_close'].shift(12))  # green bar
            ),
            'enter_long'] = 1
        # Enter Short
        dataframe.loc[
            (
 #               qtpylib.crossed_above(dataframe['ema10'], dataframe['ema20']) &
                (dataframe["rsi"] < 51) &
                (dataframe['ema5'] < dataframe['ema10']) &
                (dataframe['ema10'] < dataframe['ema15']) &
                (dataframe['ema15'] < dataframe['ema20']) &
                (dataframe['ema20'] < dataframe['ema25']) &
                (dataframe['ema25'] < dataframe['ema30']) &
                (dataframe['ema30'] < dataframe['ema35']) &
                (dataframe['ema35'] < dataframe['ema40']) &
                (dataframe['ema40'] < dataframe['ema45']) &
                (dataframe['ema45'] < dataframe['ema50']) &
                (dataframe['ema50'] < dataframe['ema55']) &
                (dataframe['ema55'] < dataframe['ema60']) &
                (dataframe['ema60'] < dataframe['ema65']) &
                (dataframe['ema65'] < dataframe['ema70']) &
                (dataframe['ema70'] < dataframe['ema75']) &
                (dataframe['ema75'] < dataframe['ema80']) &
                (dataframe['ema80'] < dataframe['ema85']) &
                (dataframe['ema85'] < dataframe['ema90']) &
                (dataframe['ema90'] < dataframe['ema95']) &
                (dataframe['ema95'] < dataframe['ema100']) &
                (dataframe['ema100'] < dataframe['ema200']) &
                (dataframe['close'] < dataframe['bb_middleband'])
#                (dataframe['ha_open'] < dataframe['ha_close'].shift(12))  # red bar

            ),
            'enter_short'] = 1


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['ema10'], dataframe['ema100']) 
#                (dataframe['ha_close'] < dataframe['ema20']) &
#                (dataframe['ha_open'] > dataframe['ha_close'])  # red bar
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['ema10'], dataframe['ema100']) 
#                (dataframe['ha_close'] > dataframe['ema20']) &
#                (dataframe['ha_open'] < dataframe['ha_close'])  # Green bar
            ),
            'exit_short'] = 1


        return dataframe
