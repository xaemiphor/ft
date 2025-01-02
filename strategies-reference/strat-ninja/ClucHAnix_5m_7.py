import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter
from pandas import DataFrame, Series
from datetime import datetime

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

class ClucHAnix_5m_7(IStrategy):
    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """

    buy_params = {
        "bbdelta_close": 0.01889,
        "bbdelta_tail": 0.72235,
        "close_bblower": 0.0127,
        "closedelta_close": 0.00916,
        "rocr_1h": 0.79492,
    }

    sell_params = {

        "pHSL": -0.35,
        "pPF_1": 0.011,
        "pPF_2": 0.064,
        "pSL_1": 0.011,
        "pSL_2": 0.062,

        'sell_fisher': 0.39075,
        'sell_bbmiddle_close': 0.99754
    }






































    buy_params = {
        "bbdelta_close": 0.00774,
        "bbdelta_tail": 0.99061,
        "close_bblower": 0.01027,
        "closedelta_close": 0.00871,
        "rocr_1h": 0.52805,
    }

    sell_params = {
        "pHSL": -0.291,
        "pPF_1": 0.009,
        "pPF_2": 0.1,
        "pSL_1": 0.012,
        "pSL_2": 0.068,
        "sell_bbmiddle_close": 1.09653,
        "sell_fisher": 0.37332,
    }





















    @property
    def plot_config(self):
        """
            There are a lot of solutions how to build the return dictionary.
            The only important point is the return value.
            Example:
                plot_config = {'main_plot': {}, 'subplots': {}}

        """

        plot_config = {}

        plot_config["main_plot"] = {
            "bb_lowerband",
            "bb_middleband",
            "ema_fast",
            "ema_slow",
        }

        plot_config["subplots"] = {
            "Mean Volume": {
                "volume_mean_slow": {"color": "#A1A2CF"}
            },
            "ROCR_1h" : {
                "rocr_1h": {"color": "#7D2C34"}
            }
        }

    minimal_roi = {
        "0": 100
    }

    stoploss = -0.99  # use custom stoploss

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 168













    order_types = {
        'buy': 'market',
        'sell': 'market',
        'emergencysell': 'market',
        'forcebuy': 'market',
        'forcesell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    rocr_1h = RealParameter(0.5, 1.0, default=buy_params['rocr_1h'], space='buy', optimize=True)
    bbdelta_close = RealParameter(0.0005, 0.02, default=buy_params['bbdelta_close'], space='buy', optimize=True)
    closedelta_close = RealParameter(0.0005, 0.02, default=buy_params['closedelta_close'], space='buy', optimize=True)
    bbdelta_tail = RealParameter(0.7, 1.0, default=buy_params['bbdelta_tail'], space='buy', optimize=True)
    close_bblower = RealParameter(0.0005, 0.02, default=buy_params['close_bblower'], space='buy', optimize=True)

    sell_fisher = RealParameter(0.1, 0.5, default=sell_params['sell_fisher'], space='sell', optimize=True)
    sell_bbmiddle_close = RealParameter(0.97, 1.1, default=sell_params['sell_bbmiddle_close'], space='sell', optimize=True)

    pHSL = DecimalParameter(-0.500, -0.040, default=sell_params['pHSL'], decimals=3, space='sell', load=True)

    pPF_1 = DecimalParameter(0.008, 0.020, default=sell_params['pPF_1'], decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=sell_params['pSL_1'], decimals=3, space='sell', load=True)

    pPF_2 = DecimalParameter(0.040, 0.100, default=sell_params['pPF_2'], decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=sell_params['pSL_2'], decimals=3, space='sell', load=True)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value




        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(self.rocr_1h.value)
            ) &
            ((
                     (dataframe['lower'].shift().gt(0)) &
                     (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.bbdelta_close.value)) &
                     (dataframe['closedelta'].gt(dataframe['ha_close'] * self.closedelta_close.value)) &
                     (dataframe['tail'].lt(dataframe['bbdelta'] * self.bbdelta_tail.value)) &
                     (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                     (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
             ) |
             (
                     (dataframe['ha_close'] < dataframe['ema_slow']) &
                     (dataframe['ha_close'] < self.close_bblower.value * dataframe['bb_lowerband'])
             )),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['fisher'] > self.sell_fisher.value) &
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_fast'] > dataframe['ha_close']) &
            ((dataframe['ha_close'] * self.sell_bbmiddle_close.value) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0),
            'sell'
        ] = 1

        return dataframe