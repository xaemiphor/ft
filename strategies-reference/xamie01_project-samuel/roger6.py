import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime
import logging  # remove after
logger = logging.getLogger(__name__)  # remove after

class roger(IStrategy):

    INTERFACE_VERSION = 3

    # Short / Long
    can_short = True

    # ROI table (config.json can't have this value as it will override): 
    minimal_roi = {
        "60":  0.01, 
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }

    # Stoploss (config.json can't have this value as it will override)
    stoploss = -0.20

    # Trailing stoploss
    trailing_stop = False
    
    # Timeframe
    timeframe = '15m'
    
    # Run "populate_indicators" only for new candle
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populates new indicators for given strategy

        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair

        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """
        
        # SMA
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = \
            (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        # Candlestick patterns bullish
        dataframe['cdl3inside'] = ta.CDL3INSIDE(dataframe)
        dataframe['cdl3outside'] = ta.CDL3OUTSIDE(dataframe)
        dataframe['cdl3starsinsouth'] = ta.CDL3STARSINSOUTH(dataframe)
        
        # Candlestick patterns bearish
        dataframe['cdl3blackcrows'] = ta.CDL3BLACKCROWS(dataframe)
        dataframe['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(dataframe)
        dataframe['cdl3linestrike'] = ta.CDL3LINESTRIKE(dataframe)
     
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populate rules for the "buy" signal

        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair

        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """
        # Entry based on positive trend and bollinger bands
        dataframe.loc[
            (
                (dataframe['sma200'] < dataframe['sma50']) &
                (dataframe['bb_percent'] < 0.1) &
                (dataframe['bb_width'] > 0.03) |
                # Candlestick patterns
                (dataframe['cdl3inside'] == 100) | # 3 Inside Up/Down
                (dataframe['cdl3outside'] == 100) | # 3 Outside Up/Down
                (dataframe['cdl3starsinsouth'] == 100) # 3 Stars In The South
            ),
        
            # If all contious indicatiors are true, buy is set to 1. Alternatively a buy signal
            # can also be set to 1 if a pattern is found.
            'buy'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populate rules for the "sell" signal
        
        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair
            
        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """
        # Exit based on negative trend and bollinger bands
        dataframe.loc[
            (
                (dataframe['sma200'] > dataframe['sma50']) &
                (dataframe['bb_percent'] > 0.9) &
                (dataframe['bb_width'] > 0.03) |
                # Candlestick patterns
                (dataframe['cdl3blackcrows'] == -100) | # 3 Black Crows
                (dataframe['cdl3whitesoldiers'] == -100) | # 3 White Soldiers
                (dataframe['cdl3linestrike'] == -100) # 3 Line Strike
            ),
            # If all conditions are True for a given row, the 'sell' column for that row is set to 1
            'sell'
        ] = 1
        
        return dataframe
