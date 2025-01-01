#!/bin/bash
pip install ta finta arrow statsmodels > /dev/null 2>&1
exec /home/ftuser/.local/bin/freqtrade "${@}"
