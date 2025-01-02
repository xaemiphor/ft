#!/bin/bash
pip install ta finta arrow py3cw > /dev/null 2>&1
exec /home/ftuser/.local/bin/freqtrade "${@}"
