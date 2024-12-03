#!/bin/bash
pip install ta > /dev/null 2>&1
exec /home/ftuser/.local/bin/freqtrade "${@}"
