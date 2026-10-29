#!/bin/bash
cd /mnt/e/LSTM_C90/build_test
make -j4 2>&1
echo "BUILD_EXIT_CODE=$?"
