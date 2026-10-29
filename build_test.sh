#!/bin/bash
cd /mnt/e/LSTM_C90
rm -rf build_test
mkdir build_test
cd build_test
cmake ..
make -j4 2>&1
echo "BUILD_EXIT_CODE=$?"
