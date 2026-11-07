cmake_minimum_required(VERSION 3.10)
project(LSTM_C90 C)

set(CMAKE_C_STANDARD 90)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/ohlc)
include_directories(${CMAKE_SOURCE_DIR}/include/direction)

# Source files
set(SOURCES
    src/utils.c
    src/data.c
    src/lstm.c
    src/model.c
    src/params.c
    src/ohlc/data_ohlc.c
    src/ohlc/predict_ohlc.c
    src/direction/data_direction.c
    src/direction/predict_direction.c
    main.c
)

add_executable(LSTM_C90 ${SOURCES})
target_link_libraries(LSTM_C90 m)

# Test executable
set(TEST_SOURCES
    tests/test_main.c
    tests/test_utils.c
    tests/test_forward.c
    tests/test_backward.c
    tests/test_update.c
    tests/test_model_io.c
    tests/test_deterministic.c
    tests/test_gradient.c
    tests/test_features_26.c
    tests/test_data_io.c
    tests/test_sdata_fixes.c
    tests/test_timeframe_align.c
    tests/test_datetime_shift.c
    tests/test_parameters.c
    tests/ohlc/test_training_ohlc.c
    tests/ohlc/test_data_ohlc.c
    tests/ohlc/test_deterministic_ohlc.c
    tests/direction/test_data_direction.c
    tests/direction/test_training_direction.c
    tests/direction/test_sigmoid_gradient.c
    tests/direction/test_deterministic_direction.c
    src/utils.c
    src/data.c
    src/lstm.c
    src/model.c
    src/params.c
    src/ohlc/data_ohlc.c
    src/ohlc/predict_ohlc.c
    src/direction/data_direction.c
    src/direction/predict_direction.c
)
add_executable(test_lstm ${TEST_SOURCES})
target_link_libraries(test_lstm m)
