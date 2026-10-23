cmake_minimum_required(VERSION 3.10)
project(LSTM_C90 C)

set(CMAKE_C_STANDARD 90)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/utils.c
    src/data.c
    src/lstm.c
    src/model.c
    main.c
)

add_executable(LSTM_C90 ${SOURCES})

# Link math library
target_link_libraries(LSTM_C90 m)

# Test executable
add_executable(test_lstm tests/test_lstm.c src/utils.c src/data.c src/lstm.c src/model.c)
target_link_libraries(test_lstm m)
