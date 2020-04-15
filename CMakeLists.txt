cmake_minimum_required(VERSION 3.8.0)
project(MnistBenchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SRC_FILES
	./MnistBenchmark/Source/Main.cpp
)

add_executable(MnistBenchmark ${SRC_FILES})

target_include_directories(MnistBenchmark
	PUBLIC ./MnistBenchmark/Include/
	PUBLIC ./External/Gsl/include/
)

add_compile_definitions(DATA_LOCATION="${CMAKE_SOURCE_DIR}/Data")