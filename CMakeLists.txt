cmake_minimum_required(VERSION 3.22)
project(Parallelism)

set(CMAKE_CXX_STANDARD 23)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx512f -mavx512dq")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mavx512f -mavx512dq")


add_executable(Parallelism main.cpp
)
