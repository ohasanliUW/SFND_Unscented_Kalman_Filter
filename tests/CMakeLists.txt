cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
set(GTEST_LIB, "../build/lib/libgtest.a")
set(GTEST_MAIN_LIB, "../build/lib/libgtest_main.a")

add_executable (test main.cpp ut.cpp ../src/ukf.cpp ../src/ukf.h)
add_test(NAME test1 COMMAND ./test)
enable_testing()
target_link_libraries(test gtest gtest_main)


