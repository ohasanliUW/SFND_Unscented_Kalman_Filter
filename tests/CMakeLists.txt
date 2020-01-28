cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

enable_testing()

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
set(GTEST_LIB, "../build/lib/libgtest.a")
set(GTEST_MAIN_LIB, "../build/lib/libgtest_main.a")

add_executable (aug_sig main.cpp aug_sig.cpp ../src/ukf.cpp ../src/ukf.h)
add_test(NAME aug_sig COMMAND ./aug_sig)
target_link_libraries(aug_sig gtest gtest_main)


