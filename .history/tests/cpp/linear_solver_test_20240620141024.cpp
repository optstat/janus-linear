#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>

// Test case for zeros method
/**
 *  g++ -g -std=c++17 tensordual_test.cpp -o dualtest   -I /home/panos/Applications/libtorch/include/torch/csrc/api/include/   -I /home/panos/Applications/libtorch/include/torch/csrc/api/include/torch/   -I /home/panos/Applications/libtorch/include/   -I /usr/local/include/gtest  -L /usr/local/lib   -lgtest  -lgtest_main -L /home/panos/Applications/libtorch/lib/ -ltorch   -ltorch_cpu   -ltorch_cuda   -lc10 -lpthread -Wl,-rpath,/home/panos/Applications/libtorch/lib
 */

// Test case for zeros_like method
TEST(LUTest, 2dTest) {

    EXPECT_TRUE(true==true);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
