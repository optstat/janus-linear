#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include "../../src/cpp/luTe.hpp"
#include "../../src/cpp/qrTe.hpp"

// Test case for zeros method
/**
 *  g++ -g -std=c++17 tensordual_test.cpp -o dualtest   -I /home/panos/Applications/libtorch/include/torch/csrc/api/include/   -I /home/panos/Applications/libtorch/include/torch/csrc/api/include/torch/   -I /home/panos/Applications/libtorch/include/   -I /usr/local/include/gtest  -L /usr/local/lib   -lgtest  -lgtest_main -L /home/panos/Applications/libtorch/lib/ -ltorch   -ltorch_cpu   -ltorch_cuda   -lc10 -lpthread -Wl,-rpath,/home/panos/Applications/libtorch/lib
 */

// Test case for zeros_like method
/*TEST(LUTest, A1x2x2) {
    int M=1;
    int N=2;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [LU,indx] = janus::LUTe(A);
    auto x = janus::solveluv(LU, indx, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}



TEST(LUTest, A1x100x100) {
    int M=1;
    int N=100;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [LU,indx] = janus::LUTe(A);
    auto x = janus::solveluv(LU, indx, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(LUTest, A10x100x100) {
    int M=10;
    int N=100;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [LU,indx] = janus::LUTe(A);
    auto x = janus::solveluv(LU, indx, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(LUTest, A1x1000x1000) {
    int M=1;
    int N=1000;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [LU,indx] = janus::LUTe(A);
    auto x = janus::solveluv(LU, indx, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(LUTest, A10x1000x1000) {
    int M=100;
    int N=1000;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [LU,indx] = janus::LUTe(A);
    auto x = janus::solveluv(LU, indx, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}*/


TEST(QRTest, A1x2x2) {
    int M=1;
    int N=2;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    std::cerr << "x=";
    janus::print_tensor(x);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        std::cerr << "Ax=";
        janus::print_tensor(Ax);
        std::cerr << "B=";
        janus::print_tensor(B);
        //EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(QRTest, A4x2x2) {
    int M=4;
    int N=2;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    std::cerr << "x=";
    janus::print_tensor(x);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        std::cerr << "Ax=";
        janus::print_tensor(Ax);
        std::cerr << "B=";
        janus::print_tensor(B);
        //EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(QRTest, A1x100x100) {
    int M=1;
    int N=100;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    std::cerr << "x=";
    janus::print_tensor(x);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        std::cerr << "Ax=";
        janus::print_tensor(Ax);
        std::cerr << "B=";
        janus::print_tensor(B);
        //EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(QRTest, A100x100x100) {
    int M=100;
    int N=100;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    std::cerr << "x=";
    janus::print_tensor(x);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        std::cerr << "Ax=";
        janus::print_tensor(Ax);
        std::cerr << "B=";
        janus::print_tensor(B);
        //EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
