#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include "../../src/cpp/lute.hpp"
#include "../../src/cpp/qr.hpp"
#include "../../src/cpp/qrte.hpp"
#include "../../src/cpp/qrted.hpp"
#include "../../src/cpp/qrtedc.hpp"
#include "../../src/cpp/gmressolver.hpp"


TEST(LUTest, A1x2x2) {
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



TEST(QRTTest, A1x2x2) {
    int N=2;
    torch::Tensor A = torch::rand({N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrt(A);
    auto x = janus::qrtsolvev(qt, r, B);
    auto Ax = torch::mv(A, x);
    EXPECT_TRUE(torch::allclose(Ax, B));
}


TEST(QRTTest, A10x10) {
    int N=10;
    torch::Tensor A = torch::rand({N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrt(A);
    auto x = janus::qrtsolvev(qt, r, B);
    auto Ax = torch::mv(A, x);
    EXPECT_TRUE(torch::allclose(Ax, B));
}

TEST(QRTTest, A1x100x100) {
    int N=100;
    torch::Tensor A = torch::rand({N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrt(A);
    auto x = janus::qrtsolvev(qt, r, B);
    auto Ax = torch::mv(A, x);
    EXPECT_TRUE(torch::allclose(Ax, B));
}







TEST(QRTeTest, A1x2x2) {
    int M=1;
    int N=2;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(QRTeTest, A4x2x2) {
    int M=4;
    int N=2;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(QRTeTest, A1x100x100) {
    int M=1;
    int N=100;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(QRTeTest, A100x10x10) {
    int M=100;
    int N=10;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}




TEST(QRTeDTest, A1x2x2) {
    int M=1;
    int N=2;
    torch::Tensor Ar = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor Ad = torch::rand({M,N,N,N}, dtype(torch::kFloat64));
    torch::Tensor Br = torch::rand({M,N}, dtype(torch::kFloat64));
    torch::Tensor Bd = torch::rand({M,N,N}, dtype(torch::kFloat64));
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrted(A);
    auto x = janus::qrtedsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(Ax.r.index({i}), B.r.index({i})));
    }
}


TEST(QRTeDTest, A1x4x4) {
    int M=1;
    int N=4;
    torch::Tensor Ar = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor Ad = torch::rand({M,N,N,N}, dtype(torch::kFloat64));
    torch::Tensor Br = torch::rand({M,N}, dtype(torch::kFloat64));
    torch::Tensor Bd = torch::rand({M,N,N}, dtype(torch::kFloat64));
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrted(A);
    auto x = janus::qrtedsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(Ax.r.index({i}), B.r.index({i})));
    }
}

TEST(QRTeDCTest, A1x2x2) {
    int M=1;
    int N=2;
    torch::Tensor Ar = torch::complex(torch::rand({M,N,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N,N}, dtype(torch::kFloat64)));
    torch::Tensor Ad = torch::complex(torch::rand({M,N,N,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N,N,N}, dtype(torch::kFloat64)));    
    torch::Tensor Br = torch::complex(torch::rand({M,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N}, dtype(torch::kFloat64)));
    torch::Tensor Bd = torch::complex(torch::rand({M,N,N}, dtype(torch::kFloat64)),
                                        torch::rand({M,N,N}, dtype(torch::kFloat64)));   
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrtedc(A);
    auto x = janus::qrtedcsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(torch::real(Ax.r.index({i})), torch::real(B.r.index({i}))));
        EXPECT_TRUE(torch::allclose(torch::imag(Ax.r.index({i})), torch::imag(B.r.index({i}))));
    }
}

TEST(QRTeDCTest, A1x100x100) {
    int M=1;
    int N=100;
    torch::Tensor Ar = torch::complex(torch::rand({M,N,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N,N}, dtype(torch::kFloat64)));
    torch::Tensor Ad = torch::complex(torch::rand({M,N,N,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N,N,N}, dtype(torch::kFloat64)));    
    torch::Tensor Br = torch::complex(torch::rand({M,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N}, dtype(torch::kFloat64)));
    torch::Tensor Bd = torch::complex(torch::rand({M,N,N}, dtype(torch::kFloat64)),
                                        torch::rand({M,N,N}, dtype(torch::kFloat64)));   
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrtedc(A);
    auto x = janus::qrtedcsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(torch::real(Ax.r.index({i})), torch::real(B.r.index({i}))));
        EXPECT_TRUE(torch::allclose(torch::imag(Ax.r.index({i})), torch::imag(B.r.index({i}))));
    }
}


TEST(QRTeDCTest, A2x2x2) {
    int M=2;
    int N=2;
    torch::Tensor Ar = torch::complex(torch::rand({M,N,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N,N}, dtype(torch::kFloat64)));
    torch::Tensor Ad = torch::complex(torch::rand({M,N,N,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N,N,N}, dtype(torch::kFloat64)));    
    torch::Tensor Br = torch::complex(torch::rand({M,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N}, dtype(torch::kFloat64)));
    torch::Tensor Bd = torch::complex(torch::rand({M,N,N}, dtype(torch::kFloat64)),
                                        torch::rand({M,N,N}, dtype(torch::kFloat64)));   
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrtedc(A);
    auto x = janus::qrtedcsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(torch::real(Ax.r.index({i})), torch::real(B.r.index({i}))));
        EXPECT_TRUE(torch::allclose(torch::imag(Ax.r.index({i})), torch::imag(B.r.index({i}))));
    }
}

TEST(QRTeDCTest, A100x100x100) {
    int M=10;
    int N=10;
    torch::Tensor Ar = torch::complex(torch::rand({M,N,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N,N}, dtype(torch::kFloat64)));
    torch::Tensor Ad = torch::complex(torch::rand({M,N,N,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N,N,N}, dtype(torch::kFloat64)));    
    torch::Tensor Br = torch::complex(torch::rand({M,N}, dtype(torch::kFloat64)),
                                      torch::rand({M,N}, dtype(torch::kFloat64)));
    torch::Tensor Bd = torch::complex(torch::rand({M,N,N}, dtype(torch::kFloat64)),
                                        torch::rand({M,N,N}, dtype(torch::kFloat64)));   
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrtedc(A);
    auto x = janus::qrtedcsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(torch::real(Ax.r.index({i})), torch::real(B.r.index({i}))));
        EXPECT_TRUE(torch::allclose(torch::imag(Ax.r.index({i})), torch::imag(B.r.index({i}))));
    }
}




// A helper function to compare two Tensors elementwise with a tolerance.
void expectTensorsClose(const torch::Tensor& actual, 
                        const torch::Tensor& expected, 
                        double tol = 1e-6)
{
    // Check same shape
    ASSERT_TRUE(actual.sizes() == expected.sizes())
        << "Tensor shapes do not match: " 
        << actual.sizes() << " vs. " << expected.sizes();

    // Convert both to CPU double (if they're not already) for easy comparison
    auto actual_cpu = actual.to(torch::kCPU).to(torch::kFloat64);
    auto expected_cpu = expected.to(torch::kCPU).to(torch::kFloat64);

    // Compare elementwise
    auto diff = (actual_cpu - expected_cpu).abs();
    auto max_diff = diff.max().item<double>();
    EXPECT_LE(max_diff, tol) 
        << "Max difference " << max_diff << " exceeds tolerance " << tol;
}

// Test a simple 3x3 system with a single right-hand side on the CPU matrix, GPU solver.
TEST(GMRESSolverTest, SolveSingleRHS_3x3)
{
    // If CUDA is not available, we can skip or fallback to CPU
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is not available. Skipping GPU test.";
    }

    // Matrix A on CPU
    torch::Tensor A = torch::tensor({
        {3.0, 1.0, 0.0},
        {1.0, 2.0, 1.0},
        {0.0, 1.0, 2.0}
    }, torch::kFloat64);

    // Known solution x_ref for the system A x = b
    // Suppose b = {5, 6, 7}, we want to see if x is correct.
    // We'll compute x_ref manually or just trust the solver and check residual.
    // Let's do a quick solve offline or guess we found x_ref = {1, 2, 3}, for instance.

    torch::Tensor b = torch::tensor({5.0, 6.0, 7.0}, torch::kFloat64).cuda();

    // We can do a small manual solve by hand or define:
    //  A = [ [3,1,0], [1,2,1], [0,1,2] ]
    //  x = [1,2,3]^T => A*x = [3*1+1*2+0*3, 1*1+2*2+1*3, 0+1*2+2*3] 
    //                  = [3+2, 1+4+3, 2+6] = [5, 8, 8], which isn't [5,6,7].
    // Let's do a direct check:
    //     x = [1,1,3]^T => A*x = [3+1, 1+2+3, 1+6] = [4,6,7] close but not exact.
    // For test, we'll just check that the solver returns a small residual.
    // Or we can do a quick solve with something like:
    //     x = [1.0, 2.0, 2.0] => A*x = [3*1+1*2, 1*1+2*2+1*2, 0+1*2+2*2]
    //                            = [5, 1+4+2, 2+4] = [5, 7, 6].
    // We'll skip exact manual solution. We'll check the residual in the test.

    GMRESSolver solver;
    solver.initialize(A);  // build GPU matrix internally

    torch::Tensor x = solver.solve(b);  // output on GPU
    ASSERT_EQ(x.device().type(), torch::kCUDA)
        << "Solution should be a CUDA tensor";

    // (Optional) Check residual A*x - b < tolerance
    // We'll copy A to GPU for quick matmul check in torch:
    torch::Tensor A_gpu = A.cuda();
    torch::Tensor r = torch::matmul(A_gpu, x) - b;
    double norm_r = r.norm().item<double>();
    EXPECT_NEAR(norm_r, 0.0, 1e-6) 
            << "Residual is too high.";
}

// Test a simple 3x3 system with multiple right-hand sides.
TEST(GMRESSolverTest, SolveBatchRHS_3x3)
{
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is not available. Skipping GPU test.";
    }

    // Matrix A on CPU
    torch::Tensor A = torch::tensor({
        {3.0, 1.0, 0.0},
        {1.0, 2.0, 1.0},
        {0.0, 1.0, 2.0}
    }, torch::kFloat64);

    // Batch of b: shape [2, 3]
    torch::Tensor B = torch::tensor({
        {5.0, 6.0, 7.0},
        {4.0, 2.0, 10.0}
    }, torch::kFloat64).cuda();

    GMRESSolver solver;
    solver.initialize(A);

    // Solve for each row in B
    torch::Tensor X = solver.solveBatch(B);
    ASSERT_EQ(X.sizes(), (std::vector<int64_t>{2, 3}))
        << "Output X should be [2, 3].";
    ASSERT_EQ(X.device().type(), torch::kCUDA)
        << "Solution batch should be a CUDA tensor";

    // Check residual for each row in B
    torch::Tensor A_gpu = A.cuda();
    torch::Tensor R = torch::matmul(A_gpu, X.transpose(0,1)).transpose(0,1) - B;
    // R now shape [2, 3], each row is A*x_i - b_i
    torch::Tensor norms = R.norm(2, 1); // norm per row => shape [2]
    for (int i = 0; i < 2; ++i) {
        double rnorm = norms[i].item<double>();
        EXPECT_NEAR(rnorm, 0.0, 1e-6) 
            << "Residual for RHS " << i << " is too high.";
    }
}


// If you want an exact solution test for a known system, you can do something like:
TEST(GMRESSolverTest, SolveKnownSystemExactCheck)
{
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is not available. Skipping GPU test.";
    }

    // Let's pick a system where the solution is easy to compute by hand:
    // A = [[2,0],[0,3]], b = [4,6] => x = [2,2].
    torch::Tensor A = torch::tensor({
        {2.0, 0.0},
        {0.0, 3.0}
    }, torch::kFloat64);

    torch::Tensor b = torch::tensor({4.0, 6.0}, torch::kFloat64).cuda();
    torch::Tensor x_expected = torch::tensor({2.0, 2.0}, torch::kFloat64).cuda();

    GMRESSolver solver;
    solver.initialize(A);
    torch::Tensor x = solver.solve(b);

    // Compare with the known exact solution
    expectTensorsClose(x, x_expected, 1e-12);
}


int main(int argc, char **argv) {
    //Initialize PETSc
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    //check to make sure PETSc is initialized
    PetscBool petsc_initialized;
    PetscErrorCode ierr = PetscInitialized(&petsc_initialized);
    if (!petsc_initialized) {
        throw std::runtime_error("PETSc is not initialized.");        
    }
    ::testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
    PetscFinalize();
}
