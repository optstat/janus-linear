#include <gtest/gtest.h>
#include <torch/torch.h>
#include <petscsys.h>

// Include your QRcuComplex implementation header
// e.g. #include "QRcuComplex.h"
#include "../../src/cpp/qrcucomplex.cu"  // Adjust path as needed

// Helper function to generate a random square complex matrix, right-hand side x,
// and then compute b = A*x for verification. This will allow us to check that
// solving the system returns the original x.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> generateSystemSquare(
    int64_t N,
    torch::Device device = torch::kCPU
) {
    // Create random matrix A (N x N) in complex float
    // For reproducibility, you could set a manual seed
    auto A = torch::rand({N, N}, torch::dtype(torch::kComplexFloat).device(device));
    
    // Create a random solution x (N)
    auto x_true = torch::rand({N}, torch::dtype(torch::kComplexFloat).device(device));
    
    // Compute b = A * x_true
    // We can do a matrix-vector multiply. 
    // Note: PyTorch is row-major by default, but matmul should still work as expected logically.
    auto b = torch::matmul(A, x_true.unsqueeze(/*dim=*/1)).squeeze(/*dim=*/1);

    return std::make_tuple(A, x_true, b);
}

TEST(QRcuComplexTest, FactorizeAndSolveSquare)
{
    // Test a small square system A in C^(4x4).
    // We generate a random system A*x = b, factorize A, then see if we recover x.
    const int64_t N = 4;
    auto device = torch::kCPU;
    
    auto [A_cpu, x_true_cpu, b_cpu] = generateSystemSquare(N, device);

    // Construct the QRcuComplex class with A. 
    // (According to the prompt, it expects A_in to be NxL. We'll do NxN for simplicity.)
    QRcuComplex solver(A_cpu);

    // Perform QR decomposition on GPU
    solver.decompose();

    // Solve for x
    auto x_est_cpu = solver.solve(b_cpu);

    // Compare x_est_cpu to x_true_cpu
    // We expect the difference to be small if the solver is correct
    auto diff = (x_est_cpu - x_true_cpu).abs().sum().item<float>();
    // Arbitrary tolerance for floating point
    float tol = 1e-3;
    EXPECT_LT(diff, tol) << "Solution differs from the true x by more than " << tol;
}

TEST(QRcuComplexTest, FactorizeAndSolveMultipleRHS)
{
    // Test solving multiple right-hand-sides B simultaneously with solvev.
    const int64_t N = 5;
    auto device = torch::kCPU;

    // Generate a system A*x = b. We'll also generate multiple b's.
    auto [A_cpu, x_true_cpu, b_cpu] = generateSystemSquare(N, device);

    // We'll generate B of shape (M x N), where each row is a separate b.
    int64_t M = 3; 
    // For each row, we pick a random x and compute b = A*x
    auto B_cpu = torch::zeros({M, N}, torch::dtype(torch::kComplexFloat).device(device));
    auto X_true_cpu = torch::zeros({M, N}, torch::dtype(torch::kComplexFloat).device(device));
    for (int64_t i = 0; i < M; ++i) {
        auto xi = torch::rand({N}, torch::dtype(torch::kComplexFloat).device(device));
        auto bi = torch::matmul(A_cpu, xi.unsqueeze(1)).squeeze(1);
        B_cpu[i] = bi;
        X_true_cpu[i] = xi;
    }

    // Factorize A
    QRcuComplex solver(A_cpu);
    solver.decompose();

    // Solve all in one call
    auto X_est_cpu = solver.solvev(B_cpu);

    // Check difference
    auto diff = (X_est_cpu - X_true_cpu).abs().sum().item<float>();
    float tol = 1e-3;
    EXPECT_LT(diff, tol) << "Batched solution differs from the true X by more than " << tol;
}

TEST(QRcuComplexTest, SolveWithoutDecomposeThrows)
{
    // If we call solve without calling decompose, it should throw.
    const int64_t N = 3;
    auto device = torch::kCPU;
    auto [A_cpu, x_true_cpu, b_cpu] = generateSystemSquare(N, device);
    // Add a check to ensure PETSc is initialized
    PetscBool petsc_initialized;
    PetscInitialized(&petsc_initialized);

    if (!petsc_initialized) {
        PetscErrorCode ierr = PetscInitialize(NULL, NULL, NULL, NULL);
        if (ierr) {
            throw std::runtime_error("Error initializing PETSc.");
        }
    }
    QRcuComplex solver(A_cpu);

    // We expect an error / throw from solve if factorized_ is false.
    EXPECT_THROW({
        auto x_est_cpu = solver.solve(b_cpu);
    }, torch::Error); // or std::runtime_error, depending on your actual error type
}

TEST(QRcuComplexTest, DimensionMismatchThrows)
{
    // Provide a b that does not match the required dimension L.
    const int64_t N = 4;
    auto device = torch::kCPU;
    auto [A_cpu, x_true_cpu, b_cpu] = generateSystemSquare(N, device);

    // Actually, b_cpu is dimension N. The class wants b to be dimension L for A NxL.
    // Let's artificially make a mismatch: supply a b of dimension N+1.
    auto b_bad = torch::rand({N+1}, torch::dtype(torch::kComplexFloat).device(device));

    // Factorize
    QRcuComplex solver(A_cpu);
    solver.decompose();

    // Attempt solve
    EXPECT_THROW({
        auto x_est_cpu = solver.solve(b_bad);
    }, torch::Error);
}

// Example main for Google Test
int main(int argc, char** argv)
{  
    ::testing::InitGoogleTest(&argc, argv);
    // Initialize PETSc
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    RUN_ALL_TESTS();    

    // Finalize PETSc
    PetscFinalize();
  
}
