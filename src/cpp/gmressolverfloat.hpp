#ifndef GMRESSOLVERFLOAT_HPP
#define GMRESSOLVERFLOAT_HPP

#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <petscksp.h>

// Simple example class for a GPU-based GMRES solver with PETSc (single-process), 
// using single-precision floats (float32).
class GMRESSolverFloat {
public:
    // Constructor / Destructor
    GMRESSolverFloat();
    ~GMRESSolverFloat();

    // Initialize solver with matrix A (CPU or GPU Tensor, float32).
    // For demonstration, we do a CPU loop to set values into a GPU-based PETSc matrix.
    void initialize(const torch::Tensor& A);

    // Solve for a single vector b (float32), returns a GPU Tensor (float32).
    torch::Tensor solve(const torch::Tensor& b);

    // Solve for a batch of vectors B, shape [batch_size, N] (float32), returns GPU Tensor.
    torch::Tensor solveBatch(const torch::Tensor& B);

private:
    Mat A_;   // PETSc matrix (on GPU) - single precision
    KSP ksp_; // PETSc linear solver
    bool initialized_;
};

// Constructor: Initialize PETSc (do this once per application).
GMRESSolverFloat::GMRESSolverFloat()
    : A_(nullptr), ksp_(nullptr), initialized_(false)
{
    PetscBool petsc_initialized;
    PetscErrorCode ierr = PetscInitialized(&petsc_initialized);

    if (!petsc_initialized) {
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    }
}

// Destructor: Destroy PETSc objects and finalize (if you own PETSc lifetime).
GMRESSolverFloat::~GMRESSolverFloat()
{
    if (ksp_) {
        KSPDestroy(&ksp_);
        ksp_ = nullptr;
    }
    if (A_) {
        MatDestroy(&A_);
        A_ = nullptr;
    }
}

// Initialize solver with matrix A; we store it in a GPU-based PETSc matrix (single precision).
void GMRESSolverFloat::initialize(const torch::Tensor& A)
{
    std::cerr << "Initializing single-precision GMRESSolver with matrix A.\n";
    TORCH_CHECK(A.dim() == 2, "Matrix A must be 2D.");
    TORCH_CHECK(A.dtype() == torch::kFloat32,
                "Matrix A must be single precision (float32).");

    const int rows = A.size(0);
    const int cols = A.size(1);
    TORCH_CHECK(rows == cols, "Matrix A must be square for this example.");

    // Destroy old objects if re-initializing
    if (ksp_) { KSPDestroy(&ksp_); ksp_ = nullptr; }
    if (A_)   { MatDestroy(&A_);   A_   = nullptr; }

    // Create a GPU-based AIJ matrix (cuSPARSE), single precision
    PetscErrorCode ierr;
    ierr = MatCreate(PETSC_COMM_WORLD, &A_);
    ierr = MatSetSizes(A_, rows, cols, rows, cols);
    // Choose the single-process AIJ cuSPARSE type:
    ierr = MatSetType(A_, MATSEQAIJCUSPARSE);
    // Or automatically pick up user-defined run-time options:
    // ierr = MatSetFromOptions(A_);

    ierr = MatSetUp(A_);

    // For demonstration, we do a CPU-side loop to insert values.
    // If A is a GPU tensor, copy to CPU, otherwise read directly from CPU memory.
    torch::Tensor A_cpu = A.device().is_cuda() ? A.cpu() : A;

    const float* A_data = A_cpu.data_ptr<float>();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = A_data[i * cols + j];
            ierr = MatSetValue(A_, i, j, static_cast<PetscScalar>(val), INSERT_VALUES);
        }
    }

    // Assemble the matrix (this will trigger transfers to GPU internally)
    ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY);

    // Create KSP
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp_);
    ierr = KSPSetOperators(ksp_, A_, A_);
    ierr = KSPSetType(ksp_, KSPGMRES);
    // Let user or code set solver options at runtime:
    ierr = KSPSetFromOptions(ksp_);

    initialized_ = true;
}

// Solve for a single vector b (1D, float32). Return a GPU torch::Tensor (float32).
torch::Tensor GMRESSolverFloat::solve(const torch::Tensor& b)
{
    TORCH_CHECK(initialized_, "GMRESSolver is not initialized with a matrix A.");
    TORCH_CHECK(b.dim() == 1, "Vector b must be 1D.");
    TORCH_CHECK(b.dtype() == torch::kFloat32,
                "Vector b must be single precision (float32).");

    int n = b.size(0);

    // Create PETSc vectors on the GPU (single precision)
    Vec b_petsc, x_petsc;
    PetscErrorCode ierr = VecCreateSeqCUDA(PETSC_COMM_WORLD, n, &b_petsc);
    ierr = VecCreateSeqCUDA(PETSC_COMM_WORLD, n, &x_petsc);

    // If b is on GPU, copy it to CPU first (simple approach).
    torch::Tensor b_cpu = b.device().is_cuda() ? b.cpu() : b;

    const float* b_data = b_cpu.data_ptr<float>();
    for (int i = 0; i < n; ++i) {
        ierr = VecSetValue(b_petsc, i, static_cast<PetscScalar>(b_data[i]), INSERT_VALUES);
    }
    ierr = VecAssemblyBegin(b_petsc);
    ierr = VecAssemblyEnd(b_petsc);

    // Solve
    ierr = KSPSetUp(ksp_);
    ierr = KSPSolve(ksp_, b_petsc, x_petsc);

    // Create an output tensor on GPU
    torch::Tensor x_gpu = torch::empty({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Get array from PETSc vector (on GPU).
    const PetscScalar* x_vals = nullptr;
    ierr = VecGetArrayRead(x_petsc, &x_vals);

    // Copy from PETSc device pointer to x_gpu
    cudaMemcpy(x_gpu.data_ptr<float>(), x_vals, sizeof(float) * n, cudaMemcpyDeviceToDevice);

    // Restore array
    ierr = VecRestoreArrayRead(x_petsc, &x_vals);

    // Cleanup
    VecDestroy(&b_petsc);
    VecDestroy(&x_petsc);

    return x_gpu;
}

// Solve for a batch of vectors B, shape [batch_size, N] in float32. Return GPU tensor.
torch::Tensor GMRESSolverFloat::solveBatch(const torch::Tensor& B)
{
    TORCH_CHECK(initialized_, "GMRESSolver is not initialized with a matrix A.");
    TORCH_CHECK(B.dim() == 2, "Tensor B must be 2D: [batch_size, N].");
    TORCH_CHECK(B.dtype() == torch::kFloat32,
                "Tensor B must be single precision (float32).");

    const int batch_size = B.size(0);
    const int n = B.size(1);

    // We'll create output X of shape [batch_size, N] on GPU
    torch::Tensor X = torch::empty({batch_size, n},
                                   torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // For each row in B (which is a separate RHS), solve and store the result in X.
    for (int i = 0; i < batch_size; ++i) {
        torch::Tensor b_i = B[i];
        torch::Tensor x_i = solve(b_i);  // returns GPU tensor [n]
        X[i].copy_(x_i);
    }

    return X;
}

#endif  // GMRESSOLVERFLOAT_HPP
