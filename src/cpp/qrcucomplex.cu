#include <torch/torch.h>
#include <cusolverDn.h>
#include <cuComplex.h>    // For cuComplex / cuDoubleComplex

#include <cublas_v2.h>
#include <cuda_runtime.h>   
#include <complex>
#include <iostream>

// cuComplex is the CUDA complex float type.
// If you need double precision complex, use cuDoubleComplex and related cuSOLVER routines.
#include <cuComplex.h>

// Error-checking helpers (for brevity, we provide simple macros here).
// In production, prefer robust error checking.
#define CHECK_CUDA(func)                                                     \
  {                                                                          \
    cudaError_t status = (func);                                            \
    if (status != cudaSuccess) {                                            \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
      throw std::runtime_error("CUDA Error");                               \
    }                                                                        \
  }

#define CHECK_CUSOLVER(func)                                                \
  {                                                                          \
    cusolverStatus_t status = (func);                                       \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                \
      std::cerr << "cuSOLVER Error: " << status << std::endl;               \
      throw std::runtime_error("cuSOLVER Error");                           \
    }                                                                        \
  }

#define CHECK_CUBLAS(func)                                                  \
  {                                                                          \
    cublasStatus_t status = (func);                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "cuBLAS Error: " << status << std::endl;                 \
      throw std::runtime_error("cuBLAS Error");                             \
    }                                                                        \
  }

class QRcuComplex {
public:
    // Constructor: store A in a GPU tensor, create handles, etc.
    // A is expected to be an N x L complex float Tensor on device or CPU.
    // If on CPU, we move it to CUDA. If it's already on CUDA, we keep it there.
    QRcuComplex(const torch::Tensor& A_in)
    {
        // Ensure A_in is complex float (c10::kComplexFloat) and on CUDA
        TORCH_CHECK(A_in.dtype() == torch::kComplexFloat,
                    "QRcuComplex requires a complex float (kComplexFloat) tensor.");
        
        // Move to CUDA if not already
        A_ = A_in.is_cuda() ? A_in.clone() : A_in.cuda();


        // Make sure it's contiguous (important for cuSOLVER column-major expectation)
        A_ = A_in.contiguous();

        // Extract sizes: A is (N x L)
        N_ = A_.size(0);
        L_ = A_.size(1);
        

        // Create cuSolver and cuBLAS handles
        CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle_));
        CHECK_CUBLAS(cublasCreate(&cublas_handle_));
        
        // We'll allocate tau_ to hold Householder scalars. 
        // For complex QR, tau is also complex. 
        tau_ = torch::zeros({std::min(N_, L_)}, 
                            torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
        
        // Initially, we have not done the factorization.
        factorized_ = false;
    }

    // Destructor: destroy handles
    ~QRcuComplex()
    {
        if (cusolver_handle_) {
            cusolverDnDestroy(cusolver_handle_);
        }
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
    }

    // Perform the QR decomposition in-place on A_ (device side)
    void decompose()
    {
        // If already factorized once, you may want to skip or re-factor. 
        // We'll just do it again here.
        factorized_ = false;
        
        // Query working space of geqrf
        int lwork = 0;
        // Using single-precision complex routine: cusolverDnCgeqrf_bufferSize
        CHECK_CUSOLVER(
            cusolverDnCgeqrf_bufferSize(
                cusolver_handle_,
                N_,
                L_,
                reinterpret_cast<cuComplex*>(A_.data_ptr<c10::complex<float>>()),
                N_,
                &lwork
            )
        );
        
        // Allocate workspace
        auto workspace = torch::empty({static_cast<long>(lwork)}, 
                                      torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
        
        // device scalar to hold info
        auto devInfo = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        
        // Perform QR factorization: A = Q * R
        CHECK_CUSOLVER(
            cusolverDnCgeqrf(
                cusolver_handle_,
                N_,
                L_,
                reinterpret_cast<cuComplex*>(A_.data_ptr<c10::complex<float>>()),
                N_,
                reinterpret_cast<cuComplex*>(tau_.data_ptr<c10::complex<float>>()),
                reinterpret_cast<cuComplex*>(workspace.data_ptr<c10::complex<float>>()),
                lwork,
                devInfo.data_ptr<int>()
            )
        );
        
        // Optionally, check devInfo for success (== 0)
        
        factorized_ = true;
    }

    // Solve for a single vector b (dimension L_). 
    // Returns solution x (also dimension L_) on the CPU.
    // NOTE: In a typical Ax=b scenario for A(N x L), we'd expect b to be N-dimensional if A x = b. 
    // Here we follow the prompt literally. 
    torch::Tensor solve(const torch::Tensor& b_in)
    {
        TORCH_CHECK(factorized_, "You must call decompose() before solve().");
        TORCH_CHECK(b_in.size(0) == static_cast<long>(L_),
                    "b must have dimension L (as per the prompt).");
        
        // Move b to GPU (complex float), contiguous
        auto b_gpu = b_in.to(torch::kCUDA, /*dtype=*/torch::kComplexFloat).contiguous();
        
        // We have factorized A_ in-place.  A_ contains R in the upper triangle
        // and Householder vectors in the lower part. tau_ has Householder scalars.
        // We want to do x = R^{-1} Q^H b.  
        // Steps:
        //  1) Apply Q^H to b using cusolverDnCunmqr (or cublas) 
        //  2) Solve R x = temporary result
        
        int M = N_; // number of rows in A
        int N = L_; // number of cols in A
        int K = std::min(M, N); // = L if N_ >= L_, etc.
        
        // 1) Apply Q^H b => let y = Q^H b
        //    We'll call cusolverDnCunmqr with side='L', trans='C' (conjugate transpose).
        {
            int lwork = 0;
            // Query workspace
            CHECK_CUSOLVER(
                cusolverDnCunmqr_bufferSize(
                    cusolver_handle_,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_OP_C,  // Q^H
                    M,            // rows of side=left
                    1,            // number of columns in b
                    K,            // min(M,N)
                    reinterpret_cast<const cuComplex*>(A_.data_ptr<c10::complex<float>>()),
                    M,  // leading dimension for A
                    reinterpret_cast<const cuComplex*>(tau_.data_ptr<c10::complex<float>>()),
                    reinterpret_cast<cuComplex*>(b_gpu.data_ptr<c10::complex<float>>()),
                    M,  // leading dimension for b
                    &lwork
                )
            );
            
            auto work_cunmqr = torch::empty({lwork}, 
                                            torch::dtype(torch::kComplexFloat).device(torch::kCUDA));
            auto devInfo = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
            
            CHECK_CUSOLVER(
                cusolverDnCunmqr(
                    cusolver_handle_,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_OP_C,
                    M,
                    1,
                    K,
                    reinterpret_cast<const cuComplex*>(A_.data_ptr<c10::complex<float>>()),
                    M,
                    reinterpret_cast<const cuComplex*>(tau_.data_ptr<c10::complex<float>>()),
                    reinterpret_cast<cuComplex*>(b_gpu.data_ptr<c10::complex<float>>()),
                    M,
                    reinterpret_cast<cuComplex*>(work_cunmqr.data_ptr<c10::complex<float>>()),
                    lwork,
                    devInfo.data_ptr<int>()
                )
            );
            // Optionally check devInfo
        }
        
        // 2) Solve R x = y
        // R is the upper triangle of A_ (dimension N x N if square, or top-left L x L).
        // We'll do a triangular solve using cuBLAS. 
        // The portion of A_ that is R is the top-left (N x N) if square or (L x L).
        // Actually, the leading dimension is M, so we interpret the top-left corner of A_ size N x N or L x L.
        
        // We want x in the top L entries of b_gpu after this solve. 
        // We'll do cublasCtrsv or cublasCtrsm. For a single vector, cublasCtrsv is simpler.
        {
            // R is upper triangular
            const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
            
            CHECK_CUBLAS(
                cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_HOST)
            );

            // We solve R x = y. 
            //   - matrix R is stored in A_ (upper triangle). 
            //   - x overwrites y in b_gpu.
            // Because A_ has dimensions (M x N) = (N_ x L_), the leading dimension is M = N_.
            // The top-left triangular block is dimension (L_ x L_), i.e. R is L_ x L_.
            
            // We do the solve in reverse order because cublas TRSV expects 
            // one column vector. 
            // cublasCtrsv: R * x = b
            // R is upper triangular, so we specify CUBLAS_FILL_MODE_UPPER.
            
            CHECK_CUBLAS(
                cublasCtrsv(
                    cublas_handle_,
                    CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N,  // no transpose (R as is)
                    CUBLAS_DIAG_NON_UNIT,
                    L_,
                    reinterpret_cast<const cuComplex*>(A_.data_ptr<c10::complex<float>>()),
                    N_,  // leading dimension of A_ is N_
                    reinterpret_cast<cuComplex*>(b_gpu.data_ptr<c10::complex<float>>()),
                    1
                )
            );
        }
        
        // Now b_gpu[0..L_-1] should contain x. If N != L, you'd have to think about partial solutions, etc.
        // Return x to CPU (dimension L_)
        auto x_cpu = b_gpu.slice(0, 0, L_).cpu();
        return x_cpu;
    }

    // Solve for multiple vectors B (dimension M x L_)
    // Return X of dimension M x L_ on CPU.
    // For each row in B, we interpret that as a separate "b" (the prompt is ambiguous).
    // We'll apply the same Q^H and R solve to each row. 
    torch::Tensor solvev(const torch::Tensor& B_in)
    {
        TORCH_CHECK(factorized_, "You must call decompose() before solvev().");
        TORCH_CHECK(B_in.size(1) == static_cast<long>(L_),
                    "B must have shape (M x L).");
        
        // Move B to GPU (complex float), contiguous
        auto B_gpu = B_in.to(torch::kCUDA, /*dtype=*/torch::kComplexFloat).contiguous();
        
        // Let M_ = B_in.size(0). So B is M_ x L_.
        // We want to apply Q^H to each "column" of B in a batched sense if possible. 
        // But for demonstration, we'll do it as a single matrix multiply. 
        //   - Q^H has dimension (N_ x N_) logically, 
        //   - B_gpu has dimension (N_ x M_) if each column is a b-vector. 
        // 
        // However, the prompt says B is (M x L). It's unclear how that lines up with A(N x L).
        // We'll assume each row is a separate b, meaning we might need to do it in a loop 
        // or reinterpret the layout so that Q^H * B is well-defined. 
        // 
        // In a real scenario, you'd reorder B to have shape (N_ x M_) if you wanted to multiply Q^H * B directly. 
        // For brevity, we do a row-wise approach:
        
        int M_ = B_in.size(0);
        
        // We'll create a loop over each row [0..M_-1], apply the same steps as solve(...).
        // Then store the result in X.
        auto X_gpu = torch::zeros_like(B_gpu); // (M_ x L_), same device/dtype
        
        for (int64_t i = 0; i < M_; i++) {
            // b_i is row i of B_gpu (size L_).
            auto b_i = B_gpu.slice(/*dim=*/0, i, i+1);   // shape = (1 x L_)
            // We want to treat it as a vector of length L_ but along dimension 1.
            // Usually for Ax=b with A(N x L), b is (N). There's a mismatch. 
            // We'll do exactly the same steps as `solve(...)`, but we must copy b_i 
            // to a separate buffer that is dimension (N_). 
            // For demonstration, let's just call solve(...) on the row.
            
            // We'll flatten that row into 1D (L_) for the method.
            auto b_i_flat = b_i.view({L_});
            
            auto x_i_cpu = solve(b_i_flat);
            // x_i_cpu is (L_)
            
            // Place it back into X_gpu row i
            // Move x_i_cpu to GPU
            auto x_i_gpu = x_i_cpu.to(B_gpu.device()).contiguous().view({1, L_});
            X_gpu.slice(/*dim=*/0, i, i+1).copy_(x_i_gpu);
        }
        
        // Return X to CPU
        auto X_cpu = X_gpu.cpu();
        return X_cpu;
    }

private:
    // Data members
    torch::Tensor A_;    // (N x L) factorized in-place by geqrf
    torch::Tensor tau_;  // (min(N,L)) Householder scalars
    bool factorized_;

    cusolverDnHandle_t cusolver_handle_ = nullptr;
    cublasHandle_t      cublas_handle_   = nullptr;

    int64_t N_;
    int64_t L_;
};

