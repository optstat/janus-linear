#ifndef GMRES_HPP_INCLUDED
#define GMRES_HPP_INCLUDED

#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include <tuple>
using Slice = torch::indexing::Slice;


// Function to solve the Hessenberg system
torch::Tensor solve_hessenberg(const torch::Tensor& H, const torch::Tensor& beta, int k) {
    // Extract the top-left k x k submatrix from H and the first k elements of beta
    auto H_sub = H.index({Slice(), Slice(0, k), Slice(0, k)});
    auto beta_sub = beta.index({Slice(), Slice(0, k)});
    int M = H.size(0);

    // Initialize the solution vector y
    auto y = torch::zeros({M, k}, H.options());       

    // Perform back substitution
    for (int i = k - 1; i >= 0; --i) {
        //y[i] = beta_sub[i];
        y.index_put_({Slice(), i}, beta_sub.index({Slice(), i}));
        for (int j = i + 1; j < k; ++j) {
            //y[i] -= H_sub[i][j] * y[j];
            y.index_put_({Slice(), i}, y.index({Slice(), i}).contiguous() - 
                                       H_sub.index({Slice(), i, j}).contiguous() * 
                                       y.index({Slice(), j}).contiguous());
        }
        //y[i] /= H_sub[i][i];
        y.index_put_({Slice(), i}, y.index({Slice(), i}) / H_sub.index({i, i}));
    }

    return y;
}



// Function to compute the Givens rotation
std::tuple<torch::Tensor, torch::Tensor> givens_rotation(torch::Tensor& v1, torch::Tensor& v2) {
    auto r = (v1.square() + v2.square()).sqrt();
    auto cs_k = v1 / r;
    auto sn_k = v2 / r;
    return std::make_tuple(cs_k, sn_k);
}



std::tuple<torch::Tensor, torch::Tensor> arnoldi(const torch::Tensor& A, const torch::Tensor& Q, int k, double eps = 1e-12) {
    // Check if the input vector Q is non-zero

    int M = A.size(0);

    torch::Tensor h = torch::zeros({M,k + 1}, A.options());
    torch::Tensor q = torch::bmm(A, Q.index({Slice(), 1, k}));  // Krylov Vector

    for (int i = 0; i <= k; ++i) {  // Modified Gram-Schmidt, keeping the Hessenberg matrix
        h.index_put_({Slice(), i}, torch::einsum("mj,mj->m", {Q.index({Slice(), 1, i}), q}));
        q.index_put_({Slice()},  torch::einsum("", {q - h.index({Slice(),i}) ,Q.index({Slice(),1, i})}));
    }

    h.index_put_({Slice(), k + 1}, torch::norm(q,1,true));
    auto m1 = h.index({Slice(), k + 1}).abs() > eps;
    if (m1.any().item<bool>()) {
        //q = q / h[k + 1];
        q.index_put_({m1}, q.index({m1}) / h.index({m1, k + 1}));
    }
    auto m2 = ~m1;
    if ( m2.index({Slice()}).any().item<bool>()) {
        q.index_put_({m2}, 0.0);  // Avoid division by zero if norm is too small
    }

    return std::make_tuple(h, q);
}


// Function to apply Givens rotation to a column of H
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> apply_givens_rotation(torch::Tensor& h, const torch::Tensor& cs, const torch::Tensor& sn, int k) {
    // Apply Givens rotation for the ith column
    for (int i = 0; i < k - 1; ++i) {
        auto temp = cs.index({Slice(), i}) * h.index({Slice(), i}) + sn.index({Slice(), i}) * h.index({Slice(), i+1});
        h.index_put_({Slice(),i + 1},  -sn.index({Slice(), i}) * h.index({Slice(), i}) + cs.index({Slice(), i}) * h.index({Slice(), i+1}));
        h.index_put_({Slice(),}, temp);
    }

    // Update the next sin and cos values for rotation
    auto h_k = h.index({Slice(), k});
    auto h_k1 = h.index({Slice(), k + 1});
    auto [cs_k, sn_k] = givens_rotation(h_k, h_k1);

    // Eliminate H(i + 1, i)
    h.index_put_({Slice(), k}, cs_k * h.index({Slice(),k}) + sn_k * h.index({Slice(),k+1}));
    h.index_put_({Slice(), k} , 0.0);

    // Return updated values
    return std::make_tuple(h, cs_k, sn_k);
}







std::tuple<torch::Tensor, torch::Tensor> gmres(const torch::Tensor& A, const torch::Tensor &b, 
                                               const torch::Tensor &x, int maxiter, const torch::Tensor &tol)
{
    int M = A.size(0);
    int n = A.size(1);
    int m = maxiter;
    auto r = b - torch::einsum("mij, mj->mi", {A, x}); //Calculate the residual
    auto b_norm = torch::norm(b, 1, true);
    auto error = torch::norm(r, 1, true) / b_norm;
    auto e = error.clone();
    //Initialize 1D vectors   
    auto sn = torch::zeros({M, m, 1}, torch::dtype(torch::kFloat64)).to(A.device());
    auto cs = torch::zeros({M, m, 1}, torch::dtype(torch::kFloat64)).to(A.device());
    auto e1 = torch::zeros({M, m+1, 1}, torch::dtype(torch::kFloat64)).to(A.device());
    e1.index_put_({Slice(), 0}, 1.0);
    auto e = torch::zeros({M, m}, torch::dtype(torch::kFloat64)).to(A.device());
    auto r_norm = torch::norm(r, 1, true);
    auto Q = torch::zeros({M, n, m+1}, torch::dtype(torch::kFloat64)).to(A.device());
    auto H = torch::zeros({M, m+1, m}, torch::dtype(torch::kFloat64)).to(A.device());
    Q.index_put_({Slice(), Slice(), 0}, r / r_norm);
    auto beta = r_norm*e1;
    int k = 1;
    auto mask = error < tol;

    while ( mask.any().item<bool>() && k <=m)
    {
      auto [h, q] = arnoldi(A, Q, k);
      H.index_put_({mask, Slice(0,k+1), k-1}, h);
      Q.index_put_({mask, Slice(), k}, q);
      //eliminate the last element in H ith row and update the rotation matrix
      auto [h, cs_k, sn_k] = apply_givens_rotation(H, cs, sn, k);
      H.index_put_({mask, Slice(0,k), k-1}, h);
      cs.index_put_({mask, k-1}, cs_k);
      sn.index_put_({mask, k-1}, sn_k);
      beta.index_put_({mask, k}, -sn.index({Slice(), k-1})*beta.index({Slice(), k-1}));
      beta.index_put_({mask, k-1}, cs.index({Slice(), k-1})*beta.index({Slice(), k-1}));
      error = torch::abs(beta.index({mask, k}))/b_norm;
      e.index_put_({mask, k-1}, error);
      //update the mask
      mask = error < tol;
    }
    auto y = solve_hessenberg(H, beta, k);
    auto xres = x + torch::einsum("mij, mj->mi", {Q.index({Slice(), Slice(), Slice()}), y});
    return std::make_tuple(xres, e);
}



#endif