#ifndef LU_HPP_INCLUDED
#define LU_HPP_INCLUDED
#include <torch/torch.h>
#include <iostream>
namespace janus
{

  using namespace torch::indexing;
  using Slice = torch::indexing::Slice;

  std::tuple<torch::Tensor, torch::Tensor> LUTe(torch::Tensor &a)
  {
    torch::Tensor lu = a.clone();
    int M = a.size(0);
    int N = a.size(1);
    assert(a.size(1) == a.size(2)); // Square matrix
    torch::Tensor tiny = torch::ones({1}, torch::dtype(torch::kFloat64)).to(a.device()) * 1.0e-40;
    torch::Tensor vv = torch::zeros({M, N}, torch::dtype(torch::kFloat64)).to(a.device());
    torch::Tensor d = torch::ones({M}, torch::dtype(torch::kFloat64)).to(a.device());
    auto true_t = torch::tensor({true}, torch::dtype(torch::kBool)).to(a.device());
    torch::Tensor imax = torch::zeros({M}, torch::dtype(torch::kInt64)).to(a.device());
    auto big = torch::zeros({M}, torch::dtype(torch::kFloat64)).to(a.device());
    auto indx = torch::zeros({M, N}, torch::dtype(torch::kInt64)).to(a.device());
    /**
     * 	for (i=0;i<n;i++) {
    big=0.0;
    for (j=0;j<n;j++)
      if ((temp=abs(lu[i][j])) > big) big=temp;
    if (big == 0.0) throw("Singular matrix in LUdcmp");
    vv[i]=1.0/big;
  }
    */
    // Loop across all rows
    for (int i = 0; i < N; i++)
    {
      // Find the maximum value of absolute values in the row
      auto maxres = torch::max(lu.index({Slice(), i, Slice()}).abs(), 1);
      auto temp = std::get<0>(maxres);
      big.index_put_({Slice()}, temp);
      vv.index_put_({Slice(), i}, big.index({Slice()}).reciprocal());
      // Loop over the columns of the matrix
    }
    //std::cerr << "vv=" << vv << std::endl;

    for (int k = 0; k < N; k++)
    {
      imax.index_put_({Slice()}, k);
      auto temp = torch::einsum("mi, mi->mi", {vv.index({Slice(), Slice(k)}),
                                               lu.index({Slice(), Slice(k), k}).abs()});
      auto maxres = torch::max(temp, 1);
      auto maxvals = std::get<0>(maxres);
      imax = std::get<1>(maxres) + k; // Need to add k to get the absolute index

      //std::cerr << "At k=" << k << " imax= " << imax << std::endl;
      auto m2 = (k != imax);
      if (m2.eq(true_t).any().item<bool>())
      {
        // Swap the rows
        auto nnz = m2.nonzero();
        for (int j = 0; j < nnz.size(0); j++)
        {
          auto sample = nnz.index({j});
          auto row = imax.index({sample});
          auto temp = lu.index({sample, row, Slice()}).clone();
          lu.index_put_({sample, row, Slice()}, lu.index({sample, k, Slice()}));
          lu.index_put_({sample, k, Slice()}, temp);
          vv.index_put_({sample, row}, vv.index({sample, k}));
        }

        d.index_put_({m2}, -d.index({m2}));
      }
      //std::cerr << "After row swap " << "k=" << k << " lu=" << lu << std::endl;
      //std::cerr << "After row swap " << "k=" << k << " d=" << d << std::endl;
      //std::cerr << "After row swap " << "k=" << k << " vv=" << vv << std::endl;
      //		indx[k]=imax;
      indx.index_put_({Slice(), k}, imax);
      auto m3 = lu.index({Slice(), k, k}) == 0.0;
      if (m3.eq(true_t).any().item<bool>())
      {
        lu.index_put_({m3, k, k}, tiny);
      }
      /*
          for (i=k+1;i<n;i++) {
         temp=lu[i][k] /= lu[k][k];
      for (j=k+1;j<n;j++)
        lu[i][j] -= temp*lu[k][j];
    }*/
      for (int i = k + 1; i < N; i++)
      {
        auto lures = lu.index({Slice(), i, k})/ lu.index({Slice(), k, k});
        lu.index_put_({Slice(), i, k}, lures);
        auto templu = torch::einsum("m, mi->mi", {lu.index({Slice(), i, k}), 
                                                  lu.index({Slice(), k, Slice(k + 1)})});
        auto templu2 = lu.index({Slice(), i, Slice(k + 1)}) - templu;
        lu.index_put_({Slice(), i, Slice(k + 1)}, templu2);
      }
      //std::cerr << "After row reduction " << "k=" << k << " lu=" << lu << std::endl;
    }

    return std::make_tuple(lu, indx);
  }

  static torch::Tensor solveluv(const torch::Tensor &lu, const torch::Tensor &indx, const torch::Tensor &b)
  {
    torch::Tensor ii, ip, j;
    int M = lu.size(0);
    int n = lu.size(1);
    assert(b.size(0) == M);
    torch::Tensor x = b.clone();
    auto sum = torch::zeros({M}, torch::dtype(torch::kFloat64)).to(b.device());
    ii = torch::zeros({M}, torch::dtype(torch::kInt64)).to(b.device());
    torch::Tensor indices = torch::arange(M, torch::kLong);
    for (int i = 0; i < n; i++)
    {
      ip = indx.index({Slice(), i});
      sum.index_put_({indices}, x.index({indices, ip}));
      x.index_put_({indices, ip}, x.index({indices, i}).clone());
      auto mii = (ii != 0);
      if (mii.eq(true).any().item<bool>())
      {
        auto temp = torch::einsum("mi,mi->m", {lu.index({Slice(), i, Slice(0, i)}),
                                               x.index({Slice(), Slice(0, i)})});
        sum = sum - temp;
      }
      auto mii2 = (ii == 0) & (sum != 0);
      if (mii2.eq(true).any().item<bool>())
      {
        ii.index_put_({mii2}, i + 1);
      }
      x.index_put_({Slice(), i}, sum);
      // x[i]=sum;
    }
    for (int i = n - 1; i >= 0; i--)
    {
      sum = x.index({Slice(), i});
      auto temp = torch::einsum("mi,mi->m", {lu.index({Slice(), i, Slice(i + 1)}),
                                             x.index({Slice(), Slice(i + 1)})});
      sum = sum - temp;
      x.index_put_({Slice(), i}, sum / lu.index({Slice(), i, i}));
      // x[i]=sum/lu[i][i];
    }
    return x;
  }

} // namespace janus
#endif // LU_HPP_INCLUDED