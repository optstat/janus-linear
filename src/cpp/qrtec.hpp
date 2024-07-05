#ifndef QRTEC_HPP_INCLUDED
#define QRTEC_HPP_INCLUDED

#include <torch/torch.h>
#include "janus_util.hpp"

namespace janus
{
   using Slice = torch::indexing::Slice;



    /**
     * Class generates QR decomposition of a complex matrix across multiple samples in a vectorized form
    */
    class QRTeC
    {
    public:
        int M, n;
        bool isComplex = true;
        bool sing = false;
        torch::Tensor a, r, qt, c, d;
        torch::Tensor scale, sm, tau, sigma, zero, one, Rkek, Qt_b, x;
        torch::Tensor qt_real, qt_imag, c_real, c_imag, d_real, d_imag;

        QRTeC(const torch::Tensor &a)
        {
            M = a.size(0); //Batch dimension
            n = a.size(1);



            this->a = a.clone(); // copy of the original matrix
            zero = torch::complex(torch::zeros({1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())), 
                                  torch::zeros({1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())));
     	    one = torch::complex(torch::ones({1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())), 
                                        torch::zeros({1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())));
           
            r = a.clone(); //Make a deep copy
            qt_real = torch::zeros({M, n, n}, torch::kDouble).to(a.device());
            qt_imag = torch::zeros({M, n, n}, torch::kDouble).to(a.device());
            qt = torch::complex(qt_real, qt_imag).to(torch::kComplexDouble);
            c_real = torch::zeros({M, n}, torch::kDouble).to(a.device());
            c_imag = torch::zeros({M, n}, torch::kDouble).to(a.device());
            c = torch::complex(c_real, c_imag).to(torch::kComplexDouble);
            d_real = torch::zeros({M, n}, torch::kDouble).to(a.device());
            d_imag = torch::zeros({M, n}, torch::kDouble).to(a.device());
            d = torch::complex(d_real, d_imag).to(torch::kComplexDouble);
            for (int k = 0; k < n - 1; k++)
            {
                scale = torch::amax(torch::abs(r.index({Slice(), Slice(k), k})), 1);

                //r.index_put_({Slice(), Slice(k), k}, r.index({Slice(), Slice(k), k})*scale.reciprocal());
                for ( int i=k; i < n; i++)
                {
                    r.index_put_({Slice(), i, k}, r.index({Slice(), i, k})/scale);
                }
                //for (sum=0.0,i=k;i<n;i++) sum += SQR(r[i][k]);
                auto sum = torch::zeros({M}, torch::kComplexDouble).to(a.device());
                //auto r_square = r.index({Slice(), Slice(k), k}).square();
                //auto sum_sq = torch::einsum("mi->m", {r_square}); 

                sum = torch::sum(r.index({Slice(), Slice(k), k}).square(), 1);
  
                //sigma=SIGN(sqrt(sum),r[k][k]);
                //sigma is a real number NOT COMPLEX of dimension M
                auto temp = torch::sqrt(sum);
                sigma = signcond(temp, r.index({Slice(), k, k}));
     
                r.index_put_({Slice(), k, k}, r.index({Slice(), k, k}) + sigma);
                c.index_put_({Slice(), k}, sigma * r.index({Slice(), k, k}));
                d.index_put_({Slice(), k}, -scale * sigma);
                //sigma=SIGN(sqrt(sum),r[k][k]);
                //sigma = signcondc(sum_sq.sqrt(), r.index({Slice(), k, k}));
                //r.index_put_({Slice(), k, k}, r.index({Slice(), k, k}) + sigma);
                //c.index_put_({Slice(), k}, sigma * r.index({Slice(), k, k}));
                //d[k] = -scale*sigma;
                //d.index_put_({Slice(), k}, -scale * sigma);


                for (int j = k + 1; j < n; j++)
                {
                    sum = torch::sum(r.index({Slice(), Slice(k), k}) * r.index({Slice(), Slice(k), j}), 1);
                    //This has the dimensions of batch size M
                    auto tau = sum / c.index({Slice(), k});
                    r.index_put_({Slice(), Slice(k), j}, r.index({Slice(), Slice(k), j}) - torch::einsum("m, mi->mi",{tau , r.index({Slice(), Slice(k), k})}));
                }

            } // for k



            d.index_put_({Slice(), n-1}, r.index({Slice(), n-1, n-1}));
            // Recombine the updated real and imaginary part
            for (int i=0;i<n;i++) 
            {
                qt.index_put_({Slice(), i, i}, one);
	        }


            for ( int k=0; k < n-1; k++)
            {
                for ( int j=0; j < n; j++)
                {
                    auto sum = torch::zeros({M}, torch::kComplexDouble).to(a.device());
                    sum = torch::sum(r.index({Slice(), Slice(k), k}) * qt.index({Slice(), Slice(k), j}), 1);
                    sum.index_put_( {Slice()}, sum * c.index({Slice(), k}).reciprocal());
                    //for ( int i=k; i < n; i++)
                    //{
                    //    qt.index_put_({Slice(), i, j}, qt.index({Slice(), i, j}) - sum * r.index({Slice(), i, k}));
                    //}
                    qt.index_put_({Slice(), Slice(k), j}, qt.index({Slice(), Slice(k), j}) -torch::einsum("m,mi->mi", {sum,r.index({Slice(), Slice(k), k})}));
   
                }
            }
            for ( int i=0; i < n; i++ )
            {
                r.index_put_({Slice(), i, i}, d.index({Slice(), i}));
                r.index_put_({Slice(), i, Slice(0, i)}, zero);//Zero out the lower triangular part
            }


        } // QR constructor

        /*
        Use the class stored Q^T and R matrices to solve for the linear system of equations
        */
        torch::Tensor solvev(torch::Tensor &bin)
        {
            // qtx = self.qt*b
            //In this formulation the Q matrix is already transposed
            Qt_b = torch::einsum("mij,mj->mi", {qt, bin});
            //We need to solve for R*x = QT*b backwards
            x = torch::zeros_like(Qt_b);
            for (int i=n-1;i>-1; i--) {
              x.index_put_({Slice(), i}, Qt_b.index({Slice(), i}));
              auto sm = torch::sum(r.index({Slice(), i, Slice(i+1, n)}) * x.index({Slice(), Slice(i+1, n)}), 1);
              x.index_put_({Slice(), i}, x.index({Slice(), i})-sm);
              //If x[i] is zero, set the value to zero
              if ((x.index({Slice(), i}) == 0.0).all().item<bool>()) {
                  x.index_put_({Slice(), i}, 0.0);
              } else {
                  x.index_put_({Slice(), i}, x.index({Slice(), i}) / r.index({Slice(), i, i}));
              }
            }
            return x;
        }

        /*
        Use the user provided Q^T and R matrices to solve for the linear system of equations
        */
        static torch::Tensor solvev(const torch::Tensor& qt, const torch::Tensor &r,  const torch::Tensor &bin)
        {
            // qtx = self.qt*b
            //In this formulation the Q matrix is already transposed
            auto Qt_b = torch::einsum("mij,mj->mi", {qt, bin});
            //We need to solve for R*x = QT*b backwards
            auto x = torch::zeros_like(Qt_b);
            int n = r.size(1);
            int M = r.size(0);
            for (int i=n-1;i>-1; i--) {
              x.index_put_({Slice(), i}, Qt_b.index({Slice(), i}));
              auto sm = torch::sum(r.index({Slice(), i, Slice(i+1, n)}) * x.index({Slice(), Slice(i+1, n)}), 1);
              x.index_put_( {Slice(), i}, x.index({Slice(), i})-sm);    
              if ((x.index({Slice(), i}) == 0.0).all().item<bool>()) {
                  x.index_put_({Slice(), i}, 0.0);
              } else {
                  x.index_put_({Slice(), i}, x.index({Slice(), i}) / r.index({Slice(), i, i}));
              }
            }
            return x;
        }


    }; // class QR
};     // namespace janust
#endif // QR_HPP_INCLUDED