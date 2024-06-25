#ifndef QR_HPP_INCLUDED
#define QR_HPP_INCLUDED

#include <torch/torch.h>
#include "tensordual.hpp"
#include "janus_util.hpp"

namespace janus
{
   using Slice = torch::indexing::Slice;



    /**
     * Class generates QR decomposition of a Dual complex tensor across multiple samples in a vectorized form
    */
    class QRTeDC
    {
    public:
        int M, n, nd;
        bool isComplex = true;
        bool sing = false;
        TensorMatDual a, r, qt;
        TensorDual c, d;
        TensorDual scale, sm, tau, sigma, zero, one, Rkek, Qt_b, x;

        QRTeDC(const TensorMatDual &a)
        {
            M = a.r.size(0); //Batch dimension
            n = a.r.size(1);
            assert(a.r.size(2) == n && "Matrix is not square");
            //assert(dynamic_cast<TensorDual*>(a) != nullptr && "Object is not an instance of TensorDual");
            nd = a.d.size(3);//In this instance this is a 4D tensor [M,D,D,N] where M is the batch size and D 
            //is the dimension of the dual number and N is the dimension of the dual number

            this->a = a.clone(); // copy of the original matrix
            auto zeroc = torch::complex(torch::zeros({M, 1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())), 
                                  torch::zeros({M, 1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())));
     	    auto onec = torch::complex(torch::ones({M, 1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())), 
                                        torch::zeros({M, 1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())));



            zero = TensorDual(zeroc.clone(), zeroc.unsqueeze(-1).repeat({1,1,nd}));
            one = TensorDual(onec.clone(), zeroc.unsqueeze(-1).repeat({1,1,nd}));
            r = a.clone(); //Make a deep copy
            auto qt_real = torch::zeros({M, n, n}, torch::kDouble).to(a.device());
            auto qt_imag = torch::zeros({M, n, n}, torch::kDouble).to(a.device());
            auto qtc = torch::complex(qt_real, qt_imag).to(torch::kComplexDouble);
            qt = TensorMatDual(qtc, qtc.unsqueeze(-1).repeat({1,1,1,nd})); 
            auto c_real = torch::zeros({M, n}, torch::kDouble).to(a.device());
            auto c_imag = torch::zeros({M, n}, torch::kDouble).to(a.device());
            auto cc = torch::complex(c_real, c_imag).to(torch::kComplexDouble);
            c = TensorDual(cc.clone(), cc.unsqueeze(-1).repeat({1,1,nd}));
            
            auto d_real = torch::zeros({M, n}, torch::kDouble).to(a.device());
            auto d_imag = torch::zeros({M, n}, torch::kDouble).to(a.device());
            auto dc = torch::complex(d_real, d_imag).to(torch::kComplexDouble);

            d = TensorDual(dc.clone(), dc.unsqueeze(-1).repeat({1,1,nd}));
            for (int k = 0; k < n - 1; k++)
            {
                //scale = torch::amax(torch::abs(r.index({Slice(), Slice(k), k})), 1);
                //convert back to complex at the end

                scale = r.index({Slice(), Slice(k), Slice(k, k+1)}).squeeze(2).abs().max();
                //Make sure the scale is never zero
                if (scale.r.any().item<double>() == 0.0)
                {
                    std::cerr << "Singularity detected" << std::endl;
                    exit(1);
                }

                //At this point the scale is a real number.  We need to convert it to a complex number


                //r.index_put_({Slice(), Slice(k), k}, r.index({Slice(), Slice(k), k})*scale.reciprocal());
                for ( int i=k; i < n; i++)
                {
                    auto ros = r.index({Slice(), Slice(i, i+1), Slice(k, k+1)}) / scale;
                    r.index_put_({Slice(), Slice(i,i+1), Slice(k,k+1)}, ros); 
                }
                //for (sum=0.0,i=k;i<n;i++) sum += SQR(r[i][k]);
                //sum = torch::sum(r.index({Slice(), Slice(k), k}).square(), 1);
                auto sum = r.index({Slice(), Slice(k), Slice(k, k+1)}).square().sum(1).squeeze();
  
                //sigma=SIGN(sqrt(sum),r[k][k]);
                //sigma is a real number NOT COMPLEX of dimension M
                sigma = signcond(sum.sqrt(), r.index({Slice(), Slice(k, k+1), Slice(k, k+1)}).squeeze());
                r.index_put_({Slice(), Slice(k,k+1), Slice(k,k+1)}, r.index({Slice(), Slice(k, k+1), Slice(k, k+1)}) + sigma);
                auto sigmar =  sigma * r.index({Slice(), Slice(k, k+1), Slice(k, k+1)});
                //This is a bug in pytorch.  index_put_ should not reduce the slice dimensions
                c.index_put_({Slice(), Slice(k, k+1)}, sigmar);
                auto scales = -scale * sigma;
                d.index_put_({Slice(), Slice(k, k+1)}, scales);
                //sigma=SIGN(sqrt(sum),r[k][k]);
                //sigma = signcondc(sum_sq.sqrt(), r.index({Slice(), k, k}));
                //r.index_put_({Slice(), k, k}, r.index({Slice(), k, k}) + sigma);
                //c.index_put_({Slice(), k}, sigma * r.index({Slice(), k, k}));
                //d[k] = -scale*sigma;
                //d.index_put_({Slice(), k}, -scale * sigma);


                for (int j = k + 1; j < n; j++)
                {
                    sum = (r.index({Slice(), Slice(k), Slice(k, k+1)}).squeeze() * r.index({Slice(), Slice(k), Slice(j, j+1)}).squeeze()).sum();
                    //This has the dimensions of batch size M
                    //auto tau = sum / c.index({Slice(), k});
                    //r.index_put_({Slice(), Slice(k), j}, r.index({Slice(), Slice(k), j}) - torch::einsum("m, mi->mi",{tau , r.index({Slice(), Slice(k), k})}));

                    auto tau = sum / c.index({Slice(), Slice(k, k+1)});
                    auto tauTr = TensorMatDual::einsum("mi, mji->mji",tau , r.index({Slice(), Slice(k), Slice(k, k+1)}));
                    r.index_put_({Slice(), Slice(k), Slice(j,j+1)}, r.index({Slice(), Slice(k), Slice(j, j+1)}) - tauTr);
                }

            } // for k



            d.index_put_({Slice(), Slice(n-1, n)}, r.index({Slice(), Slice(n-1, n), Slice(n-1, n)}).squeeze());
            // Recombine the updated real and imaginary part
            for (int i=0;i<n;i++) 
            {
                //Here one is a dual number
                qt.index_put_({Slice(), Slice(i,i+1), Slice(i,i+1)}, one.unsqueeze(2));
	        }


            for ( int k=0; k < n-1; k++)
            {
                for ( int j=0; j < n; j++)
                {
                    auto sum = (r.index({Slice(), Slice(k), Slice(k, k+1)}).squeeze(2)* qt.index({Slice(), Slice(k), Slice(j, j+1)}).squeeze(2)).sum();
                    sum.index_put_( Slice(), sum * c.index({Slice(), Slice(k, k+1)}).reciprocal());
                    //for ( int i=k; i < n; i++)
                    //{
                    //    qt.index_put_({Slice(), i, j}, qt.index({Slice(), i, j}) - sum * r.index({Slice(), i, k}));
                    //}
                    auto sumTqt =sum*r.index({Slice(), Slice(k), Slice(k, k+1)});
                    qt.index_put_({Slice(), Slice(k), Slice(j,j+1)}, qt.index({Slice(), Slice(k), Slice(j, j+1)}) -sumTqt.unsqueeze(2));
   
                }
            }
            for ( int i=0; i < n; i++ )
            {
              r.index_put_({Slice(), Slice(i,i+1), Slice(i,i+1)}, TensorMatDual::unsqueeze(d.index({Slice(), Slice(i, i+1)}), 1));
              r.index_put_({Slice(), Slice(i,i+1), Slice(0,i)}, zero.unsqueeze(2));//Zero out the lower triangular part
            }


        } // QR constructor

        /*
        Use the class stored Q^T and R matrices to solve for the linear system of equations
        */
        TensorDual solvev(TensorDual &bin)
        {
            // qtx = self.qt*b
            //In this formulation the Q matrix is already transposed
            Qt_b = TensorMatDual::einsum("mij,mj->mi", qt, bin.complex());
            //We need to solve for R*x = QT*b backwards
            x = TensorDual::zeros_like(Qt_b);
            for (int i=n-1;i>-1; i--) {
              x.index_put_({Slice(), Slice(i,i+1)}, Qt_b.index({Slice(), Slice(i,i+1)}));
              auto sm = (r.index({Slice(), Slice(i, i+1), Slice(i+1, n)}) * x.index({Slice(), Slice(i+1, n)})).sum();
              x.index_put_({Slice(), Slice(i, i+1)}, x.index({Slice(), Slice(i, i+1)})-sm);
              //If x[i] is zero, set the value to zero
              if ((x.index({Slice(), Slice(i, i+1)}) == 0.0).all().item<bool>()) {
                  x.index_put_({Slice(), Slice(i, i+1)}, 0.0);
              } else {
                  x.index_put_({Slice(), Slice(i, i+1)}, x.index({Slice(), Slice(i, i+1)}) / r.index({Slice(), Slice(i, i+1), Slice(i, i+1)}).squeeze());
              }
            }
            return x.real(); //Remove the complex part
        }

        /*
        Use the user provided Q^T and R matrices to solve for the linear system of equations
        */
        static TensorDual solvev(const TensorMatDual& qt, const TensorMatDual &r,  TensorDual &bin)
        {
            // qtx = self.qt*b
            //In this formulation the Q matrix is already transposed
            auto Qt_b = TensorMatDual::einsum("mij,mj->mi", qt, bin.complex());
            //We need to solve for R*x = QT*b backwards
            auto x = TensorDual::zeros_like(Qt_b);
            int n = bin.r.size(1);
            auto rc = r.clone();
            for (int i=n-1;i>-1; i--) {
              x.index_put_({Slice(), Slice(i,i+1)}, Qt_b.index({Slice(), Slice(i,i+1)}));
              auto sm = TensorMatDual::einsum("mij, mj->mi",rc.index({Slice(), Slice(i, i+1), Slice(i+1, n)}) , x.index({Slice(), Slice(i+1, n)})).sum();
              x.index_put_({Slice(), Slice(i, i+1)}, x.index({Slice(), Slice(i, i+1)})-sm);
              //If x[i] is zero, set the value to zero
              if ((x.index({Slice(), Slice(i, i+1)}) == 0.0).all().item<bool>()) {
                  x.index_put_({Slice(), Slice(i, i+1)}, 0.0);
              } else {
                //x.index_put_({Slice(), i}, x.index({Slice(), i}) / r.index({Slice(), i, i}));
                  TensorDual rcatir = rc.index({Slice(), Slice(i, i+1), Slice(i, i+1)}).squeeze(2);
                  x.index_put_({Slice(), Slice(i, i+1)}, x.index({Slice(), Slice(i, i+1)})/rcatir);
              }
            }
            return x; 

        }


    }; // class QR
};     // namespace janust
#endif // QR_HPP_INCLUDED