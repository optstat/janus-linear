#ifndef QRTE_HPP_INCLUDED
#define QRTE_HPP_INCLUDED

#include <torch/torch.h>
#include <janus/janus_util.hpp>

namespace janus
{
   using Slice = torch::indexing::Slice;



    /**
     * Class generates QR decomposition of a matrix across multiple samples in a vectorized form
    */
    class QRTe
    {
    public:
        int M, n;
        bool isComplex = false;
        bool sing = false;
        torch::Tensor a, r, qt, c, d;
        torch::Tensor scale, sm, tau, sigma, zero, Rkek, Qt_b, x;
        torch::Tensor qt_real, qt_imag, c_real, c_imag, d_real, d_imag;

        QRTe(const torch::Tensor &a)
        {
            M = a.size(0); //Batch dimension
            n = a.size(1);

            //check to see if a is not double precision
            if (a.dtype() != torch::kDouble && a.dtype() != torch::kComplexDouble)
            {
                std::cerr << ("Input into QRTe is not double precision");
                exit(1);
            }

            if ( a.isreal().all().item<bool>() ) {
                if (a.isnan().any().item<bool>() || a.isinf().any().item<bool>())
                {
                    throw std::runtime_error("a has nan or inf");
                }
            }
            if ( a.is_complex() ) {
                if (at::real(a).isnan().any().item<bool>() || at::real(a).isinf().any().item<bool>()
                    || at::imag(a).isnan().any().item<bool>() || at::imag(a).isinf().any().item<bool>())
                {
                    throw std::runtime_error("a has nan or inf");
                }
            }

            this->a = a.clone(); // copy of the original matrix
            // check to see if a is complex
            if (a.is_complex())
            {
                isComplex = true;
                zero = torch::complex(torch::zeros({1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())), 
                                      torch::zeros({1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device())));
                
            }
            else {
                zero = torch::zeros({1}, torch::TensorOptions().dtype(torch::kDouble).device(a.device()));
            }
            assert(a.dim() == 3);
            assert(a.size(1) == a.size(2));
            r = a.clone(); //Make a deep copy
            //check for nan or inf in a
            if (!isComplex)
            {
                qt = torch::zeros({M, n, n}, torch::TensorOptions().dtype(torch::kDouble).device(a.device()));
                c = torch::zeros({M, n}, torch::TensorOptions().dtype(torch::kDouble).device(a.device()));
                d = torch::zeros({M, n}, torch::TensorOptions().dtype(torch::kDouble).device(a.device()));
            }
            else
            {
                qt_real = torch::zeros({M, n, n}, torch::kDouble).to(a.device());
                qt_imag = torch::zeros({M, n, n}, torch::kDouble).to(a.device());
                qt = torch::complex(qt_real, qt_imag).to(torch::kComplexDouble);
                c_real = torch::zeros({M, n}, torch::kDouble).to(a.device());
                c_imag = torch::zeros({M, n}, torch::kDouble).to(a.device());
                c = torch::complex(c_real, c_imag).to(torch::kComplexDouble);
                d_real = torch::zeros({M, n}, torch::kDouble).to(a.device());
                d_imag = torch::zeros({M, n}, torch::kDouble).to(a.device());
                d = torch::complex(d_real, d_imag).to(torch::kComplexDouble);
            }
            for (int k = 0; k <= n - 2; k++)
            {
                Rkek = r.index({Slice(), Slice(k), k});
                //scale = max(abs(Rkek));
                auto [scale, scaleindices]=Rkek.abs().max(1, true);
                #ifdef DEBUG
                std::cerr << "At k=" << k << " scale=" << scale << std::endl;
                #endif
                r.index_put_({Slice(), Slice(k), k}, r.index({Slice(), Slice(k), k})/ scale);
                //for (sum=0.0,i=k;i<n;i++) sum += SQR(r[i][k]);
                auto rsq = r.index({Slice(), Slice(k), k}).square();
                auto smsqrt= torch::sum(rsq,1).sqrt();
                //sigma=SIGN(sqrt(sum),r[k][k]);
                //r = flip_epsilon(r);
                //The signcond statement is sensitive to numbers that are negative 
                //even if they are very small so zero them out before calling signcond
                sigma = signcond(smsqrt, r.index({Slice(), k, k}));
                #ifdef DEBUG
                std::cerr << "smsqrt=";
                janus::print_tensor(smsqrt);
                std::cerr << "r.index({Slice(), k, k})=";
                janus::print_tensor(r.index({Slice(), k, k}));
                std::cerr << "sigma=";
                janus::print_tensor(sigma);
                #endif
                //Avoid potential memory conflict
                r.index_put_({Slice(), k, k}, r.index({Slice(), k, k}).clone() + sigma);
                c.index_put_({Slice(), k}, sigma * r.index({Slice(), k, k}));
                //d[k] = -scale*sigma;
                #ifdef DEBUG
                //std::cerr << "sigma=";
                //janus::print_tensor(sigma);
                #endif
                d.index_put_({Slice(), k}, -scale.squeeze(1)*sigma);
                #ifdef DEBUG
                std::cerr <<"At the end of outer loop k=" << k << std::endl;
                std::cerr << "d=";
                janus::print_tensor(d);
                std::cerr << "c=";
                janus::print_tensor(c);
                std::cerr << "r=";
                janus::print_tensor(r);
                #endif

                for (int j = k + 1; j <= n-1; j++)
                {
                    sm = torch::sum(r.index({Slice(), Slice(k), k}) * r.index({Slice(), Slice(k), j}), 1);
                    #ifdef DEBUG
                    std::cerr << "For k=" << k << " j=" << j << " sm=";
                    std::cerr << "sm=";
                    janus::print_tensor(sm);
                    std::cerr << "r=";
                    janus::print_tensor(r);
                    #endif
                    tau = sm / c.index({Slice(), k});
                    #ifdef DEBUG
                    std::cerr << "tau=";
                    janus::print_tensor(tau);
                    #endif
                    if (torch::isnan(tau).any().item<bool>() || torch::isinf(tau).any().item<bool>()) {
                        std::cerr << "tau=" << tau << std::endl;
                        throw std::runtime_error("tau has nan or inf");
                    }
                    r.index_put_({Slice(), Slice(k), j}, r.index({Slice(), Slice(k), j}) - tau.unsqueeze(1) * r.index({Slice(), Slice(k), k}));
                    #ifdef DEBUG
                    //std::cerr << "r at k=" << k << " j=" << j;
                    //print_tensor(r);
                    #endif
                }
            } // for k
            #ifdef DEBUG
              std::cerr << "After first k loop " << std::endl;
              std::cerr << "r=";
              janus::print_tensor(r);
              std::cerr << "c=";
              janus::print_tensor(c);
              std::cerr << "d=";
              janus::print_tensor(d);
              exit(1);
            #endif


            if (!isComplex)
            {
                d.index_put_({Slice(), n - 1} ,  r.index({Slice(), n - 1, n - 1}));
            }
            else
            {
                // Extract real and imaginary parts of r
                auto r_real = at::real(r);
                auto r_imag = at::imag(r);

                // Extract real and imaginary parts of d
                auto d_real = at::real(d);
                auto d_imag = at::imag(d);
                // Update the real and imaginary parts separately
                d_real.index_put_({Slice(), n - 1}, r_real.index({Slice(), n - 1, n - 1}));
                d_imag.index_put_({Slice(), n - 1}, r_imag.index({Slice(), n - 1, n - 1}));

                // Recombine the updated real and imaginary parts
                d = at::complex(d_real, d_imag);

            }

            if (!isComplex)
            {
                //set the diagonal of qt to 1 for all batches
                qt = torch::eye(n, torch::kDouble).repeat({M, 1, 1}).to(a.device()).view({M, n, n});
            }
            else
            {
                qt_real = torch::eye(n, torch::kDouble).repeat({M, 1, 1}).to(a.device()).view({M, n, n});
                qt_imag = torch::zeros({n, n}, torch::kDouble).repeat({M, 1, 1}).to(a.device()).view({M, n, n});
                qt = torch::complex(qt_real, qt_imag);
            }
            for (int k = 0; k <= n - 2; k++)
            {
                torch::Tensor catk;
                if ( isComplex) 
                   catk = at::real(c.index({Slice(), k}));
                else 
                   catk = c.index({Slice(), k});
                auto mask = (catk != 0.0);
                if (mask.any().item<bool>())
                {
                    for (int j = 0; j <= n-1; j++)
                    {
                      auto sm = torch::sum(r.index({mask, Slice(k), k}) * qt.index({mask, Slice(k), j}), 1, true); 
                    
                      sm = sm / (c.index({mask, k}).unsqueeze(1));
                      qt.index_put_({mask, Slice(k), j}, qt.index({mask, Slice(k), j}) - 
                                                            sm * r.index({mask, Slice(k), k}));
                    }
                }
            }
            
            //D = diag(obj.d);
            auto D = torch::diag_embed(d);
            #ifdef DEBUG
            //std::cerr << "D=";
            //janus::print_tensor(D);
            //std::cerr << "d=";
            //janus::print_tensor(d);
            //std::cerr << "r=";
            //janus::print_tensor(r);
            #endif
            //obj.r = obj.r - diag(diag(obj.r)) + D;
            auto single_mask = 1 - torch::eye(r.size(1)).to(r.options());
            auto mask = single_mask.expand_as(r);
            r = r*mask + D;        

            if ( !isComplex ) {
                //check for nan or inf in qt
                if (qt.isnan().any().item<bool>() || qt.isinf().any().item<bool>())
                {
                    std::cerr << "qt=" << qt << std::endl;
                    throw std::runtime_error("qt has nan or inf");
                }
                //check for nan or inf in r
                if (r.isnan().any().item<bool>() || r.isinf().any().item<bool>())
                {
                    std::cerr << "r=" << r << std::endl;
                    throw std::runtime_error("r has nan or inf");
                }

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

    std::tuple<torch::Tensor, torch::Tensor> qrte(const torch::Tensor &a)
    {
        QRTe qr{a};
        return std::make_tuple(qr.qt, qr.r);
    }

    torch::Tensor qrtesolvev(const torch::Tensor &qt, const torch::Tensor &r, const torch::Tensor &bin)
    {
        return QRTe::solvev(qt, r, bin);
    }
};     // namespace janust
#endif // QR_HPP_INCLUDED