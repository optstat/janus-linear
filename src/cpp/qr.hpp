#ifndef QR_HPP_INCLUDED
#define QR_HPP_INCLUDED
// def signcond(a, b):
//     return torch.where(b >= 0, torch.where(a >= 0, a, -a), torch.where(a >= 0, -a, a))

#include <torch/torch.h>
#include <janus/janus_util.hpp>


using Slice = torch::indexing::Slice;

namespace janus
{



    class QR
    {
    public:
        int n, m;
        bool isComplex = false;
        bool sing = false;
        torch::Tensor a, r, qt, c, d;
        torch::Tensor scale, sm, tau, sigma, zero, Rkek, Qt_b, x;
        torch::Tensor qt_real, qt_imag, c_real, c_imag, d_real, d_imag;

        QR(const torch::Tensor &a)
        {
            if ( a.isreal().all().item<bool>() ) {
                if (a.isnan().any().item<bool>() || a.isinf().any().item<bool>())
                {
                    std::cerr << "a=" << a << std::endl;
                    throw std::runtime_error("a has nan or inf");
                }
            }
            if ( a.is_complex() ) {
                if (at::real(a).isnan().any().item<bool>() || at::real(a).isinf().any().item<bool>()
                    || at::imag(a).isnan().any().item<bool>() || at::imag(a).isinf().any().item<bool>())
                {
                    std::cerr << "a=" << a << std::endl;
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
            n = a.size(0);
            assert(a.size(0) == a.size(1));
            r = a.clone();
            std::cerr << "r=";
            print_matrix(r);
            //Print the dimensions of r
            std::cerr << "r.size(0)=" << r.size(0) << std::endl;
            std::cerr << "r.size(1)=" << r.size(1) << std::endl;
            //check for nan or inf in a
            if (!isComplex)
            {
                qt = torch::zeros({n, n}, torch::TensorOptions().dtype(a.dtype()).device(a.device()));
                c = torch::zeros({n}, torch::TensorOptions().dtype(a.dtype()).device(a.device()));
                d = torch::zeros({n}, torch::TensorOptions().dtype(a.dtype()).device(a.device()));
            }
            else
            {
                qt_real = torch::zeros({n, n}, torch::kDouble).to(a.device());
                qt_imag = torch::zeros({n, n}, torch::kDouble).to(a.device());
                qt = torch::complex(qt_real, qt_imag).to(torch::kComplexDouble);
                c_real = torch::zeros({n}, torch::kDouble).to(a.device());
                c_imag = torch::zeros({n}, torch::kDouble).to(a.device());
                c = torch::complex(c_real, c_imag).to(torch::kComplexDouble);
                d_real = torch::zeros({n}, torch::kDouble).to(a.device());
                d_imag = torch::zeros({n}, torch::kDouble).to(a.device());
                d = torch::complex(d_real, d_imag).to(torch::kComplexDouble);
            }
            for (int k = 0; k < n - 1; k++)
            {
                scale = zero;
                Rkek = r.index({Slice(k), k});
                std::cerr << "Rkek=";
                print_vector(Rkek);
                //scale = max(abs(Rkek));
                auto res = torch::max(Rkek.abs(), 0);
                scale = std::get<0>(res);
                std::cerr << "scale=" << scale << std::endl;
                
                auto scaled_r = r.index({Slice(k), k}) / scale;
                r.index_put_({Slice(k), k}, scaled_r);
                //for (sum=0.0,i=k;i<n;i++) sum += SQR(r[i][k]);
                sm = torch::sum(r.index({Slice(k), k}).square());
                //sigma=SIGN(sqrt(sum),r[k][k]);
                torch::Tensor sqrtsm = torch::sqrt(sm);
                torch::Tensor rkk = r.index({k, k});
                sigma = janus::signcond(sqrtsm, rkk);
                std::cerr << "sigma=" << sigma << std::endl;
                std::cerr << "r before index put=";
                print_matrix(r);
                r.index_put_({k, k}, r.index({k, k}) + sigma);
                std::cerr << "r after index put=";
                print_matrix(r);
                c.index_put_({k}, sigma * r.index({k, k}));
                assert ((c.index({k}).abs() > 0).item<bool>());
                //d[k] = -scale*sigma;
                d.index_put_({k}, -scale * sigma);
                std::cerr << "c=";
                print_vector(c);
                std::cerr << "d=";
                print_vector(d);

                for (int j = k + 1; j < n; j++)
                {
                    sm = torch::sum(r.index({Slice(k), k}) * r.index({Slice(k), j}));
                    tau = sm / c.index({k});
                    std::cerr << "c at k=" << k << " =" << c.index({k}) << std::endl;
                    std::cerr << "tau at k=" << k << " =" << tau << std::endl;
                    if (torch::isnan(tau).any().item<bool>() || torch::isinf(tau).any().item<bool>()) {
                        std::cerr << "tau=" << tau << std::endl;
                        throw std::runtime_error("tau has nan or inf");
                    }
                    auto ratj = r.index({Slice(k), j});
                    auto ratk = r.index({Slice(k), k});
                    r.index_put_({Slice(k), j}, ratj - tau * ratk);
                }
                std::cerr << "r=";
                print_matrix(r);
            } // for k

            if (!isComplex)
            {
                d[n - 1] = r[n - 1][n - 1];
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
                d_real.index_put_({n - 1}, r_real.index({n - 1, n - 1}));
                d_imag.index_put_({n - 1}, r_imag.index({n - 1, n - 1}));

                // Recombine the updated real and imaginary parts
                d = at::complex(d_real, d_imag);

            }
            std::cerr << "d=";
            print_vector(d);

            if (!isComplex)
            {
                qt = torch::eye(n, torch::TensorOptions().dtype(a.dtype()).device(a.device()));
            }
            else
            {
                qt_real = torch::eye(n, torch::kDouble).to(a.device());
                qt_imag = torch::zeros({n, n}, torch::kDouble).to(a.device());
                qt = torch::complex(qt_real, qt_imag);
            }
            for (int k = 0; k < n - 1; k++)
            {
                torch::Tensor catk;
                if ( isComplex) 
                   catk = at::real(c.index({k}));
                else 
                   catk = c.index({k});
                if ((catk != 0.0).item<bool>())
                {
                    torch::Tensor sm;
                    for (int j = 0; j < n; j++)
                    {
                      sm = torch::sum(r.index({Slice(k), k}) * qt.index({Slice(k), j})); 
                    
                      sm = sm / c.index({k});
                      qt.index_put_({Slice(k), j}, qt.index({Slice(k), j}) - sm * r.index({Slice(k), k}));
                    }
                }
            }
            std::cerr << "qt=";
            print_matrix(qt);
            for (int i=0;i<n;i++) {
		      //r[i][i]=d[i];
              r.index_put_({i, i}, d[i]);
              r.index_put_({i, Slice(0, i)}, zero);
            }
            std::cerr << "r=";
            print_matrix(r);

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

        torch::Tensor solvev(torch::Tensor &bin)
        {
            // qtx = self.qt*b
            //In this formulation the Q matrix is already transposed
            std::cerr << "qt=" << qt << std::endl;
            std::cerr << "bin=" << bin << std::endl;
            Qt_b = torch::einsum("ij,j->i", {qt, bin});
            //We need to solve for R*x = QT*b backwards
            x = torch::zeros_like(Qt_b);
            for (int i=n-1;i>-1; i--) {
              x[i] = Qt_b[i];
              auto sm = torch::sum(r.index({i, Slice(i+1, n)}) * x.index({Slice(i+1, n)}));
              x[i] = x[i]-sm;


              x[i] = x[i] / r[i][i];
            }
            return x.clone();
        }

    }; // class QR
};     // namespace janust
#endif // QR_HPP_INCLUDED