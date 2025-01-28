#ifndef QRTED_HPP_INCLUDED
#define QRTED_HPP_INCLUDED

#include <torch/torch.h>
#include <janus/tensordual.hpp>

namespace janus 
{


// Translated signcond function
/*TensorDual signcond(const TensorDual& a, const TensorDual& b) 
{
    return TensorDual::where(b.r >= 0,
                             TensorDual::where(a.r >= 0, a, -a),
                             TensorDual::where(a.r >= 0, -a, a));
}*/


using Slice = torch::indexing::Slice;




class QRTeD 
{
private:


public:
    int D;
    int M;
    int N;
    TensorMatDual r;
    TensorMatDual qt;
    //set EPS to tbe the smallet number such that 1.0 + EPS != 1.0
    double EPS = std::numeric_limits<double>::epsilon();

    // Constructor
    QRTeD(const TensorMatDual& a) {
        D = a.d.size(1);
        M = a.d.size(0);
        N = a.d.size(-1);
                // Set up tensor options for creating new tensors
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(a.device());

        // Create zero tensors with the specified options
        auto qtr = torch::zeros({M, D, D}, options);
        auto qtd = torch::zeros({M, D, D, N}, options);

        // Initialize qt with the zero tensors
        qt = TensorMatDual(qtr, qtd);




        r = a.clone();

        TensorDual c(torch::zeros({M, D}, torch::TensorOptions().dtype(torch::kFloat64).device(a.r.device())),
             torch::zeros({M, D, N}, torch::TensorOptions().dtype(torch::kFloat64).device(a.r.device())));
        TensorDual d= c.clone();
        for (int k = 0; k < D - 1; ++k) {
            // Ensure the TensorDual class and its constructor are appropriately defined.
            // Assuming TensorDual is a custom class that takes two tensors as arguments
            TensorDual Rkek(r.r.index({Slice(), Slice(k), k}),
                            r.d.index({Slice(), Slice(k), k}));
            auto scale = Rkek.abs().max();

            auto Rkeks = Rkek / scale;


            r.r.index_put_({Slice(), Slice(k), k}, Rkeks.r);
            r.d.index_put_({Slice(), Slice(k), k}, Rkeks.d);

            Rkek = TensorDual{r.r.index({Slice(), Slice(k), k}),
                              r.d.index({Slice(), Slice(k), k})};
            auto sm = Rkek.square().sum();


            TensorDual Rkk(torch::unsqueeze(r.r.index({Slice(), k, k}), 1),
                           torch::unsqueeze(r.d.index({Slice(), k, k}), 1));

            auto sigma = signcond(sm.sqrt(), Rkk);  // assuming signcond is defined elsewhere
            r.r.index_put_({Slice(), k, k}, r.r.index({Slice(), k, k}) + sigma.r.squeeze());
            r.d.index_put_({Slice(), k, k}, r.d.index({Slice(), k, k}) + sigma.d.squeeze());

            Rkk = TensorDual{torch::unsqueeze(r.r.index({Slice(), k, k}), 1),
                             torch::unsqueeze(r.d.index({Slice(), k, k}), 1)};
            auto sRkk = sigma * Rkk;

            c.r.index_put_({Slice(), k}, sRkk.r.squeeze());
            c.d.index_put_({Slice(), k}, sRkk.d.squeeze());

            auto ss = scale * sigma;

            d.r.index_put_({Slice(), k}, -ss.r.squeeze());
            d.d.index_put_({Slice(), k}, -ss.d.squeeze());

            Rkek = TensorDual{r.r.index({Slice(), Slice(k), k}),
                              r.d.index({Slice(), Slice(k), k})};
            TensorDual ck(torch::unsqueeze(c.r.index({Slice(), k}), 1), torch::unsqueeze(c.d.index({Slice(), k}), 1));

            TensorMatDual Rkej(r.r.index({Slice(), Slice(k), Slice(k+1)}),
                               r.d.index({Slice(), Slice(k), Slice(k+1)}));
            sm = Rkek * Rkej;
            std::cerr << "sm: " << sm << std::endl;
            auto tau = sm / ck;
            std::cerr << "tau: " << tau << std::endl;

            auto Rkejn = Rkej - ger(tau, Rkek);  // Assuming ger is a static method of TensorDual
            r.r.index_put_({Slice(), Slice(k), Slice(k + 1)}, Rkejn.r);
            r.d.index_put_({Slice(), Slice(k), Slice(k + 1)}, Rkejn.d);
        }

        d.r.index_put_({Slice(), D - 1}, r.r.index({Slice(), D - 1, D - 1}));
        d.d.index_put_({Slice(), D - 1}, r.d.index({Slice(), D - 1, D - 1}));

        qt.r = torch::eye(D, torch::TensorOptions().dtype(torch::kFloat64).
                 device(a.r.device())).unsqueeze(0).repeat({M, 1, 1});

        for (int k = 0; k < D - 1; ++k) {
            //mask = c.r[:, k] != 0
            auto mask = c.r.index({Slice(), k}) != 0;
            if (torch::any(mask).item<bool>()) {
                //Rk = TensorDual(self.r.r[mask, k:, k], self.r.d[mask, k:, k])
                TensorDual Rk(r.r.index({mask, Slice(k), k}),
                              r.d.index({mask, Slice(k), k}));
                //Qt = TensorMatDual(self.qt.r[mask, k:], self.qt.d[mask, k:])
                TensorMatDual Qt(qt.r.index({mask, Slice(k)}),
                                 qt.d.index({mask, Slice(k)}));

                //cmk = TensorDual(c.r[mask, k], c.d[mask, k, :])
                TensorDual cmk(torch::unsqueeze(c.r.index({mask, k}), 1), torch::unsqueeze(c.d.index({mask, k}), 1));
                //sm = Rk * Qt/cmk
                auto sm = Rk * Qt / cmk;
                //smRk = TensorDual.ger(sm, Rk)
                auto smRk = ger(sm, Rk);
                //self.qt.r[mask, k:, :] = self.qt.r[mask, k:, :] - smRk.r
                qt.r.index_put_({mask, Slice(k)}, qt.r.index({mask, Slice(k)}) - smRk.r);
                //self.qt.d[mask, k:, :, :] = self.qt.d[mask, k:, :, :] - smRk.d
                qt.d.index_put_({mask, Slice(k)}, qt.d.index({mask, Slice(k)}) - smRk.d);
            }
        }
        //self.r.r[:, torch.arange(self.D), torch.arange(self.D)] = d.r
        r.r.index_put_({Slice(), torch::arange(D), torch::arange(D)}, d.r);
        //self.r.d[:, torch.arange(self.D), torch.arange(self.D),:] = d.d
        r.d.index_put_({Slice(), torch::arange(D), torch::arange(D), Slice()}, d.d);
        //mask = torch.tril(torch.ones(self.D, self.D), diagonal=-1).bool().to(self.device)
        auto mask = torch::tril(torch::ones({D, D}, torch::TensorOptions().dtype(torch::kFloat64)), -1).to(torch::kBool).to(a.device());
        //mask_r = mask.unsqueeze(0)
        auto mask_r = mask.unsqueeze(0);
        //mask_d = mask.unsqueeze(0).unsqueeze(-1)
        auto mask_d = mask.unsqueeze(0).unsqueeze(-1);
        //self.r.r[:, :, :self.D] *= ~mask_r
        r.r.index_put_({Slice(), Slice(), Slice(0, D)}, r.r.index({Slice(), Slice(), Slice(0, D)}) * ~mask_r);
        //self.r.d[:, :, :self.D, :] *= ~mask_d
        r.d.index_put_({Slice(), Slice(), Slice(0, D), Slice()}, r.d.index({Slice(), Slice(), Slice(0, D), Slice()}) * ~mask_d);

    }

    //def solvev(self, b):
    TensorDual solvev(const TensorDual& b) {
        //qtx = self.qt*b
        auto qtx = qt * b;
        //std::cerr << "qtx: " << qtx << std::endl;
        //for i in range(self.D-1, -1, -1):
        for (int i = D - 1; i >= 0; --i) {
            //sm = TensorDual(qtx.r[:, i:i+1], qtx.d[:, i:i+1,:])
            TensorDual sm(qtx.r.index({Slice(), Slice(i, i + 1)}),
                          qtx.d.index({Slice(), Slice(i, i + 1)}));
            // if i+1 < self.D:
            //    rij = TensorMatDual(self.r.r[:, i:i+1, i+1:], self.r.d[:, i:i+1, i+1:])
            //    xj = TensorDual(qtx.r[:, i+1:], qtx.d[:, i+1:])
            //    sm = sm-rij*xj
            if (i + 1 < D) {
                TensorMatDual rij(r.r.index({Slice(), Slice(i, i + 1), Slice(i + 1)}),
                                  r.d.index({Slice(), Slice(i, i + 1), Slice(i + 1)}));
                TensorDual xj(qtx.r.index({Slice(), Slice(i + 1)}),
                              qtx.d.index({Slice(), Slice(i + 1)}));
                sm = sm - rij * xj;
            }
            //rii = TensorDual(self.r.r[:, i, i], self.r.d[:, i, i,:])
            TensorDual rii(torch::unsqueeze(r.r.index({Slice(), i, i}), 1),
                           torch::unsqueeze(r.d.index({Slice(), i, i}), 1));

            //std::cerr << "sm: " << sm << std::endl;
            //std::cerr << "rii: " << rii << std::endl;
            auto smorii = sm/rii;
            //std::cerr << "smorii: " << smorii << std::endl;
            //std::cerr << "qtx: " << qtx << std::endl;
            //use mask to find all element for which sm is zero
            //auto mask = (sm > this->EPS);
            //auto smorii = sm*0.0;
            //Make sure there is no singularity
            //auto smm = TensorDual(sm.r().index({mask}), sm.d().index({mask}));
            //auto riim = TensorDual(rii.r().index({mask}), rii.d().index({mask}));
            //auto smorrim = smm / riim;
            //smorii.r().index_put_({mask}, smorrim.r());
            //smorii.d().index_put_({mask}, smorrim.d());

            //qtx.r[:, i] = smorii.r
            //std::cerr << qtx.r.index({Slice(), Slice(i, i+1)}).sizes() << std::endl;
            qtx.r.index_put_({Slice(), Slice(i, i+1)}, smorii.r);
            //std::cerr << "qtx.r(): " << qtx.r << std::endl;
            //qtx.d[:, i,:] = smorii.d
            qtx.d.index_put_({Slice(), Slice(i, i+1), Slice()}, smorii.d);
            //std::cerr << "qtx.d: " << qtx.d << std::endl;
        }
        return qtx;

    }
    
    static TensorDual solvev(const TensorMatDual& qt, const TensorMatDual& r, const TensorDual& b) {
        //qtx = self.qt*b
        auto qtx = qt * b;
        int D = qt.r.size(1);
        //std::cerr << "qtx: " << qtx << std::endl;
        //for i in range(self.D-1, -1, -1):
        for (int i = D - 1; i >= 0; --i) {
            //sm = TensorDual(qtx.r[:, i:i+1], qtx.d[:, i:i+1,:])
            TensorDual sm(qtx.r.index({Slice(), Slice(i, i + 1)}),
                          qtx.d.index({Slice(), Slice(i, i + 1)}));
            // if i+1 < self.D:
            //    rij = TensorMatDual(self.r.r[:, i:i+1, i+1:], self.r.d[:, i:i+1, i+1:])
            //    xj = TensorDual(qtx.r[:, i+1:], qtx.d[:, i+1:])
            //    sm = sm-rij*xj
            if (i + 1 < D) {
                TensorMatDual rij(r.r.index({Slice(), Slice(i, i + 1), Slice(i + 1)}),
                                  r.d.index({Slice(), Slice(i, i + 1), Slice(i + 1)}));
                TensorDual xj(qtx.r.index({Slice(), Slice(i + 1)}),
                              qtx.d.index({Slice(), Slice(i + 1)}));
                sm = sm - rij * xj;
            }
            //rii = TensorDual(self.r.r[:, i, i], self.r.d[:, i, i,:])
            TensorDual rii(torch::unsqueeze(r.r.index({Slice(), i, i}), 1),
                           torch::unsqueeze(r.d.index({Slice(), i, i}), 1));

            //std::cerr << "sm: " << sm << std::endl;
            std::cerr << "rii: " << rii << std::endl;
            auto smorii = sm/rii;
            std::cerr << "smorii: " << smorii << std::endl;
            //std::cerr << "qtx: " << qtx << std::endl;
            //use mask to find all element for which sm is zero
            //auto mask = (sm > this->EPS);
            //auto smorii = sm*0.0;
            //Make sure there is no singularity
            //auto smm = TensorDual(sm.r().index({mask}), sm.d().index({mask}));
            //auto riim = TensorDual(rii.r().index({mask}), rii.d().index({mask}));
            //auto smorrim = smm / riim;
            //smorii.r().index_put_({mask}, smorrim.r());
            //smorii.d().index_put_({mask}, smorrim.d());

            //qtx.r[:, i] = smorii.r
            //std::cerr << qtx.r.index({Slice(), Slice(i, i+1)}).sizes() << std::endl;
            qtx.r.index_put_({Slice(), Slice(i, i+1)}, smorii.r);
            //std::cerr << "qtx.r(): " << qtx.r << std::endl;
            //qtx.d[:, i,:] = smorii.d
            qtx.d.index_put_({Slice(), Slice(i, i+1), Slice()}, smorii.d);
            //std::cerr << "qtx.d: " << qtx.d << std::endl;
        }
        return qtx;

    }


};

    std::tuple<TensorMatDual, TensorMatDual> qrted(const TensorMatDual &a)
    {
        QRTeD qr(a);
        return std::make_tuple(qr.qt, qr.r);
    }

    TensorDual qrtedsolvev(const TensorMatDual &qt, const TensorMatDual &r, const TensorDual &bin)
    {
        return QRTeD::solvev(qt, r, bin);
    }

} //namespace janus

#endif