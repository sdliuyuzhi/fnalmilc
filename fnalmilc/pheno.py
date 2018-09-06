import numpy as np
from scipy.integrate import quad
import gvar as gv
from tabulate import tabulate
import constants.constants_bs2k_2018 as const
import constants.constants_b2d_2012 as const_b2d
import constants.constants_bs2ds_2012 as const_bs2ds
from formfactors import Bs2DsFormFactors
from formfactors import Bs2KFormFactors

Bs2K = Bs2KFormFactors()


class Bs2KPheno(object):
    """
    Construct pheno observables from Bs -> k l nu form factors class.
    """

    def factor_overall(self, ml, qsq):
        """
        Return the overall factor for the differential decay rate.
        The factor ``|p_K|'' is not included.
        """
        GF = const.GF
        MBS = const.MBS
        PK = Bs2K.q_sq2PK(qsq)
        return GF**2 / (24*np.pi**3*MBS**2) * (1 - ml**2 / qsq)**2 * PK

    def factor_fplus(self, ml, qsq):
        MBS = const.MBS
        PK = Bs2K.q_sq2PK(qsq)
        return (1 + ml**2/2./qsq) * MBS**2 * PK**2

    def factor_fzero(self, ml, qsq):
        MBS = const.MBS
        MK = const.MK
        return 3.0 * ml**2 / 8.0 / qsq * (MBS**2 - MK**2)**2


    def diff_decay_rate_func(self, qsq, lepton='mu'):
        if lepton == 'mu':
            ml = const.MMU
        elif lepton == 'tau':
            ml = const.MTAU
        else:
            print "Only the 'mu' or 'tau' lepton is allowed."
            return
        z = Bs2K.q_sq2z(qsq)
        fp = Bs2K.fcn(z, Bs2K.params())['f+'] / Bs2K.Pphi(qsq, 'f+')
        f0 = Bs2K.fcn(z, Bs2K.params())['f0'] / Bs2K.Pphi(qsq, 'f0')
        ans_GeV = self.factor_overall(ml, qsq) * (self.factor_fplus(ml, qsq) * fp**2 +
                                              self.factor_fzero(ml, qsq) * f0**2)
        ans_PS = ans_GeV / const.GEV_TO_PS
        return ans_PS

    # https://helloacm.com/how-to-compute-numerical-integration-in-numpy-python/
    def integrate(self, f, lepton, a, b, N):
        x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
        fx = f(x, lepton)
        area = np.sum(fx)*(b-a)/N
        return area

    def total_decay_rate(self, lepton='mu', err=10**(-5)):
        if lepton == 'mu':
            ml = const.MMU
        elif lepton == 'tau':
            ml = const.MTAU
        else:
            print "Only the 'mu' or 'tau' lepton is allowed."
            return
        num_points = 100
        res_err = 1.0
        diff = 1.0
        res = gv.gvar(0.0,0.0)
        while diff > err:
            res_old = res
            res = self.integrate(self.diff_decay_rate_func, lepton, ml**2,
                                 const.TMINUS, num_points)
            num_points *= 2
            diff = abs(gv.mean(res - res_old))
            #print num_points, res, diff
        return gv.gvar(res), gv.sdev(res)

    def diff_decay_rate_fixed_q_sq(self, ml, qsq):
        z = Bs2K.q_sq2z(qsq)
        fp = Bs2K.fcn(z, Bs2K.params())['f+'] / Bs2K.Pphi(qsq, 'f+')
        f0 = Bs2K.fcn(z, Bs2K.params())['f0'] / Bs2K.Pphi(qsq, 'f0')
        ans_GeV = self.factor_overall(ml, qsq) * (self.factor_fplus(ml, qsq) * fp**2 +
                                              self.factor_fzero(ml, qsq) * f0**2)
        ans_PS = ans_GeV / const.GEV_TO_PS
        return [qsq, gv.mean(ans_PS), gv.sdev(ans_PS)]

    def diff_decay_rate(self, lepton='mu', num_points=500):
        """
        Calculate the differential decay rate for B -> K mu nu or B -> K tau
        nu.
        """
        if lepton == 'mu':
            ml = const.MMU
        elif lepton == 'tau':
            ml = const.MTAU
        else:
            print "Only the 'mu' or 'tau' lepton is allowed."
            return
        start = ml**2
        end = const.TMINUS

        res = []
        step = (end - start) / num_points
        for qsq in np.arange(start, end, step):
            res.append(self.diff_decay_rate_fixed_q_sq(ml, qsq))
        return res

    def diff_decay_rate_bk(self, lepton='mu', start=const.MMU**2,
                        end=const.TMINUS, num_points=500):
        """
        Calculate the differential decay rate for B -> K mu nu or B -> K tau
        nu.
        """
        if lepton == 'mu':
            ml = const.MMU
        elif lepton == 'tau':
            ml = const.MTAU
        else:
            print "Only the 'mu' or 'tau' lepton is allowed."
            return
        start = ml**2

        res = []
        step = (end - start) / num_points
        for qsq in np.arange(start, end + step, step):
            i = Bs2K.q_sq2z(qsq)
            E = Bs2K.q_sq2E(qsq)
            PK = Bs2K.q_sq2PK(qsq)
            fac_overall = const.GF**2 / (24*np.pi**3*const.MBS**2) * (1 - ml**2 / qsq)**2 * PK
            fac_plus = (1 + ml**2/2./qsq) * const.MBS**2 * PK**2
            fac_zero =  3.0 * ml**2 / 8.0 / qsq * (const.MBS**2 - const.MK**2)**2
            form = 'f+'
            fp = Bs2K.fcn(i, Bs2K.params())[form] / Bs2K.Pphi(qsq, form)
            form = 'f0'
            f0 = Bs2K.fcn(i, Bs2K.params())[form] / Bs2K.Pphi(qsq, form)
            #print qsq,gv.mean(ans), gv.sdev(ans)
            ans = fac_overall * ( fac_plus * fp**2 + fac_zero * f0**2 ) / const.GEV_TO_PS
            #print i, qsq, E, PK, fac_overall, fac_plus, fac_zero, fp, f0, ans
            res.append([qsq, gv.mean(ans), gv.sdev(ans)])
        return res

    def ratio_diff_decay_rate_fixed_qsq(self, qsq):
        # tau
        ml = const.MTAU
        fac_overall_tau = self.factor_overall(ml, qsq)
        fac_fplus_tau = self.factor_fplus(ml, qsq)
        fac_fzero_tau = self.factor_fzero(ml, qsq)

        # mu
        ml = const.MMU
        fac_overall_mu = self.factor_overall(ml, qsq)
        fac_fplus_mu = self.factor_fplus(ml, qsq)
        fac_fzero_mu = self.factor_fzero(ml, qsq)

        ratio = Bs2K.form_factor_pr_fixed_qsq('f+', 'f0', qsq)[1]

        res = (fac_overall_tau * (fac_fplus_tau * ratio + fac_fzero_tau)) / (fac_overall_mu * (fac_fplus_mu * ratio + fac_fzero_mu))
        return res

    def ratio_diff_decay_rate(self, num_points=500):
        ml = const.MTAU
        start = ml**2
        end = const.TMINUS

        res = []
        step = (end - start) / num_points
        for qsq in np.arange(start, end, step):
            r = self.ratio_diff_decay_rate_fixed_qsq(qsq)
            res.append([qsq, gv.mean(r), gv.sdev(r)])
        return res


    def ratio_diff_decay_rate_bk(self, start=const.MTAU**2, end=const.TMINUS,
                              num_points=500, integrated=False):
        """
        Calculate the ratio of the differential decay rate:
            dGamma(B -> K tau nu) / dGamma(B -> K mu nu)
        """
        res = []
        step = (end - start) / num_points
        for qsq in np.arange(start, end + step, step):
            i = self.q_sq2z(qsq)
            E = self.q_sq2E(qsq)
            #PK = abs(E**2 - const.MK**2)**0.5
            PK = self.q_sq2PK(qsq)

            # tau decay rate
            ml = const.MTAU
            fac_overall = const.GF**2 / (24*np.pi**3*const.MBS**2) * (1 - ml**2 / qsq)**2 * PK
            fac_plus = (1 + ml**2/2./qsq) * const.MBS**2 * PK**2
            fac_zero =  3.0 * ml**2 / 8.0 / qsq * (const.MBS**2 - const.MK**2)**2
            form = 'f+'
            fp = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)
            form = 'f0'
            f0 = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)
            #print qsq,gv.mean(ans), gv.sdev(ans)
            ans_tau = fac_overall * ( fac_plus * fp**2 + fac_zero * f0**2 ) / const.GEV_TO_PS


            # mu decay rate
            ml = const.MMU
            fac_overall = const.GF**2 / (24*np.pi**3*const.MBS**2) * (1 - ml**2 / qsq)**2 * PK
            fac_plus = (1 + ml**2/2./qsq) * const.MBS**2 * PK**2
            fac_zero =  3.0 * ml**2 / 8.0 / qsq * (const.MBS**2 - const.MK**2)**2
            form = 'f+'
            fp = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)
            form = 'f0'
            f0 = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)
            ans_mu = fac_overall * ( fac_plus * fp**2 + fac_zero * f0**2 ) / const.GEV_TO_PS
            ans_ratio = ans_tau / ans_mu
            res.append([qsq, gv.mean(ans_ratio), gv.sdev(ans_ratio)])

        return res

    def asymmetry(self, lepton='mu', exclusive=True, normalize=True,
                  start=const.MTAU**2, end=const.TMINUS, num_points=500):
        if lepton == 'mu':
            ml = const.MMU
        elif lepton == 'tau':
            ml = const.MTAU
        else:
            print "Only the 'mu' or 'tau' lepton is allowed."
            return
        start = ml**2

        if exclusive:
            vub = gv.gvar(const.VUB_EX)
        else:
            vub = gv.gvar(const.VUB_INC)

        res = []
        step = (end - start) / num_points
        for qsq in np.arange(start, end + step, step):
            i = self.q_sq2z(qsq)
            #E = self.q_sq2E(qsq)
            #PK = abs(E**2 - const.MK**2)**0.5
            PK = self.q_sq2PK(qsq)
            form = 'f+'
            fp = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)
            form = 'f0'
            f0 = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)

            fac_overall = const.GF**2 / (24*np.pi**3*const.MBS**2) * (1 - ml**2 / qsq)**2 * PK
            fac_plus = (1 + ml**2/2./qsq) * const.MBS**2 * PK**2
            fac_zero =  3.0 * ml**2 / 8.0 / qsq * (const.MBS**2 - const.MK**2)**2
            fac_plus_zero = 3.0 / 4.0 * (ml**2 / qsq) * (const.MBS**2 - const.MK**2)*const.MBS * PK
            #print qsq, i, E, PK, fp, f0, fac_overall, fac_plus, fac_zero, fac_plus_zero
            if normalize:
                ans = (fac_plus_zero * fp * f0) / (fac_plus * fp**2 + fac_zero * f0**2)
            else:
                ans = fac_overall * vub**2 * fac_plus_zero * fp * f0 / const.GEV_TO_PS
            res.append([qsq, gv.mean(ans), gv.sdev(ans)])
        return res

    def integrated_asymmetry(self, lepton='mu', normalized=False, start=const.MMU**2, end=const.TMINUS, num_points=500):
        if lepton == 'mu':
            ml = const.MMU
        elif lepton == 'tau':
            ml = const.MTAU
        else:
            print "Only the 'mu' or 'tau' lepton is allowed."
            return
        start = ml**2

        res = 0.0
        num = 0.0
        den = 0.0
        step = (end - start) / num_points
        for qsq in np.arange(start, end + step, step):
            i = self.q_sq2z(qsq)
            #E = self.q_sq2E(qsq)
            #PK = abs(E**2 - const.MK**2)**0.5
            PK = self.q_sq2PK(qsq)
            form = 'f+'
            fp = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)
            form = 'f0'
            f0 = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)

            fac_overall = const.GF**2 / (24*np.pi**3*const.MBS**2) * (1 - ml**2 / qsq)**2 * PK
            fac_plus = (1 + ml**2/2./qsq) * const.MBS**2 * PK**2
            fac_zero =  3.0 * ml**2 / 8.0 / qsq * (const.MBS**2 - const.MK**2)**2
            fac_plus_zero = 3.0 / 4.0 * (ml**2 / qsq) * (const.MBS**2 - const.MK**2)*const.MBS * PK
            if normalized:
                num += fac_plus_zero * fp * f0 * step
                den += (fac_plus * fp + fac_zero * f0) * step
            else:
                res += step * fac_overall * fac_plus_zero * fp * f0 / const.GEV_TO_PS
            #res += ans * step
        if normalized:
            return num / den
        else:
            return res

    def print_dGdqsq(self, lepton='mu'):
        dGdqsq = self.diff_decay_rate(lepton)
        print "-" * 20
        print tabulate(dGdqsq, headers=['qsq',"dGdqsq(" + lepton + ") / |V_ub|^2", "err"])
        return

    def print_R(self):
        R = self.ratio_diff_decay_rate()
        print "-" * 20
        print tabulate(R, headers=['qsq',"R_tau/mu", "err"])
        return


def main():
    pheno = Bs2KPheno()


#    ##########################
#    # dGdqsq mu
#    ##########################
#    var = 'qsq'
#    ml = const.MMU
#    #dGdqsq = pheno.diff_decay_rate(lepton='mu')
#
#    print "-" * 20
#    pheno.print_dGdqsq('mu')
#
#    ##########################
#    # G mu
#    ##########################
#    Gamma_mu =  pheno.integrate(pheno.diff_decay_rate_func, 'mu', ml**2, const.TMINUS, 5000)
#    print "Total decay rate mu:", Gamma_mu
#    print pheno.total_decay_rate('mu', 10**(-5))
#
#    ##########################
#    # dGdqsq tau
#    ##########################
#    ml = const.MTAU
#    #dGdqsq = pheno.diff_decay_rate(lepton='tau')
#
#    print "-" * 20
#    pheno.print_dGdqsq('tau')
#
#    ##########################
#    # G tau
#    ##########################
#    Gamma_tau =  pheno.integrate(pheno.diff_decay_rate_func, 'tau', ml**2,
#                          const.TMINUS, 5000)
#    print "Total decay rate tau:", Gamma_tau
#    print pheno.total_decay_rate('tau', 10**(-5))
    ##########################
    # R tau nu
    ##########################
    print pheno.ratio_diff_decay_rate_fixed_qsq(1.0)
    print pheno.print_R()

if __name__ == '__main__':
    main()