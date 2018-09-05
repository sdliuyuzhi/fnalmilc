"""
This is a code to reconstruct the Bs -> K l nu form factors.
"""

import numpy as np
import gvar as gv
import argparse
from tabulate import tabulate
#import bs2k_form_factors.constants as const
import constants as const
import constants_b2d as const_b2d
import constants_bs2ds as const_bs2ds

class Bs2DsFormFactors(object):
    """
    Reconstruct Bs -> K, B -> D and Bs -> Ds form factors class.
    """

    def __init__(self, MB, MD, nz):
        self.MB = MB
        self.MD = MD
        self.r = MD / MB
        self.nz = nz

    def w2z(self, w):
        """
        Convert from w to z.
        """
        return (np.sqrt(1 + w) - np.sqrt(2.0)) / (np.sqrt(1 + w) + np.sqrt(2.0))

    def q_sq2z(self, q_sq):
        """
        Convert from q_sq to z.
        """
        w = self.q_sq2w(q_sq)
        return self.w2z(w)

    def w2q_sq(self, w):
        """
        Convert from w to q^2.
        """
        MB = self.MB
        MD = self.MD
        return MB**2 + MD**2 - 2 * w * MB * MD

    def q_sq2w(self, q_sq):
        """
        Convert from q^2 to w.
        """
        MB = self.MB
        MD = self.MD
        return (MB**2 + MD**2 - q_sq) / (2 * MB * MD)

    def phi_f0(self, z):
        """
        f0 outer function.
        """
        r = self.r
        return (1 + z) * (1 - z)**1.5 / ((1 + r) * (1 - z) + 2*np.sqrt(r)*(1 + z))**4

    def phi_fp(self, z):
        """
        f+ outer function.
        """
        r = self.r
        return (1 + z)**2 * (1 - z)**0.5 / ((1 + r) * (1 - z) + 2*np.sqrt(r)*(1 + z))**5


    def Phi(self, z, form='f+'):
        """
        Outer functions.
        """
        if form == 'f+':
            return self.phi_fp(z)
        elif form == 'f0':
            return self.phi_f0(z)
        else:
            return 1.0

    def Pole(self, q_sq, form='f+'):
        """
        The Blaschke pole factors.
        """
        if form == 'f+':
            return 1.0
        elif form == 'f0':
            return 1.0
        else:
            return 1.0

    def Pphi(self, q_sq, form='f+'):
        """
        The Blaschke pole factors (and the outer function).
        """
        z = self.q_sq2z(q_sq)
        return self.Pole(q_sq, form) * self.Phi(z, form)

    def fcn(self, z, p, nz):
        """
        Functional form of the form factors.
        """
        nz = self.nz
        ans = {}
        ans['f+'] = sum(p['b'][i] * z**i for i in range(nz))
        ans['f0'] = sum(p['b0'][i] * z**i for i in range(nz))

        return ans

    def params(self, decay='Bs2Ds'):
        """
        Read in fit results.
        """
        p = gv.BufferDict()
        dir_str = "./results/"
        if decay == 'Bs2Ds':
            tail_str = '_bs2ds_2012.txt'
        elif decay == 'B2D':
            tail_str = '_b2d_2012.txt'
        elif decay == 'B2D2015':
            tail_str = '_b2d_2015.txt'
        else:
            return "Only B2D and Bs2Ds form factors are provided."

        pmean = np.loadtxt(dir_str + "pmean" + tail_str)
        p0mean = np.loadtxt(dir_str + "p0mean" + tail_str)
        perr = np.loadtxt(dir_str + "pmean_err" + tail_str)
        p0err = np.loadtxt(dir_str + "p0mean_err" + tail_str)
        pcorr = np.loadtxt(dir_str + "corr" + tail_str)
        pcov = np.loadtxt(dir_str + "cov" + tail_str)
        #pcov = np.loadtxt("/home/yuzhi/analysis/fits/xzfit_git/cov.txt")
        # print pmean, p0mean
        n1 = len(pmean)
        pmean = np.append(pmean, p0mean)
        perr = np.append(perr, p0err)
        D = np.diag(perr)
        ppcov = np.matmul(D, np.matmul(pcorr,D))
        #print pcov
        #print ppcov
        #print pmean
        #print pcov
        #print pcorr
        #print D
        #DInv = np.linalg.inv(D)
        #correlationMat = np.matmul(DInv, np.matmul(pcov,DInv))
        #print correlationMat - pcorr
        x = gv.gvar(pmean, pcov)
        p['b'] = x[:n1]
        p['b0'] = x[n1:]
        #print gv.evalcov(p)
        #print pcov
        return p


    def form_factor_fixed_qsq(self, form='f+', qsq=0.0,
              var='qsq', decay='Bs2Ds', withpole=False, nz=3, gvar=True):
        """
        Construct the f+ or f0 form factors at a fixed q^2 value.
        """
        if form == 'f0':
            PHI = const_b2d.PHI0
        elif form == 'f+':
            PHI = const_b2d.PHI_PLUS
        #if form not in ('f+', 'f0'):
        else:
            print "Only f+ and f0 form factors are provided."
            return

        z = self.q_sq2z(qsq)
        formfactor = self.fcn(z, self.params(decay), nz)[form] / self.Pphi(qsq, form)
        formfactor /= PHI
        if var == 'qsq':
            if gvar:
                res = [qsq, formfactor]
            else:
                res = [qsq, gv.mean(formfactor), gv.sdev(formfactor)]
        elif var == 'z':
            if gvar:
                res = [qsq, formfactor]
            else:
                res = [z, gv.mean(formfactor), gv.sdev(formfactor)]
        return res


    def form_factor(self, form='f+', start=0, end=const.TMINUS, num_points=500,
              var='qsq', decay='Bs2Ds', withpole=False, nz=3, gvar=True):
        """
        Construct the f+ or f0 form factors.
        """
        if form == 'f0':
            PHI = const_b2d.PHI0
        elif form == 'f+':
            PHI = const_b2d.PHI_PLUS
        #if form not in ('f+', 'f0'):
        else:
            print "Only f+ and f0 form factors are provided."
            return

        res = []
        step = (end - start) / num_points
        for qsq in np.arange(start, end + step, step):
            res.append(self.form_factor_fixed_qsq(form, qsq, var, decay,
                                                  withpole, nz, gvar))
        return res

class Bs2KFormFactors(object):
    """
    Reconstruct Bs -> k l nu form factors class.
    """

    #Pphi = {}
    #Pphi['f+'] = lambda self, q_sq: (1.0 - q_sq / const.MBSTAR**2)
    #Pphi['f0'] = lambda q_sq: 1.0
    # with poles
    #Pphi['f0'] = lambda self, q_sq: (1.0 - q_sq / const.MBSTAR0**2)
    def Pphi(self, q_sq, form='f+'):
        """
        The Blaschke pole factors (and the outer function).
        """
        if form == 'f+':
            return 1.0 - q_sq / const.MBSTAR**2
        elif form == 'f0':
            return 1.0 - q_sq / const.MBSTAR0**2
        else:
            return 1.0


    def E2q_sq(self, EK):
        """
        Convert from E to q^2.
        """
        return (const.MBS**2 + const.MK**2 - 2 * const.MBS * EK )

    def q_sq2E(self, q_sq):
        """
        Convert from q^2 to E.
        """
        return (const.MBS**2 + const.MK**2 - q_sq) / (2 * const.MBS)

    def q_sq2PK(self, q_sq):
        """
        Convert from q^2 to the kaon momentum P_K.
        """
        return (self.q_sq2E(q_sq)**2 - const.MK**2)**0.5

#    def z2Er1(self, z):
#        """
#        Convert from z to E in r1 unit.
#        """
#        E = ( const.MB**2 + const.MP**2 - const.TCUT + (const.TCUT -
#            const.T0)*((1 + z) / (1 - z))**2 ) / (2 * const.MB)
#        return E * const.GEV_TO_R1

#    def z2q_sq(self, z, t0 = None):
#        """
#        Convert from z to q^2.
#        """
#        if t0 == None:
#            t0 = const.T0
#        temp1 = 1.0 - t0 / const.TCUT
#        temp2 = (1 + z) / (1 - z)
#        return const.TCUT * (1.0 - temp2**2 * temp1)

    def q_sq2z(self, q_sq, t0 = None):
        """
        Convert from q^2 to z.
        """
        if t0 == None:
            t0 = const.T0
        part1 = np.sqrt(const.TCUT - q_sq)
        part2 = np.sqrt(const.TCUT - t0)
        return (part1 - part2) / (part1 + part2)


    def fcn(self, z, p, nz=4):
        """
        BCL parametrization functional form of the form factors.
        """
        ans = {}
        ans['f+'] = sum(p['b'][i] * (z**i - (-1)**(i - nz) * (1.0*i / nz) * z**nz) for i in range(nz))
        ans['f0'] = sum(p['b0'][i] * z**i for i in range(nz))
        return ans

    def params(self):
        """
        Read in fit results.
        """
        p = gv.BufferDict()
        dir_str = "./results/"
        pmean = np.loadtxt(dir_str + "pmean.txt")
        p0mean = np.loadtxt(dir_str + "p0mean.txt")
        pcorr = np.loadtxt(dir_str + "corr.txt")
        pcov = np.loadtxt(dir_str + "cov.txt")
        # print pmean, p0mean
        pmean = np.append(pmean, p0mean)
        #print pmean
        #print pcov
        #print pcorr
        D = np.sqrt(np.diag(np.diag(pcov)))
        DInv = np.linalg.inv(D)
        correlationMat = np.matmul(DInv, np.matmul(pcov,DInv))
        #print correlationMat - pcorr
        x = gv.gvar(pmean, pcov)
        p['b'] = x[:4]
        p['b0'] = x[4:]
        #print gv.evalcov(p)
        #print pcov
        return p

    def form_factor_fixed_qsq(self, form='f+', qsq=0.0,
              var='qsq', withpole=True, nz=4, gvar=True):
        """
        Construct the f+ or f0 form factors at a fixed q^2 value.
        """
        if form not in ('f+', 'f0'):
            print "Only f+ and f0 form factors are provided."
            return

        z = self.q_sq2z(qsq)
        if withpole:
            formfactor = self.fcn(z, self.params())[form] / self.Pphi(qsq, form)
        else:
            formfactor = self.fcn(z, self.params())[form]
        if var == 'qsq':
            if gvar:
                res = [qsq, formfactor]
            else:
                res = [qsq, gv.mean(formfactor), gv.sdev(formfactor)]
        elif var == 'z':
            if gvar:
                res = [qsq, formfactor]
            else:
                res = [z, gv.mean(formfactor), gv.sdev(formfactor)]
        return res

    def form_factor(self, form='f+', start=0, end=const.TMINUS, num_points=500,
              var='qsq', withpole=True, nz=4, gvar=True):
        """
        Construct the f+ or f0 form factors.
        """
        if form not in ('f+', 'f0'):
            print "Only f+ and f0 form factors are provided."
            return

        res = []
        step = (end - start) / num_points
        for qsq in np.arange(start, end + step, step):
            res.append(self.form_factor_fixed_qsq(form, qsq, var, withpole, nz,
                                                  gvar))
        return res

#Class Bs2KPheno(object):
#    """
#    Construct pheno observables from Bs -> k l nu form factors class.
#    """

    def factor_overall(self, ml, qsq):
        """
        Return the overall factor for the differential decay rate.
        The factor ``|p_K|'' is not included.
        """
        PK = self.q_sq2PK(qsq)
        return const.GF**2 / (24*np.pi**3*const.MBS**2) * (1 - ml**2 / qsq)**2 * PK

    def factor_fplus(self, ml, qsq):
        PK = self.q_sq2PK(qsq)
        return (1 + ml**2/2./qsq) * const.MBS**2 * PK**2

    def factor_fzero(self, ml, qsq):
        return 3.0 * ml**2 / 8.0 / qsq * (const.MBS**2 - const.MK**2)**2

    def diff_decay_rate_fixed_q_sq(self, ml, qsq):
        z = self.q_sq2z(qsq)
        fp = self.fcn(z, self.params())['f+'] / self.Pphi(qsq, 'f+')
        f0 = self.fcn(z, self.params())['f0'] / self.Pphi(qsq, 'f0')
        ans_GeV = self.factor_overall(ml, qsq) * (self.factor_fplus(ml, qsq) * fp**2 +
                                              self.factor_fzero(ml, qsq) * f0**2)
        ans_PS = ans_GeV / const.GEV_TO_PS
        return [qsq, gv.mean(ans_PS), gv.sdev(ans_PS)]

    def diff_decay_rate(self, lepton='mu', start=const.MMU**2,
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
            res.append(self.diff_decay_rate_fixed_q_sq(qsq, ml))
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
            i = self.q_sq2z(qsq)
            E = self.q_sq2E(qsq)
            PK = self.q_sq2PK(qsq)
            fac_overall = const.GF**2 / (24*np.pi**3*const.MBS**2) * (1 - ml**2 / qsq)**2 * PK
            fac_plus = (1 + ml**2/2./qsq) * const.MBS**2 * PK**2
            fac_zero =  3.0 * ml**2 / 8.0 / qsq * (const.MBS**2 - const.MK**2)**2
            form = 'f+'
            fp = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)
            form = 'f0'
            f0 = self.fcn(i, self.params())[form] / self.Pphi(qsq, form)
            #print qsq,gv.mean(ans), gv.sdev(ans)
            ans = fac_overall * ( fac_plus * fp**2 + fac_zero * f0**2 ) / const.GEV_TO_PS
            #print i, qsq, E, PK, fac_overall, fac_plus, fac_zero, fp, f0, ans
            res.append([qsq, gv.mean(ans), gv.sdev(ans)])
        return res

    def ratio_diff_decay_rate(self, start=const.MTAU**2, end=const.TMINUS,
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


'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print 'dGamma/dqsq'
print fp
print f0


GF = 1.1663787 * 0.00001 # GeV^-2 PDG16 p119
# muon mass
ml = 105.6583745 * 0.001 # GeV PDG16 p 32
# tau mass
#ml = 1776.86 * 0.001 #GeV PDG 16 p32

#1GeV-1 = 6.58*10**-25 s = 6.58*10**-13 ps

conversion = 6.58*10**(-13)

#for qsq in np.arange(0.01,qsqmax*1.02, qsqmax/50.):
for qsq in np.arange(ml**2,qsqmax*1.002, qsqmax/500.):
    i = C().q_sq2z(qsq)
    fp = fcn(i,params())['f+'] / Pphi['f+'](qsq)
    f0 = fcn(i,params())['f0'] / Pphi['f0'](qsq)
    E = C().q_sq2E(qsq)
    #PK = abs(E**2-MPION**2)**0.5
    PK = self.q_sq2PK(qsq)

    fac_overall = GF**2 / (24*np.pi**3*MBS**2) * (1 - ml**2 / qsq)**2 * PK
    fac_plus = (1 + ml**2/2./qsq) * MBS**2 * PK**2
    fac_zero =  3.0 * ml**2 / 8.0 / qsq * (MBS**2 - MK**2)**2

    ans = fac_overall * ( fac_plus * fp**2 + fac_zero * f0**2 ) / conversion
    print qsq,gv.mean(ans), gv.sdev(ans)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print 'R_tau_mu'

# muon mass
ml_mu = 105.6583745 * 0.001 # GeV PDG16 p 32
# tau mass
ml_tau = 1776.86 * 0.001 #GeV PDG 16 p32

#1GeV-1 = 6.58*10**-25 s = 6.58*10**-13 ps

conversion = 6.58*10**(-13)

#for qsq in np.arange(0.01,qsqmax*1.02, qsqmax/50.):
for qsq in np.arange(ml_tau**2,qsqmax*1.002, qsqmax/500.):
    i = C().q_sq2z(qsq)
    fp = fcn(i,params())['f+'] / Pphi['f+'](qsq)
    f0 = fcn(i,params())['f0'] / Pphi['f0'](qsq)
    E = C().q_sq2E(qsq)
    PK = abs(E**2-MPION**2)**0.5

    fac_overall = GF**2 / (24*np.pi**3*MBS**2) * (1 - ml_tau**2 / qsq)**2 * PK
    fac_plus = (1 + ml_tau**2/2./qsq) * MBS**2 * PK**2
    fac_zero =  3.0 * ml_tau**2 / 8.0 / qsq * (MBS**2 - MK**2)**2

    ans_tau = fac_overall * ( fac_plus * fp**2 + fac_zero * f0**2 ) / conversion
    #print qsq,gv.mean(ans_tau), gv.sdev(ans_tau)

    fac_overall = GF**2 / (24*np.pi**3*MBS**2) * (1 - ml_mu**2 / qsq)**2 * PK 
    fac_plus = (1 + ml_mu**2/2./qsq) * MBS**2 * PK**2 
    fac_zero =  3.0 * ml_mu**2 / 8.0 / qsq * (MBS**2 - MK**2)**2 

    ans_mu = fac_overall * ( fac_plus * fp**2 + fac_zero * f0**2 ) / conversion
    #print qsq,gv.mean(ans_mu), gv.sdev(ans_mu)

    ans_ratio = ans_tau / ans_mu

    print qsq,gv.mean(ans_ratio), gv.sdev(ans_ratio)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print 'A_FB'

# muon mass
ml_mu = 105.6583745 * 0.001 # GeV PDG16 p 32
# tau mass
ml_tau = 1776.86 * 0.001 #GeV PDG 16 p32

#1GeV-1 = 6.58*10**-25 s = 6.58*10**-13 ps
conversion = 6.58*10**(-13)

Vubex = gv.gvar(3.47,0.22)
Vubin = gv.gvar(4.41,0.22)


ml = ml_tau
Vub = Vubex
#for qsq in np.arange(0.01,qsqmax*1.02, qsqmax/50.): 
for qsq in np.arange(ml**2,qsqmax*1.02, qsqmax/50.): 
    i = C().q_sq2z(qsq)
    fp = fcn(i,params())['f+'] / Pphi['f+'](qsq)
    f0 = fcn(i,params())['f0'] / Pphi['f0'](qsq)
    E = C().q_sq2E(qsq)
    PK = abs(E**2-MPION**2)**0.5

    fac_overall = GF**2 / (32*np.pi**3*MBS) * (1 - ml**2 / qsq)**2 * PK**2 
    fac_1 = (ml**2 / qsq) * (MBS**2 - MK**2) 

    ans = fac_overall * Vub**2 * fac_1 * fp * f0 / conversion
    print qsq,gv.mean(ans), gv.sdev(ans)

'''

def main():

    #   Arguments  #
    parser = argparse.ArgumentParser(description='Bs -> K l nu form factors')
    parser.add_argument('-f', '--form', type=str, default='f+',
                        help="form factor: f+ or f0")
    parser.add_argument('-v', '--var', type=str, default='qsq',
                        help="variable: qsq or z")
    parser.add_argument('-p', '--withpole', type=bool, default=False,
                        help="whether to take the pole factor out or not")
    args = parser.parse_args()


    decay = 'B2D'
    decay = 'B2D2015'

    if decay == 'B2D':
#        qmin = const_b2d.TK
        MB = const_b2d.MB
        MD = const_b2d.MD
        nz = 3


    #decay = 'Bs2Ds'
    if decay == 'Bs2Ds':
#        qmin = const_bs2ds.TPI
        MB = const_bs2ds.MBS
        MD = const_bs2ds.MDS
        nz = 3

    if decay == 'B2D2015':
        qmin = const_b2d.TK
        MB = const_b2d.MB
        MD = const_b2d.MD
        nz = 4

    MB = const_b2d.MB
    MD = const_b2d.MD
    MBS = const_bs2ds.MBS
    MDS = const_bs2ds.MDS
    nz = 3

    formfactors_b2d = Bs2DsFormFactors(MB, MD, nz)
    formfactors_bs2ds = Bs2DsFormFactors(MBS, MDS, nz)

    nz = 4
    formfactors_b2d2015 = Bs2DsFormFactors(MB, MD, nz)

    formfactors_bs2k = Bs2KFormFactors()

#    decay = 'B2D'
#    qmin = 0
#    fp_b2d = formfactors_b2d.form_factor('f+', qmin, const_b2d.TMINUS, 500, 'qsq', decay)
#    f0_b2d = formfactors_b2d.form_factor('f0', qmin, const_b2d.TMINUS, 500, 'qsq', decay)
#
#
#    decay = 'Bs2Ds'
#    fp_bs2ds = formfactors_bs2ds.form_factor('f+', qmin, const_b2d.TMINUS, 500, 'qsq', decay)
#    f0_bs2ds = formfactors_bs2ds.form_factor('f0', qmin, const_b2d.TMINUS, 500, 'qsq', decay)
#
#
#    decay = 'B2D2015'
#    fp_b2d2015 = formfactors_b2d2015.form_factor('f+', qmin, const_b2d.TMINUS, 500, 'qsq', decay)
#    f0_b2d2015 = formfactors_b2d2015.form_factor('f0', qmin, const_b2d.TMINUS, 500, 'qsq', decay)
#

    qmin = 0
    fp_bs2k = formfactors_bs2k.form_factor('f+', qmin, const_b2d.TMINUS, 500,
                                           'qsq', withpole=True, gvar=False)
    f0_bs2k = formfactors_bs2k.form_factor('f0', qmin, const_b2d.TMINUS, 500,
                                           'qsq', withpole=True, gvar=False)

    print tabulate(fp_bs2k, headers=['qsq',"f+", "err"])
    print tabulate(f0_bs2k, headers=['qsq',"f0", "err"])
#    fp_b2d_gvar = np.array([[qsq, gv.gvar(mean, err)] for qsq, mean, err in fp_b2d])
#    f0_b2d_gvar = np.array([[qsq, gv.gvar(mean, err)] for qsq, mean, err in f0_b2d])
#
#    fp_bs2ds_gvar = np.array([[qsq, gv.gvar(mean, err)] for qsq, mean, err in
#                              fp_bs2ds])
#    f0_bs2ds_gvar = np.array([[qsq, gv.gvar(mean, err)] for qsq, mean, err in
#                              f0_bs2ds])
#
#    fp_b2d2015_gvar = np.array([[qsq, gv.gvar(mean, err)] for qsq, mean, err in
#                                fp_b2d2015])
#    f0_b2d2015_gvar = np.array([[qsq, gv.gvar(mean, err)] for qsq, mean, err in
#                                f0_b2d2015])
#
#    fp_bs2k_gvar = np.array([[qsq, gv.gvar(mean, err)] for qsq, mean, err in
#                             fp_bs2k])
#    f0_bs2k_gvar = np.array([[qsq, gv.gvar(mean, err)] for qsq, mean, err in
#                             f0_bs2k])
#
#
#    fp_ratio = np.array([[qsq, gv.mean(bs2k/bs2ds), gv.sdev(bs2k/bs2ds)] for qsq,
#                         bs2k, bs2ds in zip(fp_bs2k_gvar[:,0], fp_bs2k_gvar[:,1],
#                                          fp_bs2ds_gvar[:,1])])

#    print 'qsq=0', fp_bs2k_gvar[0], fp_bs2ds_gvar[0], fp_b2d_gvar[0], fp_b2d2015_gvar[0]
#    print tabulate(fp_ratio, headers=['qsq',"f+", "err"])

#    qsq = fp_bs2k_gvar[:,0]
#    fp_ratio_corrected = fp_bs2k_gvar[:,1] * fp_b2d_gvar[:,1] / fp_bs2ds_gvar[:,1] / fp_b2d2015_gvar[:,1]
#    fp_ratio_corrected = [[qsq, gv.mean(ratio), gv.sdev(ratio)] for qsq, ratio
#                          in zip(qsq, fp_ratio_corrected)]

#    print tabulate(fp_ratio_corrected, headers=['qsq',"f+", "err"])

#    print tabulate(fp, headers=['z',"f+", "err"])
#    print "-" * 20
#    print tabulate(f0, headers=['z',"f0", "err"])

#    formfactors = Bs2KFormFactors()
#    var='qsq'
#    #for var in ('qsq', 'z'):
#    for varr in ('q'):
#        var = 'qsq'
#        #fp = formfactors.form_factor(form=args.form, var=args.var,
#        #        withpole=args.withpole)
#        fp = formfactors.form_factor(form='f+', var=var,
#                withpole=True)
#        f0 = formfactors.form_factor(form='f0', var=var, withpole=True)
#        print tabulate(fp, headers=[var,"f+", "err"])
#        print "-" * 20
#        print tabulate(f0, headers=[var,"f0", "err"])

    #dGdqsq = formfactors.diff_decay_rate(lepton='mu')
    #print "-" * 20
    #print tabulate(dGdqsq, headers=[var,"dGdqsq(mu) / |V_ub|^2", "err"])
#    total_decay_rate_mu = sum([row[1] for row in dGdqsq]) * (const.TMINUS - const.MMU) / len(dGdqsq)
#    total_decay_rate_mu_err = sum([row[2] for row in dGdqsq]) * (const.TMINUS - const.MMU) / len(dGdqsq)
#    print total_decay_rate_mu, total_decay_rate_mu_err
#
#
#
#    dGdqsq = formfactors.diff_decay_rate(lepton='tau')
#    print "-" * 20
#    print tabulate(dGdqsq, headers=[var,"dGdqsq(tau) / |V_ub|^2", "err"])
#    total_decay_rate_tau = sum([row[1] for row in dGdqsq]) * (const.TMINUS - const.MTAU) / len(dGdqsq)
#    total_decay_rate_tau_err = sum([row[2] for row in dGdqsq]) * (const.TMINUS - const.MTAU) / len(dGdqsq)
#    print total_decay_rate_tau, total_decay_rate_tau_err


#    ratio_tau_mu_diff_decay_rate = formfactors.ratio_diff_decay_rate()
#    print "-" * 20
#    print tabulate(ratio_tau_mu_diff_decay_rate,
#                   headers=[var,"R^{\\tau}_{\mu}", "err"])
#
#    asymmetry = formfactors.asymmetry(lepton='mu', exclusive=False,
#                                      normalize=False)
#    print "-" * 20
#    print tabulate(asymmetry, headers=[var,"asymmetry mu inclusive", "err"])
#
#    asymmetry = formfactors.asymmetry(lepton='mu', exclusive=True,
#                                      normalize=False)
#    print "-" * 20
#    print tabulate(asymmetry, headers=[var,"asymmetry mu exclusive", "err"])
#
#    asymmetry = formfactors.asymmetry(lepton='tau', exclusive=False, normalize=False)
#    print "-" * 20
#    print tabulate(asymmetry, headers=[var,"asymmetry tau inclusive", "err"])
#
#    asymmetry = formfactors.asymmetry(lepton='tau', exclusive=True, normalize=False)
#    print "-" * 20
#    print tabulate(asymmetry, headers=[var,"asymmetry tau exclusive", "err"])
#
#
#    asymmetry = formfactors.asymmetry(lepton='mu', normalize=True)
#    print "-" * 20
#    print tabulate(asymmetry, headers=[var,"asymmetry mu normalized", "err"])
#
#    asymmetry = formfactors.asymmetry(lepton='tau', normalize=True)
#    print "-" * 20
#    print tabulate(asymmetry, headers=[var,"asymmetry tau normalized", "err"])

#    isym=formfactors.integrated_asymmetry(lepton='tau', num_points=3000)
#    print "-" * 20
#    print "integrated_asymmetry tau", isym
#
#    isym=formfactors.integrated_asymmetry(lepton='mu', num_points=3000)
#    print "-" * 20
#    print "integrated_asymmetry mu", isym


    #isym=formfactors.integrated_asymmetry(lepton='tau', num_points=3000)
    #print "-" * 20
    #print "integrated_asymmetry tau", isym

    #isym=formfactors.integrated_asymmetry(lepton='mu', num_points=3000)
    #print "-" * 20
    #print "integrated_asymmetry mu", isym
if __name__ == '__main__':
    main()

