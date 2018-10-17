"""
This is a code to reconstruct the semileptonic form factors.
"""

import abc
import numpy as np
import gvar as gv
import argparse
import textwrap
from tabulate import tabulate
import constants.constants_bs2k_2018 as const
import constants.constants_b2d_2012 as const_b2d
import constants.constants_bs2ds_2012 as const_bs2ds

class FormFactors(object):

    __metaclass__  = abc.ABCMeta

    def __init__(self, MB, MP, MB_cut, MP_cut, t0, nz, dir_str='./results/',
                 tail_str='.txt') :
        self.MB = MB
        self.MP = MP
        self.MB_cut = MB_cut
        self.MP_cut = MP_cut
        self.nz = nz
        self.t_cut = (MB_cut + MP_cut)**2
        self.t_minus = (MB - MP)**2
        self.t0 = t0
        self.dir_str = dir_str # these are ugly ...
        self.tail_str = tail_str

    def w_to_qsq(self, w):
        """Convert from w to q^2."""
        MB = self.MB
        MP = self.MP
        return MB**2 + MP**2 - 2 * w * MB * MP

    def qsq_to_w(self, qsq):
        """Convert from q^2 to w."""
        MB = self.MB
        MP = self.MP
        return (MB**2 + MP**2 - qsq) / (2 * MB * MP)

    def qsq_to_z(self, qsq):
        """Convert from q^2 to z."""
        t_cut = self.t_cut
        t_minus = self.t_minus
        t0 = self.t0
        if t0 == None:
            t0 = t_cut - (t_cut * (t_cut - t_minus))**0.5
        part1 = (t_cut - qsq)**0.5
        part2 = (t_cut - t0)**0.5
        return (part1 - part2) / (part1 + part2)

    def z_to_qsq(self, z):
        """Convert from z to q^2."""
        t_cut = self.t_cut
        t_minus = self.t_minus
        t0 = self.t0
        if t0 == None:
            t0 = t_cut - (t_cut * (t_cut - t_minus))**0.5
        part1 = (1 + z) / (1 - z)
        part2 = 1.0 - t0 / t_cut
        return t_cut * (1.0 - part1**2 * part2)

    def qsq_to_e(self, qsq):
        """Convert from q^2 to E."""
        MB = self.MB
        MP = self.MP
        return (MB**2 + MP**2 - qsq) / (2 * MB)

    def e_to_qsq(self, E):
        """Convert from E to q^2."""
        MB = self.MB
        MP = self.MP
        return MB**2 + MP**2 - 2 * MB * E

    def w_to_z(self, w):
        """Convert from w to z."""
        t0 = self.t0
        qsq = self.w_to_qsq(w)
        return self.qsq_to_z(qsq, t0)

    def z_to_w(self, w):
        """Convert from z to w."""
        t0 = self.t0
        qsq = self.z_to_qsq(z, t0)
        return self.qsq_to_w(qsq)

    def w_to_e(self, w):
        """Convert from w to E."""
        return w * self.MP

    def e_to_w(self, e):
        """Convert from E to w."""
        return e / self.MP

    def z_to_e(self, z):
        """Convert from z to E."""
        t0 = self.t0
        qsq = self.z_to_qsq(z, t0)
        return self.qsq_to_e(qsq)

    def e_to_z(self, e):
        """Convert from E to z."""
        t0 = self.t0
        qsq = self.e_to_qsq(e)
        return self.qsq_to_z(qsq, t0)

    def variable_change(self, x=0.0, var='qsq'):
        """Get variables 'qsq', 'w', and 'z'."""
        if var == 'qsq':
            return x, self.qsq_to_w(x), self.qsq_to_z(x)
        if var == 'w':
            qsq = self.w_to_qsq(x)
            return qsq, x, self.qsq_to_z(qsq)
        if var == 'z':
            qsq = self.z_to_qsq(x)
            return qsq, self.qsq_to_w(qsq), x
        print "Only 'qsq', 'w', and 'z' variables are supported."
        return

    @abc.abstractmethod
    def pole(self, qsq, form='f_plus'):
        """The Blaschke pole factors."""
        return None

    @abc.abstractmethod
    def phi(self, z, form='f_plus'):
        return None

    def params(self):
        """Read in fit results."""
        dir_str = self.dir_str
        tail_str = self.tail_str
        p = gv.BufferDict()
        pmean = np.loadtxt(dir_str + "/pmean" + tail_str)
        p0mean = np.loadtxt(dir_str + "/p0mean" + tail_str)
        perr = np.loadtxt(dir_str + "/pmean_err" + tail_str)
        p0err = np.loadtxt(dir_str + "/p0mean_err" + tail_str)
        pcorr = np.loadtxt(dir_str + "/corr" + tail_str)
        pcov = np.loadtxt(dir_str + "/cov" + tail_str)
        nz = len(pmean)
        if nz != self.nz:
            print "The number of the input parameter should be equal to nz."
            return
        pmean = np.append(pmean, p0mean)
        d_mat = np.sqrt(np.diag(np.diag(pcov)))
        d_mat_inv = np.linalg.inv(d_mat)
        correlation_mat = np.matmul(d_mat_inv, np.matmul(pcov, d_mat_inv))
        #print correlationMat - pcorr
        x = gv.gvar(pmean, pcov)
        p['b'] = x[:nz]
        p['b0'] = x[nz:]
        #print gv.evalcov(p)
        return p

    @abc.abstractmethod
    def fcn(self, z, p):
        """Functional form of the form factors."""
        return

    def fcn_ratio(self, z, p, nz=4):
        res = self.fcn(z, p, nz)
        return res['f_plus'] / res['f_zero']

    def fcn_product(self, z, p, nz=4):
        res = self.fcn(z, p, nz)
        return res['f_plus'] * res['f_zero']

    def form_factor_fixed_x(self, form='f_plus', x=0.0,
              var='qsq', withpole=True, gvar=False):
        """
        Construct the f_plus or f_zero form factors at a fixed x value.
        """
        if form not in ('f_plus', 'f_zero'):
            print "Only f_plus and f_zero form factors are provided."
            return

        qsq, w, z = self.variable_change(x, var)

        p = self.params()
        formfactor = self.fcn(z, p)[form]
        formfactor /= self.phi(z, form)
        if withpole and var != 'z':
        #if withpole:
            formfactor /= self.pole(qsq, form)
        if gvar:
            return x, formfactor
        return x, gv.mean(formfactor), gv.sdev(formfactor)

    def form_factor(self, form='f_plus', start=0, end=const.TMINUS, num_points=500,
              var='qsq', withpole=True, gvar=False):
        """
        Construct the f_plus or f_zero form factors.
        """
        if form not in ('f_plus', 'f_zero'):
            print "Only f_plus and f_zero form factors are provided."
            return

        step = (end - start) / num_points
        xlst = np.arange(start, end + step, step)
        return [
            self.form_factor_fixed_x(
                form, x, var,
                withpole, gvar) for x in xlst
        ]


class BsToK(FormFactors):
    """
    Reconstruct Bs -> K form factors class.
    t0 = t_c - \sqrt(t_c*(t_c - t_-)) for Bs2K analysis
    """

    def pole(self, qsq, form='f_plus'):
        """The Blaschke pole factors."""
        mb_star = const.MBSTAR
        mb_star_0 = const.MBSTAR0
        if form == 'f_plus':
            return 1.0 - qsq / mb_star**2
        elif form == 'f_zero':
            return 1.0 - qsq / mb_star_0**2
        else:
            return 1.0

    def phi(self, z, form='f_plus'):
        if form == 'f_plus':
            return 1.0
        elif form == 'f_zero':
            return 1.0
        else:
            return 1.0

    def fcn(self, z, p):
        """Functional form of the form factors."""
        nz = self.nz
        res = {}
        res['f_plus'] = sum(p['b'][i] * (z**i - (-1)**(i-nz) * (1.0*i/nz) * z**nz) for i in range(nz))
        res['f_zero'] = sum(p['b0'][i] * z**i for i in range(nz))
        return res


class BsToDs(FormFactors):
    """
    Reconstruct B -> D form factors class.
    t0 = t_ for Bs2Ds and B2D analysis
    """

    def pole(self, qsq, form='f_plus'):
        """The Blaschke pole factors."""
        if form == 'f_plus':
            return 1.0
        elif form == 'f_zero':
            return 1.0
        else:
            return 1.0

    def phi(self, z, form='f_plus'):
        r = self.MP / self.MB
        phi_plus = const_bs2ds.PHI_PLUS
        phi_0 = const_bs2ds.PHI0
        if form == 'f_plus':
            res = (1+z)**2 * (1-z)**0.5 / ((1+r) * (1-z) + 2 * r**0.5 * (1+z))**5
            res *= phi_plus
        elif form == 'f_zero':
            res = (1+z) * (1-z)**1.5 / ((1+r)*(1-z) + 2 * r**0.5 * (1+z))**4
            res *= phi_0
        return res

    def fcn(self, z, p):
        """Functional form of the form factors."""
        nz = self.nz
        res = {}
        res['f_plus'] = sum(p['b'][i] * z**i for i in range(nz))
        res['f_zero'] = sum(p['b0'][i] * z**i for i in range(nz))
        return res


def main():

    #   Arguments  #
    parser = argparse.ArgumentParser(
        description='B(s) -> K(D(s)) l nu form factors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@',
        epilog=textwrap.dedent("""
                               You can read the parameters from a local file:

                               $ formfactors @ff_args.txt

                               with ff_args.txt having the following content:
                               -d=bsk18
                               -f=f_plus
                               -v=qsq
                               -p=True
                               -xmin=0.0
                               -xmax=23.0

                               You can add all arguments you want to that file,
                               just remember to have one argument per line.
                               """)
    )
    parser.add_argument('-d', '--decay', type=str, default='bsk',
                        help="Decay type: bsk18(bsk), bsds12, bd12, bd15")
    parser.add_argument('-f', '--form', type=str, default='f_plus',
                        help="Form factor: f_plus, f_zero, or f_plus_zero")
    parser.add_argument('-v', '--var', type=str, default='qsq',
                        help="Variable: qsq or z (or w)")
    parser.add_argument('-p', '--withpole', type=bool, default=False,
                        help="Whether to take the pole factor out or not")
    parser.add_argument('-xmin', '--xmin', type=float, default=0.0,
                        help="The mininum variable: xmin")
    parser.add_argument('-xmax', '--xmax', type=float, default=1.0,
                        help="The maximum variable: xmax")
    args = parser.parse_args()


    if args.decay not in ['bsk', 'bsk18', 'bsds12', 'bd12', 'bd15']:
        parser.print_help()
        raise ValueError('Must pick the available decay mode.')

    if args.form not in ['f_plus', 'f_zero']:
        parser.print_help()
        raise ValueError('Must pick the available form factors.')


    if args.decay == 'bsk18' or 'bsk':
        m_b = const.MBS
        m_p = const.MK
        m_b_cut = const.MB
        m_p_cut = const.MPION
        t0 = None
        #tmin = (m_b - m_p)**2
        nz = 4
        dir_str = './results/'
        tail_str = '.txt'
        form_factor = BsToK(m_b, m_p, m_b_cut, m_p_cut, t0, nz, dir_str, tail_str)
    elif args.decay == 'bsds12':
        m_b = const_bs2ds.MBS
        m_p = const_bs2ds.MDS
        m_b_cut = const_bs2ds.MBS # should be MB in principle
        m_p_cut = const_bs2ds.MDS # should be MD in principle
        t0 = (m_b - m_p)**2
        #tmin = t0
        nz = 3
        dir_str = './results/'
        tail_str = '_bs2ds_2012.txt'
        form_factor = BsToDs(m_b, m_p, m_b_cut, m_p_cut, t0, nz, dir_str, tail_str)
    elif args.decay == 'bd12':
        m_b = const_b2d.MB
        m_p = const_b2d.MD
        m_b_cut = const_b2d.MB
        m_p_cut = const_b2d.MD
        t0 = (m_b - m_p)**2
        #tmin = t0
        nz = 3
        dir_str = './results/'
        tail_str = '_b2d_2012.txt'
        form_factor = BsToDs(m_b, m_p, m_b_cut, m_p_cut, t0, nz, dir_str, tail_str)
    elif args.decay == 'bd15':
        m_b = const_b2d.MB
        m_p = const_b2d.MD
        m_b_cut = const_b2d.MB
        m_p_cut = const_b2d.MD
        t0 = (m_b - m_p)**2
        #tmin = t0
        nz = 4
        dir_str = './results/'
        tail_str = '_b2d_2015.txt'
        form_factor = BsToDs(m_b, m_p, m_b_cut, m_p_cut, t0, nz, dir_str, tail_str)
    else:
        print "-decay type can only be 'bsk18', 'bd12', 'bsds12', 'bd15'"
        return

    qmin = 0.0 if args.xmin < 0 else args.xmin

    f_plus = form_factor.form_factor('f_plus', args.xmin, args.xmax, 50,
                                 var=args.var, withpole=args.withpole, gvar=False)
    f_zero = form_factor.form_factor('f_zero', args.xmin, args.xmax, 50,
                                 var=args.var, withpole=args.withpole, gvar=False)
    print tabulate(f_plus, headers=['qsq',"f_plus", "err"])
    print tabulate(f_zero, headers=['qsq',"f_zero", "err"])

if __name__ == '__main__':
    main()
