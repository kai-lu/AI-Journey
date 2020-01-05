'''
https://github.com/PhilReinhold/pozar/blob/master/pozar.py
'''
import numpy as np
from sympy import Symbol, Matrix, Identity, lambdify, simplify, cancel, collect
import operator as op
from skrf.plotting import plot_smith
import matplotlib.pyplot as plt
from functools import reduce

Hz = Symbol("Hz")
ohm = Symbol("Ohm")
rad = Symbol("rad")
farad = 1 / (ohm * Hz * 2 * rad)
henry = ohm / (Hz * 2 *rad)
pi = rad

fF, pF, nF = [pre*farad for pre in (1e-15, 1e-12, 1e-9)]
pH, nH, uH = [pre*henry for pre in (1e-12, 1e-9, 1e-6)]
kHz, mHz, gHz = [pre*Hz for pre in (1e3, 1e6, 1e9)]

FREQ = Symbol('FREQ') * Hz

def parallel(*args):
    return collect(1/sum(1/a for a in args), ohm)

def capacitance(c):
    return collect(1/(2j * pi * FREQ * c), ohm)

def inductance(l):
    return collect((2j * pi * FREQ * l), ohm)

def resonator_series_lcr(l, c, r=0):
    return collect(capacitance(c) + inductance(l) + r, ohm)

def resonator_parallel_lcg(l, c, g=0):
    return collect(parallel(capacitance(c), inductance(l), 1/g), ohm)

def resonator_series_fzq(f, z, q=None):
    w = 2*pi*f
    l = cancel(z / w)
    c = cancel(1 / (w*z))
    if q is not None:
        r = z / q
    else:
        r = 0
    return resonator_series_lcr(l, c, r)

def resonator_parallel_fzq(f, z, q=None):
    w = 2*pi*f
    l = cancel(z / w)
    c = cancel(1 / (w*z))
    if q is not None:
        g = 1 / (q*z)
    else:
        g = 0
    return resonator_parallel_lcg(l, c, g)

class Impedance(Matrix):
    def __init__(self, *args, **kwargs):
        super(Impedance, self).__init__(*args, **kwargs)
        self.applyfunc(lambda m: collect(m, ohm))

    def to_Z(self):
        return self

    def to_Y(self):
        return Admittance(self.inv())

    def to_S(self, z=50*ohm):
        zn = (self/z).applyfunc(cancel)
        one = Matrix(np.identity(self.shape[0]))
        return Scattering((zn+one).inverse_ADJ() * (zn-one), z0=z)

    def to_T(self):
        (z11, z12), (z21, z22) = self.tolist()
        if self.shape != (2, 2):
            raise ValueError
        det = z11*z22 - z12*z21
        return Transmission([[z11/z21, det/z21, 1/z21, z22/z21]]).applyfunc(cancel)

class Admittance(Matrix):
    def __init__(self, *args, **kwargs):
        super(Admittance, self).__init__(*args, **kwargs)
        self.applyfunc(lambda m: collect(m, 1/ohm))
    def to_Z(self):
        return Impedance(self.inv())

    def to_Y(self):
        return self

    def to_S(self, z=50*ohm):
        return self.to_Z().to_S(z)

    def to_T(self):
        return self.to_Z().to_T()

class Scattering(Matrix):
    def __init__(self, *args, **kwargs):
        self.z0 = kwargs.pop('z0', 50*ohm)
        super(Scattering, self).__init__(*args, **kwargs)

    def to_Z(self):
        return Impedance((1 + self)*((1 - self).inv())*self.z0)

    def to_Y(self):
        return self.to_Z().to_Y()

    def to_T(self):
        return self.to_Z().to_T()

    def to_S(self, z=50*ohm):
        self.z0 = z
        return self

class Transmission(Matrix):
    def __init__(self, m, **kwargs):
        (a, b), (c, d) = m
        a = cancel(a)
        b = collect(b, ohm)
        c = collect(c, 1/ohm)
        d = cancel(d)
        super(Transmission, self).__init__([[a, b], [c, d]], **kwargs)
        if self.shape != (2, 2):
            raise ValueError("Transmission Matrix must be 2x2")

    def to_Z(self):
        (a, b), (c, d) = self.tolist()
        return Impedance([[a/c, (a*d-b*c)/c], [1/c, d/c]]).applyfunc(cancel)

    def to_Y(self):
        (a, b), (c, d) = self.to_list()
        return Admittance([[d/b, (b*c-a*d)/b], [-1/b, a/b]]).applyfunc(cancel)

    def to_S(self, z=50*ohm):
        (a, b), (c, d) = self.tolist()
        denom = a + cancel(b/z) + cancel(c*z) + d
        return Scattering([[a + b/z-c*z-d, 2*(a*d-b*c)], [2, -a + b/z - c*z + d]], z0=z)/denom

    def to_T(self):
        return self


def chain(*args):
    return reduce(op.mul, (a.to_T() for a in args))

def series(z):
    return Transmission([[1, z], [0/ohm, 1]])

def shunt(y):
    return Transmission([[1, 0*ohm], [y, 1]])

def t_network_silly(za, zb, zc):
    return chain(series(za), shunt(zb), series(zc))

def t_network(series1, shunt, series2):
    z11 = collect(series1 + shunt, ohm)
    z22 = collect(series2 + shunt, ohm)
    return Impedance([[z11, shunt], [shunt, z22]])

def eval_freqs(expr, start, stop, n=100):
    print(expr)
    expr = simplify(expr)
    print(expr)
    try:
        start = float(start / Hz)
    except TypeError:
        pass
    try:
        stop = float(stop / Hz)
    except TypeError:
        pass
    freqs = np.linspace(start, stop, n)
    return freqs, lambdify(FREQ/Hz, expr, "numpy")(freqs)

def test_units():
    c = capacitance(10 * pF).subs(FREQ/Hz, 1)
    l = inductance(10 * nH).subs(FREQ/Hz, 1)
    r = resonator_series_fzq(9 * gHz, 50*ohm, 1e6).subs(FREQ/Hz, 1)
    try:
        complex(c / ohm)
    except:
        print(c)
        raise
    try:
        complex(l / ohm)
    except:
        print(l)
        raise
    try:
        complex(r / ohm)
    except:
        print(r)
        raise

def test_make_resonator():
    coupler = capacitance(10 * fF)
    res = resonator_parallel_fzq(9 * gHz, 50*ohm, 1000)
    netwk = t_network(coupler, res, coupler)
    smat = netwk.to_S()

    s21 = smat[1,0]
    freqs, trace = eval_freqs(s21, 3*gHz, 15*gHz)
    plt.plot(freqs, np.log(np.abs(trace)))
    plt.show()

    s11 = smat[0,0]
    freqs, trace = eval_freqs(s11, 8*gHz, 10*gHz)
    plot_smith(trace)
    plt.show()
    plt.plot(freqs, np.angle(trace))
    plt.show()

if __name__ == "__main__":
    test_units()
    test_make_resonator()
