from sys import stdin
import sympy as sym
import numpy as np
import matplotlib.pyplot as ptl
import time
from ode import eulers_method, taylor_series_method
from utils import eval_func

def main():
    t = sym.Symbol('t')
    y = sym.Function('y')(t)
    dydt = y.diff(t)
    eq1, eq1_ = sym.Eq(dydt, -2*t*(y**2)), -2*t*(sym.Symbol('y')**2)
    eq2, eq2_ = sym.Eq(dydt, y), sym.Symbol('y')
    eqs = [(eq1, eq1_), (eq2, eq2_)]
    i = 1
    for eq in eqs:
        print(f"{i}. {eq[0]}")
        i += 1
    good = False
    option = None
    while(not good):
        option = int(input("Please choose the function you want to work with: ")); option -= 1
        if(option < 0 or option >= len(eqs)):
            print("Invalid option. Please try again")
        else:
            good = True
    eq = eqs[option][0]
    eq_ = eqs[option][1]

    # Numerical calculation
    t0 = 0
    y0 = 1
    n = 3
    h = 0.5
    y_approx = eulers_method(eq_, t0, y0, h, n)

    # Analytical calculation
    t_ans = [None for _ in range(n + 1)]
    t_ans[0] = t0
    tk = t0 + h
    k = 1
    while(k <= n):
        t_ans[k] = tk
        tk += h
        k += 1

    eq_C1 = sym.Eq(y0, sym.dsolve(eq).rhs.subs(sym.Symbol('t'), t0))
    C1 = sym.solvers.solve(eq_C1, sym.Symbol('C1'))[0]
    f_analytic = sym.dsolve(eq).rhs.subs(sym.Symbol('C1'), C1)
    y_analytic = eval_func(f_analytic, t_ans)

    print(t_ans)
    print(y_approx)
    print(y_analytic)
    
    return 0

if __name__ == '__main__':
    main()