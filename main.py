from inspect import FullArgSpec
from sys import stdin
import sympy as sym
import numpy as np
import matplotlib.pyplot as ptl
import time

from sympy.core.numbers import E
from ode import eulers_method, taylor_series_method, runge_kutta_2_method, runge_kutta_4_method, two_steps_method, adams_bashforth, higher_order_method
from utils import eval_func

def main():
    t = sym.Symbol('t')
    y = sym.Function('y')(t)
    x = sym.Function('x')(t)
    dydt = y.diff(t)
    dy2dt2 = y.diff(t).diff(t)
    dxdt = x.diff(t)
    eq1, eq1_ = sym.Eq(dydt, y), sym.Symbol('y')
    eq2, eq2_ = sym.Eq(dydt, -2*t*(y**2)), -2*t*(sym.Symbol('y')**2)
    eq3, eq3_ = sym.Eq(dydt, 2*t), 2*t
    eq4, eq4_ = sym.Eq(dydt, 1 + t**2), 1 + t**2
    eq5, eq5_ = sym.Eq(dydt, 1 + (t**3)/3 + t), 1 + (t**3)/3 + t
    eq6, eq6_ = sym.Eq(dydt, 2*t + y), 2*t
    eqs = [(eq1, eq1_), (eq2, eq2_), (eq3, eq3_), (eq4, eq4_), (eq5, eq5_), (eq6, eq6_)]
    i = 1
    for eq in eqs:
        print(f"{i}. {eq[0]}")
        i += 1
    good = False
    option = None
    while(not good):
        option = int(input("Please choose the equation you want to work with: ")); option -= 1
        if(option < 0 or option >= len(eqs)):
            print("Invalid option. Please try again")
        else:
            good = True
    eq = eqs[option][0]
    eq_ = eqs[option][1]

    print("1. PVI for first order ODE")
    print("2. PVI for higher order ODE")
    good = False
    option = None
    while(not good):
        option = int(input("Please choose the kind of equation you are working with: "))
        if(option < 1 or option > 2):
            print("Invalid option. Please try again")
        else:
            good = True
    if(option == 1):
        # Numerical calculation
        t0 = 0
        y0 = 1
        n = 3 # Number of steps
        h = 0.005
        y_approx = adams_bashforth(eq_, t0, y0, h, n)

        # Analytical calculation
        t = [None for _ in range(n + 1)]
        t[0] = t0
        tk = t0 + h
        k = 1
        while(k <= n):
            t[k] = tk
            tk += h
            k += 1
        eq_C1 = sym.Eq(y0, sym.dsolve(eq).rhs.subs(sym.Symbol('t'), t0))
        C1 = sym.solvers.solve(eq_C1, sym.Symbol('C1'))[0]
        y = sym.dsolve(eq).rhs.subs(sym.Symbol('C1'), C1)
        y_analytic = eval_func(y, t)

        print(t)
        print(y_approx)
        print(y_analytic)
    else:
        order = 4
        # Numerical calculation
        initial_values = [(0, 2), (0, 1), (0, 3), (0, 6)]
        n = 10 # Number of steps
        h = 0.0005
        y_approx = higher_order_method(eq_, initial_values, h, n, order)

        # Analytical calculation
        t = [None for _ in range(n + 1)]
        t[0] = initial_values[0][0]
        tk = initial_values[0][0] + h
        k = 1
        while(k <= n):
            t[k] = tk
            tk += h
            k += 1
        y = None
        for i in range(order):
            t0 = initial_values[i][0]
            y0 = initial_values[i][1]
            eq_C = sym.Eq(y0, sym.dsolve(eq).rhs.subs(sym.Symbol('t'), t0))
            C = sym.solvers.solve(eq_C, sym.Symbol('C1'))[0]
            eq = sym.Eq(dydt, sym.dsolve(eq).rhs.subs(sym.Symbol('C1'), C))
            if(i == order - 1):
                y = eq.rhs
        y_analytic = eval_func(y, t)
        print(t)
        print(y_approx)
        print(y_analytic)
        
    
    return 0

if __name__ == '__main__':
    main()