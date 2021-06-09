from inspect import FullArgSpec
from sys import stdin
import sympy as sym
import numpy as np
import matplotlib.pyplot as ptl
import time
import random
from sympy.core.numbers import E
from ode import eulers_method, taylor_series_method, runge_kutta_2_method, runge_kutta_4_method, two_steps_method, adams_bashforth, higher_order_method
from utils import eval_func

def main():
    t = sym.Symbol('t')
    y = sym.Function('y')(t)
    dydt = y.diff(t)
    eq1, eq1_ = sym.Eq(dydt, y), sym.Symbol('y')
    eq2, eq2_ = sym.Eq(dydt, -2*t*(y**2)), -2*t*(sym.Symbol('y')**2)
    eq3, eq3_ = sym.Eq(dydt, 2*t), 2*t
    eq4, eq4_ = sym.Eq(dydt, 1 + t**2), 1 + t**2
    eq5, eq5_ = sym.Eq(dydt, 1 + (t**3)/3 + t), 1 + (t**3)/3 + t
    eq6, eq6_ = sym.Eq(dydt, 2*t), 2*t
    eqs = [(eq1, eq1_), (eq2, eq2_), (eq3, eq3_), (eq4, eq4_), (eq5, eq5_), (eq6, eq6_)]
    i = 1
    for eq in eqs:
        print(f"{i}. {eq[0].lhs} = {eq[0].rhs}")
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
        option = int(input("Please choose the kind of ODE you are working with: "))
        if(option < 1 or option > 2):
            print("Invalid option. Please try again")
        else:
            good = True
    if(option == 1):
        # Numerical calculation
        t0 = 0
        y0 = 1
        n = 3 # Number of steps after the initial value
        h = 0.005
        print("1. Euler's method")
        print("2. Taylor series method")
        print("3. Runge-Kutta order 2 method")
        print("4. Runge-Kutta order 4 method")
        print("5. Two steps method")
        print("6. Adams Bashforth method")
        good = False
        option = None
        while(not good):
            option = int(input("Please choose the method you want to work with: "))
            if(option < 1 or option > 6):
                print("Invalid option. Please try again")
            else:
                good = True
        y_approx = None
        if(option == 1):
            y_approx = eulers_method(eq_, t0, y0, h, n)
        elif(option == 2):
            y_approx = taylor_series_method(eq_, t0, y0, h, n)
        elif(option == 3):
            y_approx = runge_kutta_2_method(eq_, t0, y0, h, n)
        elif(option == 4):
            y_approx = runge_kutta_4_method(eq_, t0, y0, h, n)
        elif(option == 5):
            y_approx = two_steps_method(eq_, t0, y0, h, n)
        elif(option == 6):
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
        order = int(input("Please type the order of the ODE: "))

        # Numerical calculation
        initial_values = [None for _ in range(order)]
        for i in range(order):
            initial_values[i] = (0, random.randint(1, 6))
        print("Initial values:", initial_values)
        n = 3 # Number of steps after the initial value
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
        print(y_approx)
        print(y_analytic)
        
    
    return 0

if __name__ == '__main__':
    main()