from inspect import FullArgSpec
from sys import stdin
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
from sympy.core.numbers import E
from sympy.polys.numberfields import to_number_field
from ode import eulers_method, taylor_series_method, runge_kutta_2_method, runge_kutta_4_method, two_steps_method, adams_bashforth, higher_order_method, finite_difference_method, finite_element_method
from utils import eval_func, abs_error

def main():
    t = sym.Symbol('t')
    y = sym.Function('y')(t)
    dydt = y.diff(t)
    eq1, eq1_ = sym.Eq(dydt, t/2 + t**2 + sym.exp(t)*sym.sin(t)), t/2 + t**2 + sym.exp(t)*sym.sin(t)
    eq2, eq2_ = sym.Eq(dydt, 2*t**2 + 10), 2*t**2 + 10
    eq3, eq3_ = sym.Eq(dydt, sym.sin(t**2)), sym.sin(t**2)
    eq4, eq4_ = sym.Eq(dydt, sym.cos(t) + 10),  sym.cos(t) + 10
    eq5, eq5_ = sym.Eq(dydt, 2*y), 2*sym.Symbol('y')
    eq6, eq6_ = sym.Eq(dydt, 6*t), 6*t
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

    print("1. IVP for first order ODEs")
    print("2. IVP for higher order ODEs")
    print("3. FVP (for 2-order ODEs)")
    good = False
    option = None
    while(not good):
        option = int(input("Please choose the kind of problem you want to solve: "))
        if(option < 1 or option > 3):
            print("Invalid option. Please try again")
        else:
            good = True
    t = None
    t_analytic = None
    y_approx = None
    y_analytic = None
    y_analytic_error = None
    elapsed = None
    tmax = 20 # A value greater than 0
    h = 0.5
    if(option == 1):
        # Numerical calculation
        t0 = 0
        y0 = 1
        n = int(math.ceil(tmax/h)) # Number of steps after the initial value
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
        start = time.time()
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
        end = time.time()
        elapsed = end - start

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
        t_analytic = list(np.linspace(t0, tmax, num = 1000))
        y_analytic = eval_func(y, t_analytic)
        y_analytic_error = eval_func(y, t)
    elif(option == 2):
        order = int(input("Please type the order of the ODE: "))

        # Numerical calculation
        initial_values = [(0, 6), (0, 2), (0, 2), (0, 1)]
        # for i in range(order):
        #     initial_values[i] = (0, random.randint(1, 6)) # The t value must be the same for all the equations
        # print("Initial values:", initial_values)
        n = int(math.ceil(tmax/h)) # Number of steps after the initial value
        start = time.time()
        y_approx = higher_order_method(eq_, initial_values, h, n, order)
        end = time.time()
        elapsed = end - start

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
        t_analytic = list(np.linspace(initial_values[0][0], tmax, num = 1000))
        y_analytic = eval_func(y, t_analytic)
        y_analytic_error = eval_func(y, t)
    elif(option == 3):
        order = 2
        # Numerical calculation
        t0, y0 = 0, 0
        tn, yn = 20, None
        n = 100

        # Analytical calculation
        t = list(np.linspace(t0, tn, num = n))
        y = None
        for i in range(order):
            eq_C = sym.Eq(y0, sym.dsolve(eq).rhs.subs(sym.Symbol('t'), t0))
            C = sym.solvers.solve(eq_C, sym.Symbol('C1'))[0]
            eq = sym.Eq(dydt, sym.dsolve(eq).rhs.subs(sym.Symbol('C1'), C))
            if(i == order - 1):
                y = eq.rhs
        t_analytic = list(np.linspace(t0, tn, num = 1000))
        y_analytic = eval_func(y, t_analytic)
        y_analytic_error = eval_func(y, t)
        yn = y_analytic[-1]

        print("1. Finite difference method")
        print("2. Finite element method")
        good = False
        option = None
        while(not good):
            option = int(input("Please choose the method you want to work with: "))
            if(option < 1 or option > 2):
                print("Invalid option. Please try again")
            else:
                good = True
        y_approx = None
        start = time.time()
        if(option == 1):
            y_approx = finite_difference_method(eq_, t0, tn, y0, yn, n)
        elif(option == 2):
            y_approx = finite_element_method(eq_, t0, tn, y0, yn, n)
        end = time.time()
        elapsed = end - start
    error = abs_error(y_approx, y_analytic_error)
    mean_error = np.mean(error)
    error_std = np.std(error)
    print(f"Mean error: {mean_error}")
    print(f"Error std: {error_std}")
    print(f"Time: {elapsed} sec")
    # Plotting
    fig1, ax1 = plt.subplots()
    ax1.plot(t, y_approx, label="Approximate solution")
    ax1.plot(t_analytic, y_analytic, label="Analytic solution")
    ax1.legend()
    ax1.set_xlabel("t")
    ax1.set_ylabel("y(t)")
    scalar_offset = 0 # 1.3
    x0_1, xmax_1 = plt.xlim()
    y0_1, ymax_1 = plt.ylim()
    text_x_pos_1 = 10.1
    text_y_pos_1 = y0_1 + (ymax_1 - y0_1) + scalar_offset
    ax1.text(text_x_pos_1, text_y_pos_1, "Mean error: {:e} \nStandard deviation: {:e} \nRunning time: {:e} sec".format(mean_error, error_std, elapsed), bbox=dict(boxstyle="round",
                ec=(1., 0.5, 0.5),
                fc=(1., 0.9, 0.8),
                ))
    plt.show()
    return 0

if __name__ == '__main__':
    main()