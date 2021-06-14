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
    eqs = [(eq1, eq1_), (eq2, eq2_), (eq3, eq3_), (eq4, eq4_), (eq5, eq5_)]
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
    y_approx = None
    y_analytic = None
    elapsed = None
    tmax = 20 # A value greater than 0

    if(option == 1):
        x_axis = list()

        y_axis_mean_error_eulers_method = list()
        y_axis_mean_error_taylor_series_method = list()
        y_axis_mean_error_runge_kutta_2_method = list()
        y_axis_mean_error_runge_kutta_4_method = list()
        y_axis_mean_error_two_steps_method = list()
        y_axis_mean_error_adams_bashforth = list()

        y_axis_error_std_eulers_method = list()
        y_axis_error_std_taylor_series_method = list()
        y_axis_error_std_runge_kutta_2_method = list()
        y_axis_error_std_runge_kutta_4_method = list()
        y_axis_error_std_two_steps_method = list()
        y_axis_error_std_adams_bashforth = list()

        running_time_eulers_method = list()
        running_time_taylor_series_method = list()
        running_time_runge_kutta_2_method = list()
        running_time_runge_kutta_4_method = list()
        running_time_two_steps_method = list()
        running_time_adams_bashforth = list()
        # Numerical calculation
        t0 = 0
        y0 = 1
        h_values = list(np.linspace(0.05, 2, num = 10))
        for h in h_values:
            print(h)
            x_axis.append(h)
            n = int(math.ceil(tmax/h)) # Number of steps after the initial value
            y_approx = None
            start = time.time()
            y_approx_eulers_method = eulers_method(eq_, t0, y0, h, n)
            end = time.time()
            elapsed_eulers_method = end - start
            start = time.time()
            y_approx_taylor_series_method = taylor_series_method(eq_, t0, y0, h, n)
            end = time.time()
            elapsed_taylor_series_method = end - start
            start = time.time()
            y_approx_runge_kutta_2_method = runge_kutta_2_method(eq_, t0, y0, h, n)
            end = time.time()
            elapsed_runge_kutta_2_method = end - start
            start = time.time()
            y_approx_runge_kutta_4_method = runge_kutta_4_method(eq_, t0, y0, h, n)
            end = time.time()
            elapsed_runge_kutta_4_method = end - start
            start = time.time()
            y_approx_two_steps_method = two_steps_method(eq_, t0, y0, h, n)
            end = time.time()
            elapsed_two_steps_method = end - start
            start = time.time()
            y_approx_adams_bashforth = adams_bashforth(eq_, t0, y0, h, n)
            end = time.time()
            elapsed_adams_bashforth = end - start
            
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

            error = abs_error(y_approx_eulers_method, y_analytic)
            mean_error = np.mean(error)
            error_std = np.std(error)
            y_axis_mean_error_eulers_method.append(mean_error)
            y_axis_error_std_eulers_method.append(error_std)
            running_time_eulers_method.append(elapsed_eulers_method)

            error = abs_error(y_approx_taylor_series_method, y_analytic)
            mean_error = np.mean(error)
            error_std = np.std(error)
            y_axis_mean_error_taylor_series_method.append(mean_error)
            y_axis_error_std_taylor_series_method.append(error_std)
            running_time_taylor_series_method.append(elapsed_taylor_series_method)

            error = abs_error(y_approx_runge_kutta_2_method, y_analytic)
            mean_error = np.mean(error)
            error_std = np.std(error)
            y_axis_mean_error_runge_kutta_2_method.append(mean_error)
            y_axis_error_std_runge_kutta_2_method.append(error_std)
            running_time_runge_kutta_2_method.append(elapsed_runge_kutta_2_method)

            error = abs_error(y_approx_runge_kutta_4_method, y_analytic)
            mean_error = np.mean(error)
            error_std = np.std(error)
            y_axis_mean_error_runge_kutta_4_method.append(mean_error)
            y_axis_error_std_runge_kutta_4_method.append(error_std)
            running_time_runge_kutta_4_method.append(elapsed_runge_kutta_4_method)

            error = abs_error(y_approx_two_steps_method, y_analytic)
            mean_error = np.mean(error)
            error_std = np.std(error)
            y_axis_mean_error_two_steps_method.append(mean_error)
            y_axis_error_std_two_steps_method.append(error_std)
            running_time_two_steps_method.append(elapsed_two_steps_method)

            error = abs_error(y_approx_adams_bashforth, y_analytic)
            mean_error = np.mean(error)
            error_std = np.std(error)
            y_axis_mean_error_adams_bashforth.append(mean_error)
            y_axis_error_std_adams_bashforth.append(error_std)
            running_time_adams_bashforth.append(elapsed_adams_bashforth)
        
        # Plotting
        fig1, ax1 = plt.subplots()
        ax1.plot(x_axis, y_axis_mean_error_eulers_method, label="Euler's method")
        ax1.plot(x_axis, y_axis_mean_error_taylor_series_method, label="Taylor series method")
        ax1.plot(x_axis, y_axis_mean_error_runge_kutta_2_method, label="Runge-Kutta order 2 method")
        ax1.plot(x_axis, y_axis_mean_error_runge_kutta_4_method, label="Runge-Kutta order 4 method")
        ax1.plot(x_axis, y_axis_mean_error_two_steps_method, label="Two steps method")
        ax1.plot(x_axis, y_axis_mean_error_adams_bashforth, label="Adams-Bashforth method")
        ax1.legend()
        ax1.set_xlabel("h")
        ax1.set_ylabel("Mean error")

        fig2, ax2 = plt.subplots()
        ax2.plot(x_axis, y_axis_error_std_eulers_method, label="Euler's method")
        ax2.plot(x_axis, y_axis_error_std_taylor_series_method, label="Taylor series method")
        ax2.plot(x_axis, y_axis_error_std_runge_kutta_2_method, label="Runge-Kutta order 2 method")
        ax2.plot(x_axis, y_axis_error_std_runge_kutta_4_method, label="Runge-Kutta order 4 method")
        ax2.plot(x_axis, y_axis_error_std_two_steps_method, label="Two steps method")
        ax2.plot(x_axis, y_axis_error_std_adams_bashforth, label="Adams-Bashforth method")
        ax2.legend()
        ax2.set_xlabel("h")
        ax2.set_ylabel("Standard deviation")
    
        fig3, ax3 = plt.subplots()
        ax3.plot(x_axis, running_time_eulers_method, label="Euler's method")
        ax3.plot(x_axis, running_time_taylor_series_method, label="Taylor series method")
        ax3.plot(x_axis, running_time_runge_kutta_2_method, label="Runge-Kutta order 2 method")
        ax3.plot(x_axis, running_time_runge_kutta_4_method, label="Runge-Kutta order 4 method")
        ax3.plot(x_axis, running_time_two_steps_method, label="Two steps method")
        ax3.plot(x_axis, running_time_adams_bashforth, label="Adams-Bashforth method")
        ax3.legend()
        ax3.set_xlabel("h")
        ax3.set_ylabel("Time (seconds)")

        plt.show()

    elif(option == 2):
        order = int(input("Please type the order of the ODE: "))

        # Numerical calculation
        initial_values = [None for _ in range(order)]
        for i in range(order):
            initial_values[i] = (0, random.randint(1, 6)) # The t value must be the same for all the equations
        print("Initial values:", initial_values)
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
        print(y)
        y_analytic = eval_func(y, t)
    elif(option == 3):
        order = 2
        # Numerical calculation
        t0, y0 = 0, 0
        tn, yn = 20, 1
        n = 5
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
        # Analytical calculation
        t = list(np.linspace(t0, tn, num = n))
        y = None
        for i in range(order):
            eq_C = sym.Eq(y0, sym.dsolve(eq).rhs.subs(sym.Symbol('t'), t0))
            C = sym.solvers.solve(eq_C, sym.Symbol('C1'))[0]
            eq = sym.Eq(dydt, sym.dsolve(eq).rhs.subs(sym.Symbol('C1'), C))
            if(i == order - 1):
                y = eq.rhs
        print(y)
        y_analytic = eval_func(y, t)
    return 0

if __name__ == '__main__':
    main()