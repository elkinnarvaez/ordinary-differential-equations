import sympy as sym
import numpy as np
from utils import build_polynomial_matrix, evaluate_polynomial


def eulers_method(f, t0, y0, h, n):
    """
        Input:
            f: A function f that represents the right-hand-side of the ordinary differential equation
            t0: Initial value of t
            y0: Initial value of y
            h: An increment value
            n: Number of steps after the initial value t
        Output:
            Approximation of the solution of the IVP for the ordinary differential equation for the values t0, t0 + h, t0 + 2h, ... , t0 + nh using the Euler's method
    """
    t = [None for _ in range(n + 1)]
    y = [None for _ in range(n + 1)]
    t[0] = t0
    y[0] = y0
    tk = t0 + h
    k = 1
    while(k <= n):
        yk = float(y[k - 1] + f.subs([(sym.Symbol('t'), t[k - 1]), (sym.Symbol('y'), y[k - 1])]).evalf()*h)
        t[k] = tk
        y[k] = yk
        k += 1
        tk += h
    return y

def taylor_series_method(f, t0, y0, h, n):
    """
        Input:
            f: A function f that represents the right-hand-side of the ordinary differential equation
            t0: Initial value of t
            y0: Initial value of y
            h: An increment value
            n: Number of steps after the initial value t
        Output:
            Approximation of the solution of the IVP for the ordinary differential equation for the values t0, t0 + h, t0 + 2h, ... , t0 + nh using the Taylor series method
    """
    t = [None for _ in range(n + 1)]
    y = [None for _ in range(n + 1)]
    t[0] = t0
    y[0] = y0
    tk = t0 + h
    k = 1
    f_derivative = sym.diff(f, sym.Symbol('t')) + sym.diff(f, sym.Symbol('y'))*f
    while(k <= n):
        yk = float(y[k - 1] + f.subs([(sym.Symbol('t'), t[k - 1]), (sym.Symbol('y'), y[k - 1])]).evalf()*h + (f_derivative.subs([(sym.Symbol('t'), t[k - 1]), (sym.Symbol('y'), y[k - 1])]).evalf()/2)*(h**2))
        t[k] = tk
        y[k] = yk
        k += 1
        tk += h
    return y

def runge_kutta_2_method(f, t0, y0, h, n):
    """
        Input:
            f: A function f that represents the right-hand-side of the ordinary differential equation
            t0: Initial value of t
            y0: Initial value of y
            h: An increment value
            n: Number of steps after the initial value t
        Output:
            Approximation of the solution of the IVP for the ordinary differential equation for the values t0, t0 + h, t0 + 2h, ... , t0 + nh using the method of Runge-Kutta of order 2
    """
    t = [None for _ in range(n + 1)]
    y = [None for _ in range(n + 1)]
    t[0] = t0
    y[0] = y0
    tk = t0 + h
    k = 1
    while(k <= n):
        k1 = float(f.subs([(sym.Symbol('t'), t[k - 1]), (sym.Symbol('y'), y[k - 1])]).evalf()*h)
        k2 = float(f.subs([(sym.Symbol('t'), t[k - 1] + h), (sym.Symbol('y'), y[k - 1] + k1)]).evalf()*h)
        yk = float(y[k - 1] + (k1 + k2)/2)
        t[k] = tk
        y[k] = yk
        k += 1
        tk += h
    return y

def runge_kutta_4_method(f, t0, y0, h, n):
    """
        Input:
            f: A function f that represents the right-hand-side of the ordinary differential equation
            t0: Initial value of t
            y0: Initial value of y
            h: An increment value
            n: Number of steps after the initial value t
        Output:
            Approximation of the solution of the IVP for the ordinary differential equation for the values t0, t0 + h, t0 + 2h, ... , t0 + nh using the method of Runge-Kutta of order 4
    """
    t = [None for _ in range(n + 1)]
    y = [None for _ in range(n + 1)]
    t[0] = t0
    y[0] = y0
    tk = t0 + h
    k = 1
    while(k <= n):
        k1 = float(f.subs([(sym.Symbol('t'), t[k - 1]), (sym.Symbol('y'), y[k - 1])]).evalf()*h)
        k2 = float(f.subs([(sym.Symbol('t'), t[k - 1] + h/2), (sym.Symbol('y'), y[k - 1] + k1/2)]).evalf()*h)
        k3 = float(f.subs([(sym.Symbol('t'), t[k - 1] + h/2), (sym.Symbol('y'), y[k - 1] + k2/2)]).evalf()*h)
        k4 = float(f.subs([(sym.Symbol('t'), t[k - 1] + h), (sym.Symbol('y'), y[k - 1] + k3)]).evalf()*h)
        yk = float(y[k - 1] + (k1 + 2*k2 + 2*k3 + k4)/6)
        t[k] = tk
        y[k] = yk
        k += 1
        tk += h
    return y

def two_steps_method(f, t0, y0, h, n):
    """
        Input:
            f: A function f that represents the right-hand-side of the ordinary differential equation
            t0: Initial value of t
            y0: Initial value of y
            h: An increment value
            n: Number of steps after the initial value t
        Output:
            Approximation of the solution of the IVP for the ordinary differential equation for the values t0, t0 + h, t0 + 2h, ... , t0 + nh using the two steps method
    """
    t = [None for _ in range(n + 1)]
    y = [None for _ in range(n + 1)]
    ans = runge_kutta_4_method(f, t0, y0, h, 1)
    t[0] = t0
    y[0] = ans[0]
    t[1] = t[0] + h
    y[1] = ans[1]
    tk = t[1] + h
    k = 2
    while(k <= n):
        yk = float(y[k - 1] + (1/2)*h*(3*f.subs([(sym.Symbol('t'), t[k - 1]), (sym.Symbol('y'), y[k - 1])]).evalf() - f.subs([(sym.Symbol('t'), t[k - 2]), (sym.Symbol('y'), y[k - 2])]).evalf()))
        t[k] = tk
        y[k] = yk
        k += 1
        tk += h
    return y

def adams_bashforth(f, t0, y0, h, n):
    """
        Input:
            f: A function f that represents the right-hand-side of the ordinary differential equation
            t0: Initial value of t
            y0: Initial value of y
            h: An increment value
            n: Number of steps after the initial value t
        Output:
            Approximation of the solution of the IVP for the ordinary differential equation for the values t0, t0 + h, t0 + 2h, ... , t0 + nh using the Adams-Bashforth method
    """
    t = [None for _ in range(n + 1)]
    y = [None for _ in range(n + 1)]
    ans = runge_kutta_4_method(f, t0, y0, h, 3)
    t[0] = t0
    y[0] = ans[0]
    t[1] = t[0] + h
    y[1] = ans[1]
    t[2] = t[1] + h
    y[2] = ans[2]
    t[3] = t[2] + h
    y[3] = ans[3]
    tk = t[3] + h
    k = 4
    while(k <= n):
        yk = float(y[k - 1] + (1/24)*h*(55*f.subs([(sym.Symbol('t'), t[k - 1]), (sym.Symbol('y'), y[k - 1])]).evalf() - 59*f.subs([(sym.Symbol('t'), t[k - 2]), (sym.Symbol('y'), y[k - 2])]).evalf() + 37*f.subs([(sym.Symbol('t'), t[k - 3]), (sym.Symbol('y'), y[k - 3])]).evalf() - 9*f.subs([(sym.Symbol('t'), t[k - 4]), (sym.Symbol('y'), y[k - 4])]).evalf()))
        t[k] = tk
        y[k] = yk
        k += 1
        tk += h
    return y

def higher_order_method(f, initial_values, h, n, order):
    """
        Input:
            f: A function f that represents the right-hand-side of the ordinary differential equation
            initial_values: An array that contains the initial values of the different differential equations
            h: An increment value
            n: Number of steps after the initial value t
            order: The order of the differential equation
        Output:
            Approximation of the solution of the IVP for the higher order ordinary differential equation for the values t0, t0 + h, t0 + 2h, ... , t0 + nh
    """
    t = [None for _ in range(n + 1)]
    t[0] = initial_values[0][0]
    for i in range(1, n + 1):
        t[i] = t[i - 1] + h
    t0 = initial_values[0][0]
    y0 = initial_values[0][1]
    y_prev = eulers_method(f, t0, y0, h, n) # Euler's method
    y_curr = [None for _ in range(n + 1)]
    for i in range(1, order):
        t0 = initial_values[i][0]
        y0 = initial_values[i][1]
        y_curr[0] = y0
        k = 1
        while(k <= n):
            yk = float(y_curr[k - 1] + y_prev[k - 1]*h) # Euler's method
            y_curr[k] = yk
            k += 1
        y_prev = y_curr.copy()
    return y_curr

def finite_difference_method(f, t0, tn, y0, yn, n):
    """
        Input:
            f: A function f that represents the right-hand-side of the ordinary differential equation
            t0: Initial value of t
            tn: Final value of t
            y0: Initial value of y
            yn: Final value of y
            n: Number of points in the interval [t0, tn]
        Output:
            Approximation of the solution of the BVP for the 2-order ordinary differential equation for the values t0, t0 + (tn - t0)/(n - 1), t0 + 2(tn - t0)/(n - 1), ... , tn using the finite difference method
    """
    h = (tn - t0)/(n - 1)
    t = list(np.linspace(t0, tn, num = n))
    y = [None for _ in range(n)]
    y[0] = y0
    y[n - 1] = yn
    for i in range(1, n - 1):
        y[i] = sym.Symbol(f'y{i}')
    A = np.array([[0 for _ in range(n - 2)] for _ in range(n - 2)], dtype = 'float')
    b = np.array([[0] for _ in range(n - 2)], dtype = 'float')
    for i in range(1, n - 1):
        ti = t[i]
        if(i == n - 2):
            A[i - 1][(i - 1) - 1] = 1
            A[i - 1][i - 1] = -2
            b[i - 1][0] = float(f.subs(sym.Symbol('t'), ti).evalf())*(h**2) - y[i + 1]
        elif(i == 1):
            A[i - 1][i - 1] = -2
            A[i - 1][(i + 1) - 1] = 1
            b[i - 1][0] = float(f.subs(sym.Symbol('t'), ti).evalf())*(h**2) - y[i - 1]
        else:
            A[i - 1][(i - 1) - 1] = 1
            A[i - 1][i - 1] = -2
            A[i - 1][(i + 1) - 1] = 1
            b[i - 1][0] = float(f.subs(sym.Symbol('t'), ti).evalf())*(h**2)
    x = np.linalg.solve(A, b)
    for i in range(1, n - 1):
        y[i] = float(x[i - 1][0])
    return y

def finite_element_method(f, t0, tn, y0, yn, n):
    """
        Input:
            f: A function f that represents the right-hand-side of the ordinary differential equation
            t0: Initial value of t
            tn: Final value of t
            y0: Initial value of y
            yn: Final value of y
            n: Number of points in the interval [t0, tn]
        Output:
            Approximation of the solution of the BVP for the 2-order ordinary differential equation for the values t0, t0 + (tn - t0)/(n - 1), t0 + 2(tn - t0)/(n - 1), ... , tn using the finite element method
    """
    t = list(np.linspace(t0, tn, num = n))
    A = np.array([[0 for _ in range(n)] for _ in range(n)], dtype = 'float')
    b = np.array([[0] for _ in range(n)], dtype = 'float')
    for i in range(n):
        A[0][i] = t[0]**i
    b[0][0] = y0
    for i in range(n):
        A[n - 1][i] = t[n - 1]**i
    b[n - 1][0] = yn
    for i in range(1, n - 1):
        A[i][0] = 0
        A[i][1] = 0
        for j in range(n - 2):
            coeff = (j + 2)*(j + 1)
            A[i][j + 2] = coeff*(t[i]**j)
        b[i][0] = float(f.subs(sym.Symbol('t'), t[i]).evalf())
    x = np.linalg.solve(A, b)
    # Polynomial evaluation
    y = list(evaluate_polynomial(t, x).T[0])
    return y