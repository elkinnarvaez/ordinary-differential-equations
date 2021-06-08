import sympy as sym

def eulers_method(f, t0, y0, h, n):
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
    t = [None for _ in range(n + 1)]
    t[0] = initial_values[0][0]
    for i in range(1, n + 1):
        t[i] = t[i - 1] + h
    t0 = initial_values[0][0]
    y0 = initial_values[0][1]
    y_prev = runge_kutta_4_method(f, t0, y0, h, n)
    y_curr = [None for _ in range(n + 1)]
    for i in range(1, order):
        t0 = initial_values[i][0]
        y0 = initial_values[i][1]
        y_curr[0] = y0
        k = 1
        while(k <= n):
            yk = float(y_curr[k - 1] + y_prev[k - 1]*h)
            y_curr[k] = yk
            k += 1
        y_prev = y_curr.copy()
    return y_curr