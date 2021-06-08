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

def runge_kutta_method(f, t0, y0, h, n):
    