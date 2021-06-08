import sympy as sym

def euler_method(f, t0, y0, h, n):
    t = [None for _ in range(n + 1)]
    y = [None for _ in range(n + 1)]
    t[0] = t0
    y[0] = y0
    tk = t0 + h
    k = 1
    while(k <= n):
        yk = y[k - 1] + f.subs([(sym.Symbol('t'), t[k - 1]), (sym.Symbol('y'), y[k - 1])])*h
        t[k] = tk
        y[k] = yk
        k += 1
        tk += h
    return t, y