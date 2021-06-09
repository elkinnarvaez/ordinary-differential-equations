import sympy as sym
import numpy as np

def eval_func(f, t):
    ans = list()
    for tk in t:
        ans.append(float(f.subs(sym.Symbol('t'), tk).evalf()))
    return ans

def build_polynomial_matrix(n, t):
    A = np.array([[None for _ in range(n)] for _ in range(n)], dtype='float')
    for i in range(n):
        for j in range(n):
            A[i][j] = t[i]**j
    return A

def evaluate_polynomial(t, x):
    n = len(x)
    A = build_polynomial_matrix(n, t)
    b = np.matmul(A, x)
    return b

def abs_error(a, b):
    n = len(a)
    c = [None for _ in range(n)]
    for i in range(n):
        c[i] = abs(a[i] - b[i])
    return c