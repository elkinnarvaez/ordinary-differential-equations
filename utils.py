import sympy as sym

def eval_func(f, t):
    ans = list()
    for tk in t:
        ans.append(float(f.subs(sym.Symbol('t'), tk).evalf()))
    return ans
