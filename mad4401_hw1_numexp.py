"""
mad4401_hw1_numexp.py

Single-file implementation of the numerical experiments requested:
- Bisection method experiments (Problems 1 & 2)
- Fixed-point iteration for g(x) = x - arctan(x) (Problem 3)
- Newton's method for f(x) = arctan(x) (Problem 6)
- Secant method for f(x) = arctan(x) (Problem 7)

Uses mpmath for arbitrary precision so we can handle extremely small tolerances.
"""

from mpmath import mp, mpf, atan, fabs, nstr
import math

# ---------- Configuration ----------
# Default high precision: this will be adjusted dynamically based on smallest tolerance used.
mp.dps = 80  # decimal digits; may be increased in functions based on requested tolerances

# Utility to ensure mp.dps large enough for a given smallest tolerance
def ensure_precision_for_tol(tol):
    try:
        tol = mp.mpf(tol)
    except Exception:
        tol = mp.mpf(str(tol))
    if tol <= 0:
        return
    digits_needed = max(20, int(mp.ceil(-mp.log10(tol))) + 10)
    if mp.dps < digits_needed:
        mp.dps = digits_needed

# ---------- Functions ----------

def f_arctan(x):
    return atan(x)

def df_arctan(x):
    # derivative of arctan(x) is 1 / (1 + x^2)
    return mp.mpf(1) / (mp.mpf(1) + x * x)

def bisection(f, a, b, tol, max_iter=10000, return_history=False):
    ensure_precision_for_tol(tol)
    a = mp.mpf(a); b = mp.mpf(b); tol = mp.mpf(tol)
    fa = f(a); fb = f(b)
    if fa == 0:
        return a, 0, [(0, a, b, a, fa)] if return_history else (a, 0)
    if fb == 0:
        return b, 0, [(0, a, b, b, fb)] if return_history else (b, 0)
    if fa * fb > 0:
        raise ValueError("Bisection precondition failed: f(a) and f(b) must have opposite signs.")
    history = []
    for k in range(1, max_iter + 1):
        mid = (a + b) / 2
        fmid = f(mid)
        history.append((k, a, b, mid, fmid))
        if fabs(b - a) / 2 <= tol or fmid == 0:
            if return_history:
                return mid, k, history
            else:
                return mid, k
        
        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid
    if return_history:
        return (mid, max_iter, history)
    else:
        return (mid, max_iter)

def predicted_bisection_iterations(a, b, tol):
    if tol <= 0:
        return float('inf')
    ratio = (b - a) / tol
    if ratio <= 0:
        return 0
    return int(math.ceil(math.log2(ratio)))

def fixed_point_iter(g, x0, n_iter=10):
    x = mp.mpf(x0)
    seq = [x]
    for k in range(1, n_iter + 1):
        x = g(x)
        seq.append(x)
    return seq

def newton_method(f, df, x0, tol=1e-10, max_iter=20):
    ensure_precision_for_tol(tol)
    x = mp.mpf(x0)
    records = [(0, x, f(x))]
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            records.append((k, x, fx))
            break
        x_new = x - fx / dfx
        records.append((k, x_new, f(x_new)))
        if fabs(f(x_new)) < mp.mpf(tol) or fabs(x_new - x) < mp.mpf(tol):
            break
        x = x_new
    return records

def secant_method(f, x0, x1, tol=1e-10, max_iter=20):
    ensure_precision_for_tol(tol)
    x_prev = mp.mpf(x0)
    x_curr = mp.mpf(x1)
    records = [(0, x_prev, f(x_prev)), (1, x_curr, f(x_curr))]
    for k in range(2, max_iter + 1):
        f_prev = f(x_prev)
        f_curr = f(x_curr)
        denom = (f_curr - f_prev)
        if denom == 0:
            records.append((k, x_curr, f_curr))
            break
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        records.append((k, x_next, f(x_next)))
        if fabs(x_next - x_curr) < mp.mpf(tol) or fabs(f(x_next)) < mp.mpf(tol):
            break
        x_prev, x_curr = x_curr, x_next
    return records

# ---------- Problem-specific experiments ----------

def problem_1_and_2_bisection_prints():
    print("=== Problem 1 (theory + k=5 manual bisection) and Problem 2 (computational experiments) ===")
    a = mp.mpf(-4.9)
    b = mp.mpf(5.1)
    print(f"Function: f(x) = arctan(x); initial interval [a,b] = [{a}, {b}]")

    fa = f_arctan(a); fb = f_arctan(b)
    print(f"f(a) = arctan({a}) = {nstr(fa,40)}")
    print(f"f(b) = arctan({b}) = {nstr(fb,40)}")
    sign_check = "opposite signs" if fa * fb < 0 else "same sign"
    print(f"Check: f(a) and f(b) have {sign_check}. (Hence bisection hypotheses are valid if they are opposite.)")
    if fa * fb >= 0:
        print("WARNING: f(a) and f(b) do not have opposite signs — bisection would not be guaranteed.")
    else:
        print("Since f(a) and f(b) have opposite signs and f is continuous, the bisection method hypotheses are satisfied.")
    
    tol_1 = 1e-2
    k_needed = predicted_bisection_iterations(a, b, tol_1)
    print(f"Predicted minimal bisection iterations to get error <= {tol_1}: k >= {k_needed}")
 
    print("\nManual (simulated) bisection steps for k = 1..5:")
    mid = None
    a_k, b_k = mp.mpf(a), mp.mpf(b)
    for k in range(1, 6):
        mid = (a_k + b_k) / 2
        f_mid = f_arctan(mid)
        print(f"k={k:2d}: interval = [{nstr(a_k,15)}, {nstr(b_k,15)}], mid = {nstr(mid,20)}, f(mid) = {nstr(f_mid,20)}")
        if f_arctan(a_k) * f_mid < 0:
            b_k = mid
        else:
            a_k = mid
   
    eps_list = [1e-2, 1e-4, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128]
    print("\nProblem 2: run bisection for epsilons:", eps_list)
    ensure_precision_for_tol(min(eps_list))
    for eps in eps_list:
        try:
            predicted_k = predicted_bisection_iterations(a, b, eps)
            root, iters = bisection(f_arctan, a, b, eps, max_iter=10000, return_history=False)
            print(f"eps = {eps:0.0e} | predicted k >= {predicted_k:3d} | achieved iterations = {iters:3d} | root ~ {nstr(root,30)}")
        except Exception as e:
            print(f"eps = {eps:0.0e} | ERROR: {e}")

def problem_3_fixed_point():
    print("\n=== Problem 3: Fixed point method for g(x) = x - arctan(x) ===")
    def g(x):
        return x - atan(x)
    initials = [5, -5, 1, -1, 0.1]
    for x0 in initials:
        seq = fixed_point_iter(g, x0, n_iter=10)
        print(f"\nInitial x0 = {x0}:")
        for k, xk in enumerate(seq):
            print(f" k={k:2d} -> x_{k} = {nstr(xk,30)}")
    print("\nShort theoretical note (printed): derivative g'(x) = 1 - 1/(1+x^2) = x^2/(1+x^2).")
    print("For small x near 0, g' ~ 0 -> contraction; for large |x|, g' -> 1 so contraction is not guaranteed globally.")
    print("This explains potential convergence from small starts and slow/divergent behavior from large starts.")

def problem_6_newton():
    print("\n=== Problem 6: Newton's method for f(x) = arctan(x) ===")
    tol = 1e-10
    ensure_precision_for_tol(tol)
    initials = [0.5, 1, 1.3, 1.4, 1.35, 1.375, 1.3875, 1.39375, 1.390625, 1.3921875]
    for x0 in initials:
        print(f"\nInitial x0 = {x0}:")
        records = newton_method(f_arctan, df_arctan, x0, tol=tol, max_iter=20)
        for k, xk, fxk in records:
            print(f" k={k:2d}: x = {nstr(xk,30)}, f(x) = {nstr(fxk,30)}")
        last_x = records[-1][1]
        print(f" -> Stopped at k={records[-1][0]}, approx root = {nstr(last_x,30)}, f(root)={nstr(records[-1][2],30)}")

def problem_7_secant():
    print("\n=== Problem 7: Secant method for f(x) = arctan(x) ===")
    tol = 1e-10
    ensure_precision_for_tol(tol)
    initial_pairs = [(0.5, 1), (1, 1.3), (1.4, 1.5), (10, 11)]
    for x0, x1 in initial_pairs:
        print(f"\nInitial pair (x0, x1) = ({x0}, {x1}):")
        records = secant_method(f_arctan, x0, x1, tol=tol, max_iter=20)
        for k, xk, fxk in records:
            print(f" k={k:2d}: x = {nstr(xk,30)}, f(x) = {nstr(fxk,30)}")
        print(f" -> Stopped at k={records[-1][0]}, approx = {nstr(records[-1][1],30)}, f(approx)={nstr(records[-1][2],30)}")


if __name__ == "__main__":
    print("MAD 4401 — HW1 numerical experiments (Python)")
    problem_1_and_2_bisection_prints()
    problem_3_fixed_point()
    problem_6_newton()
    problem_7_secant()
    print("\nAll done. If you want output to a file, redirect stdout to a file when running:")
    print("  python3 mad4401_hw1_numexp.py > hw1_results.txt")

