import math

# ------------------
# Problem 2 (1,2a): Bisection Method
# ------------------
def bisection(f, a, b, eps, max_iter=1000):
    if f(a) * f(b) >= 0:
        raise ValueError("Bisection method fails: f(a) and f(b) must have opposite signs.")
    
    k = 0
    while (b - a) / 2 > eps and k < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c, k+1
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        k += 1
    return (a + b) / 2, k

# ------------------
# Problem 3 (1): Fixed Point Iteration
# ------------------
def fixed_point(g, x0, n_iters):
    results = [x0]
    x = x0
    for _ in range(n_iters):
        x = g(x)
        results.append(x)
    return results

# ------------------
# Problem 6: Newton's Method
# ------------------
def newton(f, df, x0, tol=1e-10, max_iter=20):
    results = [x0]
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            break
        x_new = x - fx / dfx
        results.append(x_new)
        if abs(x_new - x) < tol or abs(f(x_new)) < tol:
            break
        x = x_new
    return results

# ------------------
# Problem 7: Secant Method
# ------------------
def secant(f, x0, x1, tol=1e-10, max_iter=20):
    results = [x0, x1]
    for _ in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        results.append(x2)
        if abs(x2 - x1) < tol or abs(f(x2)) < tol:
            break
        x0, x1 = x1, x2
    return results

# ------------------
# runs each problem
# ------------------
if __name__ == "__main__":
    f = lambda x: math.atan(x)
    g = lambda x: x - math.atan(x)
    df = lambda x: 1 / (1 + x*x)

    print("=== Problem 2(1,2a): Bisection Method ===")
    a, b = -4.9, 5.1
    eps_list = [1e-2, 1e-4, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128]
    for eps in eps_list:
        try:
            root, k = bisection(f, a, b, eps)
            print(f"eps={eps}, iterations={k}, root={root}")
        except ValueError as e:
            print(f"eps={eps}, error: {e}")

    print("\n=== Problem 3(1): Fixed Point Iteration ===")
    starts = [5, -5, 1, -1, 0.1]
    for x0 in starts:
        results = fixed_point(g, x0, 10)
        print(f"x0={x0}: {results}")

    print("\n=== Problem 6: Newton's Method ===")
    starts_newton = [0.5, 1, 1.3, 1.4, 1.35, 1.375, 1.3875, 1.39375, 1.390625, 1.3921875]
    for x0 in starts_newton:
        results = newton(f, df, x0)
        print(f"x0={x0}: {results}")

    print("\n=== Problem 7: Secant Method ===")
    starts_secant = [(0.5,1), (1,1.3), (1.4,1.5), (10,11)]
    for x0, x1 in starts_secant:
        results = secant(f, x0, x1)
        print(f"(x0,x1)=({x0},{x1}): {results}")
