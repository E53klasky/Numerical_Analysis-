A = [
    [1, 0, 1],
    [-1, 1, 0],
    [1, 2, -3]
]

b = [0, 0, 0]

x0 = [1.0, 1.0, 1.0]

num_iter = 10


def jacobi(A, b, x0, num_iter):
    n = len(b)
    x_old = x0[:]
    x_new = [0.0] * n

    print("Jacobi Iterations:")
    for k in range(num_iter):
        for i in range(n):
            sum_ = 0.0
            for j in range(n):
                if j != i:
                    sum_ += A[i][j] * x_old[j]
            x_new[i] = (b[i] - sum_) / A[i][i]

        print(f"Iteration {k+1}: x = {[round(val, 6) for val in x_new]}")
        x_old = x_new[:]
    return x_new


def gauss_seidel(A, b, x0, num_iter):
    n = len(b)
    x = x0[:]

    print("\nGauss–Seidel Iterations:")
    for k in range(num_iter):
        for i in range(n):
            sum1 = 0.0
            sum2 = 0.0
            for j in range(i): 
                sum1 += A[i][j] * x[j]
            for j in range(i + 1, n): 
                sum2 += A[i][j] * x[j]
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
    
        print(f"Iteration {k+1}: x = {[round(val, 6) for val in x]}")
    return x



x_jacobi = jacobi(A, b, x0, num_iter)
x_gs = gauss_seidel(A, b, x0, num_iter)

