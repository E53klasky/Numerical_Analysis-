import numpy as np

# Nodes
x1 = -np.sqrt(3/5)
x2 = 0.0
x3 = np.sqrt(3/5)
nodes = [x1, x2, x3]

# Weights
w1 = 5/9
w2 = 8/9
w3 = 5/9
weights = [w1, w2, w3]

print(f"Nodes: {nodes}")
print(f"Weights: {weights}")




def gaussian_quadrature_3_point(f_func):
    """
    Computes the 3-point Gaussian quadrature for a given function f_func
    on the interval [-1, 1].
    """
    return w1 * f_func(x1) + w2 * f_func(x2) + w3 * f_func(x3)

# Test with monomials f(x) = x^k
print("\n--- Testing Monomials ---")
for k in range(6): # For k = 0, 1, 2, 3, 4, 5
    # Define the monomial function
    def f_monomial(x, power=k):
        return x**power

    # Calculate exact integral for x^k on [-1, 1]
    if k % 2 == 0: # Even power
        exact_integral = 2 / (k + 1)
    else: # Odd power
        exact_integral = 0

    # Calculate approximation using Gaussian quadrature
    approx_integral = gaussian_quadrature_3_point(f_monomial)

    print(f"\nf(x) = x^{k}:")
    print(f"  Exact Integral: {exact_integral}")
    print(f"  Approx Integral: {approx_integral}")
    print(f"  Difference: {abs(exact_integral - approx_integral):.1e}")

    # Check for exactness (allowing for floating point errors)
    assert np.isclose(exact_integral, approx_integral), \
        f"Mismatch for x^{k}! Exact={exact_integral}, Approx={approx_integral}"

print("\nAll checks passed! The 3-point Gaussian quadrature is exact for monomials up to x^5.")
