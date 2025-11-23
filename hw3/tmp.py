import math

# ------------------------------------------------------------
#  Integrand:  f(x) = cos(pi/4 * x) / sin^2(pi/4 * x)
# ------------------------------------------------------------

def f(x):
    return math.cos((math.pi/4)*x) / (math.sin((math.pi/4)*x)**2)

# Exact analytic value
J_exact = -4/math.pi + 4/math.pi*math.sqrt(2)

# ------------------------------------------------------------
# 1. Midpoint Rule on [1, 2]
# ------------------------------------------------------------
xm = (1 + 2) / 2
Q_mid = f(xm)

# ------------------------------------------------------------
# 2. Two-point Open Rule (Open Newton–Cotes)
# Nodes at 1 + 1/3 and 1 + 2/3
# ------------------------------------------------------------
x1 = 1 + 1/3
x2 = 1 + 2/3
Q_open2 = (f(x1) + f(x2)) / 2   # factor (b-a) = 1

# ------------------------------------------------------------
# 3. Simpson’s Rule on [1, 2]
# ------------------------------------------------------------
Q_simpson = (f(1) + 4*f(xm) + f(2)) / 6   # (b-a)/6 = 1/6

# ------------------------------------------------------------
# 4. Closed 4-Point Newton-Cotes Rule
# Integral approx = sum w_j f(x_j)
# weights: 1/8, 3/8, 3/8, 1/8
# nodes: 1, 4/3, 5/3, 2
# ------------------------------------------------------------
x0 = 1
x1 = 1 + 1/3
x2 = 1 + 2/3
x3 = 2

Q_NC4 = (1/8)*f(x0) + (3/8)*f(x1) + (3/8)*f(x2) + (1/8)*f(x3)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print("Exact value:")
print(J_exact)
print()

print("1. Midpoint Rule:")
print(Q_mid)
print("Error:", abs(Q_mid - J_exact))
print()

print("2. Two-point Open Rule:")
print(Q_open2)
print("Error:", abs(Q_open2 - J_exact))
print()

print("3. Simpson's Rule:")
print(Q_simpson)
print("Error:", abs(Q_simpson - J_exact))
print()

print("4. Closed 4-node Newton–Cotes Rule:")
print(Q_NC4)
print("Error:", abs(Q_NC4 - J_exact))
print()

