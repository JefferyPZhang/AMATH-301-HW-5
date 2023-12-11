import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

# Question 1

x = lambda t: (11 / 6) * (np.exp(-t / 12) - np.exp(-t))
x_neg = lambda t: -(11 / 6) * (np.exp(-t / 12) - np.exp(-t))
x_prime = lambda t: (11 / 6) * (np.exp(-t) - (np.exp(-t / 12) / 12))
x_prime_neg = lambda t: -(11 / 6) * (np.exp(-t) - (np.exp(-t / 12) / 12))
x_double_prime = lambda t: (11 / 6) * ((np.exp(-t / 12) / 144) - np.exp(-t))

A1 = x_prime(2)
A2 = x_double_prime(2)

t = np.arange(0, 10.01, 0.01)

A3 = scipy.optimize.fminbound(x_neg, 0, 10)
A4 = x(A3)

init_guess = 1
tol = 1e-8

def newton(init_guess, tol):
    t = init_guess
    k = 0
    while (abs(x_prime(t) >= tol)):
        t = t - (x_prime(t) / x_double_prime(t))
        k += 1
    return t, k - 1

A5, A6 = newton(init_guess, tol)

def section_search(f, a, b, c, tol):
    k = 0
    while (abs(b - a) >= tol):
        x = c * a + (1 - c) * b
        y = (1 - c) * a + c * b
        if (f(x) < f(y)):
            b = y
        else:
            a = x
        print(a)
        print(b)
        k += 1
    return (a + b) / 2, k - 1

A7, A8 = section_search(x_neg, 0.0, 5.0, 0.51, tol)
A9, A10 = section_search(x_neg, 0.0, 5.0, 0.9, tol)
A11, k = section_search(x_neg, 0.0, 5.0, 0.25, tol)

print(A7)
print(A9)

def gradient_descent(t_start, max_iterations, learning_rate, tol):
    k = 0
    t = t_start
    value = x_prime_neg(t)
    while (abs(value) >= tol):
        t = t - (learning_rate * value)
        k += 1
        value = x_prime_neg(t)
        if (k > max_iterations):
            break
    return t, k + 1

def gradient_descent_k_only(t_start, max_iterations, learning_rate, tol):
    k = 0
    t = t_start
    value = x_prime_neg(t)
    while (abs(value) >= tol):
        t = t - (learning_rate * value)
        k += 1
        value = x_prime_neg(t)
        if (k > max_iterations):
            break
    return k + 1

A12, A13 = gradient_descent(1.0, 20000, 1.0, tol)
A14 = []

learning_rates = np.logspace(-2.5, 1.5, 10)

for i in range (len(learning_rates)):
    A14.append(gradient_descent_k_only(1.0, 20000, learning_rates[i], tol))
    
# Question 2

f = lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
del_f0 = lambda x, y: 4 * x ** 3 - 42 * x + 4 * x * y + 2 * y ** 2 - 14
del_f1 = lambda x, y: 4 * y ** 3 - 26 * y + 4 * x * y + 2 * x ** 2 - 22

A15 = f(3, 4)
A16 = [del_f0(3, 4), del_f1(3, 4)]

x = np.linspace(-7, 7)
y = np.linspace(-7, 7)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

A17 = X
A18 = Y
A19 = Z

def grad_descent_variable(p, tol):
    grad = np.array((del_f0(p[0], p[1]), del_f1(p[0], p[1])))
    k = 0
    phi = lambda t: p - t * grad
    f_of_phi = lambda t: f(phi(t)[0], phi(t)[1])
    while (scipy.linalg.norm(grad) >= tol):
        grad = np.array((del_f0(p[0], p[1]), del_f1(p[0], p[1])))
        tmin = scipy.optimize.fminbound(f_of_phi, 0, 1)
        p = phi(tmin)
        k += 1
        if (k > 2000):
            break
    return p, k + 1

def grad_descent_constant(p, tol, learning_rate):
    grad = np.array((del_f0(p[0], p[1]), del_f1(p[0], p[1])))
    k = 0
    while (scipy.linalg.norm(grad) >= tol):
        p = p - (learning_rate * grad)
        grad = np.array((del_f0(p[0], p[1]), del_f1(p[0], p[1])))
        k += 1
        if (k > 2000):
            break
    return p, k + 1

p = np.array([-3, -2])
q = np.array([3, 2])
r = np.array([-3, 2])
s = np.array([3, -2])
tol = 1e-7
learning_rate = 0.01

A20, A21 = grad_descent_variable(p, tol)
A22, A23 = grad_descent_constant(p, tol, learning_rate)
A24, k = grad_descent_constant(q, tol, learning_rate)
A25, k = grad_descent_constant(r, tol, learning_rate)
A26, k = grad_descent_constant(s, tol, learning_rate)