# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 08:38:40 2022

@author: Ziyuan Gu
"""

"""demo"""

import matplotlib.pyplot as plt
import numpy as np
from noisy_direct import NoisyDirect
plt.rcParams['figure.dpi'] = 600

#%%
# deterministic six-hump camel function
def func(x):
    x1 = x[0]
    x2 = x[1]
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
l = [-3, -2]
u = [3, 2]
# solve the problem
solver = NoisyDirect(noisy_direct=False, robust_obj=False, min_eval_per_point=1, max_eval_per_point=1)
solver.minimize(func, l, u)
# scatter plot for sampled points
x, y = [], []
for point in solver.sim_data:
    x.append(solver._back_transform(point)[0])
    y.append(solver._back_transform(point)[1])
plt.scatter(x, y, s=3, c='k')
plt.show()

#%%
# stochastic six-hump camel function contaminated with Gaussian white noise
def func(x):
    x1 = x[0]
    x2 = x[1]
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2 + 0.1 * np.random.normal()
l = [-3, -2]
u = [3, 2]
# solve the problem
solver = NoisyDirect(noisy_direct=True, robust_obj=False, max_iter=20)
solver.minimize(func, l, u)
# scatter plot for sampled points
x, y, c = [], [], []
for point in solver.sim_data:
    x.append(solver._back_transform(point)[0])
    y.append(solver._back_transform(point)[1])
    c.append(solver.sim_data[point]['res'].size)
plt.scatter(x, y, s=3, c=c, cmap='jet')
plt.colorbar()
plt.show()

#%% 
# deterministic Goldstein-Price function
def func(x):
    x1 = x[0]
    x2 = x[1]
    f1 = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
    f2 = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
    return f1 * f2
l = [-2, -2]
u = [2, 2]
# solve the problem
solver = NoisyDirect(noisy_direct=False, robust_obj=False, min_eval_per_point=1, max_eval_per_point=1)
solver.minimize(func, l, u)
# scatter plot for sampled points
x, y = [], []
for point in solver.sim_data:
    x.append(solver._back_transform(point)[0])
    y.append(solver._back_transform(point)[1])

plt.scatter(x, y, s=1, c='k')
plt.axis('square')
plt.xlim([-2,2])
plt.xticks([-2,-1,0,1,2])
plt.ylim([-2,2])
plt.yticks([-2,-1,0,1,2])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()

#%%
# stochastic Goldstein-Price function contaminated with Gaussian white noise
def func(x):
    x1 = x[0]
    x2 = x[1]
    f1 = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
    f2 = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
    return f1 * f2 + np.random.normal(scale=np.sqrt(10))
l = [-2, -2]
u = [2, 2]
# solve the problem
solver = NoisyDirect(noisy_direct=True, robust_obj=False, max_iter=20)
solver.minimize(func, l, u)
# scatter plot for sampled points
x, y, c = [], [], []
for point in solver.sim_data:
    x.append(solver._back_transform(point)[0])
    y.append(solver._back_transform(point)[1])
    c.append(solver.sim_data[point]['res'].size)

plt.scatter(x, y, s=1, c=c, cmap='jet')
plt.axis('square')
plt.colorbar()
plt.xlim([-2,2])
plt.xticks([-2,-1,0,1,2])
plt.ylim([-2,2])
plt.yticks([-2,-1,0,1,2])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()