import numpy as np

# question c
ns = 2.5
na = 1
lam = 500e-9
Lam = 300e-9
theta_i = np.deg2rad(60)
m = 1

theta_m = np.arcsin(ns/na*np.sin(theta_i)-m*lam/(na*Lam))
theta_c = np.arcsin(na/ns)
print(np.rad2deg(theta_m), np.rad2deg(theta_c))
