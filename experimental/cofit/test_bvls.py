import bvls
import numpy as np

G = np.random.random((10, 2))
m = np.array([1.0, 2.0])
d = G.dot(m)

lower_bounds = np.array([0.0, 0.0])
upper_bounds = np.array([1.5, 1.5])

lower_bounds = np.array([-np.inf, -np.inf])
upper_bounds = np.array([np.inf, np.inf])

bounds = [lower_bounds, upper_bounds]

print(bounds)
soln = bvls.bvls(G, d, bounds)

print(soln)