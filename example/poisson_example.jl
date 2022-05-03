
using LinearAlgebra, IntegralCovariance, Plots

n = 2^11
kernel(xi, xj) = exp(-norm(xi - xj))
xs = range(0, stop=1.0, length=n)
A = [kernel(xi, xj) for xi in xs, xj in xs]
slv(x) = A*x
adj(x) = A'*x

tree = IndexTree(n, 5)

k = 64
p = 0

H = HODLRMatrix(slv, adj, tree, k, p)

@show norm(A - Matrix(H))

pl = heatmap(Matrix(H), yflip=true)
display(pl)