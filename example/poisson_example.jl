
using LinearAlgebra, IntegralCovariance, BenchmarkTools, Plots

n = 2^12
xs = range(0, stop=1.0, length=n)

h = 1/(n-1)
c(x) = 1 # + 0.2x
cv = c.(xs .+ h/2)
L = 1/h^2 * Tridiagonal(
  -cv[1:end-1], 
  vcat(2cv[1], cv[2:end-1]+cv[3:end], 2cv[end]), 
  -cv[2:end]
  )
slv(M) = L\M
adj(M) = L'\M

tree = IndexTree(n, Int64(log2(n))-1)

r = 2

H = HODLRMatrix(slv, adj, tree, r)

# @show norm(inv(L) - Matrix(H))

# gr(size=(1000, 500))
# lo = @layout [a b]
# p1 = heatmap(inv(L), yflip=true, ticks=((),()), aspect_ratio=1)
# p2 = heatmap(Matrix(H), yflip=true, ticks=((),()), aspect_ratio=1)
# pl = plot(p1, p2, layout=lo)
# display(pl)