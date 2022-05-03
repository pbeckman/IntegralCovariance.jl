
function IndexTree(n::Int64, lvl::Int64)::IndexTree
  if lvl > floor(Int64, log2(n))
    throw(ArgumentError("The requested tree level is greater than the number of points n allows. Please choose a tree level of at most log2(n) = $(floor(Int64, log2(n)))."))
  end
  leafszs = leafsizes(n, lvl)
  idx = Vector{Vector{SVector{2,Int64}}}(undef, lvl+1)

  idx[end] = []
  start = 1
  for sz in leafszs
    push!(idx[end], SVector{2,Int64}(start, start+sz-1))
    start += sz
  end
  
  for i = lvl:-1:1
    idx[i] = Vector{SVector{2,Int64}}([])
    for j=1:2:length(idx[i+1])
      push!(idx[i], (idx[i+1][j][1], idx[i+1][j+1][2]))
    end
  end
  
  return IndexTree(lvl, idx)
end

function HODLRMatrix(
  slv::Function, adj::Function, 
  tree::IndexTree, k::Int64, p::Int64
  )::HODLRMatrix{Float64}
  T = Float64
  n = tree.idx[1][1][2] # matrix dimension
  r = k+p # off-diagonal block rank 

  if r > minimum(leafsizes(tree))
    throw(ArgumentError("The requested off-diagonal rank at the lowest level is greater than the block size. Please decrease the tree level or the rank."))
  end

  # initialize factors and leaves
  U = Vector{Vector{Matrix{T}}}(undef, tree.lvl)
  V = Vector{Vector{Matrix{T}}}(undef, tree.lvl)
  L = Vector{Matrix{T}}(undef, length(tree.idx[end]))

  # initialize HODLR matrix
  A = HODLRMatrix(tree, U, V, L)

  # initialize random matrices
  S1 = Matrix{T}(undef, n, r)
  S2 = Matrix{T}(undef, n, r)

  for l=1:tree.lvl
    # initialize factors at this level
    A.U[l] = Vector{Matrix{T}}(undef, 2^l)
    A.V[l] = Vector{Matrix{T}}(undef, 2^l)

    # construct random matrices
    for b=1:2:2^l
      idx1 = indexiter(tree, l, b)
      n1   = length(idx1)
      S1[idx1, :] .= randn(n1, r)
      S2[idx1, :] .= zeros(n1, r)

      idx2 = indexiter(tree, l, b+1)
      n2   = length(idx2)
      S1[idx2, :] .= zeros(n2, r)
      S2[idx2, :] .= randn(n2, r)
    end

    # apply operator to random matrices
    # and subtract products with already compressed blocks
    Y1 = slv(S2) - mul!(Matrix{T}(undef, size(S2)), A, S2, maxlvl=l-1)
    Y2 = slv(S1) - mul!(Matrix{T}(undef, size(S1)), A, S1, maxlvl=l-1)

    # compute and store column space matrices
    for b=1:2:2^l
      idx1 = indexiter(tree, l, b)
      A.U[l][b]    = qr(Y1[idx1,:]).Q
      S1[idx1, :] .= A.U[l][b]

      idx2 = indexiter(tree, l, b+1)
      A.U[l][b+1]  = qr(Y2[idx2,:]).Q
      S2[idx2, :] .= A.U[l][b+1]
    end

    # apply adjoint to sample matrices
    # and subtract products with adjoints of already compressed blocks
    Z1 = adj(S2) - mul!(Matrix{T}(undef, size(S2)), A', S2, maxlvl=l-1)
    Z2 = adj(S1) - mul!(Matrix{T}(undef, size(S1)), A', S1, maxlvl=l-1)

    # compute and store row space matrices
    for b=1:2:2^l
      idx1 = indexiter(tree, l, b)
      F1 = qr(Z1[idx1,:])
      A.V[l][b]    = F1.Q
      A.U[l][b+1] .= A.U[l][b+1] * F1.R'

      idx2 = indexiter(tree, l, b+1)
      F2 = qr(Z2[idx2,:])
      A.V[l][b+1]  = F2.Q
      A.U[l][b]   .= A.U[l][b] * F2.R'
    end
  end

  # initialize identity block matrix
  S = Matrix{T}(undef, n, maximum(leafsizes(tree)))

  # construct identity block matrix
  # and subtract products with already compressed blocks
  for b=1:2^tree.lvl
    idx = indexiter(tree, tree.lvl, b)
    nl  = length(idx)
    S[idx, 1:nl] .= Diagonal(ones(nl))
  end

  # apply operator to identity block matrix
  Y = slv(S) - mul!(Matrix{T}(undef, size(S)), A, S, maxlvl=tree.lvl)

  # extract diagonal blocks
  for b=1:2^tree.lvl
    idx  = indexiter(tree, tree.lvl, b)
    nl   = length(idx)
    L[b] = Y[idx, 1:nl]
  end

  return HODLRMatrix(tree, U, V, L)
end
