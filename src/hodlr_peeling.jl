
function HODLRMatrix(
  slv::Function, adj::Function, 
  tree::IndexTree, r::Int64; T=Float64
  )::HODLRMatrix{T}
  n = tree.idx[1][1][2] # matrix dimension

  if r > minimum(leafsizes(tree))
    throw(ArgumentError("The requested off-diagonal rank at the lowest level is greater than the block size. Please decrease the tree level or the rank."))
  end

  # initialize factors and leaves
  U = Vector{Vector{Matrix{T}}}(undef, tree.lvl)
  S = Vector{Vector{AbstractMatrix{T}}}(undef, tree.lvl)
  V = Vector{Vector{Matrix{T}}}(undef, tree.lvl)
  L = Vector{Matrix{T}}(undef, length(tree.idx[end]))
  for l=1:tree.lvl
    U[l] = Vector{Matrix{T}}(undef, 2^l)
    S[l] = Vector{AbstractMatrix{T}}(undef, 2^l)
    V[l] = Vector{Matrix{T}}(undef, 2^l)
  end

  # initialize HODLR matrix
  A = HODLRMatrix(tree, U, S, V, L)

  # initialize random matrices
  M1 = Matrix{T}(undef, n, r)
  M2 = Matrix{T}(undef, n, r)

  for l=1:tree.lvl
    # construct random matrices
    for b=1:2:2^l
      idx1 = indexiter(tree, l, b)
      n1   = length(idx1)
      M1[idx1, :] .= randn(n1, r)
      M2[idx1, :] .= zeros(n1, r)

      idx2 = indexiter(tree, l, b+1)
      n2   = length(idx2)
      M1[idx2, :] .= zeros(n2, r)
      M2[idx2, :] .= randn(n2, r)
    end

    # apply operator to random matrices
    # and subtract products with already compressed blocks
    Y1 = slv(M2) - mul!(Matrix{T}(undef, size(M2)), A, M2, maxlvl=l-1)
    Y2 = slv(M1) - mul!(Matrix{T}(undef, size(M1)), A, M1, maxlvl=l-1)

    # compute and store column space matrices
    for b=1:2:2^l
      idx1 = indexiter(tree, l, b)
      A.U[l][b]    = qr(Y1[idx1,:]).Q
      M1[idx1, :] .= A.U[l][b]

      idx2 = indexiter(tree, l, b+1)
      A.U[l][b+1]  = qr(Y2[idx2,:]).Q
      M2[idx2, :] .= A.U[l][b+1]
    end

    # apply adjoint to sample matrices
    # and subtract products with adjoints of already compressed blocks
    Z1 = adj(M2) - mul!(Matrix{T}(undef, size(M2)), A', M2, maxlvl=l-1)
    Z2 = adj(M1) - mul!(Matrix{T}(undef, size(M1)), A', M1, maxlvl=l-1)

    # compute and store row space matrices
    for b=1:2:2^l
      idx1 = indexiter(tree, l, b)
      U1, S1, V1t = svd(Z1[idx1,:])
      A.V[l][b]    = U1
      A.S[l][b+1]  = Diagonal(S1)
      A.U[l][b+1] .= A.U[l][b+1] * V1t'

      idx2 = indexiter(tree, l, b+1)
      U2, S2, V2t = svd(Z2[idx2,:])
      A.V[l][b+1]  = U2
      A.S[l][b]    = Diagonal(S2)
      A.U[l][b]   .= A.U[l][b] * V2t'
    end
  end

  # initialize identity block matrix
  M = Matrix{T}(undef, n, maximum(leafsizes(tree)))

  # construct identity block matrix
  # and subtract products with already compressed blocks
  for b=1:2^tree.lvl
    idx = indexiter(tree, tree.lvl, b)
    nl  = length(idx)
    M[idx, 1:nl] .= Diagonal(ones(nl))
  end

  # apply operator to identity block matrix
  Y = slv(M) - mul!(Matrix{T}(undef, size(M)), A, M, maxlvl=tree.lvl)

  # extract diagonal blocks
  for b=1:2^tree.lvl
    idx    = indexiter(tree, tree.lvl, b)
    nl     = length(idx)
    A.L[b] = Y[idx, 1:nl]
  end

  return A
end

function SymmetricHBSMatrix(
  slv::Function, tree::IndexTree, r::Int64; s::Int64=3r, T=Float64
  )::SymmetricHBSMatrix{T}

  n = tree.idx[1][1][2] # matrix dimension

  if r > minimum(leafsizes(tree))
    throw(ArgumentError("The requested off-diagonal rank at the lowest level is greater than the block size. Please decrease the tree level or the rank."))
  end

  # initialize factors and leaves
  U = Vector{Vector{Matrix{T}}}(undef, tree.lvl)
  D = Vector{Vector{AbstractMatrix{T}}}(undef, tree.lvl)
  for l=1:tree.lvl
    U[l] = Vector{Matrix{T}}(undef, 2^l)
    S[l] = Vector{AbstractMatrix{T}}(undef, 2^l)
    V[l] = Vector{Matrix{T}}(undef, 2^l)
  end

  # initialize HBS matrix
  A = SymmetricHBSMatrix(tree, U, D)

  # sample Gaussian random matrix
  M = randn(n, s)
  Y = slv(M)

  for l=tree.lvl:-1:1
    for b=1:2:2^l
      idx = indexiter(tree, l, b)
      if l == tree.lvl
        Mt = M[idx,:]
        Yt = Y[idx,:]
      else
        idx = indexiter(tree, l+1, b)
        Mt = [U[l+1][2b-1]' * M[idx,:]; ]
        Yt = [Y[idx,:]; ]
      end
    end
  end
end