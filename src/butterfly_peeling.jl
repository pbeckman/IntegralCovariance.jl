
using Plots, LaTeXStrings

function ColumnButterflyMatrix(
  slv::Function, adj::Function, 
  tree::IndexTree, k::Int64, p::Int64; T=ComplexF64, plt=false
  )::ColumnButterflyMatrix{T}
  r = k + p
  n = tree.idx[1][1][2] # matrix dimension

  if r > minimum(leafsizes(tree))
    throw(ArgumentError("The requested off-diagonal rank with oversampling ($r) is greater than the leaf size ($(minimum(leafsizes(tree)))). Please decrease the tree level ($(tree.lvl)), the rank ($k), or the oversampling ($p)."))
  end

  # initialize factors
  U = Vector{SparseMatrixCSC{T}}(undef, tree.lvl + 2)

  # initialize HODLR matrix
  A = ColumnButterflyMatrix(tree, U)

  # initialize random matrix
  M  = zeros(T, n, r*(2^(tree.lvl+1) - 1))
  c0 = 1 # which column of the sample matrix to look at
  for l=0:tree.lvl
    for b=1:2^l
      idx = indexiter(tree, l, b)
      M[idx, c0:c0+r-1] .= randn(length(idx), r)
      c0 += r
    end
  end
  plt && display(heatmap(
    ifelse.(M .== 0, NaN, M), yflip=true, title=L"\Omega", 
    ))

  # apply operator to random matrix
  Y = slv(M)

  # tree in the reduced rank matrix factor space
  nr = r*2^tree.lvl
  rtree = IndexTree(nr, tree.lvl)

  # compute and store column basis matrices
  A.U[1] = spzeros(n, nr)
  for b=1:2^tree.lvl
    idx  = indexiter(tree, tree.lvl,  b)
    idxr = indexiter(rtree, tree.lvl, b)
    A.U[1][idx, idxr] .= qr(Y[idx,1:r]).Q[:, 1:r]
  end

  plt && display(heatmap(
    abs.(Y), 
    yflip=true,
    title=L"$|Y_0\,|$"
    # size=(size(Y,1)/size(Y,2)*500, 500)
    ))

  # Project sample matrix to column space
  @show size(Y)
  Y = A.U[1]' * Y[:, r+1:end]

  for l=1:rtree.lvl
    @show size(Y)
    plt && display(heatmap(
      abs.(Y), 
      yflip=true,
      title=L"$|Y_%$l\,|$"
      # size=(500,size(Y,1)/size(Y,2)*500)
      ))

    A.U[l+1] = spzeros(nr, nr)
    c0 = 1 # which columns of the sample matrix to look at
    l0 = 0 # which column to store factors at
    w0 = 0 # which row    to store factors at

    # compute and store transfer matrices
    Y2 = zeros(T, r*2^(tree.lvl-l), size(Y, 2) - r*2^l)
    
    for b2=1:2^l
      for b1=1:2^(tree.lvl-l)
        # index of basis of parent in first dimension
        idx1 = indexiter(rtree, tree.lvl-1, b1)
        idx2 = indexiter(rtree, tree.lvl, b1)
        # @show b2, b1, w0 .+ idx1, l0 .+ idx2
        A.U[l+1][w0 .+ idx1, l0 .+ idx2] = qr(Y[idx1, c0:c0+r-1]).Q[:, 1:r]
      end

      indvec = nested_ind(l, tree.lvl-1, b2, r)
      ind  = findall(==(1), indvec) 
      sind = findall(==(1), indvec[r*2^l+1:end])
      idx1b = indexiter(rtree, l-1, 1)
      idx2b = indexiter(rtree, l,   1)
      # @show indvec
      # @show indvec[r*2^l+1:end]
      # @show r*2^l .+ sind
      # @show length(indvec)
      # @show length(ind)
      # @show size(A.U[l+1][w0 .+ idx1b, l0 .+ idx2b]')
      # @show size(Y[:, ind])
      # @show size(Y[:, r*2^l .+ sind])
      Y2[:, sind] = A.U[l+1][w0 .+ idx1b, l0 .+ idx2b]' * Y[:, r*2^l .+ sind]
      
      c0 += r
      l0 += r*2^(tree.lvl-l) 
      if iseven(b2)
        w0 += r*2^(tree.lvl-l+1) 
      end
    end

    Y = Y2
  end

  M = randn(n, r)

  Z = adj(M)

  # # compute and store row basis matrices
  # A.U[end] = spzeros(nr, n)
  # for b=1:2^tree.lvl
  #   idx  = indexiter(tree, tree.lvl,  b)
  #   idxr = indexiter(rtree, tree.lvl, b)
  #   @show idxr
  #   @show idx
  #   @show size(Z[idx, :])
  #   @show size(M)
  #   @show size(foldl(*, A.U[1:end-1]))
  #   @show size(M \ foldl(*, A.U[1:end-1]))
  #   @show size((Z[idx, :] * (M \ foldl(*, A.U[1:end-1])))')
  #   A.U[end][idxr, idx] .= Matrix((Z[idx, :] * (foldl(*, A.U[1:end-1])))')
  # end

  A.U[end] = spzeros(nr, n)
  for b=1:2^tree.lvl
    idx  = indexiter(tree, tree.lvl, b)
    idxr = indexiter(rtree, tree.lvl, b)
    A.U[end][idxr, idx] .= (Z[idx, :] * pinv(foldl(*, A.U[1:end-1])[:, idxr]' * M))'
  end

  return A
end

function nested_ind(l0, L, b, r)
  v  = zeros(Int64, 2r*sum([2^l for l=l0-1:L]))
  c0 = 1
  for l=l0-1:L 
    v[c0+(b-1)*r*2^(l-l0+1):c0+b*r*2^(l-l0+1)-1] .= 1
    c0 += r*2^(l+1)
  end
  return v 
end