
function Base.adjoint(A::HODLRMatrix{T})::HODLRMatrix{T} where{T<:Number}
  # don't take adjoints of undef elements
  return HODLRMatrix(
    A.tree, 
    A.V, 
    Vector{Vector{AbsMatOrAdj{T}}}([
      isassigned(Sl, 1) ? adjoint.(
        Sl[[isodd(i) ? i+1 : i-1 for i in eachindex(Sl)]]
        ) : Sl for Sl in A.S
    ]), 
    A.U, 
    isassigned(A.L, 1) ? adjoint.(A.L) : A.L
    )
end

function LinearAlgebra.mul!(target::Array{T}, A::HODLRMatrix{T}, src::Array{T}; maxlvl=nothing)::Array{T} where{T<:Number}
  # zero out target
  fill!(target, zero(T))
  # add products with low-rank factors
  for l=1:(isnothing(maxlvl) ? A.tree.lvl : maxlvl)
    for b=1:2:2^l
      idx1 = indexiter(A.tree, l, b)
      idx2 = indexiter(A.tree, l, b+1)
      target[idx1,:] .+= A.U[l][b] * (A.S[l][b] * (A.V[l][b+1]' * src[idx2,:]))
      target[idx2,:] .+= A.U[l][b+1] * (A.S[l][b+1] * (A.V[l][b]' * src[idx1,:]))
    end
  end
  # add product with leaves
  if isnothing(maxlvl)
    for b=1:2^A.tree.lvl
      idx  = indexiter(A.tree, A.tree.lvl, b)
      target[idx,:] .+= A.L[b]*src[idx,:]
    end
  end
  return target
end

function LinearAlgebra.:*(A::HODLRMatrix{T}, src::Array{T})::Array{T} where{T<:Number}
  target = Array{T}(undef, size(src))
  return mul!(target, A, src)
end

function Base.size(A::HODLRMatrix{T})::Tuple{Int64, Int64} where{T<:Number} 
  return (A.tree.idx[1][1][2], A.tree.idx[1][1][2])
end

function Base.size(A::HODLRMatrix{T}, i::Int64)::Int64 where{T<:Number} 
  return A.tree.idx[1][1][2]
end

function Base.Matrix(A::Union{HODLRMatrix{T}, Adjoint{T, HODLRMatrix{T}}})::Matrix{T} where{T<:Number}
  # M = Matrix{T}(undef, size(A)...)
  # for l=1:A.tree.lvl
  #   for b=1:2:2^l
  #     idx1 = indexiter(A.tree, l, b)
  #     idx2 = indexiter(A.tree, l, b+1)
  #     M[idx1, idx2] .= A.U[l][b]*A.V[l][b]'
  #     M[idx2, idx1] .= A.U[l][b+1]*A.V[l][b+1]'
  #   end
  # end
  # for b=1:2^A.tree.lvl
  #   idx = indexiter(A.tree, A.tree.lvl, b)
  #   M[idx, idx] .= A.L[b]
  # end
  return A*Matrix(Diagonal(ones(size(A, 1))))
end

function leafsizes(n::Int64, lvl::Int64)::Vector{Int64}
  leafszs = ones(Int64, 2^lvl)*floor(Int64, n/(2^lvl))
  remandr = n - 2^lvl*floor(Int64, n/(2^lvl))
  while remandr > 0
    for i in eachindex(leafszs)
      leafszs[i] += 1
      remandr    -= 1
      if remandr == 0
        break
      end
    end
  end
  return leafszs
end

function leafsizes(tree::IndexTree)::Vector{Int64}
  return [tree.idx[tree.lvl+1][b][2]-tree.idx[tree.lvl+1][b][1]+1 for b=1:2^tree.lvl]
end

indexiter(tree, l, b) = (tree.idx[l+1][b][1]:tree.idx[l+1][b][2])