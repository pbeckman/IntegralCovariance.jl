
struct IndexTree
  lvl :: Int64
  idx :: Vector{Vector{SVector{2,Int64}}}
end

struct HODLRMatrix{T} <: AbstractMatrix{T}
  tree :: IndexTree
  U    :: Vector{Vector{Matrix{T}}}
  V    :: Vector{Vector{Matrix{T}}}
  L    :: Vector{Matrix{T}}
end