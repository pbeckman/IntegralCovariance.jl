
AbsMatOrAdj{T} = Union{<:AbstractMatrix{T}, Adjoint{T, <:AbstractMatrix{T}}}

struct IndexTree
  lvl :: Int64
  idx :: Vector{Vector{SVector{2,Int64}}}
end

mutable struct HODLRMatrix{T} <: AbstractMatrix{T}
  tree :: IndexTree
  U    :: Vector{Vector{Matrix{T}}}
  S    :: Vector{<:Vector{<:AbsMatOrAdj{T}}}
  V    :: Vector{Vector{Matrix{T}}}
  L    :: Vector{<:AbsMatOrAdj{T}}
end

mutable struct SymmetricHBSMatrix{T} <: AbstractMatrix{T}
  tree :: IndexTree
  U    :: Vector{Vector{Matrix{T}}}
  D    :: Vector{<:Vector{<:AbsMatOrAdj{T}}}
end

mutable struct ColumnButterflyMatrix{T} <: AbstractMatrix{T}
  tree :: IndexTree
  U    :: Vector{SparseMatrixCSC{T}}
end