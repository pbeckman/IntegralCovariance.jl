
module IntegralCovariance

  using StaticArrays, LinearAlgebra, NearestNeighbors, SparseArrays

  export IndexTree, HODLRMatrix, ColumnButterflyMatrix

  include("src/structstypes.jl")
  include("src/constructors.jl")
  include("src/baseoverloads.jl")
  include("src/hodlr_peeling.jl")
  include("src/butterfly_peeling.jl")

end 
