
module IntegralCovariance

  using StaticArrays, LinearAlgebra, NearestNeighbors

  export IndexTree, HODLRMatrix

  include("src/structstypes.jl")
  include("src/baseoverloads.jl")
  include("src/peeling_constructor.jl")

end 
