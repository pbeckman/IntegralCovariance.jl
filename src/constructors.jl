
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