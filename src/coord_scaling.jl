
"""Linearly coordinate scaling """
function _coord_scaling(loc, scales)
  output = loc[1]*scales[1], loc[2]*scales[2], loc[3]*scales[3]
  return(round.(Int, output))
end

"""Linearly coordinate scaling """
function coord_scaling(locs::Vector, scales)
  output = Vector{Tuple{Int,Int,Int}}(undef, length(locs))
  for (i, loc) in enumerate(locs)
    output[i] = _coord_scaling(loc, scales)
  end
  return(output)
end

