#### Mark images
"""
Mark a coordinate. 
img is should be RBG format. If not, try `colorImg = convert.(RGB, img)`
It will mark a given x and y position with a red dot.
change red dot size with 'r'.
"""
function markImg!(img, y0, x0, r)
  x1 = max(x0 - r, 1)
  y1 = max(y0 - r, 1)
  x2 = min(x0 + r, size(img, 1))
  y2 = min(y0 + r, size(img, 2))
  for x in x1:x2
    for y in y1:y2
      if r^2 > (x0 - x)^2 + (y0 - y)^2
        img[x, y] = typeof(img[1])(typemax(eltype(img[1])), 0, 0)
      end
    end
  end
  img
end    

function markImg!(img, y0, x0, z, r) # Maybe mapped array can be used
  x1 = max(x0 - r, 1)
  y1 = max(y0 - r, 1)
  x2 = min(x0 + r, size(img, 1))
  y2 = min(y0 + r, size(img, 2))
  for x in x1:x2
    for y in y1:y2
      if r^2 > (x0 - x)^2 + (y0 - y)^2
        img[x, y, z] = typeof(img[1])(typemax(eltype(img[1])), 0, 0)
      end
    end
  end
  img
end    

function markImg!(img, pos::AbstractVector, rad)
  for i in 1:length(pos)
  markImg!(img, pos[i][2], pos[i][1], pos[i][3], rad)	
  end
end

function markImg!(img, pos::Vector{<:BlobLoG}, rad)
  for i in 1:length(pos)
  markImg!(img, pos[i].location[2], pos[i].location[1], pos[i].location[3], rad)	
  end
end

function markImg!(img, pos::Vector{<:BlobLoG}, rad)
  for i in 1:length(pos)
  markImg!(img, pos[i].location[2], pos[i].location[1], pos[i].location[3], rad)	
  end
end

""" Overlap two gray images, Color becomes red and green"""
function checkOverlap(imgred, imggreen)
  mri = zeros(RGB, size(imgred))
  mri = AxisArray(mri, AxisArrays.axes(imgred))
  for i in eachindex(mri)
    if isfinite(imgred[i]) && isfinite(imggreen[i])
      mri[i] = RGB(imgred[i], imggreen[i], 0)
    else
      mri[i] = RGB(NaN)
    end
  end
  mri
end

"""Check morphology: Use guassian `Normal` fit"""
function fittest(model::UnionAll, img, pos, r)
  co = (pos[1]-r:pos[1]+r, pos[2]-r:pos[2]+r, pos[3]-r:pos[3]+r) 
  output = fit(model, img[co...])
  return(output)
end


