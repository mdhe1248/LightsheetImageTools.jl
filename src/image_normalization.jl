"""Image Normalization"""
function normalizeImg(img::AbstractArray, thresh; f = mean)
  denum = f(filter(x -> x >thresh, img))
  img = img ./ denum 
  return(img)
end

""" Only mean"""
function normalizeImg!(img::AbstractArray, thresh)
  denum = 0
  for k in 1:size(img,3)
    for j in 1:size(img, 2)
      for i in 1:size(img, 1)
        if img[i, j, k] > thresh
          denum += img[i,j,k]
        end
      end
    end
  end
  for i in 1:length(img)
    img[i] = img[i] ./ denum 
  end
  return(img)
end

""" Across each row of an image, get a statistics among values above `thresh`. Default is `median`.
If all values are below thresh, then the maximum(img) will be obtained.
"""
function threshFilterFunc(img, thresh; f = median)
  m = zeros(eltype(img), Base.axes(img, 1))
  mx = maximum(img)
  for i in Base.axes(img,1) 
    v = filter(x -> !isnan(x) , img[i, :,:])
    if sum(v .> thresh) != 0
      m[i] = f(filter(x -> x > thresh, v))
    else
      m[i] = mx 
    end
  end
  return(m) 
end

"""Lightsheet intensity correction"""
function lightsheetNorm(img, thresh; f = mean)
  denum = threshFilterFunc(img, thresh; f = f)
  imgc = img./denum
  return(imgc, denum)
end
 
""" Lightsheet intensity correction; use mean"""
function lightsheetNorm!(img, denuml::Vector)
  for k in axes(img, 3)
    for j in axes(img, 2)
      for i in axes(img, 1)
        img[i,j,k] = img[i,j,k]/denuml[i]
      end
    end
  end
  return(img)
end

