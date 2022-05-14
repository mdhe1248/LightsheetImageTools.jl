module LightsheetImageTools

using Images, NRRD, FileIO, Mmap
using Distributions, Interpolations, Statistics
using JLD2
#LinearAlgebra
# Write your package code here.

export lightsheetNorm, lightsheetNorm!, normalizeImg!
export mmap_mapwindow!, mmap_fun!, _mean, create_NRRD_header1
export blob_LoG_split, blobSelection, blobSelectEdge, replaceBlobPos
export markImg!
export coord_scaling, coordTransform_nrrd_jl2elx
export saveroi4Elastix
export voxelize_roi

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

""" 
`A`is an array compatible with mmap.
`fn` is a file name stored in hard drive. 
`B` is a mapped array
After some operation to B, run `Mmap.sync!(B); close(io)` to the end.
"""
function prepare_mmap(fn, A)
  dims = size(A)
  t = typeof(A)
  io = open(fn, "w+")
  B = mmap(io, t, dims)
  return(io, B)
end

"""mmap convert and store in the hard drive
`M` is an array 
`fn = filename.bin`
`arraytype =  Array{Float32, 3}'
"""
function mmap_convert!(fn, arraytype, M)
  io, B = prepare_mmap(fn, M)
  for i in 1:prod(dims)
    B[i] = convert(t, M[i]) 
  end
  Mmap.sync!(B)
  close(io)
end

""" pixelwise memory-mapped function between two images. For example the value after subtraction between two large images can be stored in the hard drive.
`fn` is a save file name. 
`f` is a function whose input is `img1[i]` and `img2[i]`, where i is for-looping number.
`dims` is tuple of dimention e.g. size(img)
image type shoule be compatible with `Mmap.mmap` e.g. Float64

e.g.)
f(x, y) = x - y
mmap_fun!("test.bin", f, a, b)  
io = open("test.bin", "r+") #Check values
c = mmap(io, Array{Float64,2}, (3,3))
close(io)
"""
function mmap_fun!(fn, f, img1, img2)
  io, B = prepare_mmap(fn, img1)
  dims = size(img1)
  for i in 1:prod(dims)
    B[i] = f(img1[i], img2[i]) 
  end
  Mmap.sync!(B)
  close(io)
end  

""" This function can replace most applciation of `mmap_mapwindow!`. A lot more memory will be used, but maybe faster"""
function mmap_imfilter!(fn, img, kernel)
  io, B = prepare_mmap(fn, img)
  imfilter!(B, img, kernel)
  Mmap.sync!(B)
  close(io)
  GC.gc()
end

"""filter image and save directly to the hard drive"""
function mmap_mapwindow!(fn, f, img, window)
  io, B = prepare_mmap(fn, img)
  mapwindow!(f, B, img, window)
  Mmap.sync!(B) #See ?Mmap
  close(io)
end

"""## Make NRRD header
`type` file type (e.g. Float32)
`dim` dimension of data (e.g. size(img))
`datafilefn` file name to be linked
`headerfn` file header to be saved
"""
function create_NRRD_header1(type, dims, datfilefn, headerfn)
  header = NRRD.headerinfo(type, dims)
  header["datafile"] = datfilefn
  open(headerfn, "w") do io
      write(io, magic(format"NRRD"))
      NRRD.write_header(io, "0004", header)
  end
end

"""## Make NRRD header with axis info"""
function create_NRRD_header(imgaxes, type, px_spacing, datfilefn, headerfn)
  axy = Axis{:y}((imgaxes[1])*px_spacing[1]) # Note that the axis changes
  axx= Axis{:x}((imgaxes[2])*px_spacing[2])
  if length(imgaxes) == 3
    axz = Axis{:z}((imgaxes[3])*px_spacing[3])
    header = NRRD.headerinfo(type, (axy, axx, axz))  # assuming Float64 data
  else
    header = NRRD.headerinfo(type, (axy, axx))  # assuming Float64 data
  end
  header["datafile"] = datfilefn
  open(headerfn, "w") do io
      write(io, magic(format"NRRD"))
      NRRD.write_header(io, "0004", header)
  end
end

""" 
weighted mean function, which behaves like mean(buf, weight(w)). (see `?mean`)
`kv` is a vectorized kernel. 
"""
function _mean(buf, kv::AbstractVector)
  output = eltype(buf)(0)
  for i in 1:length(buf)
    val = buf[i]*kv[i] 
    output = output + val
  end
  output/length(buf)
  return(output)
end 

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

function markImg!(img, pos::Vector{BlobLoG{Float64, Float64, 3}}, rad)
  for i in 1:length(pos)
  markImg!(img, pos[i].location[2], pos[i].location[1], pos[i].location[3], rad)	
  end
end

function markImg!(img, pos::Vector{BlobLoG{Float32, Tuple{Float64, Float64, Float64}, 3}}, rad)
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


"""blob_LoG cell detection. If image is too large, this may reduce the memory consumption. Possbily the edge might not be counted."""
function blob_LoG_split(win::AbstractVector, img, σscales, edges)
  xi = round.(Int, collect(range(1, size(img, 1), length = win[1]+1)))
  yi = round.(Int, collect(range(1, size(img, 2), length = win[2]+1)))
  zi = round.(Int, collect(range(1, size(img, 3), length = win[3]+1)))
  results = [];
  #n = prod(map(length, [xi, yi, zi]))
  #p = Progress(n, 1)
  for k in 1:win[3]
    for j in 1:win[2]
      Threads.@threads for i in 1:win[1]
        coords = xi[i]:xi[i+1], yi[j]:yi[j+1], zi[k]:zi[k+1]
        subimg = view(img, coords...)
        result = blob_LoG(subimg, σscales,)
        #result coordinate correction
        for (idx, res) in enumerate(result)
          loc = res.location + CartesianIndex(xi[i]-1, yi[j]-1, zi[k]-1)
          σ = res.σ
          amp = res.amplitude
          result[idx] = BlobLoG(loc, σ, amp)
        end
        push!(results, result)
      end
      #next!(p)
    end
  end
  return(results)  
end

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

"""select cells by intensity threshold"""
function blobSelection(results::Vector, thresh::Number)
  keep = falses(length(results))
  i = 1
  for res in results
    keep[i] = res.amplitude > thresh 
    i += 1
  end
  return(keep)
end

""" remove cells near the edges of the image"""
function blobSelection(results::Vector, imgaxes::Tuple, boundary::Int)
  fi = first.(imgaxes) .+ boundary 
  la = last.(imgaxes) .- boundary 
  keep = falses(length(results))
  i = 1
  for res in results
    loc = convert(Tuple, res.location)
    outoffirst = map(<, fi, loc)
    outoflast = map(>, la, loc)
    outof = all(outoffirst), all(outoflast)
    keep[i] = all(outof)
    i += 1
  end
  return(keep)
end

"""blobselection another way: using intensity thresholding nearby"""
function blobSelectEdge(results::Vector, img::AbstractArray, threshold, r)
  keep = falses(length(results))
  Threads.@threads for i in eachindex(results)
    Ifirst = max(results[i].location - CartesianIndex(r,r,r), CartesianIndex(1,1,1))
    Ilast = min(results[i].location + CartesianIndex(r,r,r), CartesianIndex(size(img)...))
    if quantile(vec(img[Ifirst:Ilast]), 0.1) > threshold
      keep[i] = true
    end
  end
  return(keep)
end


"""
Create a new BlobLoG. Replace location in `blobs` with `pos`
"""
function replaceBlobPos(blobs::BlobLoG, pos::CartesianIndex)
  newblob =  BlobLoG(pos, blobs.σ, blobs.amplitude)
  return(newblob)
end
function replaceBlobPos(blobs::Vector, locs::Vector{CartesianIndex{3}})
  newblobs = similar(blobs)
  for i in eachindex(blobs)
    newblobs[i] = replaceBlobPos(blobs[i], locs[i])
  end
  return(newblobs)
end 

""" find the subbrain labeling of blobs within target brain region
`annotationImg` is brain with labeling. It is from ClearMap2 reference (`ABA_25um_annotation.tif`).
`blobs` are obtained from blob_LoG; possibly contains the coordinate of c-fos.
"""
function label_blobs(annotationImg, blobs)
  fos_coords = map(x -> x.location, blobs)
  keep_lbl = zeros(Int, length(fos_coords))
  for (i, fos_coord) in enumerate(fos_coords)
    lbl = annotationImg[fos_coord]
    keep_lbl[i] = lbl 
  end
  keep_lbl
end

""" nrrd coordinate is different between julia and elastix. From julia (row, column) index coordinates, this function obtains coordinates for elastix and FIJI"""
coordTransform_nrrd_jl2elx(img, x, y, z) = (x-1), (y-1), (z-1) 
function saveroi4Elastix(filename::String, roi_coords::Vector)
  open(filename, "w") do io
    write(io, "index\n")
    write(io, string(length(roi_coords), "\n"))
    for coord in roi_coords
      #roix, roiy, roiz = roi_julia2elastix(img, coord)
      #write(io, string(roix, " ", roiy, " ", roiz), "\n")
      #x, y, z = coordTransform_nrrd(coord[1], coord[2], coord[3])
      if length(coord) == 3
        write(io, string(coord[1], " ", coord[2], " ", coord[3]), "\n")
      else if length(coord) == 2
        write(io, string(coord[1], " ", coord[2]), "\n")
      end
    end
  end
end

"""
`r` is a tuple of pixel lenths in each dimention e.g.) (2,2,2)
`rois` is a vector of Cartesian index in 3D
"""
function voxelize_roi(rois, img, r)
  img1 = zeros(size(img))
  for roi in rois
    voxelize_roi!(img1, roi, img, r)
  end
  return(img1)
end
function voxelize_roi!(img1, roi, img, r)
  v = img[roi]
  fi = max(roi-CartesianIndex(r), CartesianIndex(1,1,1))
  li = min(roi+CartesianIndex(r), CartesianIndex(size(img)))
  for i in fi:li
    img1[i] = img1[i]+v
  end
  return(img1)
end


end
