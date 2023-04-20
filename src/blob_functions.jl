
"""blob_LoG cell detection. If image is too large, this may reduce the memory consumption. Possbily the edge might not be counted."""
function blob_LoG_split(win::AbstractVector, img::AbstractArray{T, N}, σscales; edges::Union{Bool,Tuple{Bool,Vararg{Bool,N}}}=(true, ntuple(d->false, Val(N))...), σshape::NTuple{N,Real}=ntuple(d->1, Val(N)), rthresh::Real=1//1000) where {T<:Union{AbstractGray,Real},N}
  xi = round.(Int, collect(range(1, size(img, 1), length = win[1]+1)))
  yi = round.(Int, collect(range(1, size(img, 2), length = win[2]+1)))
  zi = round.(Int, collect(range(1, size(img, 3), length = win[3]+1)))
  results = Vector{Vector{BlobLoG}}();
  #n = prod(map(length, [xi, yi, zi]))
  #p = Progress(n, 1)
  for k in 1:win[3]
    for j in 1:win[2]
      Threads.@threads for i in 1:win[1]
        coords = xi[i]:xi[i+1], yi[j]:yi[j+1], zi[k]:zi[k+1]
        subimg = view(img, coords...)
        result = blob_LoG(subimg, σscales; edges = edges, σshape = σshape, rthresh = rthresh)
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

function blob_LoG_split1(tilesz::NTuple{N, Int}, img::AbstractArray{T, N}, σscales; edges::Union{Bool,Tuple{Bool,Vararg{Bool,N}}}=(true, ntuple(d->false, Val(N))...), σshape::NTuple{N,Real}=ntuple(d->1, Val(N)), rthresh::Real=1//1000) where {T<:Union{AbstractGray,Real}, N}
  tileids_all = collect(TileIterator(axes(img), tilesz))
  blobs = Vector{Vector{BlobLoG}}()
  n = length(tileids_all)
  p = Progress(n, "blob_LoG computing...", 50)
  Threads.@threads for i in 1:n
    tileaxs = CartesianIndices(tileids_all[i])
    blobs1 = blob_LoG(img[tileaxs], σscales; edges = edges, σshape = σshape, rthresh = rthresh)
    _update_blob_loc!(blobs1, first(tileaxs))
    push!(blobs, blobs1)
    next!(p)
  end
  return(vcat(blobs...))
end

function _update_blob_loc!(blobs, pos::CartesianIndex{N}) where {N}
  for (i, blob) in enumerate(blobs)
    newpos = blob.location+pos-CartesianIndex(ntuple(n->1, N))
    blobs[i] = BlobLoG(newpos, blob.σ, blob.amplitude)
  end
  return(blobs)
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
function blobSelection(results::Vector{<:BlobLoG}, imgaxes::Tuple, boundary::Int)
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
function blobSelectEdge(results::Vector{<:BlobLoG}, img::AbstractArray, threshold, r)
  keep = falses(length(results))
  n = length(results)
  p = Progress(n, "Edge blob filtering...", 50)
  Threads.@threads for i in eachindex(results)
    Ifirst = max(results[i].location - CartesianIndex(r,r,r), CartesianIndex(1,1,1))
    Ilast = min(results[i].location + CartesianIndex(r,r,r), CartesianIndex(size(img)))
    if quantile(vec(img[Ifirst:Ilast]), 0.1) > threshold
      keep[i] = true
    end
    next!(p)
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

# PyPlot or PythonPlot is unstable. Make local function
#""" plot blob amplitudes and low and high threasholds """
#function plot_thresh(blobs, thresh; ttl = "Threshold")
#  amps = [x.amplitude for x in blobs]
#  sorted_amps = sort(amps)
#  nblobs = length(blobs)
#  fig1 = figure(ttl); #Visualize thresholding
#  plot1 = fig1.add_subplot(2,1,1);
#  plot1.plot(sorted_amps);
#  yf, yl = 0, 80;
#  plot1.set_ylim([yf, yl]);
#  plot1.axhline(thresh[1], color = "red");
#  plot2 = fig1.add_subplot(2,1,2);
#  plot2.plot(sorted_amps)
#  xf, xl = nblobs-(thresh[2]+200), nblobs
#  plot2.set_xlim([xf, xl])
#  plot2.axhline(thresh[2], color = "red")
#  yf, yl = sorted_amps[nblobs-(thresh[2]+200)], last(sorted_amps)
#  plot2.set_ylim([yf, yl])
#  tight_layout()
#end
