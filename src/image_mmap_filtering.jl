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

