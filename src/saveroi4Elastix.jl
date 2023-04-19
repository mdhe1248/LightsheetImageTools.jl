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
      elseif length(coord) == 2
        write(io, string(coord[1], " ", coord[2]), "\n")
      end
    end
  end
end


