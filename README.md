# LightsheetImageTools
My functions use image filtering and feature detection functions in the `Images` package.
Because the image size is large (several Gb, could be hundreds of Gb), I used `mmap` arrays to store processed images directly in the hard drive.
