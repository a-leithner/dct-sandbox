# DCT Python Sandbox

This repository compiles some Python snippets created while working on my Bachelor's
thesis in Mathematics on the discrete cosine transform (DCT). They are by no means
optimised (e.g., they don't exploit the FFT) and are intended for illustrative
purposes only.

An overview of the included modules is given below. Where image files are to be taken
as an input, they ***MUST*** be in PGM (portable graymap) format and ***MUST*** have
widths and heights divisible by 8. See the example images in `images/` for reference.

## Prerequisites

Make sure to have Pillow (a fork of PIL), numpy, and Matplotlib installed. Any fairly
recent version should suffice. Modules will only run on Python 3 (tested with
Python 3.13.2).

## Modules

### `dct.py`

The main file is `dct.py`, providing all necessary definitions to apply the DCT to
a numpy vector or matrix. DCT-I through DCT-IV are supported for any dimension
greater than or equal to 1. This module provides the methods `make_C_x(n)`, where `x`
is any of the flavours `I`, `II`, `III`, or `IV`, which generate the DCT matrix of
dimension `n x n` with orthonormal normalisation factor.

To apply the DCT to a numpy vector, use `make_C_x (n) @ v`, where `x` is the desired
variant, `n` the desired dimension and `v` the vector to transform. If you're working
with 8x8 matrices, you may use `compute_dct` or `compute_dct_orth` for the DCT-II
directly (the former computing the transform by means of a Kronecker product, the
latter as a similarity transform), or `compute_idct` or `compute_idct_orth` for the
DCT-III (inverse of the DCT-II).

### `averages.py`

This module calculates the average DCT-II coefficient matrix of all 8x8 blocks
over a given set of images and displays the magnitudes of this average matrix
as a Matplotlib image.

To compute the average matrix of a single image file, use
```shell
python3 averages.py image.pgm
```
or replace `image.pgm` by an arbitrary number of files to compute the average
matrix across all files.

### `reduce.py`

This module transforms a given image using the DCT-II one 8x8 pixel block at a
time, removing a given number of coefficients and reversing the transform, resulting
in a modified image assembled from less information.

Three patterns for removing coefficients are available:
  * `linear` enumerates the coefficients one after the other; the top left coefficient
    is index 0 and the bottom right (last coefficient) is index 64.
  * `diag` only applies a mask to those coefficients above the antidiagonal of the
    coefficient matrix, enumerating one row after the other. Maximum: 32 coefficients.
  * `1q` applies only to the top left quadrant of the matrix. Maximum: 16 coefficients.
Any pattern only keeps the first `x` coefficients as specified upon invocation.

Usage:
```shell
python3 reduce.py coeffs [pattern] file
```
where `coeffs` is the number of coefficients to *keep*, `pattern` is the optional
parameter defining the discarding pattern (see above, if none is given this defaults
to `linear`), and `file` is the PGM image to operate on.

So, for example, to keep the first 17 coefficients (that is, the first two rows in
full and then 1 coefficient from the third), use
```shell
python3 reduce.py 17 image.pgm # or
python3 reduce.py 17 linear image.pgm
```
Outputs PGM files with 16 bit colour depth.

### `quantise.py`

Applies JPEG-style quantisation to the given image file, also writing an image file
which will not be dequantised, i.e. which will appear "washed-out".

Usage:
```shell
python3 quantise.py image.pgm
```
(but replace `image.pgm` with the file you want to operate on). Outputs PGM files
with 16 bit colour depth.

### `basisimages.py`

Writes the 64 basis images of the 8x8 DCT-II to the directory `basis/`. Make sure
that this folder exists before running the script. Outputs PGM files with 16 bit
colour depth.

Usage:
```shell
python3 basisimages.py
```

### `cropy.py`

Crops an image such that its width and height both are multiples of 8. This will
keep the largest possible rectangle whose top-left corner coincides with the top-left
corner of the original image. Cropped files will be put in the subdirectory `cropped/`
of the parent directory the original files are located in.

If the original image already has width and height divisible by 8 this script only
copies the file. Requires PGM files, may output files with only 8 bit colour depth.

Usage:
```shell
python3 crop.py image.pgm
```
(supports operating on multiple files at once, simply specify any number of files).

### `images.py`

Provides some convenience functions for the other modules of this repository, such
as a simple method to write 16-bit PGM files, or computing the DCT-II or DCT-III of
images read by Pillow.

Provides no CLI.

### `plots.py`

Provides a method to display the absolute values of a given matrix as a Matplotlib
image with a fixed colour bar, fixed colour scheme, and fixed scale. Also draws
a dashed line between the four quadrants of 8x8 matrices.


## Copyright

The Python modules are licensed under the MIT license. The provided sample images
come from Unsplash and are subject to their terms and license; specifically, the
images remain in their respective authors' copyright.

For the Python code:  
Copyright &copy; 2025 Alexander Leithner.

