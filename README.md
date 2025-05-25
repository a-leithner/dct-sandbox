# DCT Python Sandbox

This repository compiles some Python snippets created while working on my Bachelor's
thesis in Mathematics on the discrete cosine transform (DCT). They are by no means
optimised (e.g., they don't exploit the FFT) and are intended ***for illustrative
purposes only.***

An overview of the included modules is given below.

> [!NOTE]
> Where image files are to be taken as an input, they ***MUST*** be in
> [PGM (portable graymap) format](https://en.wikipedia.org/wiki/Netpbm#File_formats)
> and ***MUST*** have widths and heights divisible by 8. See the example images in
> `images/` for reference.

Converting an image to PGM is as simple as running [ImageMagick](https://imagemagick.org)
against it:
```shell
magick input.jpg output.pgm
```
where `input.jpg` is the input file (in any of the formats supported by ImageMagick)
and `output.pgm` is the file to write to.

The reference images in `images/` have been scaled using ImageMagick, too:
```shell
magick image.pgm -sharpen 0x1.2 -resize 25\% -quality 95 output.pgm
```

> [!NOTE]
> Any claim of exactness or equivalence of methods given regarding this project
> shall be understood ***up to numerical inaccuracies***.

## Prerequisites

Make sure to have Pillow (a fork of PIL), NumPy, and Matplotlib installed. Any fairly
recent version should suffice. Modules will only run on Python 3. All scripts and
modules have been tested to work with the following versions:
  * Python 3.13.2
  * Pillow 11.1.0
  * NumPy 2.2.2
  * Matplotlib 3.10.1

## Modules

### `dct.py`

The main file is `dct.py`, providing all necessary definitions to apply the DCT to
a NumPy vector or matrix. DCT-I through DCT-IV are supported for any dimension
greater than or equal to 1. This module provides the methods `make_C_x(n)`, where `x`
is any of the flavours `I`, `II`, `III`, or `IV`, which generate the DCT matrix of
dimension `n x n` with orthonormal normalisation factor.

To apply the DCT to a NumPy vector, use `make_C_x (n) @ v`, where `x` is the desired
variant, `n` the desired dimension and `v` the vector to transform. If you're working
with 8x8 matrices, you may use `compute_dct` or `compute_dct_orth` for the DCT-II
directly (the former computing the transform by means of a Kronecker product, the
latter as a similarity transform), and `compute_idct` or `compute_idct_orth` for the
DCT-III (inverse of the DCT-II).

If `M` is an 8x8 NumPy matrix, calling `dct.compute_dct (M)` or `dct.compute_dct_orth (M)`
will be equivalent to `scipy.fft.dctn (M, norm="ortho")`, and calling
`dct.compute_idct (M)` or `dct.compute_idct_orth (M)` will be equivalent to
`scipy.fft.idctn (M, norm="ortho")`.

Module only, provides no CLI.

### `ccquad.py`

`ccquad.py` is a module implementing
[Clenshaw–Curtis quadrature](https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature)
and adjacent functions using NumPy and `dct.py`. Clenshaw–Curtis works by
interpolating the function which is to be integrated using
[Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) of the
first kind up to degree $n$ such that the interpolation is exact at the extrema of the
$n$-th Chebyshev polynomials and then integrating this interpolation.

This quadrature always calculates $`\int_{-1}^1 f(x)\,\mathrm{d}x`$; it may be
necessary to scale your function to integrate over an arbitrary interval.

The module provides no CLI but the following functions:

#### `ccquad.integrate`

Signature:  
```python
def integrate (f: Callable[[np.float64], np.float64], n: int) -> np.float64:
```

Computes $`\int_{-1}^1 f(x)\,\mathrm{d}x`$. Here, `f` is the function to be
integrated, which must have domain at least $`[-1, 1]`$, and `n` is the order
of integration, i.e. the order of the highest Chebyshev polynomial to be used
in the interpolation.

If `f` is a polynomial and `n` fulfills $`n\geq\deg f`$, the integration result
will be exact.

#### `ccquad.chebyshev_interpolation`

Signature:  
```python
def chebyshev_interpolation (f: Callable[[np.float64], np.float64], n: int) -> Callable[[np.float64], np.float64]:
```

Returns a callable taking one input `x` returning the value of the Chebyshev
interpolation polynomial at `x`. Here, `f` is the function to be interpolated,
which must have domain at least $`[-1, 1]`$, and `n` is the order of interpolation,
i.e. the degree of the last Chebyshev polynomial to use.

The returned callable will compute the value of `f` exactly at the Chebyshev
extrema of order `n`, that is, at $`\cos(k\pi/n)`$ for $`0\leq k\leq n`$.

#### `ccquad.chebyshev_sample`

Signature:  
```python
def chebyshev_sample (f: Callable [[np.float64], np.float64], n: int) -> np.array:
```

Samples the callable `f` at the Chebyshev extrema of order `n`, i.e. at the points
$`\cos(k\pi/n)`$ for $`0\leq k\leq n`$, and returns the results in a NumPy array.
Note that the first and last entry will be multiplied by $`1/\sqrt{2}`$ to achieve
correct normalisation when used as an input to DCT-I (as is the case in all functions
calling this function from within `ccquad.py`).

#### `ccquad.chebyshev_transform`

Signature:  
```python
def chebyshev_transform (f: Callable [[np.float64], np.float64], n: int) -> np.array:
```

Computes the DCT-I of the output of [`ccquad.chebyshev_sample`](#ccquadchebyshev_sample).
Used in [`ccquad.integrate`](#ccquadintegrate) and
[`ccquad.chebyshev_interpolation`](#ccquadchebyshev_interpolation).

#### `ccquad.chebyshev_extrema`

Signature:  
```python
def chebyshev_extrema (n: int) -> np.ndarray:
```

Computes the extrema of the Chebyshev polynomial of the first kind of order `n`
and returns them in a NumPy array. As stated above, these extrema are the points
$`\cos(k\pi/n)`$ for $`0\leq k\leq n`$. Used in
[`ccquad.chebyshev_sample`](#ccquadchebyshev_sample).

### `averages.py`

This module calculates the average DCT-II and DFT coefficient matrices of all
8x8 blocks over a given set of images and displays the magnitudes of these
average matrices as Matplotlib images, DFT first, DCT-II second.

To compute the average matrix of a single image file, use
```shell
python3 averages.py image.pgm
```
or replace `image.pgm` by an arbitrary number of files to compute the average
matrix across all files.

Use `--3d` to generate 3D bar plots instead of 2D image plots. This allows
for easier visualisation of the magnitudes of the coefficients and their
relationships.

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

When called with `--display`, this module will only display the generated
reduced image in a Matplotlib window and will not save it. This option is
not positional, meaning it may be positioned anywhere in the commandline.

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

### `crop.py`

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

Provides no CLI.

### `dft.py`

Provides methods for calculating the [Fourier matrices](https://en.wikipedia.org/wiki/DFT_matrix)
and for applying the DFT to arbitrary NumPy matrices (which must be square).
To compute the DFT of a matrix `M`, call `dft.compute_dft (M)`. This will be
equivalent to `np.fft.fft2 (M, norm="forward")`. To compute the IDFT of a
matrix `D`, e.g. to transform a DFT-transformed matrix back to its original
form, call `dft.compute_idft (D)`. This will be equivalent to
`np.fft.ifft2 (D, norm="forward")`.

Provides no CLI.

## Copyright

The Python modules are licensed under the MIT license. The provided sample images
come from Unsplash and are subject to their terms and license; specifically, the
images remain in their respective authors' copyright.

For the Python code:  
Copyright &copy; 2025 Alexander Leithner.

