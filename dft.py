import numpy as np
import cmath

def make_Omega (n: int) -> np.matrix:
	"""
	Compute the Fourier matrix of order n
	
	The Fourier matrix is the complex matrix whose entries are given by
	`exp((2 pi ikl)/n)`, where (k, l) is the (row, column) index
	pair. Returns a NumPy matrix (of type matrix) to allow for .H
	
	:n:       The order of the Fourier matrix to compute
	:returns: The requested matrix
	"""
	return np.matrix ([
		[ cmath.exp ((-2 * np.pi * k * l * 1j) / n) for k in range (n) ] for l in range (n)
	])

# OMEGA8 is the 8x8 Fourier matrix and OMEGA8H its conjugate transpose
# (Hermitian transpose)
OMEGA8 = make_Omega (8)
OMEGA8H = OMEGA8.H

def compute_dft (matrix: np.ndarray | np.matrix) -> np.matrix:
	"""
	Compute the discrete Fourier transform (DFT) of the given matrix.
	
	This amounts to left-multiplication with OmegaN and right-mutliplication
	with OmegaN.H, scaled by 1/n**2, where n is the dimension of the matrix
	and OmegaN the n-th Fourier matrix as computed by make_Omega(n).
	
	The given matrix may contain real or complex values and must be square.
	The resulting matrix will be of type np.matrix since in general it will
	contain complex entries.
	
	:matrix:  The matrix to apply the DFT to
	:returns: The DFT of the given matrix
	"""
	shape = matrix.shape
	n = shape [0]
	if n != shape [1]:
		raise ValueError ("Non-square matrices are not supported")
	
	Omega = make_Omega (n)
	OmegaH = Omega.H
	return (Omega @ matrix @ OmegaH) / n**2

def compute_dft_8 (matrix: np.ndarray | np.matrix) -> np.matrix:
	"""
	Compute the discrete Fourier transform (DFT) of the given 8x8 matrix.
	
	The given matrix may contain real or complex values and must have shape
	(8,8). Any other shape will be rejected but may be computed by means
	of compute_dft(matrix). The resulting matrix will be of type np.matrix
	since in general it will contain complex entries.
	
	:matrix:  The matrix to apply the DFT to
	:returns: The DFT of the given matrix
	"""
	if matrix.shape != (8, 8):
		raise ValueError ("Unsupported matrix shape, must be (8,8)!")
	
	return (OMEGA8 @ matrix @ OMEGA8H) / 64
