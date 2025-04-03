import numpy as np
from math import isqrt

SQRT2 = np.sqrt(2)
INVSQRT2 = 1/SQRT2
LAM = lambda k: INVSQRT2 if k == 0 else 1
LAMN = lambda k, n: INVSQRT2 if k in [0, n] else 1

def make_C_I (n: int) -> np.ndarray:
	return np.array ([
		[ SQRT2 * LAMN (k, n) * LAM (l, n) * np.cos ((np.pi * k * l) / n) for l in range (n + 1) ] for k in range (n + 1)
	])

def make_C_II (n: int) -> np.ndarray:
	norm = SQRT2 / np.sqrt (n)
	n2 = 2 * n
	return np.array ([
		[ norm * LAM (k) * np.cos (np.pi * k * (2 * l + 1) / n2) for l in range (n) ] for k in range (n)
	])

def make_C_III (n: int) -> np.ndarray:
	# Though C[II] is unitary, C_II @ C_III is closer to eye(8) if we
	# compute C_III as the inverse matrix of C_II.
	return np.linalg.inv (make_C_II (n))

def make_C_IV (n: int) -> np.ndarray:
	norm = SQRT2 / np.sqrt (n)
	return np.array ([
		[ norm * cos (np.pi * (k + 0.5) * (l + 0.5) / n) for l in range (n) ] for k in range (n)
	])

C_II_8 = make_C_II (8)
C_III_8 = make_C_III (8)

DCT_KRON_8 = np.kron (C_II_8, C_II_8)
IDCT_KRON_8 = np.kron (C_III_8, C_III_8)

def matrix_to_vector (m: np.ndarray) -> np.ndarray:
	shape = m.shape
	return m.reshape (shape [0] * shape [1])

def vector_to_matrix (v: np.ndarray) -> np.ndarray:
	if len (v.shape) != 1:
		raise ValueError ("Vector must be one-dimensional")
	
	d = isqrt (v.shape [0])
	if d ** 2 != v.shape [0]:
		raise ValueError ("Unsupported vector length, shape must be (n^2,) for an integer n")
	
	return v.reshape ((d, d))

# compute_* methods apply the DCT via matrix-vector-product with the Kronecker
# product of two copies of the same C_* matrix.
#
# compute_*_orth methods are more straight forward; they apply the DCT
# as a similarity transform, i.e. C_* @ m @ C_*.T
def compute_dct (matrix: np.ndarray) -> np.ndarray:
	if matrix.shape != (8, 8):
		raise RuntimeError ("Unsupported matrix dimensions, must be 8x8!")
	
	return vector_to_matrix (DCT_KRON_8 @ matrix_to_vector (matrix))

def compute_dct_orth (matrix: np.ndarray) -> np.ndarray:
	if matrix.shape != (8, 8):
		raise RuntimeError ("Unsupported matrix dimensions, must be 8x8!")
	
	return C_II_8 @ matrix @ C_III_8

def compute_idct (matrix: np.ndarray) -> np.ndarray:
	if matrix.shape != (8, 8):
		raise RuntimeError ("Unsupported matrix dimensions, must be 8x8!")
	
	return vector_to_matrix (IDCT_KRON_8 @ matrix_to_vector (matrix))

def compute_idct_orth (matrix: np.ndarray) -> np.ndarray:
	if matrix.shape != (8, 8):
		raise RuntimeError ("Unsupported matrix dimensions, must be 8x8!")
	
	return C_III_8 @ matrix @ C_II_8
