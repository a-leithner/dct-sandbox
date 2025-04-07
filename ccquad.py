import numpy as np
import dct

from collections.abc import Callable

def chebyshev_extrema (n: int) -> np.ndarray:
	"""
	Compute the n+1 extrema of the n-th Chebyshev polynomial of the first kind
	
	:n:       The order of the Chebyshev polynomial to operate for
	:returns: NumPy array containing all n+1 extrema
	"""
	return np.array ([ np.cos ((k * np.pi) / n) for k in range (n + 1) ])

def chebyshev_sample (f: Callable [[np.float64], np.float64], n: int) -> np.array:
	"""
	Sample the given callable f at the n+1 extrema of the first-kind Chebyshev
	polynomial of order n, with the first and last elements multiplied by 1/np.sqrt(2)
	
	The given function MUST accept values in [-1, 1] as an input and MAY have
	a strictly larger domain, though sampling will only occur in [-1, 1].
	
	:f:       The (real-valued) function to sample
	:n:       The order of the Chebyshev polynomial at whose extrema to sample
	:returns: The values of f at the extrema of the n-th Chebyshev polynomial
	"""
	extr = chebyshev_extrema (n)
	return [ dct.LAMN (k, n) * f (extr [k]) for k in range (n + 1) ]

def chebyshev_transform (f: Callable [[np.float64], np.float64], n: int) -> np.array:
	"""
	Sample the given callable f at the n+1 extrema of the first-kind Chebyshev
	polynomial of order n and apply DCT-I to the resulting vector.
	
	The given function MUST accept values in [-1, 1] as an input and MAY have
	a strictly larger domain, though sampling will only occur in [-1, 1].
	
	:f:       The (real-valued) function to sample
	:n:       The order of the Chebyshev polynomial at whose extrema to sample
	:returns: The DCT-I of the output of chebyshev_sample(f, n)
	"""
	return dct.make_C_I (n) @ chebyshev_sample (f, n)

def chebyshev_interpolation (f: Callable[[np.float64], np.float64], n: int) -> Callable[[np.float64], np.float64]:
	"""
	Compute the polynomial interpolation of order n for the given callable f
	by means of Chebyshev polynomials of the first kind.
	
	This is equivalent to sampling f, computing the DCT-I of the sample vector,
	and then constructing a lambda expression which computes the inner product
	of that DCT sample vector with the vector containing the values of the
	k-th Chebyshev polynomial for 0 <= k <= n.
	
	The given function MUST accept values in [-1, 1] as an input and MAY have
	a strictly larger domain, though sampling will only occur in [-1, 1]. The
	resulting expression will yield results for all of the real number line,
	though these will generally only be exact the the Chebyshev extrema.
	
	:f:       The (real-valued) function to interpolate
	:n:       The order of the Chebyshev polynomial at whose extrema the
	          interpolation is ought to be exact.
	:returns: Callable lambda expression computing the value of the interpolation
	"""
	dct_f = chebyshev_transform (f, n)
	return lambda x: np.inner (dct_f, np.array ([ np.cos (k * np.arccos (x)) for k in range (n + 1) ]))

def integrate (f: Callable[[np.float64], np.float64], n: int) -> np.float64:
	"""
	Compute the definite integral over the interval [-1, 1] of the given
	callable f by means of Clenshaw-Curtis quadrature.
	
	The given function MUST accept values in [-1, 1] as an input and MAY have
	a strictly larger domain, though integration will only compute the definite
	integral in [-1, 1] (you may want to scale your function for arbitrary
	intervals).
	
	:f:       The (real-valued) function to integrate
	:n:       The order of the integration, i.e. the order of the Chebyshev
			  interpolation used under the hood.
	:returns: The computed definite integral over [-1, 1]
	"""
	dct_f = chebyshev_transform (f, n)
	coeffs = np.array ([ (-2 * dct.LAMN (2 * k, n)) / (4 * k**2 - 1) for k in range ((n // 2) + 1) ])
	return np.sqrt (2 / n) * np.inner (coeffs, dct_f [::2])

