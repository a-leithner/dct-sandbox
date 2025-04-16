import numpy as np
import dct
import dft

import os

from PIL import Image
from collections.abc import Callable

def image_to_matrix (patch: Image) -> np.ndarray:
	"""
	Converts the given 8x8 PIL/Pillow image to a 8x8 NumPy matrix
	
	:patch:   The Pillow image, must have size 8x8
	:returns: An 8x8 NumPy matrix containing the Pillow image's data
	"""
	if patch.size != (8, 8):
		raise RuntimeError ("Image patch has unsupported dimensions")
	return np.array (patch.getdata ()).reshape ((8, 8))

def compute_dct (patch: Image) -> np.ndarray:
	"""
	Computes the DCT-II of the given 8x8 PIL/Pillow image via Kronecker product.
	
	:patch:   The Pillow image to process, must have size 8x8
	:returns: An 8x8 NumPy matrix containing the image's DCT-II coefficients.
	"""
	if patch.size != (8, 8):
		raise RuntimeError ("Image patch has unsupported dimensions; crop to 8x8 pixels first")
	
	return dct.compute_dct (image_to_matrix (patch))

def compute_dct_orth (patch: Image) -> np.ndarray:
	"""
	Computes the DCT-II of the given 8x8 PIL/Pillow image via similarity transform.
	
	:patch:   The Pillow image to process, must have size 8x8
	:returns: An 8x8 NumPy matrix containing the image's DCT-II coefficients.
	"""
	if patch.size != (8, 8):
		raise RuntimeError ("Image patch has unsupported dimensions; crop to 8x8 pixels first")
	
	return dct.compute_dct_orth (image_to_matrix (patch))

def compute_dft (patch: Image) -> np.ndarray:
	"""
	Computes the DFT of the given 8x8 PIL/Pillow image.
	
	:patch:   The Pillow image to process, must have size 8x8
	:returns: An 8x8 NumPy matrix containing the image's DFT coefficients.
	"""
	if patch.size != (8, 8):
		raise RuntimeError ("Image patch has unsupported dimensions; crop to 8x8 pixels first")
	
	return dft.compute_dft_8 (image_to_matrix (patch))

def valid_image (im: Image) -> tuple[bool, str]:
	"""
	Checks whether the given image can be processed by this suit of scripts.
	
	Specifically, the image must be a PGM image (though we can only test
	for PPM images) with side lengths divisible by 8.
	
	:im:      The image to test
	:returns: (True, None) if im can be processed; (False, str) otherwise
		where str is a discriptive message of the problem
	"""
	if im.format != "PPM":
		return False, "Given image is not in PGM format"
	
	width, height = im.size
	if width % 8 != 0 or height % 8 != 0:
		return False, "Given image has unsupported dimensions: Both width and height need to be multiples of 8."
	
	return True, None

def num_to_bytes (num: int) -> list:
	"""
	Converts a number in the range [0, 65535] to a list of bytes.
	
	The resulting list will always contain two elements to ensure that
	PGM files are written correctly.
	
	:num:     The number to convert
	:returns: The number as a list of bytes (big-endian)
	"""
	if num <= 255: return [0, num]
	else: return [(num & 0xFF00) >> 8, num & 0xFF]

def write_pgm (filename: str, data: np.ndarray) -> None:
	"""
	Writes a given matrix as a PGM image to the given file.
	
	The matrix MUST NOT contain any entry greater than 65535. All entries
	will be converted to integers; either make sure that the matrix only
	contains integers or live with the lost resolution otherweise.
	
	:filename: Where to store the image
	:data:     The image's data to store
	"""
	if len(data.shape) != 2:
		raise ValueError ("Writing images requires two-dimensional arrays")
	
	height, width = data.shape
	with open (filename, "bw+") as f:
		f.write (bytes(f"P5\n{width} {height}\n65535\n", "ascii"))
		for entry in data.reshape (width * height):
			l = num_to_bytes (int (entry))
			f.write (bytes (l))

def invert_color (data: np.ndarray) -> np.ndarray:
	"""
	Inverts an image's colour, i.e. flips the data's range [0, 65535]
	
	:data:    The image to invert, MUST NOT contain entries less than 0 or greater than 65535
	:returns: The inverted image
	"""
	return np.full (data.shape, 65535) - data

def execute_for_all_files (files: list[str], func: Callable[[int, int, Image], None], printinfo: bool = True) -> None:
	"""
	Executes the given function against all given files, either printing information
	on its progress, problems it encounters and info about the file or not.
	
	:files:     The list of files to operate on
	:func:      The function to execute against every file
	:printinfo: Whether to print progress/error/metadata information (default: True)
	"""
	for infile in files:
		if not os.path.isfile (infile):
			if printinfo: print (f"FILE: {infile} - IS DIRECTORY")
			continue
		
		if printinfo: print (f"FILE: {infile}")
		with Image.open (infile) as im:
			if im.format != "PPM":
				if printinfo: print ("Not a valid PPM/PGM file")
				continue
			
			width, height = im.size
			if width % 8 != 0 or height % 8 != 0:
				if printinfo: print ("Unsupported dimensions: Width and height must be divisible by 8.")
				continue
			
			if printinfo: print (f"Width: {width}, Height: {height}")
			
			func (width, height, im)

def shift_into_16bit (matrix: np.ndarray) -> np.ndarray:
	"""
	Transforms the given matrix' values from [-1, 1] to [0, 65535].
	
	:matrix:  The matrix to shift
	:returns: The shifted matrix
	"""
	return np.rint ((matrix + 1) * 65535 / 2)
