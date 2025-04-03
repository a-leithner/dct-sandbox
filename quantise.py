import numpy as np
import dct
import images

import sys
import os

from PIL import Image

# See Rao, Ochoa-Dominguez, Subbarayappa: "JPEG Series", p. 89
LUMINANCE_QUANT = np.array(
	[
		[ 16,  11,  10,  16,  24,  40,  51,  61],
		[ 12,  12,  14,  19,  26,  58,  60,  55],
		[ 14,  13,  16,  24,  40,  57,  69,  56],
		[ 14,  17,  22,  29,  51,  87,  80,  62],
		[ 18,  22,  37,  56,  68, 109, 103,  77],
		[ 24,  35,  55,  64,  81, 104, 113,  92],
		[ 49,  64,  78,  87, 103, 121, 120, 101],
		[ 72,  92,  95,  98, 112, 100, 103,  99]
	]
)

def idct (patch: np.ndarray, depth: int) -> np.ndarray:
	"""
	Computes the IDCT of the given image patch JPEG-style.
	
	The given patch (coefficient matrix) must have shape (8,8). The
	resulting image matrix will have a bit depth/precision as given,
	though this will only produce correct results if the original
	image has the same bit depth.
	
	The original image is expected to have had data in range
	[-2^(depth - 1), 2^(depth - 1)] before being transformed. The
	output of this method will already be shifted back into
	[0, 2^(depth) - 1].
	
	:patch:   The coefficient matrix to transform to an image
	:depth:   The bit depth, typically 8 or 16.
	:returns: The IDCT of the given patch
	"""
	if patch.shape != (8, 8):
		raise RuntimeError ("Unsupported image patch dimensions, must be (8, 8)")
	
	if depth not in [8, 16]:
		raise ValueError ("Unsupported bit depth: Must be 8 or 16")
	
	newpatch = np.rint (dct.compute_idct_orth (patch) + 2**(depth - 1)) * (257 if depth == 8 else 1)
	
	# Due to numerical inaccuracies, we may end up with entries slightly below 0
	# or slightly above 65535. We simply clip these to the required range.
	return np.clip (newpatch, 0, 65535)

if len (sys.argv) != 2:
	print ("usage: python3 quantise.py file")
	print ()
	print ("Applies JPEG quantisation after DCT to the given image and reassembles")
	print ("it using IDCT.")
	print ()
	print ("Arguments:")
	print (" file        The file to operate on, must be in PGM format")
	sys.exit ()

filename = sys.argv [1]

path_parts = filename.split("/")
target = "/".join(path_parts[:-1]) + f"/quantised_" + path_parts[-1]
target_noreco = "/".join(path_parts[:-1]) + f"/quantised_noreco_" + path_parts[-1]

with open (filename, "rb") as f:
	im = Image.open (f)
	
	validim, msg = images.valid_image (im)
	if not validim:
		print (msg)
		sys.exit ()
	
	depth = 8 if im.mode == "L" else 16
	
	width, height = im.size
	newim = np.zeros ((height, width))
	quaim = np.zeros ((height, width))
	
	patches_h = width // 8
	patches_v = height // 8
	patches = patches_h * patches_v
	print (f"Patches: {patches}")
	
	index = 0
	for v_index in range (patches_v):
		for h_index in range (patches_h):
			# Coordinates of the currently processed patch.
			# These are in PIL coordinates, meaning that the horizontal
			# axis is x and the vertical axis is y. NumPy expects coordinates
			# in opposite order: rows first, then columns.
			top_left = (h_index * 8, v_index * 8)
			bottom_right = (top_left[0] + 8, top_left[1] + 8)
			
			# Extract the current patch and process it; this -2**(depth-1) shifts
			# its (integer) values from range [0, 2^(depth) - 1] to range
			# [-2^(depth - 1), 2^(depth - 1)].
			patch = im.crop ((*top_left, *bottom_right))
			patch = images.image_to_matrix (patch) - 2**(depth - 1)
			dct_coeffs = dct.compute_dct_orth (patch)
			
			# Quantise and copy the quantised coefficients
			dct_coeffs = np.rint (dct_coeffs / LUMINANCE_QUANT)
			quantised_dct_coeffs = np.copy (dct_coeffs)
			
			# After this step, dct_coeffs will be the correctly de-quantised copy.
			dct_coeffs = dct_coeffs * LUMINANCE_QUANT
			
			# Put the IDCT of the correctly de-quantised coefficients into the reconstructed
			# image; put the non-de-quantised coefficients into quaim
			newpatch = idct (dct_coeffs, depth)
			newim [top_left [1]:bottom_right [1], top_left [0]:bottom_right [0]] = newpatch
			quaim [top_left [1]:bottom_right [1], top_left [0]:bottom_right [0]] = idct(quantised_dct_coeffs, depth)
			
			index += 1
			print (f"{index} of {patches}\r", end="", flush=True)
	
	# newim, saved to target, is correctly de-quantised, as would be the result of
	# a JPEG encode-decode procedure. quaim is not correctly de-quantised, resulting
	# in a washed-out image carrying the same information about the image (in some sense)
	images.write_pgm (target, newim)
	images.write_pgm (target_noreco, quaim)
