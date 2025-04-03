import numpy as np
import dct
import images

import sys
import os

from PIL import Image

def make_mask_linear (keep: int) -> np.ndarray:
	"""
	Constructs a mask selecting all but the first `keep` entries
	
	:keeps:   How many coefficients not to select (max. 64)
	:returns: A mask matrix
	"""
	if keep > 64: keep = 64
	mask = np.array ([True] * keep + [False] * (64 - keep))
	return ~(mask.reshape ((8, 8)))

def make_mask_diag (keep: int) -> np.ndarray:
	"""
	Constructs a mask selecting all but the first `keep` entries above the antidiagonal.
	
	That is, any entry below the antidiagonal will always be selected. Of those
	above the antidiagonal, this mask will only deselect the first `keep`, i.e. after
	enumerating the entries from left to right, top to bottom.
	
	:keeps:   How many coefficients not to select (max. 32)
	:returns: A mask matrix
	"""
	mask = np.fliplr (np.tri (8).T) == 1
	mask_flat = mask.reshape (64)
	positive = mask_flat [mask_flat == True]
	positive [[j >= keep for j in range (positive.shape [0])]] = False
	mask_flat [mask_flat == True] = positive
	mask = mask_flat.reshape ((8, 8))
	return ~mask

def make_mask_1q (keep: int) -> np.ndarray:
	"""
	Constructs a mask selecting all but the first `keep` entries in the first quadrant.
	
	That is, any entry not in the first (top-left) quadrant will always be selected.
	Of those in the first quadrant, this mask will only deselect the first `keep`
	after enumerating the entries in the first quadrant from left to right, top to bottom.
	
	:keeps:   How many coefficients not to select (max. 16)
	:returns: A mask matrix
	"""
	mask = np.full ((8, 8), False)
	
	submask = []
	if keep >= 16: submask = [True] * 16
	else: submask = [True] * keep + [False] * (16 - keep)
	
	mask [0:4,0:4] = np.array (submask).reshape ((4, 4))
	return ~mask

if len (sys.argv) not in [3, 4]:
	print ("Usage: python3 reduce.py coeffs [pattern] file")
	print ()
	print ("Reduces a given image file in quality by means of discarding DCT coefficients.")
	print ()
	print ("Arguments:")
	print (" coeffs    The number of DCT coefficients to keep")
	print (" pattern   (Optional) Specifies the order in which coefficients are kept")
	print (" file      The PGM file to transform")
	print ()
	print ("Patterns:")
	print (" linear    Enumerates the coefficients from left to right, top to bottom")
	print (" diag      Keeps the first n coefficients (ltr, ttb) above the antidiagonal")
	print (" 1q        Keeps the first n coefficients (ltr, ttb) in the top left quadrant")
	sys.exit ()

# Extract the command line arguments and supply default values where needed
coeffs = sys.argv [1]
filename = sys.argv [2] if len(sys.argv) == 3 else sys.argv [3]
pattern = "linear" if len(sys.argv) == 3 else sys.argv [2]

if pattern not in ["linear", "diag", "1q"]:
	print ("Invalid pattern. Must either be 'linear', 'diag', or '1q'.")
	sys.exit ()

try:
	coeffs = int(coeffs)
except ValueError:
	print ("First argument must be a valid number of coefficients (integer between 0 and 63 inclusive)")
	sys.exit ()

if not os.path.isfile (filename):
	print ("Given path is either invalid, not a file, or does not exist")
	sys.exit ()

path_parts = filename.split("/")
target = "/".join(path_parts[:-1]) + f"/reduced_{pattern}_{coeffs}_" + path_parts[-1]

# Prepare the requested mask matrix
if pattern == "linear":
	mask = make_mask_linear (coeffs)
elif pattern == "diag":
	mask = make_mask_diag (coeffs)
else:
	mask = make_mask_1q (coeffs)

with open (filename, "rb") as f:
	im = Image.open (f)
	
	validim, msg = images.valid_image (im)
	if not validim:
		print (msg)
		sys.exit ()
	
	depth = 8 if im.mode == "L" else 16
	
	width, height = im.size
	newim = np.zeros ((height, width))
	
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
			
			# Extract the patch to be processed and shift it, JPEG style,
			# from [0, 2^(depth) - 1] to [-2^(depth - 1), 2^(depth - 1)].
			patch = im.crop ((*top_left, *bottom_right))
			patch = images.image_to_matrix (patch) - 2**(depth - 1)
			
			# Compute the patch's DCT and remove all coefficients according
			# to the requested mask
			dct_coeffs = dct.compute_dct_orth (patch)
			dct_coeffs [mask] = 0
			
			# Compute IDCT and shift back into [0, 2^(depth) - 1], clipping any entry slightly off.
			newpatch = np.rint (dct.compute_idct_orth (dct_coeffs) + 2**(depth - 1)) * (257 if depth == 8 else 1)
			newim [top_left [1]:bottom_right [1], top_left [0]:bottom_right [0]] = np.clip (newpatch, 0, 65535)
			
			index += 1
			print (f"{index} of {patches}\r", end="", flush=True)
	
	# Write the resulting reduced image
	images.write_pgm (target, newim)


