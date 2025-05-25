import numpy as np
import images
import plots

import sys

from PIL import Image

dct_coeff_total = np.zeros ((8, 8))
dft_coeff_total = np.zeros ((8, 8))
patches_total = 0

def image_action (width: int, height: int, im: Image) -> None:
	global dct_coeff_total, dft_coeff_total
	global patches_total
	
	patches_h = width // 8
	patches_v = height // 8
	patches = patches_h * patches_v
	print (f"Patches: {patches}")
	
	index = 0
	for v_index in range (patches_v):
		for h_index in range (patches_h):
			# Coordinates of the currently processed patch.
			# These are in PIL coordinates, meaning that the horizontal
			# axis is x and the vertical axis is y.
			top_left = (h_index * 8, v_index * 8)
			bottom_right = (top_left[0] + 8, top_left[1] + 8)
			
			# Extract the processed patch, process, and store DFT and DCT coefficients
			patch = im.crop ((*top_left, *bottom_right))
			
			dft_coeffs = images.compute_dft (patch)
			dft_coeff_total += np.abs(dft_coeffs)
			
			dct_coeffs = images.compute_dct_orth (patch)
			dct_coeff_total += np.abs(dct_coeffs)
			
			index += 1
			print (f"{index} of {patches}\r", end="", flush=True)
	
	patches_total += patches
	
	# Clear progress indicator
	print(" " * (len (str (patches)) * 2 + 4))

plot3d = False
if "--3d" in sys.argv:
	plot3d = True
	sys.argv.remove ("--3d")

images.execute_for_all_files (sys.argv [1:], image_action)

if patches_total == 0:
	print ("No patches processed")
	sys.exit ()

dft_coeff_total /= patches_total
dct_coeff_total /= patches_total
print ()
print ("Average DFT coefficient matrix (absolute values):")
print (dft_coeff_total)
print ()
print ("Average DCT-II coefficient matrix:")
print (dct_coeff_total)
print ()
with np.printoptions (precision=3, suppress=True):
	print ("Average DFT coefficient matrix (reduced precision):")
	print (dft_coeff_total)
	print ()
	print ("Average DCT-II coefficient matrix (reduced precision):")
	print (dct_coeff_total)
print ()
print (f"Total patches: {patches_total}")

plotfunc = plots.plot_dct_coefficients
if plot3d:
	plotfunc = plots.plot_dct_coefficients_3d

# Open in Matplotlib windows
plotfunc (dft_coeff_total)
plotfunc (dct_coeff_total)
