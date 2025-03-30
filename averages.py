import numpy as np
import images
import plots

import sys
import os

from PIL import Image

coeff_total = np.zeros ((8, 8))
patches_total = 0

def image_action (width: int, height: int, im: Image) -> None:
	global coeff_total
	global patches_total
	
	patches_h = width // 8
	patches_v = height // 8
	patches = patches_h * patches_v
	print (f"Patches: {patches}")
	
	index = 0
	for v_index in range (patches_v):
		for h_index in range (patches_h):
			top_left = (h_index * 8, v_index * 8)
			bottom_right = (top_left[0] + 8, top_left[1] + 8)
			
			patch = im.crop ((*top_left, *bottom_right))
			dct_coeffs = images.compute_dct (patch)
			coeff_total += dct_coeffs
			
			index += 1
			print (f"{index} of {patches}\r", end="", flush=True)
	
	patches_total += patches
	print(" " * (len (str (patches)) * 2 + 4))

images.execute_for_all_files (sys.argv [1:], image_action)

if patches_total == 0:
	print ("No patches processed")
	sys.exit ()

coeff_total /= patches_total
print()
print(coeff_total)
print()
with np.printoptions (precision=3, suppress=True):
	print (coeff_total)
print()
print(f"Total patches: {patches_total}")

plots.plot_dct_coefficients (coeff_total)
