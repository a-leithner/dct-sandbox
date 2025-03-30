import numpy as np
import dct

import os

from PIL import Image
from collections.abc import Callable

def image_to_matrix (patch: Image) -> np.ndarray:
	if patch.size != (8, 8):
		raise RuntimeError ("Image patch has unsupported dimensions")
	return np.array (patch.getdata ()).reshape ((8, 8))

def compute_dct (patch: Image) -> np.ndarray:
	if patch.size != (8, 8):
		raise RuntimeError ("Image patch has unsupported dimensions; crop to 8x8 pixels first")
	
	return dct.compute_dct (image_to_matrix (patch))

def compute_dct_orth (patch: Image) -> np.ndarray:
	if patch.size != (8, 8):
		raise RuntimeError ("Image patch has unsupported dimensions; crop to 8x8 pixels first")
	
	return dct.compute_dct_orth (image_to_matrix (patch))

def valid_image (im: Image) -> tuple[bool, str]:
	if im.format != "PPM":
		return False, "Given image is not in PGM format"
	
	width, height = im.size
	if width % 8 != 0 or height % 8 != 0:
		return False, "Given image has unsupported dimensions: Both width and height need to be multiples of 8."
	
	return True, None

def num_to_bytes (num: int) -> list:
	if num == 0: return [0, 0]
	elif num <= 255: return [0, num]
	else: return [(num & 0xFF00) >> 8, num & 0xFF]

def write_pgm (filename: str, data: np.ndarray) -> None:
	if len(data.shape) != 2:
		raise ValueError ("Writing images requires two-dimensional arrays")
	
	height, width = data.shape
	with open (filename, "bw+") as f:
		f.write (bytes(f"P5\n{width} {height}\n65535\n", "ascii"))
		for entry in data.reshape (width * height):
			l = num_to_bytes (int (entry))
			f.write (bytes (l))

def invert_color (data: np.ndarray) -> np.ndarray:
	return np.full (data.shape, 65535) - data

def execute_for_all_files (files: list[str], func: Callable[[int, int, Image], None], printinfo: bool = True) -> None:
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

# Function used to transport the matrix' values in [-1,1] to [0,65535]
def shift_into_16bit (matrix: np.ndarray) -> np.ndarray:
	return np.rint ((matrix + 1) * 65535 / 2)
