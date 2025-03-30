import numpy as np
import dct
import images

def Delta (k: int, l: int) -> np.ndarray:
	m = np.zeros ((8, 8))
	m [k, l] = 1
	return m

for k in range (8):
	for l in range (8):
		d = dct.compute_idct_orth (Delta (k, l)) * 4
		d = images.shift_into_16bit (d)
		images.write_pgm (f"basis/basis_{k},{l}.pgm", d)

