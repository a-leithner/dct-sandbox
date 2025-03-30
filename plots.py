import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_dct_coefficients (matrix: np.ndarray) -> None:
	fig, ax = plt.subplots ()
	#im = ax.imshow (np.abs (matrix), norm="symlog", cmap="jet")
	im = ax.imshow (np.abs (matrix), cmap="jet")
	im.set_clim (0, 2.5)
	im.get_cmap ().set_over (color="w")
	plt.colorbar(im)
	
	ax.axhline (3.5, linestyle='--', color="k")
	ax.axvline (3.5, linestyle='--', color="k")
	
	plt.show()
