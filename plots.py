import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_dct_coefficients (matrix: np.ndarray) -> None:
	"""
	Unified function to generate a plot of DCT coefficients.
	
	Uses fixed colour map ("jet") and fixed lower and upper
	limits of (0, 2.5). Opens the plot in a matplotlib window.
	
	:matrix: The coefficient matrix to plot
	"""
	fig, ax = plt.subplots ()
	im = ax.imshow (np.abs (matrix), cmap="jet")
	im.set_clim (0, 2.5)
	im.get_cmap ().set_over (color="w")
	plt.colorbar(im)
	
	ax.axhline (3.5, linestyle='--', color="k")
	ax.axvline (3.5, linestyle='--', color="k")
	
	plt.show()
