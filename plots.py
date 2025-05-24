import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import SymmetricalLogLocator

def plot_dct_coefficients (matrix: np.ndarray) -> None:
	"""
	Unified function to generate a plot of DCT coefficients.
	
	Uses fixed colour map ("jet") and fixed lower and upper
	limits of (0, 2.5). Opens the plot in a matplotlib window.
	
	:matrix: The coefficient matrix to plot
	"""
	fig, ax = plt.subplots ()
	im = ax.imshow (np.abs (matrix), cmap="jet", norm="symlog")
	im.set_clim (0, 100)
	im.get_cmap ().set_over (color="w")
	plt.colorbar(im, ticks=SymmetricalLogLocator(base=10, linthresh=1, subs=range(10)))
	
	ax.axhline (3.5, linestyle='--', color="k")
	ax.axvline (3.5, linestyle='--', color="k")
	
	plt.show()

def display_image (image: np.ndarray) -> None:
	"""
	Unified function to display a black-and-white image
	
	:matrix: The image matrix to display
	"""
	plt.gray ()
	plt.axis ("off")
	plt.imshow (image, vmin=0)
	plt.show ()
