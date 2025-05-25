import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import SymmetricalLogLocator
from matplotlib.colors import LightSource
from matplotlib.colorizer import Colorizer
from matplotlib.cm import ScalarMappable

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

def plot_dct_coefficients_3d (matrix: np.ndarray) -> None:
	"""
	Unified function to generate a 3D plot of DCT coefficients.
	
	Uses a fixed colour map ("jet") and fixed lower and upper
	limits of (0, 100). Values above 100 will be clipped to 101
	and painted black. Opens the plot in a matplotlib window.
	
	:matrix: The coefficient matrix to plot
	"""
	data = np.abs (matrix)
	data [data > 100] = 101
	data = data.ravel ()
	
	xidx = np.array([[i] * 8 for i in range (8)]).ravel ()
	yidx = np.array([[j for j in range (8)] for i in range (8)]).ravel ()
	zidx = 0
	
	s = np.ones_like (xidx)
	fig, ax = plt.subplots (subplot_kw=dict(projection="3d"))
	
	ls = LightSource (270, 45)
	c = Colorizer (cmap="jet", norm="symlog")
	c.set_clim (0, 100)
	c.cmap.set_over ("k")
	rgb = c.to_rgba (data)
	
	ax.bar3d (xidx, yidx, zidx, s, s, data, color=rgb, lightsource=ls, zsort="max")
	ax.set_ylim (8, 0)
	ax.set_xlim (0, 8)
	ax.set_zlim (0, 100)
	fig.colorbar (ScalarMappable (colorizer=c), ax=ax, ticks=SymmetricalLogLocator(base=10, linthresh=1, subs=range(10)))
	
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
