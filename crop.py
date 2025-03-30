from PIL import Image
import sys
import os
import shutil

for infile in sys.argv [1:]:
	if os.path.isdir(infile): continue
	
	print (f"FILE: {infile}", end="\t", flush=True)
	path_parts = infile.split("/")
	target = "/".join(path_parts[:-1]) + "/cropped/" + path_parts[-1]
	with Image.open (infile) as im:
		if im.format != "PPM":
			print ("Not a valid PPM/PGM file")
			continue
		
		width, height = im.size
		if width % 8 != 0 or height % 8 != 0:
			im.crop ((0, 0, width - (width % 8), height - (height % 8))).save(target)
			print ()
		else:
			shutil.copy(infile, target)
			print ("Already fits required dimensions")
