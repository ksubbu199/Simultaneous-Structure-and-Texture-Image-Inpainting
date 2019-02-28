import os
from TextureSynthesis import *

inputImage = "imgs/Fill1.bmp"
os.system('rm -rf out/*')
os.system('rm -rf filled/*')

outputPath = "out/"
kernelSize = 5

textureSynthesis(inputImage,  kernelSize, outputPath,18, attenuation = 80, truncation = 0.8, snapshots = True)
