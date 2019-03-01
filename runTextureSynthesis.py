import os
from TextureSynthesis import *
from gifMaker import *

inputImage = "imgs/Fill1.bmp"
os.system('rm -rf out/*')
os.system('rm -rf filled/*')

outputPath = "out/"
kernelSize = 5

textureSynthesis(inputImage,  kernelSize, outputPath,18, attenuation = 80, truncation = 0.8, snapshots = True)

gifOutputPath = "out3.gif"
gifMaker(outputPath, gifOutputPath, frame_every_X_steps = 15, repeat_ending = 15)