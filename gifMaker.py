#for gif making
import imageio 
import numpy as np
import os
from PIL import Image
from math import floor

def gifMaker(savePath, outputPath, frame_every_X_steps = 15):
    fileCount = len(os.listdir(savePath))-1
    steps = np.arange(floor(fileCount/frame_every_X_steps)) * frame_every_X_steps
    steps = steps + (fileCount - np.max(steps))
    images = []
    for f in steps:
        filename = savePath + 'out' + str(f) + '.jpg'
        images.append(imageio.imread(filename))
    imageio.mimsave(outputPath, images)