import numpy as np
import scipy.stats as st

from random import randint
from math import floor
from skimage import io, feature, transform 

import imageio 
from PIL import Image
import matplotlib.pyplot as plt

def textureSynthesis(inputImagePath, kernelSize):
        
    if kernelSize % 2 == 0:
        kernelSize = kernelSize + 1
        
    exampleMap = readImage(inputImagePath)
    imgRows, imgCols, imgChs = np.shape(exampleMap)
    
    canvas, filledMap, filledPixelsCount = getCanvasAndFilledmap(exampleMap, backgroundThresh)
    stackOfPatches = getStackOfPatches(exampleMap, filledMap, kernelSize)
    totalPixelsCount = imgRows*imgCols
    
    gaussian = getGaussian(kernelSize, kernelSize)
    
    while filledPixelsCount < totalPixelsCount:
        curRow, curCol = getBestCoords(filledMap, 5)

def getBestCandidateCoord(bestCandidateMap):
    size = bestCandidateMap.shape
    curRow = floor(np.argmax(bestCandidateMap) / size[1])
    curCol = np.argmax(bestCandidateMap) - curRow * size[1]
    return curRow, curCol

def getBestCoords(filledMap, kernelSize):
    bestCandidateMap = np.zeros(filledMap.shape)
    x,y = np.nonzero(1-filledMap)
    for i in range(len(x)):
        r  = x[i]
        c = y[i]
        bestCandidateMap[r, c] = np.sum(getNeighbourhood(filledMap, kernelSize, r, c))
    return getBestCandidateCoord(bestCandidateMap)

def readImage(inputImagePath):
    exampleMap = io.imread(inputImagePath)
    exampleMap = exampleMap / 255.0
    if (np.shape(exampleMap)[-1] > 3): 
        exampleMap = exampleMap[:,:,:3]
    elif (len(np.shape(exampleMap)) == 2):
        exampleMap = np.repeat(exampleMap[np.newaxis, :, :], 3, axis=0)
    return exampleMap

def getCanvasAndFilledmap(exampleMap,thresh):
    threshold = 1.0*thresh/255.0
    canvas = np.array(exampleMap)
    filledMap = np.array(canvas[:,:,0])
    filledMap[filledMap > threshold] = 1
    filledMap[filledMap <= threshold] = 0
    num_filled_pixels = np.sum(filledMap)
    return canvas, filledMap, num_filled_pixels

def getStackOfPatches(exampleMap, filledMap, kernelSize):
    imgRows, imgCols, imgChs = np.shape(exampleMap)
    num_horiz_patches = imgRows - (kernelSize-1)
    num_vert_patches = imgCols - (kernelSize-1)
    stackOfPatches  = []
    for r in range(num_horiz_patches):
        for c in range(num_vert_patches):
            if np.count_nonzero(filledMap[r:r+kernelSize, c:c+kernelSize] == 0) < 5:
                stackOfPatches.append(exampleMap[r:r+kernelSize, c:c+kernelSize])

    stackOfPatches = np.array(stackOfPatches)
    print(stackOfPatches.shape)
    return stackOfPatches

def getGaussian(kern_x, kern_y, nsig=3):
    interval = (2*nsig+1.)/(kern_x)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kern_x+1)
    kern1d_x = np.diff(st.norm.cdf(x))

    interval = (2*nsig+1.)/(kern_y)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kern_y+1)
    kern1d_y = np.diff(st.norm.cdf(x))
    
    kernel_raw = np.sqrt(np.outer(kern1d_x, kern1d_y))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel