import numpy as np
import scipy.stats as st

from random import randint
from math import floor
from skimage import io, feature, transform 

import imageio 
from PIL import Image
import matplotlib.pyplot as plt

def textureSynthesis(inputImagePath, kernelSize, backgroundThresh, attenuation = 80, truncation = 0.8, snapshots = True):
        
    if kernelSize % 2 == 0:
        kernelSize = kernelSize + 1
        
    exampleMap = readImage(inputImagePath)
    imgRows, imgCols, imgChs = np.shape(exampleMap)
    
    canvas, filledMap, filledPixelsCount = getCanvasAndFilledmap(exampleMap, backgroundThresh)
    stackOfPatches = getStackOfPatches(exampleMap, filledMap, kernelSize)
    totalPixelsCount = imgRows*imgCols
    
    gaussian = getGaussian(kernelSize, kernelSize)
    
    i = 0
    while filledPixelsCount < totalPixelsCount:
        curRow, curCol = getBestCoords(filledMap, 5)
        
        curPatch = getNeighbourhood(canvas, kernelSize, curRow, curCol)

        curPatchMask = gaussian * getNeighbourhood(filledMap, kernelSize, curRow, curCol)
        curPatchMask = np.repeat(curPatchMask[:, :, np.newaxis], 3, axis=2)

        stackCountOfPatches = np.shape(stackOfPatches)[0]
        curPatchMask = np.repeat(curPatchMask[np.newaxis, :, :, :, ], stackCountOfPatches, axis=0)
        curPatch = np.repeat(curPatch[np.newaxis, :, :, :, ], stackCountOfPatches, axis=0)

        distances = curPatchMask * pow(stackOfPatches - curPatch, 2)
        distances = np.sum(np.sum(np.sum(distances, axis=3), axis=2), axis=1)

        probabilities = calcProbabilitiesFromDistances(distances, truncation, attenuation)
        sample = np.random.choice(np.arange(stackCountOfPatches), 1, p=probabilities)
        #print(sample)
        #sample = np.argmax(probabilities)
        chosenPatch = stackOfPatches[sample]
        halfKernel = floor(kernelSize / 2)
        chosenPixel = np.copy(chosenPatch[0, halfKernel, halfKernel])

        canvas[curRow, curCol, :] = chosenPixel
        filledMap[curRow, curCol] = 1

        filledPixelsCount = filledPixelsCount+1
        i = i+1
        if snapshots:
            img = Image.fromarray(np.uint8(canvas*255))
            img = img.resize((300, 300), resample=0, box=None)
            img.save(savePath + 'out' + str(i) + '.jpg')
    
    if snapshots==False:
        img = Image.fromarray(np.uint8(canvas*255))
        img = img.resize((300, 300), resample=0, box=None)
        img.save('out.jpg')

def getNeighbourhood(mapToGetNeighbourhoodFrom, kernelSize, row, col):
    
    halfKernel = floor(kernelSize / 2)
    if mapToGetNeighbourhoodFrom.ndim == 3:
        npad = ((halfKernel, halfKernel), (halfKernel, halfKernel), (0, 0))
    elif mapToGetNeighbourhoodFrom.ndim == 2:
        npad = ((halfKernel, halfKernel), (halfKernel, halfKernel))
    else:
        print('ERROR:invalid dimension!')
        return None

    paddedMap = np.lib.pad(mapToGetNeighbourhoodFrom, npad, 'constant', constant_values=0)
    return paddedMap[row:row+2*halfKernel +1, col:col+2*halfKernel+1]

def calcProbabilitiesFromDistances(distances, PARM_truncation, attenuation):
    
    probabilities = 1 - distances / np.max(distances)
    probabilities_backup = np.array(probabilities)
    probabilities *= (probabilities > PARM_truncation)
    if np.max(probabilities)==0:
        probabilities = probabilities_backup
        probabilities *= (probabilities > PARM_truncation*np.max(probabilities))
    probabilities = pow(probabilities, attenuation)
    probabilities /= np.sum(probabilities)
    return probabilities
   
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