#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import mlpy
import sys



def getParser():    
    parser = argparse.ArgumentParser(description="kernel ridge regression of N independent variables vs M observed variables")
    parser.add_argument("file", help="data file with independent variable columns first, then the observed variable columns")
    parser.add_argument("-n", "--num-indep", type=int, default=0, help="number of independent variables")
    parser.add_argument("-r", "--ranges", help="comma separated list of ranges, eg: a0:a1:Na:max(2),b0:b1,c,d,e. Overrides indep variables from FILE or RANGE-FILE")
    parser.add_argument("-f", "--range-file", help="overrides the indep variables in FILE")
    parser.add_argument("-s", "--sigma", help="gaussian kernel deviation, default=1.0, can be comma separated list for multiple observed variables", default="1.0")
    parser.add_argument("-a", "--alpha", help="penalization parameter, default=0.1, can be comma separated list for multiple observed variables", default="0.1")
    return parser


def parseArgs():
    parser = getParser()
    args = parser.parse_args()
    
    rangeFile = args.file
    ranges = None
    if (not args.ranges == None):
        rangeFile = None
        try:
            ranges = args.ranges.split(",")
            for rangeI in range(0,len(ranges)):
                rangeTokens = ranges[rangeI].split(":")
                if (len(rangeTokens) == 4):
                    ranges[rangeI] = [float(rangeTokens[0]), float(rangeTokens[1]),
                            float(rangeTokens[2]), rangeTokens[3]]
                else:
                    ranges[rangeI]=[float(i) for i in ranges[rangeI].split(":")[:-1]]
                if len(ranges[rangeI]) > 4:
                    raise Exception("")
        except:
            raise Exception("Invalid format for ranges")
    elif (args.range_file):
        rangeFile = args.range_file

    if ranges:
        numIndep = len(ranges)
    else:
        numIndep = args.num_indep

    sigma = [float(s) for s in args.sigma.split(",")]
    alpha = [float(a) for a in args.alpha.split(",")]
    return (args.file, numIndep, ranges, rangeFile, sigma, alpha)


def flattenMeshGrid(linSpaces):
    meshGrid = np.meshgrid(tuple(linSpaces))
    # Now concat these horizontally
    numPoints = meshGrid[0].size
    x = np.zeros((numPoints, len(meshGrid)))
    for i in range(0,len(meshGrid)):
        x[:,i] = np.reshape(meshGrid[i], (numPoints))
    return x


def getPredVarsFromRanges(ranges):
    defaultNum = 50
    linSpaces = []
    indexSpaces = []
    numIndep = len(ranges)
    for r in ranges:
        if len(r) == 3 or len(r) == 4:
            linSpaces.append(np.linspace(r[0],r[1],r[2],endpoint=True)[:])
            indexSpaces.append([int(i) for i in np.linspace(0,r[2]-1,r[2], endpoint=True)[:]])
        elif len(r) == 2:
            linSpaces.append(np.linspace(r[0],r[1],defaultNum, endpoint=True)[:])
            indexSpaces.append(np.array([int(i) for i in np.linspace(0,defaultNum-1,defaultNum, endpoint=True)[:]]))
        elif (len(r) == 1):
            linSpaces.append(np.array([r[0]]))
            indexSpace.append(np.array([0]))
        else:
            raise Exception("Error in ranges: "+str(r))

    predVars = flattenMeshGrid(linSpaces)
    ndIndexing = flattenMeshGrid(indexSpaces)

    return (predVars, linSpaces, ndIndexing)


def getPredVarsFromFile(numIndep , rangeFile):
    predVarData = np.loadtxt(rangeFile)
    if numIndep == 0:
        numIndep = predVarData.shape[1]-1
    predVars = predVarData[:,0:numIndep]
    return predVars


def getPredVars(numIndep, ranges, rangeFile):
    if ranges:
        (predVars, linSpaces, ndIndexing) = getPredVarsFromRanges(ranges)
    else:
        predVars = getPredVarsFromFile(numIndep, rangeFile)
        linSpaces = None
        ndIndexing = None

    if len(predVars.shape)==1:
        predVars = np.reshape(predVars, (predVars.shape[0], 1))
    return (predVars, linSpaces, ndIndexing)
        
        
def calcOptimalScaling(x):
    n = x.shape[1]
    scaling = np.zeros(n)
    for i in range(0,n):
        scaling[i] = 1.0/np.amax(x[:,i])
    return scaling


# input: 1d array
def calcTypicalJump(x):
    xs = np.sort(x)
    d = xs[1:]-xs[:-1]

    jump = np.amax(d)
    return jump


def calcTypicalJumps(x):
    n = x.shape[1]
    jumps = np.zeros(n)
    for i in range(0,n):
        jumps[i] = calcTypicalJump(x[:,i])
    return jumps


def calcOptimalOffsets(x):
    n = x.shape[1]
    offsets = np.zeros(n)
    for i in range(0,n):
        offsets[i] = np.average(x[:,i])
    return offsets


def predict(x, y, xf, sigma, alpha):
    ## not needed, but kep for flexibility
    #scaling = calcOptimalScaling(x)
    #offsets = calcOptimalOffsets(x) 

    jumps = calcTypicalJumps(x)
    scaling = 0.1/jumps

    kernel = mlpy.KernelGaussian(sigma)
    krr = mlpy.KernelRidgeRegression(kernel, alpha)
    scaleY = 1.0/np.amax(y)
    krr.learn(x*scaling, y*scaleY)
    yf = krr.pred(xf*scaling)/scaleY
    return yf


def predictObserved(fname, predVars, sigma, alpha):
    data = np.loadtxt(fname)
    numIndep = predVars.shape[1]
    indepData = data[:,0:numIndep]
    numObser = data.shape[1] - numIndep
    obserData = data[:,numIndep:]
    predOut = np.zeros((predVars.shape[0], numObser))

    if (numObser < 1):
        raise Exception("not enough columns for "+str(numIndep)+" indep variables")
    if (len(sigma)==1 and numObser > 1):
        sigma = [sigma[0]]*numObser
    if (len(alpha) == 1 and numObser> 1):
        alpha = [alpha[0]]*numObser

    if (not len(sigma)  == numObser):
        raise Exception("sigma length wrong")
    if (not len(alpha) == numObser):
        raise Exception("alpha length wrong")

    for i in range(0,numObser):
        y = obserData[:,i]
        yf = predict(indepData, y, predVars, sigma[i], alpha[i])
        predOut[:,i] = yf[:]
    return predOut


def parseOptimizationToken(string):
    tokens = string.split('(')
    tokens = [tokens[0]]+tokens[1].split(')')

    mode = tokens[0]
    colI = int(tokens[1])
    return (mode, colI)


def optimizePredVars(allData, numIndep, ranges, linSpaces, ndIndexing):
    doOptimization = False
    for r in ranges:
        if (len(r)==4):
            doOptimization = True
    if not doOptimization:
        return allData

    ## put data into matrix form
    # first get the shape of this matrix
    numAll = allData.shape[1]
    numObser = numAll - numIndep
    numPoints = allData.shape[0]

    sizes = []
    for i in range(0,numIndep):
        sizes.append(int(np.amax(ndIndexing[:,i])))

    data = []
    for j in range(0,numAll):
        # create an empty matrix
        matrix = np.zeros(tuple(sizes))

        # loop the data, putting the predOut vars in there
        for i in range(0, numPoints):
            matrix[ndIndexing[i,:]] = allData[i,j]
        data.append(matrix)

    ## maximize along certain axes, do this by looping the ranges
    for i in range(0,numIndep):
        r = ranges[i]
        if (len(r) == 4):
            if (r[3] > numIndep-1):
                (mode, colI) = parseOptmizationToken(r[3])
                optimI = colI - numIndep
                if (mode == 'max'):
                    optimIDS = np.argmax(data[optimI], axis=i)
                    data[optimI] = np.amax(data[optimI], axis=i)
                elif (mode == 'min'):
                    optimIDS = np.argmin(data[optimI], axis=i)
                    data[optimI] = np.amin(data[optimI], axis=i)
                else:
                    raise Exception("Mode \""+str(mode)+"\" not recognized")
                # Loop the other data in order to do the same reduction
                for j in range(0,numAll):
                    if (not j == int(r[3])):
                        data[j] = data[j][optimIDS]

            else:
                raise Exception("Invalid column indexing for optimizing utility")
    
    ## Convert the matrix back to the flattened form
    allData = data[0].reshape((data[0].size))
    for i in range(1,numAll):
        allData = np.c_[allData, data[i].reshape((data[1].size))]
    return allData


#TODO: automatic determination of good and robust values for sigma and alpha
def main():
    (fname, numIndep, ranges, rangeFile, sigma, alpha) = parseArgs()

    (predVars, linSpaces, ndIndexing) = getPredVars(numIndep, ranges, rangeFile)

    predOut = predictObserved(fname, predVars, sigma, alpha)

    (predVars, predOut) = optimizePredVars(predVars, predOut, ranges, linSpaces, ndIndexing)

    dataOut = np.c_[predVars, predOut]
    dataOut = optimizePredVars(dataOut, numIndep, ranges, linSpaces, ndIndexing)

    np.savetxt(sys.stdout, dataOut, fmt='%g')
    return


main()
