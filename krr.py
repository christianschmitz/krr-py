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
    parser.add_argument("-r", "--ranges", help="comma separated list of ranges, eg: a0:a1:Na,b0:b1,c,d,e. Overrides indep variables from FILE or RANGE-FILE")
    parser.add_argument("-f", "--range-file", help="overrides the indep variables in FILE")
    parser.add_argument("-s", "--sigma", help="gaussian kernel deviation, default=1.0, can be comma separated list for multiple observed variables", default="1.0")
    parser.add_argument("-a", "--alpha", help="penalization parameter, default=0.1, can be comma separated list for multiple observed variables", default="0.1")
    parser.add_argument("-o", "--optimize", help="column index from which the optimum is always selected, can only be used in combination with -r")
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
                ranges[rangeI]=[float(i) for i in ranges[rangeI].split(":")]
                if len(ranges[rangeI]) > 3:
                    raise Exception("")
        except:
            raise Exception("Invalid format for ranges")
    elif (args.range_file):
        rangeFile = args.range_file

    if ranges:
        numIndep = len(ranges)
    else:
        numIndep = args.num_indep

    optimColumns = [-1,-1]
    if args.optimize:
        if not ranges:
            raise Exception("--optimize must be used in combination with --ranges")
        else:
            optimColumns = [int(i) for i in args.optimize.split(",")]
            if (not len(optimColumns) == 2):
                raise Exception("--optimize requires argument in form INDEPC,OBSERVC")

    sigma = [float(s) for s in args.sigma.split(",")]
    alpha = [float(a) for a in args.alpha.split(",")]
    return (args.file, numIndep, ranges, rangeFile, sigma, alpha, optimColumns)


def getPredVarsFromRanges(ranges):
    defaultNum = 50
    linSpaces = []
    numIndep = len(ranges)
    for r in ranges:
        if len(r) == 3:
            linSpaces.append(np.linspace(r[0],r[1],r[2],endpoint=True)[:])
        elif len(r) == 2:
            linSpaces.append(np.linspace(r[0],r[1],defaultNum, endpoint=True)[:])
        elif (len(r) == 1):
            linSpaces.append(np.array([r[0]]))
        else:
            raise Exception("Error in ranges: "+str(r))

    meshGrid = np.meshgrid(tuple(linSpaces))
    # Now concat these horizontally
    numPoints = meshGrid[0].size
    predVars = np.zeros((numPoints, len(meshGrid)))
    for i in range(0,len(meshGrid)):
        predVars[:,i] = np.reshape(meshGrid[i], (numPoints))

    return predVars


def getPredVarsFromFile(numIndep , rangeFile):
    predVarData = np.loadtxt(rangeFile)
    if numIndep == 0:
        numIndep = predVarData.shape[1]-1
    predVars = predVarData[:,0:numIndep]
    return predVars


def getPredVars(numIndep, ranges, rangeFile):
    if ranges:
        predVars = getPredVarsFromRanges(ranges)
    else:
        predVars = getPredVarsFromFile(numIndep, rangeFile)

    if len(predVars.shape)==1:
        predVars = np.reshape(predVars, (predVars.shape[0], 1))
    return predVars
        
        
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


def predict(x, y, xf, sigma, alpha):
    #scaling = calcOptimalScaling(x)
    jumps = calcTypicalJumps(x)
    scaling = 0.1/jumps
    kernel = mlpy.KernelGaussian(sigma)
    krr = mlpy.KernelRidgeRegression(kernel, alpha)
    krr.learn(x*scaling, y)
    yf = krr.pred(xf*scaling)
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


def getSortableFieldArray(x, numIndep, indepColumn):
    nR = x.shape[0]
    nC = x.shape[1]

    dtype=[]
    dtypeSort=[]
    for i in range(0,nC):
        name=str(i)
        dtype.append((name, float))

        if i < numIndep-1 and (not i==indepColumn):
            dtypeSort.append(name)

    values = []
    for i in range(0, nR):
        l = x[i,:].tolist()
        values.append(tuple(l))
    a = np.array(values, dtype = dtype)
    return (a, dtypeSort)


def extractStructuredArray(x):
    l = x.tolist()
    nR = len(l)
    nC = len(list(l[0]))

    a = np.zeros((nR, nC))

    for i in range(0,nR):
        a[i,:] = np.array(list(l[i]))[:]

    return a


def optimize(dataOut, numIndep, optimColumns):
    indepColumn = optimColumns[0] 
    obserColumn = optimColumns[1]

    (x, dtypeSort) = getSortableFieldArray(dataOut, numIndep, indepColumn)
    xs_ = np.sort(x, order=dtypeSort)
    xs = extractStructuredArray(xs_)

    # Now loop the array looking for the optimal condition
    

    # First step is to sort over all the columns except the 
    return dataOut


#TODO: automatic determination of good and robust values for sigma and alpha
#TODO: normalization of indepVar input data so that single sigma and alpha are optimal over whole input space
#TODO: choose between maximize and minimize
def main():
    (fname, numIndep, ranges, rangeFile, sigma, alpha, optimColumns) = parseArgs()

    predVars = getPredVars(numIndep, ranges, rangeFile)

    predOut = predictObserved(fname, predVars, sigma, alpha)

    # Print the result 
    dataOut = np.c_[predVars, predOut]

    if (optimColumns[0] >= 0):
        dataOut = optimize(dataOut, numIndep, optimColumns)
    np.savetxt(sys.stdout, dataOut, fmt='%g')
    return


main()
