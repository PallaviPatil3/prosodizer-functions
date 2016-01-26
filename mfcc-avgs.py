import pandas as pd
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
import os

np.set_printoptions(threshold=np.inf)

targetPath = os.path.abspath("../csv")

zcrFiles = [os.path.join(path, name)
             for path, dirs, files in os.walk(targetPath)
             for name in files if ("angry" in name and name.endswith(("-mfcc.csv")))]

def getmean(filename):
    rms = np.array(pd.read_csv(filename, header=None, dtype='float64'))
    return np.mean(rms)

def splitSignal(MFCCFile):

    MFCC = np.nan_to_num(np.array(pd.read_csv(MFCCFile, header=None), dtype='float64'))

    divFrames = []

    jump = 25
    divFrameLength = 100
    startIndex = 0
    endIndex = divFrameLength
    fileLen = len(MFCC)
    rangeLen = fileLen // endIndex
    
    if fileLen > rangeLen:
        paddingLength = (rangeLen * endIndex) + endIndex - fileLen
        MFCC = np.lib.pad(MFCC, ((0, paddingLength), (0, 0)), 'constant', constant_values = 0)
    
    fileLen = len(MFCC)

    while (endIndex != fileLen):
        # Taking only rows from startIndex to endIndex
        tempMFCC = MFCC[startIndex:endIndex, ]
        ratio = []
        avg1 = []
        avg2 = []
        for t in tempMFCC:
            a, b, c, d, e, f = t[0], t[1], t[2], t[9], t[10], t[11]
            av1 = (a + b + c) / 3
            av2 = (d + e + f) / 3
            ratio.append(av2)
            avg1.append(av1)
            avg2.append(av2)

        intraavg1 = np.nan_to_num(np.array(ratio))
        localmean1 = np.mean(intraavg1)
        intraavg2 = np.nan_to_num(np.array(avg1))
        localmean2 = np.mean(intraavg2)
        intraavg3 = np.nan_to_num(np.array(avg2))
        localmean3 = np.mean(intraavg3)

        print(round(localmean1, 7), round(localmean2, 7), round(localmean3, 7))
        startIndex += jump
        endIndex += jump

    return divFrames

def getValues(MFCCFile):
    MFCC = np.nan_to_num(np.array(pd.read_csv(MFCCFile, header=None), dtype='float64'))
    ratio = []
    avg1 = []
    avg2 = []

    for t in MFCC:
        a, b, c, d, e, f = t[0], t[1], t[2], t[9], t[10], t[11]
        av1 = (a + b + c) / 3
        av2 = (d + e + f) / 3
        ratio.append(av2)
        avg1.append(av1)
        avg2.append(av2)
    intraavg1 = np.nan_to_num(np.array(ratio))
    intraavg2 = np.nan_to_num(np.array(avg1))
    intraavg3 = np.nan_to_num(np.array(avg2))

    return (np.mean(intraavg1), np.std(intraavg1), np.mean(intraavg2), np.std(intraavg2), np.mean(intraavg3), np.std(intraavg3))

def getZscores(globalmean1, globalmean2, globalmean3, globalstd1, globalstd2, globalstd3, intraavg1, intraavg2, intraavg3):
    zscrs = np.array([]) 
    for avg in intraavg1:
        x = (avg - globalmean1) / globalstd1
        zscrs = np.append(zscrs, x)

    p1 = (float(len(zscrs[zscrs > 3])) + float(len(zscrs[zscrs < -3])) ) / float(len(zscrs)) * 100

    zscrs = np.array([]) 
    for avg in intraavg2:
        x = (avg - globalmean2) / globalstd2
        zscrs = np.append(zscrs, x)

    p2 = (float(len(zscrs[zscrs > 3])) + float(len(zscrs[zscrs < -3])) ) / float(len(zscrs)) * 100

    zscrs = np.array([]) 
    for avg in intraavg3:
        x = (avg - globalmean3) / globalstd3
        zscrs = np.append(zscrs, x)

    p3 = (float(len(zscrs[zscrs > 3])) + float(len(zscrs[zscrs < -3])) ) / float(len(zscrs)) * 100
    return p1, p2, p3

def splitSignalZscores(MFCCFile):

    MFCC = np.nan_to_num(np.array(pd.read_csv(MFCCFile, header=None), dtype='float64'))
    globalmean1, globalstd1, globalmean2, globalstd2, globalmean3, globalstd3 = getValues(MFCCFile)

    divFrames = []

    jump = 25
    divFrameLength = 100
    startIndex = 0
    endIndex = divFrameLength
    fileLen = len(MFCC)
    rangeLen = fileLen // endIndex
    
    if fileLen > rangeLen:
        paddingLength = (rangeLen * endIndex) + endIndex - fileLen
        MFCC = np.lib.pad(MFCC, ((0, paddingLength), (0, 0)), 'constant', constant_values = 0)
    
    fileLen = len(MFCC)

    while (endIndex != fileLen):
        # Taking only rows from startIndex to endIndex
        tempMFCC = MFCC[startIndex:endIndex, ]
        ratio = []
        avg1 = []
        avg2 = []
        for t in tempMFCC:
            a, b, c, d, e, f = t[0], t[1], t[2], t[9], t[10], t[11]
            av1 = (a + b + c) / 3
            av2 = (d + e + f) / 3
            ratio.append(av2)
            avg1.append(av1)
            avg2.append(av2)

        intraavg1 = np.nan_to_num(np.array(ratio))
        intraavg2 = np.nan_to_num(np.array(avg1))
        intraavg3 = np.nan_to_num(np.array(avg2))

        print(getZscores(globalmean1, globalmean2, globalmean3, globalstd1, globalstd2, globalstd3, intraavg1, intraavg2, intraavg3))
        startIndex += jump
        endIndex += jump

    return divFrames

def main():
    
    filename = "../training_angry_1.wav-mfcc.csv"
    i = 0
    divFrames = splitSignal(filename)


if __name__ == '__main__':
    main()
