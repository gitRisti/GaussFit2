import matplotlib.pyplot as plt
import matplotlib.mlab
import xlrd
import numpy as np
from scipy import optimize
from scipy.special import erf
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import math
import copy

values_x = []
values_y = []
gaussian_x = []
gaussian_y = []
#Default distribution fitMode = 1 is Gaussian
fitMode = 2
fixedPeaks = True
#Shift ppm for better fit
chemShift = 0.016

#'r' preceding the string turns off the eight-character Unicode escape (for a raw string)
workbook = xlrd.open_workbook(r"C:\Users\robik\Desktop\FASTEDLPL.xlsx")

#Get worksheet by index
worksheet = workbook.sheet_by_index(0)

#Iterate over first two columns, starting from row 3 (skip titles)
#NMRi jaoks vahetuses -> column 1 = x ja column 0 = y
#worksheet.nrows
if fitMode == 2:
    for columnIndex in range (0, worksheet.ncols):
        for rowIndex in range (100266, 102123):
            if columnIndex == 0:
                values_y.append(float(worksheet.cell(rowIndex,columnIndex).value))
            if columnIndex == 1:
                values_x.append(float(worksheet.cell(rowIndex,columnIndex).value))
elif fitMode == 1:
    for columnIndex in range (0, worksheet.ncols):
        for rowIndex in range (3, 800):
            if columnIndex == 0:
                values_x.append(float(worksheet.cell(rowIndex,columnIndex).value))
            if columnIndex == 1:
                values_y.append(float(worksheet.cell(rowIndex,columnIndex).value))

#List of variables
pW = [] #Initial guesses for peak width
pM = [] #Initial guesses for peak mean
pH = [] #Intial guesses for peak height

def fitFormula(x,pars): 
    height = pars[0]
    center = pars[1]
    width = pars[2]
    #Gaussian
    if fitMode == 1:
        return height*np.exp(-(x - center)**2/(2*width**2))
    #Cauchy
    if fitMode == 2:
        return height*((width**2)/(((x-center)**2)+width**2))

#Substract baseline and normalization (Normalization is necessary because data will not fit when using values that are too large or too small)

maxValue = np.amax(values_y)
for i in range (0,len(values_y)):

    values_y[i] = values_y[i]/maxValue
#Show raw data
showRawData = False
showMarkedPeaks = True
if showRawData == True:
    plt.plot(values_x,values_y, label = "Raw data")
    plt.show()
#figTitle = input("Enter sample name: \n")
fitTitle = "Test"
fitMode = int(input("Choose distribution function: \n 1 - Gaussian \n 2 - Cauchy \n"))
insertPeaksManually = input("Do you want to enter peaks manually? (y/n/5/6) \n")
if insertPeaksManually == "y":
    peakNames = []
    nPeaks = int(input("Please enter # of peaks:\n"))
    aM = []
    indices = []
    for i in range(1,nPeaks+1):
        indices.append(float(input("Enter mean for peak " + str(i) + "\n")))
        #Find corresponding array index for values_x (idx = (np.abs(arr - v)).argmin())
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[-1]))
        pM.append(np.argmin(aM))
        aM.clear()
    for i in pM:
        plt.plot(values_x[i],values_y[i],"x")
        plt.plot(values_x[i],values_y[i],"x")
    for i in range(1,nPeaks+1):
        peakNames.append(input("Enter name for peak " + str(i) + "\n"))
elif insertPeaksManually == "n":
    #Find noise (standard deviation of baseline)
    noise = np.std(values_y[0:10])
    #, = tuple unpacking
    peaks, _ = (find_peaks(values_y,100))
    print(peaks)
    for i in peaks:
        plt.plot(values_x[i],values_y[i],"x")
    nPeaks = peaks.shape
    nPeaks = nPeaks[0]
    print("Found " + str(nPeaks) + " peaks")
elif insertPeaksManually == "5":
    peakNames = ["VLDL","LDL","REM","HDL","HSA"]
    aM = []
    nPeaks = 5
    indices = [7.52,11.3,12.6,15.8,16.8]
    print(len(indices))
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    for i in pM:
        plt.plot(values_x[i],values_y[i],"x")
        plt.plot(values_x[i],values_y[i],"x")
    pW = [0.3,0.8,0.8,0.5,0.3]
elif insertPeaksManually == "6":
    peakNames = ["VLDL","LDL","REM","HDL","HSA","PROD"]
    aM = []
    nPeaks = 6
    indices = [7.45,11.4,14.2,15.65,16.29,17]
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    for i in pM:
        plt.plot(values_x[i],values_y[i],"x")
        plt.plot(values_x[i],values_y[i],"x")
    pW = [0.3,0.8,0.8,0.4,0.2,0.2]
    pH = [0.4,0.4,0.4,0.7,0.7,0.4]
elif insertPeaksManually == "LDL":
    peakNames = ["LDL1","LDL2","LDL3","LDL4","LDL5","LDL6","LDL7","PROTEIN"]
    aM = []
    nPeaks = 8
    indices = [0.8454868,0.85046652,0.85448537,0.86676283,0.85954238,0.88240966,0.87473024,0.95640]
    meanPROTEIN = [0.95640]
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    if showMarkedPeaks == True:
        for i in pM:
            plt.plot(values_x[i],values_y[i],"x")
            plt.plot(values_x[i],values_y[i],"x")
    pW = [0.00644499,0.00458794,0.00441878,0.00740763,0.00537808,0.00500155,0.00449342,0.14676]
    pH = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.154]
    widthPROTEIN = [0.14676]

elif insertPeaksManually == "VLDL":
    peakNames = ["VLDL1","VLDL2","VLDL3","VLDL4","VLDL5","VLDL6","VLDL7","PROTEIN"]
    aM = []
    nPeaks = 8
    indices = [0.87900098,0.88583026,0.8888809,0.89199671,0.89528075,0.89818989,0.90332644,0.9464]
    meanPROTEIN =[0.9464]
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    if showMarkedPeaks == True:
        for i in pM:
            plt.plot(values_x[i],values_y[i],"x")
            plt.plot(values_x[i],values_y[i],"x")
    pW = [0.00751988,0.00422151,0.00324447,0.00293996,0.00273448,0.00284239,0.00228012,0.009]
    pH = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.2]
    widthPROTEIN =[0.009]
elif insertPeaksManually == "PLASMA":
    namesHDL = ["HDL1","HDL2","HDL3","HDL4","HDL5","HDL6","HDL7"]
    namesLDL = ["LDL1","LDL2","LDL3","LDL4","LDL5","LDL6","LDL7"]
    namesVLDL = ["VLDL1", "VLDL2" , "VLDL3", "VLDL4", "VLDL5","VLDL6","VLDL7"]
    peakNames = [*namesHDL,*namesLDL,*namesVLDL]
    peakNames.append("PROTEIN")
    aM = []
    #Data derived from individually fitted LP-fractions
    meanHDL = [0.82834298,0.83392093,0.84316387,0.85391685,0.848127,0.83858213,0.86173502]
    meanLDL = [0.8454868,0.85046652,0.85448537,0.86676283,0.85954238,0.88240966,0.87473024]
    meanVLDL = [0.87900098,0.88583026,0.8888809,0.89199671,0.89528075,0.89818989,0.90332644]
    meanPROTEIN = [0.9564]
    nHDL = len(meanHDL)
    nLDL = len(meanLDL)
    nVLDL = len(meanVLDL)
    nPROTEIN = len(meanPROTEIN)
    nPeaks = (nHDL+nLDL+nVLDL+nPROTEIN)
    indices = [*meanHDL,*meanLDL,*meanVLDL, *meanPROTEIN]
    for x in range (0, len(indices)):
        indices[x] -= chemShift
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    if showMarkedPeaks == True:
        for i in pM:
            plt.plot(values_x[i],values_y[i],"x")
            plt.plot(values_x[i],values_y[i],"x")
    widthHDL = [0.00418966,0.00401647,0.00438304,0.00587371,0.00495607,0.00406461,0.00710216]
    widthLDL = [0.00644499,0.00458794,0.00441878,0.00740763,0.00537808,0.00500155,0.00449342]
    widthVLDL = [0.00751988,0.00422151,0.00324447,0.00293996,0.00273448,0.00284239,0.00228012]
    widthPROTEIN = [0.2]
    pW = [*widthHDL,*widthLDL,*widthVLDL,*widthPROTEIN]
    heightHDL = [0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    heightLDL = [0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    heightVLDL = [0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    heightPROTEIN = [0.5]
    pH = [*heightHDL,*heightLDL,*heightVLDL,*heightPROTEIN]
elif insertPeaksManually == "HDL":
    peakNames = ["HDL1","HDL2","HDL3","HDL4","HDL5","HDL6","HDL7","PROTEIN"]
    aM = []
    nPeaks = 8
    indices = [0.82834298,0.83392093,0.84316387,0.85391685,0.848127,0.83858213,0.86173502,0.910]
    meanPROTEIN = [0.910]
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    if showMarkedPeaks == True:
        for i in pM:
            plt.plot(values_x[i],values_y[i],"x")
            plt.plot(values_x[i],values_y[i],"x")
    pW = [0.00418966,0.00401647,0.00438304,0.00587371,0.00495607,0.00406461,0.00710216,0.037]
    pH = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.25]
    widthPROTEIN = [0.037]

if showMarkedPeaks == True:
    plt.plot(values_x,values_y, label = "Raw data")
    plt.show()

if insertPeaksManually == "y" or insertPeaksManually == "n":
    for i in range(1,nPeaks+1):
        pW.append(float(input("Enter initial width(0...1) for peak " + str(i) + "\n")))

#When calling a function, the * operator can be used to unpack 
#an iterable into the arguments in the function call
#Repack guess after receiving using *
def nGaussians(x,*params):
    gaussianSum = 0
    for i in range (0,nPeaks*3,3):
        gaussianSum += fitFormula(x,[params[i],params[i+1],params[i+2]])
    return gaussianSum

#Calculate lorentzians using fixed peak means and half-widths
def libCalc(x,*params):
    sum = 0
    for i in range (0,nPeaks-1):
        sum += fitFormula (x,[params[i],indices[i],pW[i]])
    sum += fitFormula (x, [params[nPeaks-1],params[nPeaks],params[nPeaks+1]])
    return sum

gaussianTest = []
fit = []
gauss = []
areas = []
percentAreas = []
guess = []
totalArea = 0
if nPeaks > 0:
    #Create list of initial parameters
    if fixedPeaks == False:
        for i in range(0,nPeaks):
            #extend - for sequences, append - single elements
            guess.extend((values_y[pM[i]]*pH[i],values_x[pM[i]],pW[i]))
        for i in values_x:
            #Unpack guess when sending to nGaussians using *
            gaussianTest.append(nGaussians(i,*guess))
    elif fixedPeaks == True:
        for i in range(0,nPeaks):
            #Create guess with only two variables
            #guess.extend((values_y[pM[i]]*pH[i],pW[i]))
            guess.append((values_y[pM[i]]*pH[i]))
        #Add PROT guess
        guess.extend((meanPROTEIN[0], widthPROTEIN[0]))
        for i in values_x:
            #Unpack guess when sending to nGaussians using *
            gaussianTest.append(libCalc(i,*guess))

    #Find fitting parameters, non-negative bounds =(0,infinity)
    if fixedPeaks == False:
        popt, pcov = curve_fit(nGaussians, values_x, values_y, p0=[*guess], maxfev = 100000, bounds=(0,np.inf))
            #Create list of fitting parameters
        for i in range (0,nPeaks*3,3):
            fit.append(popt[i:i+3])
        print(popt)
        for i in range(0,nPeaks):
            gauss.append(fitFormula(values_x, fit[i]))
            areas.append(np.trapz(gauss[i],values_x))
            totalArea += areas[i]
        for i in range(0,nPeaks):
            percentAreas.append((areas[i]/totalArea)*100)
    elif fixedPeaks == True:
        if insertPeaksManually == "PLASMA":
            popt, pcov = curve_fit(libCalc, values_x, values_y, p0=[*guess], maxfev = 10000000, bounds=([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.0,   0.0,      0.0], 
                                                                                                        [0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  np.inf,np.inf,np.inf]))
        else:
            popt, pcov = curve_fit(libCalc, values_x, values_y, p0=[*guess], maxfev = 10000000, bounds=(0, np.inf))
        for i in range (0,nPeaks-1):
             fit.append(popt[i])
             fit[i] = np.append(fit[i],indices[i])
             fit[i] = np.append(fit[i],pW[i])
             print("PEAK " + str(i) +": " + str(fit[i]))
        fit.append(popt[nPeaks-1])
        fit[nPeaks-1] = np.append(fit[nPeaks-1],popt[nPeaks])
        fit[nPeaks-1] = np.append(fit[nPeaks-1],popt[nPeaks+1])
        print("PROT :"  + str(fit[nPeaks-1]))
        for i in range(0,nPeaks):
            gauss.append(fitFormula(values_x, fit[i]))
            areas.append(np.trapz(gauss[i],values_x))
            totalArea += areas[i]
        for i in range(0,nPeaks):
            percentAreas.append((areas[i]/totalArea)*100)
        if insertPeaksManually == "PLASMA":
            sumHDL = copy.copy(gauss[nHDL-1])
            sumLDL = copy.copy(gauss[nHDL+nLDL-1])
            sumVLDL = copy.copy(gauss[nHDL+nLDL+nVLDL-1])
            for i in range (0, nHDL-1):
                for y in range (0,len(gauss[i])):
                    sumHDL[y] += gauss[i][y]
            for i in range (nHDL, nHDL+nLDL-1):
                for y in range (0,len(gauss[i])):
                    sumLDL[y] += gauss[i][y]
            for i in range (nHDL+nLDL, nPeaks-1):
                for y in range (0,len(gauss[i])):
                    sumVLDL[y] += gauss[i][y]
            areaHDL = np.trapz(sumHDL,values_x)
            areaLDL = np.trapz(sumLDL,values_x)
            areaVLDL = np.trapz(sumVLDL,values_x)
            totalCombinedArea = areaHDL + areaLDL + areaVLDL
            percentAreaHDL = (areaHDL / totalCombinedArea)*100
            percentAreaLDL = (areaLDL / totalCombinedArea)*100
            percentAreaVLDL = (areaVLDL / totalCombinedArea)*100
#Draw raw data and final fit
plt.plot(values_x,values_y, label = "Raw data")
plt.plot(values_x,gaussianTest,label="Initial fit")
if fitMode == 2:
    plt.gca().invert_xaxis()
plt.show()
if fixedPeaks == True:
    plt.plot(values_x,values_y, label = "Raw data")
    final_y = []
    for i in values_x:
        final_y.append(libCalc(i,*popt))
    plt.plot(values_x, final_y, label = "Final fit", linestyle = "--")
elif fixedPeaks == False:
    plt.plot(values_x,values_y, label = "Raw data")
    final_y = []
    for i in values_x:
        final_y.append(nGaussians(i,*popt))
    plt.plot(values_x, final_y, label = "Final fit", linestyle = "--")

#Find R-squared:
ss_res = 0
ss_tot = 0
ss_mean = np.mean(values_y)
for (a,b) in zip(values_y,final_y):
    ss_res += (a-b)**2
    ss_tot += (a-ss_mean)**2
r2 = 1 - (ss_res/ss_tot)
print(r2)
#Plot single fitted gaussians (nPeaks-1 = skip protein curve)
for i in range (0, nPeaks-1):
    plt.plot(values_x, gauss[i],label = peakNames[i], alpha=0.25)

#Plot sum of VLDL,LDL and HDL
if insertPeaksManually == "PLASMA" and fixedPeaks == True:
    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.plot(values_x,sumHDL,label="HDL")
    plt.plot(values_x,sumLDL,label="LDL")
    plt.plot(values_x,sumVLDL,label="VLDL")
    plt.plot(values_x,gauss[nPeaks-1],label="PROTEIN")
    #Generate area text
    areaTexts = str("HDL (%): " + str(round(percentAreaHDL,2)) + "\nLDL (%): " + str(round(percentAreaLDL,2))+ "\nVLDL (%): " + str(round(percentAreaVLDL,2)) + "\nR-squared: " + str(round(r2,4)))
    #Create a textbox
    areaTextBox = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
    #Insert text to textbox
    plt.text(0.96,0.95, areaTexts, fontsize=14,verticalalignment='top', bbox=areaTextBox)
#Axis manipulation
plt.xlabel("[ppm]")
plt.ylabel("[rel]")
if fitMode == 2:
    #Invert x-axis for NMR
    plt.gca().invert_xaxis()
#plt.suptitle(figTitle, fontsize=16)
plt.legend()
plt.show()

#Legacy code:
    #baseline = np.average(values_y[(len(values_y)-50):len(values_y)])
    #values_y[i] -= baseline
#def fixedPeaksCalculation(x,*params):
#    sum = 0
#    for i in range (0,nPeaks*2,2):
#        peak = int(i/2)
#        sum += fitFormula (x,[params[i],indices[peak],params[i+1]])
#    return sum

#popt, pcov = curve_fit(fixedPeaksCalculation, values_x, values_y, p0=[*guess], maxfev = 100000, bounds=(0, [2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,0.02,2.,np.inf]))