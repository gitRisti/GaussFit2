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

values_x = []
values_y = []
gaussian_x = []
gaussian_y = []
#Default distribution fitMode = 1 is Gaussian
fitMode = 1
fixedPeaks = False
#Plasma data:
#VLDL   7.52 0.3 
#LDL    11.3 0.8
#REM    12.6 0.8
#HDL    15.8 0.5
#HSA    16.8 0.3
#NMRi spektri intensiivsuse kordajad
#0.0263 plasma
#0.2973 VLDL
#0.25
#1 LDL

#'r' preceding the string turns off the eight-character Unicode escape (for a raw string)
workbook = xlrd.open_workbook(r"C:\Users\robik\Desktop\HDLNMR.xlsx")

#Get worksheet by index
worksheet = workbook.sheet_by_index(0)

#Iterate over first two columns, starting from row 3 (skip titles)
#NMRi jaoks vahetuses -> column 1 = x ja column 0 = y
#worksheet.nrows
for columnIndex in range (0, worksheet.ncols):
    for rowIndex in range (100266, 102123):
        if columnIndex == 0:
            values_y.append(float(worksheet.cell(rowIndex,columnIndex).value))
        if columnIndex == 1:
            values_x.append(float(worksheet.cell(rowIndex,columnIndex).value))

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
baseline = np.average(values_y[(len(values_y)-50):len(values_y)])
maxValue = np.amax(values_y)
for i in range (0,len(values_y)):
    values_y[i] -= baseline
    values_y[i] = values_y[i]/maxValue
#Show raw data
showRawData = True
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
elif insertPeaksManually == "LDL":
    peakNames = ["LDL1","LDL2","LDL3","LDL4","LDL5","PROTEIN"]
    aM = []
    nPeaks = 6
    indices = [0.8522,0.8595,0.8673,0.8744,0.8800,0.9302]
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    if showMarkedPeaks == True:
        for i in pM:
            plt.plot(values_x[i],values_y[i],"x")
            plt.plot(values_x[i],values_y[i],"x")
    pW = [0.007,0.006,0.006,0.009,0.004,0.03]
    pH = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]

elif insertPeaksManually == "VLDL":
    peakNames = ["VLDL1","VLDL2","VLDL3","VLDL4","VLDL5","PROTEIN"]
    aM = []
    nPeaks = 6
    indices = [0.8749,0.8799,0.8861,0.8899,0.8962,0.9464]
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    if showMarkedPeaks == True:
        for i in pM:
            plt.plot(values_x[i],values_y[i],"x")
            plt.plot(values_x[i],values_y[i],"x")
    pW = [0.005,0.005,0.005,0.005,0.005,0.005]

    pH = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
elif insertPeaksManually == "PLASMA":
    peakNames = ["HDL1","HDL2","HDL3","HDL4","HDL5","HDL6","HDL7",
                 "LDL1","LDL2","LDL3","LDL4","LDL5",
                 "VLDL1","VLDL2","VLDL3","VLDL4","VLDL5",
                 "PROTEIN"]
    aM = []
    nPeaks = 18
    indices = [0.835,0.840,0.845,0.850,0.855,0.860,0.865, #HDL
               0.8522,0.8595,0.8673,0.8744,0.8800, #LDL
               0.8749,0.8799,0.8861,0.8899,0.8962, #VLDL
               0.9464] #PROTEIN
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    if showMarkedPeaks == True:
        for i in pM:
            plt.plot(values_x[i],values_y[i],"x")
            plt.plot(values_x[i],values_y[i],"x")
    pW = [0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005]
    pH = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
elif insertPeaksManually == "HDL":
    peakNames = ["HDL1","HDL2","HDL3","HDL4","HDL5","HDL6","HDL7","PROTEIN"]
    aM = []
    nPeaks = 8
    indices = [0.835,0.840,0.845,0.850,0.855,0.860,0.865,0.909]
    for a in range(0,nPeaks):
        for i in range(0, len(values_x)):
            aM.append(np.abs(values_x[i]-indices[a]))
        pM.append(np.argmin(aM))
        aM.clear()
    if showMarkedPeaks == True:
        for i in pM:
            plt.plot(values_x[i],values_y[i],"x")
            plt.plot(values_x[i],values_y[i],"x")
    pW = [0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005]
    pH = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]

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
    elif fixedPeaks == True:
        for i in range(0,nPeaks):
            guess.extend((values_y[pM[i]],values_x[fixedPM[i]],pW[i]))
    for i in values_x:
        #Unpack guess when sending to nGaussians using *
        gaussianTest.append(nGaussians(i,*guess))
    #Find fitting parameters, non-negative bounds =(0,infinity)
    popt, pcov = curve_fit(nGaussians, values_x, values_y, p0=[*guess], maxfev = 100000, bounds=(0,np.inf))
    #Create list of fitting parameters
    for i in range (0,nPeaks*3,3):
        fit.append(popt[i:i+3])
    print(popt)
    #Create lists of fitted gaussians and integrated areas
    for i in range(0,nPeaks):
        gauss.append(fitFormula(values_x,fit[i]))
        areas.append(np.trapz(gauss[i],values_x))
        totalArea += areas[i]
    for i in range(0,nPeaks):
        percentAreas.append((areas[i]/totalArea)*100)
#Draw raw data and final fit
plt.plot(values_x,values_y, label = "Raw data")
plt.plot(values_x,gaussianTest,label="Initial fit")
plt.gca().invert_xaxis()
plt.show()
plt.plot(values_x,values_y, label = "Raw data")
plt.plot(values_x, nGaussians(values_x, *popt), label = "Final fit", linestyle = "--")
#Plot single fitted gaussians
for i in range (0,nPeaks):
    plt.plot(values_x, gauss[i],label = peakNames[i])
#Plot sum of VLDL,LDL and HDL
#Axis manipulation
plt.xlabel("Volume (ml)")
plt.ylabel("OD280")
#Invert x-axis for NMR
plt.gca().invert_xaxis()
#plt.suptitle(figTitle, fontsize=16)
ylim = plt.ylim()
xlim = plt.xlim()
areaTexts = "\n"
#Generate area text
for i in range (0,nPeaks):
    areaString = str(peakNames[i] + " area: " + str(round(areas[i],2)) + " ("+str(round(percentAreas[i],2)) + " %)\n")
    areaTexts += areaString
#Create a textbox
areaTextBox = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#Insert text to textbox
#plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/10, areaTexts, fontsize=14,
 #       verticalalignment='top', bbox=areaTextBox)
plt.legend()
plt.show()
