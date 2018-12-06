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
#Plasma data:
#VLDL   7.52 0.3 
#LDL    11.3 0.8
#REM    12.6 0.8
#HDL    15.8 0.5
#HSA    16.8 0.4


#'r' preceding the string turns off the eight-character Unicode escape (for a raw string)
workbook = xlrd.open_workbook(r"C:\Users\Robert\Desktop\plasma.xls")

#Get worksheet by index
worksheet = workbook.sheet_by_index(0)

#Iterate over first two columns, starting from row 3 (skip titles)
for columnIndex in range (0, worksheet.ncols):
    for rowIndex in range (3, worksheet.nrows):
        if columnIndex == 0:
            values_x.append(float(worksheet.cell(rowIndex,columnIndex).value))
        if columnIndex == 1:
            values_y.append(float(worksheet.cell(rowIndex,columnIndex).value))


#Define gaussian, ** = power operator
def gaussian(x, pars):
    height = pars[0]
    center = pars[1]
    width = pars[2]
    return height*np.exp(-(x - center)**2/(2*width**2))

#Show raw data and peaks
plt.plot(values_x,values_y, label = "Raw data")
plt.show()
figTitle = input("Enter sample name: \n")
insertPeaksManually = input("Do you want to enter peaks manually? (y/n) \n")
#Lõpuks teha nii, et saab hiirega vajutada graafikule ja valida punktid
if insertPeaksManually == "y":
    pM = [] #Initial guesses for peak mean
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
plt.plot(values_x,values_y, label = "Raw data")
plt.show()

pW = [] #Initial guesses for peak width
peakNames = ["VLDL","LDL","HDL","REM","HSA"]

for i in range(1,nPeaks+1):
    pW.append(float(input("Enter initial width(0...1) for peak " + str(i) + "\n")))
#for i in range(1,nPeaks+1):
  #  peakNames.append(input("Enter name for peak " + str(i) + "\n"))

#Define two peaks
#When calling a function, the * operator can be used to unpack 
#an iterable into the arguments in the function call
def twoGaussians(x,height1,center1,width1,height2,center2,width2):
    f1 = gaussian(x,[height1,center1,width1])
    f2 = gaussian(x,[height2,center2,width2])
    return f1+f2
def fiveGaussians(x,height1,center1,width1,height2,center2,width2,height3,center3,width3,height4,center4,width4,height5,center5,width5):
    f1 = gaussian(x,[height1,center1,width1])
    f2 = gaussian(x,[height2,center2,width2])
    f3 = gaussian(x,[height3,center3,width3])
    f4 = gaussian(x,[height4,center4,width4])
    f5 = gaussian(x,[height5,center5,width5])
    return f1+f2+f3+f4+f5
gaussianTest = []
fit = []
gauss = []
areas = []
guess = []
if nPeaks == 2:
    guess2 = [values_y[pM[0]],values_x[pM[0]],pW[0],values_y[pM[1]],values_x[pM[1]],pW[1]]
    for i in values_x:
        gaussianTest.append(twoGaussians(i,*guess2))
    popt, pcov = curve_fit(twoGaussians, values_x, values_y, p0=[*guess2])
    fit1 = popt[0:3]
    fit2 = popt[3:7]
    gauss1 = gaussian(values_x,fit1)
    gauss2 = gaussian(values_x,fit2)
    area1 = (np.trapz(gauss1,values_x))
    area2 = (np.trapz(gauss2,values_x))
elif nPeaks == 5:
    #Create list of initial parameters
    for i in range(0,nPeaks):
        #extend - for sequences, append - single elements
        guess.extend((values_y[pM[i]],values_x[pM[i]],pW[i]))
    for i in values_x:
        gaussianTest.append(fiveGaussians(i,*guess))
    #Find fitting parameters
    popt, pcov = curve_fit(fiveGaussians, values_x, values_y, p0=[*guess])
    #Create list of fitting parameters
    for i in range (0,nPeaks*3,3):
        fit.append(popt[i:i+3])
    #Create lists of fitted gaussians and integrated areas
    for i in range(0,nPeaks):
        gauss.append(gaussian(values_x,fit[i]))
        areas.append(np.trapz(gauss[i],values_x))

#Draw raw data and final fit
plt.plot(values_x,values_y, label = "Raw data")
plt.plot(values_x,gaussianTest,label="Initial fit")
plt.show()
plt.plot(values_x,values_y, label = "Raw data")
plt.plot(values_x, fiveGaussians(values_x, *popt), label = "Final fit")
plt.plot(values_x,gauss[0], label = peakNames[0])
plt.plot(values_x,gauss[1], label = peakNames[1])
plt.plot(values_x,gauss[2], label = peakNames[2])
plt.plot(values_x,gauss[3], label = peakNames[3])
plt.plot(values_x,gauss[4], label = peakNames[4])
#Axis manipulation
plt.xlabel("Volume (ml)")
plt.ylabel("OD280")
ylim = plt.ylim()
xlim = plt.xlim()
areaTexts = str(peakNames[0] + " area: " + str(round(areas[0],2))),str(peakNames[1] + " area: " + str(round(areas[1],2))),str(peakNames[2] + " area: " + str(round(areas[2],2))),str(peakNames[3] + " area: " + str(round(areas[3],2))),str(peakNames[4] + " area: " + str(round(areas[4],2)))
areaTexts = ("\n".join(areaTexts))
#Create a text box
areaTextBox = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/10, areaTexts, fontsize=14,
        verticalalignment='top', bbox=areaTextBox)
plt.legend()
plt.suptitle(figTitle, fontsize=16)
plt.show()
