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
    
#'r' preceding the string turns off the eight-character Unicode escape (for a raw string)
workbook = xlrd.open_workbook(r"C:\Users\robik\Desktop\HDL.xls")

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
insertPeaksManually = input("Do you want to enter peaks manually? (Y/N) \n")
if insertPeaksManually == "Y":
    pM = [] #Initial guesses for peak mean
    nPeaks = int(input("Please enter # of peaks:\n"))
    for i in range(1,nPeaks+1):
        pM.append(int(input("Enter mean for peak " + str(i) + "\n")))
    for i in pM:
        plt.plot(values_x[i],values_y[i],"x")
elif insertPeaksManually == "N":
    #Find noise (standard deviation of baseline)
    noise = np.std(values_y[0:10])
    #, = tuple unpacking
    peaks, _ = (find_peaks(values_y,100))
    for i in peaks:
        plt.plot(values_x[i],values_y[i],"x")
    nPeaks = peaks.shape
    nPeaks = nPeaks[0]
    print("Found " + str(nPeaks) + " peaks")

plt.plot(values_x,values_y, label = "Raw data")
plt.show()

pW = [] #Initial guesses for peak width
peakNames = []

for i in range(1,nPeaks+1):
    pW.append(float(input("Enter initial width(0...1) for peak " + str(i) + "\n")))
peak1_name = input("Enter name for peak1 \n")
peak2_name = input("Enter name for peak2 \n")  



#Define two peaks
#When calling a function, the * operator can be used to unpack 
#an iterable into the arguments in the function call
def twoGaussians(x,height1,center1,width1,height2,center2,width2):
    f1 = gaussian(x,[height1,center1,width1])
    f2 = gaussian(x,[height2,center2,width2])
    return f1+f2

def fourGaussians(x,height1,center1,width1,height2,center2,width2,height3,center3,width3,height4,center4,width4):
    f1 = gaussian(x,[height1,center1,width1])
    f2 = gaussian(x,[height1,center1,width1])
    f3 = gaussian(x,[height1,center1,width1])
    f4 = gaussian(x,[height1,center1,width1])
    return f1+f2+f3+f4
gaussianTest = []

if nPeaks == 2:
    guess = [values_y[peaks[0]],values_x[peaks[0]],pW[0],values_y[peaks[1]],values_x[peaks[1]],pW[1]]
    for i in values_x:
        gaussianTest.append(twoGaussians(i,*guess))
elif nPeaks == 4:
    guess4 = [values_y[pM[0]],values_x[pM[0]],pW[0],values_y[pM[1]],values_x[pM[1]],pW[1],values_y[pM[2]],values_x[pM[2]],pW[2],values_y[pM[3]],values_x[pM[3]],pW[3]]
    for i in values_x:
        gaussianTest.append(fourGaussians(i,*guess))


popt, pcov = curve_fit(twoGaussians, values_x, values_y, p0=[*guess])
fit1 = popt[0:3]
fit2 = popt[3:7]

gauss1 = gaussian(values_x,fit1)
gauss2 = gaussian(values_x,fit2)
area1 = (np.trapz(gauss1,values_x))
area2 = (np.trapz(gauss2,values_x))


#Draw raw data and final fit
plt.plot(values_x,values_y, label = "Raw data")
plt.plot(values_x,gaussianTest,label="Initial fit")
#plt.plot(values_x, twoGaussians(values_x, *popt), label = "Final fit")
plt.show()
#plt.plot(values_x,values_y, label = "Raw data")
#plt.plot(values_x,gauss1, label = peak1_name)
#plt.plot(values_x,gauss2, label = peak2_name)

#Axis manipulation
plt.xlabel("Volume (ml)")
plt.ylabel("OD280")
ylim = plt.ylim()
xlim = plt.xlim()
plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/10, peak1_name + " area: " + str(round(area1,2)))
plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/6,peak2_name + " area: " + str(round(area2,2)))


plt.legend()
plt.show()
