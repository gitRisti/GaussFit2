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
#VLDL   7.48 0.3 
#LDL    11.3 0.8
#HDL    15.7 0.5
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
insertPeaksManually = input("Do you want to enter peaks manually? (Y/N) \n")
#Lõpuks teha nii, et saab hiirega vajutada graafikule ja valida punktid
if insertPeaksManually == "Y":
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
elif insertPeaksManually == "N":
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
peakNames = ["VLDL","LDL","HDL","HSA"]

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

def fourGaussians(x,height1,center1,width1,height2,center2,width2,height3,center3,width3,height4,center4,width4):
    f1 = gaussian(x,[height1,center1,width1])
    f2 = gaussian(x,[height2,center2,width2])
    f3 = gaussian(x,[height3,center3,width3])
    f4 = gaussian(x,[height4,center4,width4])
    return f1+f2+f3+f4
gaussianTest = []

if nPeaks == 2:
    guess = [values_y[pM[0]],values_x[pM[0]],pW[0],values_y[pM[1]],values_x[pM[1]],pW[1]]
    for i in values_x:
        gaussianTest.append(twoGaussians(i,*guess))
    popt, pcov = curve_fit(twoGaussians, values_x, values_y, p0=[*guess])
    fit1 = popt[0:3]
    fit2 = popt[3:7]
    gauss1 = gaussian(values_x,fit1)
    gauss2 = gaussian(values_x,fit2)
    area1 = (np.trapz(gauss1,values_x))
    area2 = (np.trapz(gauss2,values_x))
elif nPeaks == 4:
    guess4 = [values_y[pM[0]],values_x[pM[0]],pW[0],values_y[pM[1]],values_x[pM[1]],pW[1],values_y[pM[2]],values_x[pM[2]],pW[2],values_y[pM[3]],values_x[pM[3]],pW[3]]
    for i in values_x:
        gaussianTest.append(fourGaussians(i,*guess4))
    
    popt, pcov = curve_fit(fourGaussians, values_x, values_y, p0=[*guess4])
    fit1 = popt[0:3]
    fit2 = popt[3:6]
    fit3 = popt[6:9]
    fit4 = popt[9:12]

    gauss1 = gaussian(values_x,fit1)
    gauss2 = gaussian(values_x,fit2)
    gauss3 = gaussian(values_x,fit3)
    gauss4 = gaussian(values_x,fit4)
    area1 = (np.trapz(gauss1,values_x))
    area2 = (np.trapz(gauss2,values_x))
    area3 = (np.trapz(gauss3,values_x))
    area4 = (np.trapz(gauss4,values_x))

#Draw raw data and final fit
plt.plot(values_x,values_y, label = "Raw data")
plt.plot(values_x,gaussianTest,label="Initial fit")
plt.show()
plt.plot(values_x,values_y, label = "Raw data")
#plt.plot(values_x, fourGaussians(values_x, *popt), label = "Final fit")
plt.plot(values_x,gauss1, label = peakNames[0])
plt.plot(values_x,gauss2, label = peakNames[1])
plt.plot(values_x,gauss3, label = peakNames[2])
plt.plot(values_x,gauss4, label = peakNames[3])
#Axis manipulation
plt.xlabel("Volume (ml)")
plt.ylabel("OD280")
ylim = plt.ylim()
xlim = plt.xlim()
plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/10, peakNames[0] + " area: " + str(round(area1,2)))
plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/8, peakNames[1] + " area: " + str(round(area2,2)))
plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/7, peakNames[2] + " area: " + str(round(area3,2)))
plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/6, peakNames[3] + " area: " + str(round(area4,2)))

plt.legend()
plt.show()
