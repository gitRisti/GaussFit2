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

peaks, _ = (find_peaks(values_y,100))
for i in peaks:
    plt.plot(values_x[i],values_y[i],"x")
nPeaks = peaks.shape
print("Found " + str(nPeaks) + " peaks")
plt.show()

peak1_width = input("Enter initial width(0...1) for peak 1 \n")
peak2_width = input("Enter initial width(0...1) for peak 2 \n")
peak1_name = input("Enter name for peak1 \n")
peak2_name = input("Enter name for peak2 \n")  

peak1_width = float(peak1_width)
peak2_width = float(peak2_width)

#, = tuple unpacking


#Define two peaks
#When calling a function, the * operator can be used to unpack 
#an iterable into the arguments in the function call
def twoGaussians(x,height1,center1,width1,height2,center2,width2):
    f1 = gaussian(x,[height1,center1,width1])
    f2 = gaussian(x,[height2,center2,width2])
    return f1+f2

gaussianTest = []

guess = [values_y[peaks[0]],values_x[peaks[0]],peak1_width,values_y[peaks[1]],values_x[peaks[1]],peak2_width]
for i in values_x:
    gaussianTest.append(twoGaussians(i,*guess))

peakWidths = peak_widths(values_y,peaks,rel_height = 0.5)
print(peakWidths)

popt, pcov = curve_fit(twoGaussians, values_x, values_y, p0=[*guess])
fit1 = popt[0:3]
fit2 = popt[3:7]

gauss1 = gaussian(values_x,fit1)
gauss2 = gaussian(values_x,fit2)
area1 = (np.trapz(gauss1,values_x))
area2 = (np.trapz(gauss2,values_x))


#Draw raw data and final fit
plt.plot(values_x,values_y, label = "Raw data")
#plt.plot(values_x,gaussianTest,label="Initial fit")
plt.plot(values_x,gauss1, label = peak1_name)
plt.plot(values_x,gauss2, label = peak2_name)
#plt.plot(values_x, twoGaussians(values_x, *popt), label = "Final fit")

#Axis manipulation
plt.xlabel("Volume (ml)")
plt.ylabel("OD280")
ylim = plt.ylim()
xlim = plt.xlim()
plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/10, peak1_name + " area: " + str(round(area1,2)))
plt.text(xlim[0]+xlim[0]/10,ylim[1]-ylim[1]/6,peak2_name + " area: " + str(round(area2,2)))


plt.legend()
plt.show()
