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
workbook = xlrd.open_workbook(r"C:\Users\Robert\Desktop\HDL.xls")

#Get worksheet by index
worksheet = workbook.sheet_by_index(0)

#Iterate over first two columns, starting from row 3 (skip titles)
for columnIndex in range (0, worksheet.ncols):
    for rowIndex in range (3, worksheet.nrows):
        if columnIndex == 0:
            values_x.append(float(worksheet.cell(rowIndex,columnIndex).value))
        if columnIndex == 1:
            values_y.append(float(worksheet.cell(rowIndex,columnIndex).value))


#Define gaussian
def gaussian(x, pars):
    height = pars[0]
    center = pars[1]
    width = pars[2]
    offset = pars[3]
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

#, = tuple unpacking
peaks, _ = (find_peaks(values_y))

for i in peaks:
    plt.plot(values_x[i],values_y[i],"x")

#Define two peaks
def twoGaussians(x,height1,center1,width1,offset1,height2,center2,width2,offset2):
    f1 = gaussian(x,[height1,center1,width1,offset1])
    f2 = gaussian(x,[height2,center2,width2,offset2])
    return f1+f2

gaussianTest = []
guess = [500,values_x[peaks[0]],0.7,0,800,values_x[peaks[1]],0.3,0]
for i in values_x:
    gaussianTest.append(twoGaussians(i,500,values_x[peaks[0]],0.7,0,800,values_x[peaks[1]],0.3,0))

peakWidths = peak_widths(values_y,peaks,rel_height = 0.5)

popt, pcov = curve_fit(twoGaussians, values_x, values_y, p0=[500,values_x[peaks[0]],0.7,0,800,values_x[peaks[1]],0.3,0])
fit1 = popt[0:4]
fit1[3] = 0
fit2 = popt[4:8]
fit2[3] = 0
print(fit1)

gauss1 = gaussian(values_x,fit1)
gauss2 = gaussian(values_x,fit2)

#plt.plot(values_x,gaussianTest,label="Initial fit")
plt.plot(values_x,values_y, label = "Raw data")
plt.plot(values_x,gauss1, label = "Peak1")
plt.plot(values_x,gauss2, label = "Peak2")
plt.xlim(12,19)
#plt.plot(values_x, twoGaussians(values_x, *popt), label = "Final fit")
plt.legend()
plt.show()

#Draw raw data plot


