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
from sklearn import mixture
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


#Define gaussian
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#, = tuple unpacking
peaks, _ = (find_peaks(values_y))
for i in peaks:
    plt.plot(values_x[i],values_y[i],"x")

#Substract baseline
baseline = np.mean(values_y[0:10])
values_y -= baseline


#erf - Returns the error function of complex argument
def asym_peak(t, pars):
    'from Anal. Chem. 1994, 66, 1294-1301'
    a0 = pars[0]  # peak area
    a1 = pars[1]  # elution time
    a2 = pars[2]  # width of gaussian
    a3 = pars[3]  # exponential damping term
    f = (a0/2/a3*np.exp(a2**2/2.0/a3**2 + (a1 - t)/a3)
        *(erf((t-a1)/(np.sqrt(2.0)*a2) - a2/np.sqrt(2.0)/a3) + 1.0))
    return f

#Single asterisk as used in function declaration allows variable number of arguments passed from calling environment
def two_peaks(t, *pars):    
    'function of two overlapping peaks'
    a10 = pars[0]  # peak area
    a11 = pars[1]  # elution time
    a12 = pars[2]  # width of gaussian
    a13 = pars[3]  # exponential damping term
    a20 = pars[4]  # peak area
    a21 = pars[5]  # elution time
    a22 = pars[6]  # width of gaussian
    a23 = pars[7]  # exponential damping term   
    p1 = asym_peak(t, [a10, a11, a12, a13])
    p2 = asym_peak(t, [a20, a21, a22, a23])
    return p1 + p2

test = np.array(values_x)
parguess = (500, values_x[peaks[0]], 0.3, 0.05, 1000, values_x[peaks[1]], 0.5, 0.1)
plt.plot(test,two_peaks(test,*parguess))
plt.plot(values_x,values_y)

#popt, pcov = curve_fit(two_peaks, test, values_y, parguess)
#plt.plot(test, two_peaks(test, *popt), 'b-')
#Draw raw data plot

plt.show()
