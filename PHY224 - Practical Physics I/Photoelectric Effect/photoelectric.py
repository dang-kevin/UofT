'''
Photoelectric Effect

Kevin Dang
'''

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


#Exercise 1

wavelengths, stop_volt, wave_err = np.loadtxt('stop_voltage.txt',unpack=True,skiprows=1,delimiter=',')
wave_freq = 3*10**8 / (wavelengths / 10**9) #Lightspeed is 3e8, Nanometers is e-9
freq_err = wave_err * wave_freq / wavelengths

volt_err = [0.0005 for x in range(len(stop_volt))] #Reading error

#Linear Function
def linear(x,a,b):
    return a*x + b

#Reduced chi-squared
#Note that error represents the error of the DEPENDENT variable.
def chisq_red(x,y,slope,intercept,error):
    squares = []
    for i in range(len(x)):
        val = y[i] - linear(x[i],slope,intercept)
        val /= error[i]
        squares.append(val**2)
    return 1/(len(x)-2) * sum(squares)


popt, pcov = curve_fit(linear,wave_freq,stop_volt)

plt.figure(figsize=(10,6))
plt.scatter(wave_freq,stop_volt)
plt.errorbar(wave_freq,stop_volt,xerr=freq_err,yerr=volt_err,fmt='none',ecolor='orange')
plt.plot(wave_freq, linear(wave_freq,popt[0],popt[1]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Stopping Voltage (V)')
plt.title('Stopping Voltage vs Frequency')
plt.savefig('vstop_freq.pdf')
plt.show()

print('Equation: Stopping Voltage =',popt[0],'* Frequency -',abs(popt[1]))

chi1 = chisq_red(wave_freq,stop_volt,popt[0],popt[1],volt_err)
print('Reduced chi-squared =',chi1)
#large reduced chi-squared due to large freq error

a_err = np.sqrt(pcov[0,0])
b_err = np.sqrt(pcov[1,1])

electron = 1.602*10**(-19)

planck = popt[0] * electron
planck_err = a_err * electron
print('\nPlanck\'s Constant =',planck,'+/-',planck_err,'m^2 * kg/s')

cutoff_freq = -popt[1]*electron/planck
cutoff_freq_err = np.sqrt((a_err/popt[0])**2 + (b_err/popt[1])**2)*cutoff_freq
print('Cutoff Frequency =',cutoff_freq,'+/-',cutoff_freq_err,'Hz')

work_function = planck*cutoff_freq
work_function_err = (planck_err/planck + cutoff_freq_err/cutoff_freq)*work_function
print('Work function =',work_function,'+/-',work_function_err,'J\n')


#Exercise 2

intensity, stop_volt2, photocurrent = np.loadtxt('variable_intensity.txt',unpack=True,comments='#',delimiter=',')

popt2, pcov2 = curve_fit(linear,intensity,stop_volt2)

volt_err2 = [0.0005 for x in range(len(stop_volt2))]                                        
                                                 
plt.figure(figsize=(10,6))
plt.scatter(intensity,stop_volt2)
plt.errorbar(intensity,stop_volt2,yerr=volt_err2,fmt='none',ecolor='orange')
plt.plot(intensity,linear(intensity,popt2[0],popt2[1]))
plt.xlabel('Intensity')
plt.ylabel('Stopping Voltage (V)')
plt.title('Stopping Voltage vs Intensity')
plt.savefig('vstop_inten.pdf')
plt.show()

print('Equation: Stopping Voltage =',popt2[0],'* Intensity +',popt2[1])

chi2 = chisq_red(intensity,stop_volt2,popt2[0],popt2[1],volt_err2)
print('Reduced chi-squared =',chi2,'\n')

popt3, pcov3 = curve_fit(linear,intensity,photocurrent)

photo_err = [0.005 for x in range(len(photocurrent))]

plt.figure(figsize=(10,6))
plt.scatter(intensity,photocurrent)
plt.errorbar(intensity,photocurrent,yerr=photo_err,fmt='none',ecolor='orange')
plt.plot(intensity,linear(intensity,popt3[0],popt3[1]))
plt.xlabel('Intensity')
plt.ylabel('Photocurrent (V)')
plt.title('Photocurrent vs Intensity')
plt.savefig('photo_inten.pdf')
plt.show()

print('Equation: Photocurrent =',popt3[0],'* Intensity +',popt3[1])

chi3 = chisq_red(intensity,photocurrent,popt3[0],popt3[1],photo_err)
print('Reduced chi-squared =',chi3)

'''
Stopping voltage does not depend on intensity (it depends on frequency). 
Photocurrent increases with intensity.
'''

#Exercise 3
power_led = 60/1000 #watts
photocathode_area = 3.23/(100**2)  #metres squared
electron_radius = 0.3*0.5/(10**9)
electron_area = np.pi*(electron_radius**2)

electron_power = power_led*electron_area/photocathode_area
print('\nPower per electron =',electron_power,'J/s')

escape_time = work_function/electron_power
print('Escape time =',escape_time,'s')

print('\nExperimental Time Constant = 0.0002 s')


