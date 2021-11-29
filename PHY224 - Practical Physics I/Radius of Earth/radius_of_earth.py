'''
Radius of the Earth

Kevin Dang
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


_, reading, _ = np.loadtxt('gravity_test_1.txt',skiprows=1,unpack=True)

gravity = reading * 0.10055 / 1000 / 100  #convert to m/s^2

error1 = np.array([3*0.10055/1000/100 for x in range(len(reading))])  #convert error to m/s^2

building_heights = np.array([5.1]+[5.44]+[3.95]*13)
height1 = np.array([sum(building_heights[0:i])-building_heights[0] for i in range(15)])

def correct_gravity(gravity):
    for floor in range(15):
        #Every floor below it must have its influcence subtracted, vice-verca for above.
        for i in range(floor):
            dist = sum(building_heights[i:floor])
            grav = 6.67*10**(-11) * 10**6 / (dist)**2
            gravity[floor] -= grav
        for i in range(floor+1,15):
            dist = sum(building_heights[floor:i])
            grav = 6.67*10**(-11) * 10**6 / (dist)**2
            gravity[floor] += grav
    return gravity

#Note: the correction factors turns out to be very small
gravity1 = correct_gravity(gravity[:])

#Take a look at the corrected reading and compare to original
plt.figure(figsize=(10,6))
plt.plot(gravity,label='Uncorrected')
plt.plot(gravity1,label='Corrected')
plt.xlabel('Floor')
plt.ylabel('Gravitational Reading')
plt.title('Gravitational Reading vs Floor')
plt.legend()
plt.savefig('gravity-floor.pdf')
plt.show()

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


#Fit an initial model with all floors
popt1, pcov1 = curve_fit(linear, height1, gravity1, sigma=error1)

plt.figure(figsize=(10,6))
plt.scatter(height1,gravity1,label='Data')
plt.errorbar(height1,gravity1,yerr=error1,fmt='none',ecolor='lime',label='Error')
plt.plot(height1,linear(height1,popt1[0],popt1[1]),color='orange',label='Linear fit')
plt.xlabel('Height(m)')
plt.ylabel('Gravitational Reading(m/s^2)')
plt.ylim(0.000530,0.00075)
plt.title('Gravitational Reading vs Height')
plt.legend()
plt.savefig('gravity-height1.pdf')
plt.show()

print('\nEquation: g =',popt1[0],'* R +',popt1[1],'\n')

chi1 = chisq_red(height1,gravity1,popt1[0],popt1[1],error1)
print('Reduced chi-squared =',chi1,'\n')

a_err1 = np.sqrt(pcov1[0,0])
radius1 = -2*9.804253/popt1[0]
radius_err1 = radius1*np.sqrt((a_err1/popt1[0])**2)
print('Radius via curve fit of all floors:',radius1,'+/-',radius_err1,'\n')


#Index floors 3 to 13
height2 = np.array([x*3.95 for x in range(11)])
gravity2 = gravity1[3:14]
error2 = error1[3:14]

#Fit a second model containing only floors 3 to 13
popt2, pcov2 = curve_fit(linear, height2, gravity2, sigma=error2)

plt.figure(figsize=(10,6))
plt.scatter(height2,gravity2,label='Data')
plt.errorbar(height2,gravity2,yerr=error2,fmt='none',ecolor='lime',label='Error')
plt.plot(height2,linear(height2,popt2[0],popt2[1]),color='orange',label='Linear fit')
plt.xlabel('Height(m)')
plt.ylabel('Gravitational Reading(m/s^2)')
plt.ylim(0.000550,0.0007)
plt.title('Gravitational Reading vs Height (Floors 3-13)')
plt.legend()
plt.savefig('gravity-height2.pdf')
plt.show()

print('\nEquation: g =',popt2[0],'* R +',popt2[1],'\n')

chi2 = chisq_red(height2,gravity2,popt2[0],popt2[1],error2)
print('Reduced chi-squared =',chi2,'\n')

a_err2 = np.sqrt(pcov2[0,0])
radius2 = -2*9.804253/popt2[0]
radius_err2 = radius2*np.sqrt((a_err2/popt2[0])**2)


print('Radius via curve fit of floors 3-13:',radius2,'+/-',radius_err2)
