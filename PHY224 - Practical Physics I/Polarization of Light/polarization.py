"""
Polarization of Light

Kevin Dang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 2 polarizers data
theta, intensity = np.loadtxt('malus_2polarizers.txt',skiprows=2,unpack=True)

# cos squared function
def cos_sq(x,a,b,c):
    return a*(np.cos(b*x) ** 2) + c

popt, pcov = curve_fit(cos_sq,theta,intensity)

plt.figure(figsize=(10,6)) 
plt.plot(theta, cos_sq(theta,*popt))
plt.scatter(theta,intensity,color='orange',s=10)
plt.title('Intensity vs Angle - 2 Polarizer')
plt.savefig('2pol-IntvsAng.pdf')
plt.show()

# Linear function
def linear(x,a,b):
    return a*x + b

popt1, pcov1 = curve_fit(linear,np.cos(theta)**2,intensity)

plt.figure(figsize=(10,6))  
plt.plot(np.cos(theta)**2,linear(np.cos(theta)**2,*popt1))
plt.scatter(np.cos(theta)**2, intensity,color='orange',s=10)
plt.title('Intensity vs cos^2(theta) - 2 Polarizer')
plt.xlabel('cos^2(theta)')
plt.ylabel('Intensity (V)')
plt.savefig('2pol-linear.pdf')
plt.show()
print('\nIntensity (2 polarizers) =',popt1[0],'* cos^2(theta) +',popt1[1])



theta2, intensity2 = np.loadtxt('3polarizers.txt',skiprows=2,unpack=True)
def sin_sq(x,a,b):
    return a*(np.sin(2*x+np.pi/2) ** 2) + b

popt2, pcov2 = curve_fit(sin_sq, theta2, intensity2)

plt.figure(figsize=(10,6)) 
plt.plot(theta2, sin_sq(theta2,*popt2))
plt.scatter(theta2, intensity2,color='orange',s=10)
plt.title('Intensity vs Angle - 3 Polarizer')
plt.savefig('3pol-IntvsAng.pdf')
plt.show()


'''
I3 = I1/4 * sin^2 (2phi)
phi = theta + pi/4
2phi = 2theta + pi/2
'''
popt3, pcov3 = curve_fit(linear, np.sin(2*theta2+np.pi/2)**2,intensity2)

plt.figure(figsize=(10,6))
plt.plot(np.sin(2*theta2+np.pi/2)**2,linear(np.sin(2*theta2+np.pi/2)**2,*popt3))
plt.scatter(np.sin(2*theta2+np.pi/2)**2, intensity2,color='orange',s=10)
plt.title('Intensity vs sin^2(2*phi) - 3 Polarizer')
plt.xlabel('sin^2(2*phi)')
plt.ylabel('Intensity (V)')
plt.savefig('3pol-linear.pdf')
plt.show()
print('\nIntensity (3 polarizers) =',popt3[0],'* sin^2(2phi) +',popt3[1])


sensor_deg_raw, brew_intensity_raw = np.loadtxt('brewster_nopolarizer.txt',skiprows=2,unpack=True)

#This trims out regions where the sensor was in motion, leaving only the points where maxima are assured.
sensor_deg_steady = [sensor_deg_raw[i] for i in range(1,len(sensor_deg_raw)-1) if sensor_deg_raw[i-1]==sensor_deg_raw[i] and sensor_deg_raw[i]==sensor_deg_raw[i+1]]
brew_intensity_steady = [brew_intensity_raw[i] for i in range(1,len(sensor_deg_raw)-1) if sensor_deg_raw[i-1]==sensor_deg_raw[i] and sensor_deg_raw[i]==sensor_deg_raw[i+1]]


sensor_deg_maxima = [sensor_deg_steady[0]]
brew_intensity_maxima = [brew_intensity_steady[0]]
for i in range(len(sensor_deg_steady)):
    if sensor_deg_steady[i]>sensor_deg_maxima[len(sensor_deg_maxima)-1]:
        sensor_deg_maxima.append(sensor_deg_steady[i])
        brew_intensity_maxima.append(brew_intensity_steady[i])
    brew_intensity_maxima[len(brew_intensity_maxima)-1] = max(brew_intensity_steady[i], brew_intensity_maxima[len(brew_intensity_maxima)-1])

sensor_deg = []
brew_intensity = []
for i in range(len(sensor_deg_maxima)):
    if brew_intensity_maxima[i] > 0.01:
        sensor_deg.append(sensor_deg_maxima[i])
        brew_intensity.append(brew_intensity_maxima[i])

sensor_deg_raw, brew_intensity_raw = np.loadtxt('brewster_polarizer.txt',skiprows=2,unpack=True)

#This trims out regions where the sensor was in motion, leaving only the points where maxima are assured.
sensor_deg_steady = [sensor_deg_raw[i] for i in range(1,len(sensor_deg_raw)-1) if sensor_deg_raw[i-1]==sensor_deg_raw[i] and sensor_deg_raw[i]==sensor_deg_raw[i+1]]
brew_intensity_steady = [brew_intensity_raw[i] for i in range(1,len(sensor_deg_raw)-1) if sensor_deg_raw[i-1]==sensor_deg_raw[i] and sensor_deg_raw[i]==sensor_deg_raw[i+1]]


sensor_deg_maxima = [sensor_deg_steady[0]]
brew_intensity_maxima = [brew_intensity_steady[0]]
for i in range(len(sensor_deg_steady)):
    if sensor_deg_steady[i]>sensor_deg_maxima[len(sensor_deg_maxima)-1]:
        sensor_deg_maxima.append(sensor_deg_steady[i])
        brew_intensity_maxima.append(brew_intensity_steady[i])
    brew_intensity_maxima[len(brew_intensity_maxima)-1] = max(brew_intensity_steady[i], brew_intensity_maxima[len(brew_intensity_maxima)-1])

sensor_deg2 = []
brew_intensity2 = []
for i in range(len(sensor_deg_maxima)):
    if brew_intensity_maxima[i] > 0.001:
        sensor_deg2.append(sensor_deg_maxima[i])
        brew_intensity2.append(brew_intensity_maxima[i])
        
plt.figure(figsize=(10,6))
plt.scatter(sensor_deg,brew_intensity,label='No polarizer')
plt.scatter(sensor_deg2,brew_intensity2,label='Square polarizer')
plt.xlabel('Angle (Degrees)')
plt.ylabel('Intensity (V)')
plt.legend()
plt.title('Intensity vs Angle')
plt.savefig('brewster-plot.pdf')
plt.show()

brew_intensity_overI = [x/max(brew_intensity) for x in brew_intensity]
brew_intensity2_overI = [x/max(brew_intensity2) for x in brew_intensity2]


plt.figure(figsize=(10,6))
plt.scatter(sensor_deg,brew_intensity_overI,label='No polarizer')
plt.scatter(sensor_deg2,brew_intensity2_overI,label='Square polarizer')
plt.xlabel('Angle (Degrees)')
plt.ylabel('Ratio I/I0')
plt.legend()
plt.title('Ratio of I/I0 vs Angle')
plt.savefig('brewster-ratio.pdf')
plt.show()

'''
When aligning the laser beam, the computer started off at 180 degrees, while the actual start point was 150 degrees.
To correct for this, we add 30 degrees.
'''

min_deg = 260

brewster = (min_deg+30)/2 - 90
brew_rad = brewster*np.pi/180

print('\nBrewster angle:',brewster)

n2 = np.tan(brew_rad)
print('Index of refraction of acrylic:',n2)

refracted = np.arcsin(np.sin(brew_rad)/n2)

perpendicular = (np.cos(brew_rad)-n2*np.cos(refracted))/(np.cos(brew_rad)+n2*np.cos(refracted))
print('Perpendicular reflectance =',perpendicular)      

parallel = (np.cos(refracted)-n2*np.cos(brew_rad))/(np.cos(refracted)+n2*np.cos(brew_rad))
print('Parallel reflectance =',parallel)
