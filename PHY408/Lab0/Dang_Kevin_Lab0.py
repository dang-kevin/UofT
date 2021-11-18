import numpy as np
import matplotlib.pyplot as plt


def integral(y, dx):
    # function c = integral(y, dx)
    # To numerically calculate integral of vector y with interval dx:
    # c = integral[ y(x) dx]
    # ------ This is a demonstration program ------
    n = len(y) # Get the length of vector y
    nx = len(dx) if np.iterable(dx) else 1
    c = 0 # initialize c because we are going to use it
    # dx is a scalar <=> x is equally spaced
    if nx == 1: # ’==’, equal to, as a condition
        for k in range(1, n):
            c = c + (y[k] + y[k-1]) * dx / 2
    # x is not equally spaced, then length of dx has to be n-1
    elif nx == n-1:
        for k in range(1, n):
            c = c + (y[k] + y[k-1]) * dx[k-1] / 2
    # If nx is not 1 or n-1, display an error messege and terminate program
    else:
        print('Lengths of y and dx do not match!')
    return c

###############################
# Integration Function Part 1 #
###############################
    
# number of samples
nt = 100
# generate time vector
t = np.linspace(0, np.pi, nt)
# compute sample interval (evenly sampled, only one number)
dt = t[1] - t[0]
y = np.sin(t)
plt.plot(t, y, 'r+')
plt.savefig('fig1.png')
plt.show()
c = integral(y, dt)
print(c)


###############################
# Integration Function Part 2 #
###############################

nt = 50
# sampling between [0,0.5]
t1 = np.linspace(0, 0.5, nt)
# double sampling between [0.5,1]
t2 = np.linspace(0.5, 1, 2*nt)
# concatenate time vector
t = np.concatenate((t1[:-1], t2))
# compute y values 
y = np.sin(2 * np.pi * (8*t - 4*t**2))
plt.plot(t, y)
plt.savefig('fig2.png')
plt.show()
# compute sampling interval vector
dt = t[1:] - t[:-1]
c = integral(y, dt)
print(c)

# My code for c(nt)
c = []
val = [20,50,100,500,1000]
for nt in val:    
    # sampling between [0,0.5]
    t1 = np.linspace(0, 0.5, nt)
    # double sampling between [0.5,1]
    t2 = np.linspace(0.5, 1, 2*nt)
    # concatenate time vector
    t = np.concatenate((t1[:-1], t2))
    # compute y values 
    y = np.sin(2 * np.pi * (8*t - 4*t**2))
    # compute sampling interval vector
    dt = t[1:] - t[:-1] 
    c.append(integral(y, dt))

print(c)
plt.plot(val, c)
plt.savefig('fig3.png')
plt.show()


########################
# Accuracy of Sampling #
########################

# Frequency 0.25 Hz
f = 0.25
dt1 = 0.5
t1 = np.arange(0, 2*np.pi, dt1) 
y = np.cos(2*np.pi*f*t1)
dt2 = 0.04
t2 = np.arange(0, 2*np.pi, dt2)
g = np.cos(2*np.pi*f*t2)

plt.plot(t1, y, 'r+')
plt.plot(t2, g, '.')
plt.xlabel('t')
plt.ylabel('g(t)')
plt.title('Time Series of cos(2$\pi$ft) with freq 0.25')
plt.legend(['Time series','g(t)'])
plt.show()


# Frequency 0.75 Hz
f = 0.75
dt1 = 0.5
t1 = np.arange(0, 2*np.pi, dt1) 
y = np.cos(2*np.pi*f*t1)
dt2 = 0.02
t2 = np.arange(0, 2*np.pi, dt2)
g = np.cos(2*np.pi*f*t2)

plt.plot(t1, y, 'r+')
plt.plot(t2, g, '.')
plt.xlabel('t')
plt.ylabel('g(t)')
plt.title('Time Series of cos(2$\pi$ft) with freq 0.75')
plt.legend(['Time series','g(t)'])
plt.show()

