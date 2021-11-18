import numpy as np


# EXERCISE 1
h = .1
x = np.arange(10)*h
y = np.cos(x)

def rate():
    roc = []
    for i in range(len(y)-1):
        diff = (y[i+1] - y[i])
        roc.append(diff)
    return np.array(roc)/h    # Need to call rate() to return the array

print('Exercise 1: rate() =',rate(),'\n')

# EXERCISE 2
print('Exercise 2: (y[1:] - y[:-1]) / h =',(y[1:] - y[:-1]) / h,'\n')   # Returns the same thing as Exercise 1


# EXERCISE 3
m = 1.42    # +- 0.1kg
v = 35.0    # +- 0.9m/s 
r = 1.20    # +- 0.02m
g = 9.81

tension = m*np.sqrt((g**2) + ((v)**2/r)**2)
#To calculated uncertainties, take fractional uncertainties and add in quadrature.
unc = tension*np.sqrt((0.1/1.42)**2 + (0.9/35)**2 + (0.02/1.2)**2)
print('Exercise 3: ',tension,'+/-',unc,'Newtons','\n')  # Prints tension with uncertainty


# EXERCISE 4
col1, col2 = np.loadtxt('sample-correlation.txt', unpack=True)  # Load data
corr = np.corrcoef(col1,col2)   # Create correlation matrix
print('Exercise 4: corr[0][1] =',corr[0][1],'\n')   # Index to return correct value


# EXERCISE 5 
a, b = 0, np.pi/2
N = 50 # Number of intervals
dx = (b-a)/N
x = np.linspace(a, b-dx, N)
f = np.cos(x)
riemann_sum = np.sum(f * dx)
print('Exercise 5: riemann_sum =',riemann_sum,'\n')


# EXERCISE 6
def average(array):
    total = 0
    for num in array:
        total += num
    return total / len(array)

print('Exercise 6: average(np.arange(-5,5)) =',average(np.arange(-5,5)),'\n')

# EXERCISE 7
def stdev(array):
    mean = average(array)
    sumsquare = 0
    for num in array:
        sumsquare += (num-mean)**2
        
    dev = np.sqrt(sumsquare/(len(array)-1)) #Sample Standard Deviation use N-1
    return mean, dev

print('Exercise 7: stdev(np.arange(10)) = ',stdev(np.arange(10)),'\n')


# EXERCISE 8
volt, curr = np.loadtxt('sample-resistor.txt', delimiter = ',', skiprows = 1, unpack=True)
xbar = np.mean(curr)    # mean of current
ybar = np.mean(volt)    # mean of voltage
bhat = 0    # Initialize bhat
for i in range(len(curr)):
    bhat = np.sum((curr[i]-xbar) * (volt[i]-ybar)) / np.sum((curr[i]-xbar)**2)
ahat = ybar - bhat*xbar

voltage_offset = ahat + bhat*xbar

resistance = ybar / xbar    # R = V/I
print('Exercise 8: voltage_offset =',voltage_offset,'volts,','resistance =',resistance,'ohms')


