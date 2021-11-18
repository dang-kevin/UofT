import numpy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def linearized(x,a,b):
    return a*x + b

def non_linear(x,a,b):
    return a * x**b

def non_linear_tungsten(x,a):
    return a * x**0.5882

voltage, current = numpy.loadtxt('lightbulb.txt',delimiter=',',unpack=True)
current = current / 1000 #Convert from mA to A.


verr = [max(x*0.0025,0.05) for x in voltage]
cerr = [max(x*0.0075,0.00005) for x in current]

popt_lin, pcov_lin = curve_fit(linearized,numpy.log(voltage),numpy.log(current))
print('Linear Slope, Intercept = ',popt_lin)

popt_nl, pcov_nl = curve_fit(non_linear,voltage,current)
print('Nonlinear Slope, Intercept = ',popt_nl)

print('Linear Relationship: I =',numpy.e**popt_lin[1],'* V ^',popt_lin[0])
print('Nonlinear Relationship: I =',popt_nl[0],'* V ^',popt_nl[1])


const_of_prop_W = numpy.average(current/(voltage**0.5882))
popt_w, _ = curve_fit(non_linear,voltage,current)

plt.figure(figsize=(10,6))
plt.errorbar(voltage,current,xerr=verr,yerr=cerr,fmt='none',ecolor='r',label='Data')
plt.plot(voltage,non_linear(voltage,numpy.e**popt_lin[1],popt_lin[0]),label='Linear Fit')
plt.plot(voltage,non_linear(voltage,*popt_nl),label='Nonlinear Fit')
plt.plot(voltage,non_linear(voltage,popt_w[0],0.5882),label='Theoretical (Tungsten)')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Current vs Voltage')
plt.legend()
plt.savefig('current-voltage.pdf')
plt.show()

plt.figure(figsize=(10,6))
plt.errorbar(voltage,current,xerr=verr,yerr=cerr,fmt='none',ecolor='r',label='Data')
plt.plot(voltage,non_linear(voltage,numpy.e**popt_lin[1],popt_lin[0]),label='Linear Fit')
plt.plot(voltage,non_linear(voltage,*popt_nl),label='Nonlinear Fit')
plt.plot(voltage,non_linear(voltage,popt_w[0],0.5882),label='Theoretical (Tungsten)')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Current vs Voltage (log-scale)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig('log-log.pdf')
plt.show()


std_dev_lin = numpy.sqrt(pcov_lin[0][0])
print('Linear Exponent =',popt_lin[0],'+-',std_dev_lin)

std_dev_nl = numpy.sqrt(pcov_nl[1][1])
print('Nonlinear Exponent =',popt_nl[1],'+-',std_dev_nl)

chi_sq = sum([((current[i]-non_linear(voltage[i],numpy.e**popt_lin[1],popt_lin[0]))/cerr[i])**2 for i in range(len(voltage))])
chi_sq_red = chi_sq/(len(voltage)-2)

print('LINEAR Chi^2_red:',chi_sq_red)


chi_sq = sum([((current[i]-non_linear(voltage[i],*popt_nl))/cerr[i])**2 for i in range(len(voltage))])
chi_sq_red = chi_sq/(len(voltage)-2)

print('NONLINEAR Chi^2_red:',chi_sq_red)

