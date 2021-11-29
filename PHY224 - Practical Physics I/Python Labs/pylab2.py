import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


voltage, current = np.loadtxt('voltage_current.txt', delimiter=',', unpack=True)

def linear(x, a, b):
    return a*x + b

p_opt , p_cov = opt.curve_fit(linear, voltage, current/1000)
print('Slope, Intercept =',p_opt,'\n')

verr = [max(x*0.0025,0.05) for x in voltage]    #Voltage error
cerr = [max(x*0.0075,0.05)/1000 for x in current]    #Current error

plt.figure(figsize=(10,7))

plt.title('Current vs Voltage')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.plot(voltage, linear(voltage, *p_opt),color="g")
plt.errorbar(voltage, current/1000, xerr=verr, yerr=cerr,fmt='none',ecolor="r")
plt.savefig('current_voltage.pdf')
plt.show()

# Calculate resistance error
prop_verr = max(verr/voltage)
prop_cerr = max(cerr/current)
rerr = np.sqrt(prop_verr**2 + prop_cerr**2)
# Print the result with error
print('\nResistance =',1/p_opt[0],'+-',rerr/p_opt[0],'Ohms \n')


# Question 2: Force line to pass through 0

def no_intercept(x, a):
    return a*x

popt, _ = opt.curve_fit(no_intercept, voltage, current/1000)
print('(Force b = 0) Resistance =',1/popt[0],'Ohms \n')


#Question 4: Chi-squared

#current is mA. cerr is in A.
sq_list = []
for i in range(len(voltage)):
    val = current[i]/1000 - voltage[i]*p_opt[0] + p_opt[1]
    val /= cerr[i]
    sq_list.append(val**2)
    
    
#15 measurements, 2 parameters
print('Chi Squared:',sum(sq_list))
xsqred = 1/(15-2) * sum(sq_list)
print('Chi Squared Reduced:',xsqred)

