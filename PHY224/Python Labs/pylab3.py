import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def linear(x,a,b):
    return x*a + b

def exponential(x,a,b):
    return b * np.e**(-a*x)


sample_decay, radiation_decay = np.loadtxt('Radioactive Decay Cs (20sec).txt', skiprows=2, unpack=True)
time_decay = sample_decay*20

sample_background, radiation_background = np.loadtxt('Radioactive Background 20min(20secdwell).txt', skiprows=2, unpack=True)
time_background = sample_background*20

actual_decay = radiation_decay - np.mean(radiation_background)

#Calculate standard deviation for each data point
actual_unc = np.sqrt(radiation_decay+radiation_background)

    
decay_rates = actual_decay / 20
rate_unc = actual_unc / 20


#popt_lin, pcov_lin = curve_fit(linear, time_decay, np.log(actual_decay))
popt_lin, pcov_lin = curve_fit(linear, time_decay, np.log(decay_rates))


print('Linear Half-Life:',np.log(2)/(-popt_lin[0]))

popt_exp, pcov_exp = curve_fit(exponential, time_decay/20, decay_rates)
popt_exp[0] = popt_exp[0]/20    # curve_fit malfunctions with the large time_decay values, so we have to use this workaround.

print('Exponential Half-Life:',np.log(2)/popt_exp[0])


# Original Plots

plt.figure(figsize=(10,6))
plt.scatter(time_decay, decay_rates, s=15, label='Data', color='lime')
plt.errorbar(time_decay, decay_rates, yerr=rate_unc, fmt='none', ecolor='lime')
plt.plot(time_decay, exponential(time_decay, *popt_exp), label='Curve Fit')
plt.plot(time_decay, exponential(time_decay, -popt_lin[0],np.e**popt_lin[1]), label='Linear Fit')
plt.plot(time_decay, exponential(time_decay, np.log(2)/156, decay_rates[0]), label='Theoretical')
plt.xlabel('Time (s)')
plt.ylabel('Rate (Counts/s)')
plt.title('Exponential')
plt.legend()
plt.savefig('exponential.png')
plt.show()



# Log axis plots

plt.figure(figsize=(10,6))
plt.scatter(time_decay, decay_rates, s=15, label='Data', color='lime')
plt.errorbar(time_decay, decay_rates, yerr=rate_unc, fmt='none', ecolor='lime')
plt.plot(time_decay, exponential(time_decay, *popt_exp), label='Curve Fit')
plt.plot(time_decay, exponential(time_decay, -popt_lin[0],np.e**popt_lin[1]), label='Linear Fit')
plt.plot(time_decay, exponential(time_decay, np.log(2)/156, decay_rates[0]), label='Theoretical')
plt.xlabel('Time (s)')
plt.ylabel('Rate (Counts/s)')
plt.title('Exponential with Log Scale')
plt.legend()
plt.yscale('log')
plt.savefig('logscale.png')
plt.show()



# Variance of Parameters
error_lin = np.sqrt(pcov_lin[0,0]) / (popt_lin[0] ** 2)
half_life_std_lin = error_lin * np.log(2)
print('\nLinear Half-life Standard Deviation:',half_life_std_lin)
print('Linear Half-Life:',np.log(2)/(-popt_lin[0]),'+-',half_life_std_lin)

error_exp = np.sqrt(pcov_exp[0,0]) / (popt_exp[0] ** 2)
half_life_std_exp = error_exp * np.log(2)
print('\nExponential Half-life Standard Deviation:',half_life_std_exp)
print('Exponential Half-Life:',np.log(2)/(popt_exp[0]),'+-',half_life_std_exp)

#Chi Squared Reduced Calculations:
sq_list = []
for i in range(len(decay_rates)):
    val = decay_rates[i] - exponential(time_decay[i],popt_exp[0],popt_exp[1])
    val /= rate_unc[i]
    sq_list.append(val**2)
xsqred = 1/(len(decay_rates)-2) * sum(sq_list)
print('\nChi squared reduced exp:',xsqred)

sq_list = []
for i in range(len(decay_rates)):
    val = decay_rates[i] - exponential(time_decay[i],-popt_lin[0],np.e**popt_lin[1])
    val /= rate_unc[i]
    sq_list.append(val**2)
xsqred = 1/(len(decay_rates)-2) * sum(sq_list)
print('\nChi squared reduced lin:',xsqred)