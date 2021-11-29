import numpy as np
import matplotlib.pyplot as plt
import glob


# EXERCISE 1
array = []
for filename in sorted(glob.glob('../to_combine/*.txt')):
    datafile = np.loadtxt(filename)
    for number in datafile:
        array.append(number)
print('\nExercise 1: ','Mean =',np.mean(array),', Standard Dev.=',np.std(array))


# EXERCISE 2
r = np.linspace(0.1,5,200)
rm = 2**(1/6)
V = (rm/r)**12 - 2*(rm/r)**6

plt.figure(figsize=(9,6))
plt.plot(r,V)
plt.xlabel('r')
plt.ylabel('V')
plt.xlim(0,5)
plt.ylim(-1.5,5)
plt.title('Lennard-Jones potential')
plt.savefig('lennard-jones.pdf')

print('\n\nExercise 2:')
plt.show()


# EXERCISE 3
k = 1.38*(10**(-23))
m = 4.652*(10**(-23))
v = np.linspace(0,100,200)

def pdf(T):
    return np.sqrt((m/(2*np.pi*k*T))**3) * 4*np.pi*(v**2)*(np.e**((-m*(v**2))/(2*k*T)))

plt.figure(figsize=(9,6))
plt.plot(pdf(80),label='80K')
plt.plot(pdf(233),label='233K')
plt.plot(pdf(298),label='298K')
plt.xlim(0,80)
plt.xlabel('Particle speed')
plt.ylabel('Probability Density Functions (PDF)')
plt.legend(title='Temperature')
plt.title('Maxwell-Boltzmann distributions for Nitrogen')
plt.savefig('maxwell-boltzmann.pdf')    # Submit this with the code

print('\n\nExercise 3:')
plt.show()


# EXERCISE 4
data = np.array([[2, 5, 6, 2],
[0, 1, 0, 0],
[1, 1, -1, 1],
[20, 20, 1, 0],
[9, 1.6, 4.2, 2],
])
print('\n\nExercise 4:\n','data[0] + data[2] =',data[0] + data[2])


# EXERCISE 5
voltage, temperature, error = np.loadtxt('sample_data.txt', unpack=True)

plt.figure(figsize=(9,6))
plt.scatter(voltage, temperature, color='blue')
plt.errorbar(voltage, temperature, yerr=error, fmt='none', ecolor='lime')
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature (C)')
plt.title('Scatter Plot of Temperature vs Voltage')
plt.savefig('temperature-voltage.pdf')    # Submit this with the code

print('\n\nExercise 5:')
plt.show()

