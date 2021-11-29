from scipy.stats import poisson, norm
import matplotlib.pyplot as plt
import numpy as np

sample_number, count = np.loadtxt('Radioctive Activity-Fiesta Plate(3sec. Dwell).txt', skiprows=2, unpack=True)

background = np.loadtxt('Background20min.txt')
avg_radioactivity = np.mean(background)

plate_count = count - avg_radioactivity    #Subtract average radioactivity from count


#Histogram
plt.figure(figsize=(9,6))
plt.hist(plate_count, bins=20, normed=True)

# Poisson
mu = np.mean(plate_count)
pois = poisson.pmf(sample_number,mu)
plt.plot(pois,label='Poisson')


# Gaussian
sigma = np.sqrt(mu)
gaussian = norm.pdf(sample_number, mu, sigma)
plt.plot(gaussian, label='Gaussian')

plt.xlim(40,100)
plt.title('Histogram of Fiesta Plate Count Data')
plt.legend()
plt.savefig('hist1.pdf')
plt.show()

'''
Bonus
'''
#Histogram
plt.figure(figsize=(9,6))
plt.hist(background, bins=12, normed=True)

# Poisson
mewtwo = np.mean(background)
pois2 = poisson.pmf(sample_number,mewtwo)
plt.plot(pois2,label='Poisson')


# Gaussian
sigma2 = np.sqrt(mewtwo)
gaussian2 = norm.pdf(sample_number, mewtwo, sigma2)
plt.plot(gaussian2, label='Gaussian')

plt.xlim(0,15)
plt.title('Histogram of Background Count Data')
plt.legend()
plt.savefig('hist2.pdf')
plt.show()
