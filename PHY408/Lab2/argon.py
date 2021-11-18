import numpy as np

# the structure factor data for Argon
YanData = np.array([
    0.161000, 0.138000, 0.120000, 0.105000, 0.088000, 0.086000, 0.087000, 0.088000,
    0.090000, 0.094000, 0.105000, 0.125000, 0.152000, 0.277000, 0.468000, 1.248000,
    2.391000, 2.604000, 1.725000, 1.160000, 0.904000, 0.761000, 0.640000, 0.626000,
    0.633000, 0.680000, 0.744000, 0.869000, 1.038000, 1.183000, 1.225000, 1.237000,
    1.231000, 1.184000, 1.099000, 1.011000, 0.932000, 0.870000, 0.801000, 0.771000,
    0.871000, 0.961000, 1.026000, 1.075000, 1.102000, 1.107000, 1.087000, 1.064000,
    1.033000, 0.998000, 0.968000, 0.945000, 0.936000, 0.946000, 0.966000, 0.987000,
    1.004000, 1.018000, 1.027000, 1.024000, 1.022000, 1.015000, 1.006000, 0.998000,
    0.997000, 0.996000, 0.996000, 0.995000, 0.995000, 0.995000, 0.995000, 0.996000,
    1.001000, 1.001000, 1.002000, 1.003000, 1.003000, 1.003000, 1.003000, 1.002000,
    1.001000, 1.000000, 1.001000, 1.000000, 0.999000, 0.998000, 0.997000, 0.998000,
    0.998000, 0.999000, 1.000000, 1.000000, 1.001000, 1.001000, 1.001000, 1.001000,
    1.001000, 1.001000, 1.001000, 1.000000, 1.000000, 0.999000, 0.999000, 0.999000,
    1.000000, 0.999000, 0.999000, 0.999000, 1.000000, 1.000000, 1.000000, 1.002000,
    1.001000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.999000,
    0.999000, 0.999000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000
])

dk = 0.12                 # inverse Angstroms
massRho = 1.4274          # grams/cc 
molWeight = 39.948        # grams/mol
Navogadro = 6.0221367e23  # atoms/mol
