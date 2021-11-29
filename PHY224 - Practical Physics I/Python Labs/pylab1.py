import matplotlib.pyplot as plt
import numpy

'''
Displaying experimental data.
'''
data = numpy.loadtxt('position.txt', comments="#", skiprows=2, delimiter='\t')
time = data.T[0]
dist = data.T[1]

print("Experimental Data")
plt.figure(1)
plt.title('Experimental Plot 1')
plt.plot(time, dist)
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")

plt.show()

print()
print()

'''
Forward Euler Simulation
'''
dt = 0.01
omega0 = 9.24
k = 17.08
m = 0.2001 # +- 0.0001

time = numpy.arange(0,10,dt)

y = numpy.zeros(len(time))
v = numpy.zeros(len(time))
e = numpy.zeros(len(time))

y[0] = -0.03
e[0] = 0.5 * k * y[0]**2
  
for i in range(len(time)-1):
    y[i+1] = y[i] + dt*v[i]
    v[i+1] = v[i] - dt*y[i]*(omega0**2)
    e[i+1] = 0.5*m*v[i+1]**2 + 0.5*k*y[i+1]**2

print('Forward Euler Simulation')

plt.figure(2)
plt.title('Forward Euler Plot 1')
plt.plot(time,y)
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")

plt.figure(3)
plt.title('Forward Euler Plot 2')
plt.plot(time,v)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")

plt.figure(4)
plt.title('Forward Euler Plot 3')
plt.plot(y,v)
plt.xlabel("Position (m)")
plt.ylabel("Velocity (m/s)")

plt.figure(5)
plt.title('Forward Euler Plot 4')
plt.plot(time,e)
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")

plt.show()

print()
print()

'''
Sympletic Simulation
'''

dt = 0.01
omega0 = 9.24
k = 17.08
m = 0.2001

time = numpy.arange(0,10,dt)

y = numpy.zeros(len(time))
v = numpy.zeros(len(time))
e = numpy.zeros(len(time))


y[0] = -0.03
e[0] = 0.5 * k * y[0]**2
  
for i in range(len(time)-1):
    y[i+1] = y[i] + dt*v[i]
    v[i+1] = v[i] - dt*y[i+1]*(omega0**2)
    e[i+1] = 0.5*m*v[i+1]**2 + 0.5*k*y[i+1]**2

print('Sympletic Simulation')

plt.figure(6)
plt.title('Sympletic Plot 1')
plt.plot(time,y)
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")

plt.figure(7)
plt.title('Sympletic Plot 2')
plt.plot(time,v)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")

plt.figure(8)
plt.title('Sympletic Plot 3')
plt.plot(y,v)
plt.xlabel("Position (m)")
plt.ylabel("Velocity (m/s)")

plt.figure(9)
plt.title('Sympletic Plot 4')
plt.plot(time,e)
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")

plt.show()



#####################################################################
# SECOND SESSION
#####################################################################


dt = 0.01
omega0 = 9.24
k = 17.08
m = 0.2170
gamma = 0.016

time = numpy.arange(0,100,dt)

y = numpy.zeros(len(time))
v = numpy.zeros(len(time))
e = numpy.zeros(len(time))


y[0] = -0.03
e[0] = 0.5 * k * y[0]**2
  
for i in range(len(time)-1):
    y[i+1] = y[i] + dt*v[i]
    v[i+1] = v[i] - dt*(y[i+1]*omega0**2 + gamma*v[i])
    e[i+1] = 0.5*m*v[i+1]**2 + 0.5*k*y[i+1]**2

print('Sympletic Simulation with Damped System')

plt.figure(10)
plt.title('Damped Sympletic Plot 1')
plt.plot(time,y)
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")

plt.figure(11)
plt.title('Damped Sympletic Plot 2')
plt.plot(time,e)
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")

plt.show()

