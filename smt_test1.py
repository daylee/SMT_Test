from __future__ import print_function, division
import numpy as np
import os.path

from smt.methods import RBF
import matplotlib.pyplot as plt

from scipy import linalg
from smt.utils import compute_rms_error
from smt.problems import Sphere, NdimRobotArm
from smt.sampling import LHS
from smt.methods import LS, QP, KPLS, KRG, KPLSK, GEKPLS
try:
    from smt.methods import IDW, RBF, RMTC, RMTB
    compiled_available = True
except:
    compiled_available = False

try:
#    import matplotlib.pyplot as plt
    plot_status = True
except:
    plot_status = False

########### Initialization of the problem, construction of the training and validation points


# Raw data to load
fpath = os.path.dirname(os.path.realpath(__file__))
fpath = os.path.join(fpath, 'data')
solar_raw1 = np.genfromtxt(fpath + '/Solar/Area10.txt')
solar_raw2 = np.loadtxt(fpath + '/Solar/Area_all.txt')
#solar_raw3 = np.loadtxt(fpath + '/Solar/Thermal.txt')
comm_rawGdata = np.genfromtxt(fpath + '/Comm/Gain.txt')
comm_raw = (10 ** (comm_rawGdata / 10.0)
            ).reshape((361, 361), order='F')
power_raw = np.genfromtxt(fpath + '/Power/curve.dat')

na = 10
nz = 73
ne = 37

nc = 7
nPoint = 12

angle = np.zeros(na)
azimuth = np.zeros(nz)
elevation = np.zeros(ne)

index = 0
for i in range(na):
    angle[i] = solar_raw1[index]
    index += 1
for i in range(nz):
    azimuth[i] = solar_raw1[index]
    index += 1

index -= 1
azimuth[nz - 1] = 2.0 * np.pi
for i in range(ne):
    elevation[i] = solar_raw1[index]
    index += 1

angle[0] = 0.0
angle[-1] = np.pi / 2.0
azimuth[0] = 0.0
azimuth[-1] = 2 * np.pi
elevation[0] = 0.0
elevation[-1] = np.pi

"""
counter = 0
data = np.zeros((na, nz, ne, nPoint * nc))
flat_size = na * nz * ne
for p in range(nPoint):
    for c in range(nc):
        data[:, :, :, counter] = \
            solar_raw2[7 * p + c][119:119 + flat_size].reshape((na, nz, ne))

        #print(data[:, :, :, counter])
        counter += 1
"""

counter = 0
flat_size = na * nz * ne
data = np.zeros((flat_size, nPoint * nc))
for p in range(nPoint):
    for c in range(nc):
        data[:, counter] = \
            solar_raw2[7 * p + c][119:119 + flat_size]
        #print(data[:, :, :, counter])
        counter += 1


"""
xt = np.zeros((na*nz*ne, 3))
counter = 0
for i in range(elevation.shape[0]):
    for j in range(azimuth.shape[0]):
        for k in range(angle.shape[0]):
            xt[counter]= np.array([angle[k], azimuth[j], elevation[i]])
            counter += 1
"""

xt = np.zeros((flat_size, 3))
counter = 0
for i in range(elevation.shape[0]):
    for j in range(azimuth.shape[0]):
        for k in range(angle.shape[0]):
            xt[counter,:]= np.array([angle[k], azimuth[j], elevation[i]])
            counter += 1


print(xt)
print(xt.shape)
yt = data[:,1]
print(yt.shape)
sm = RBF(d0=5)
sm.set_training_values(xt, yt)
sm.train()

#xt = np.array([azimuth, elevation])
#yt = np.array(data[1,:,:,:])
#plt.plot(xt, yt,  'o')
#plt.show()
#print(yt)