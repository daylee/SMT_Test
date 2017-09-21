from __future__ import print_function, division
import numpy as np
import os.path

from smt.methods import RBF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

xlimits = np.array([[angle[0],     angle[-1] ],
                    [azimuth[0],   azimuth[-1] ],
                    [elevation[0], elevation[-1] ] ])

counter = 0
flat_size = na * nz * ne
data = np.zeros((flat_size, nPoint * nc))
for p in range(nPoint):
    for c in range(nc):
        data[:, counter] = \
            solar_raw2[7 * p + c][119:119 + flat_size]
        #print(data[:, :, :, counter])
        counter += 1

counter = 0
data2 = np.zeros((na, nz, ne, nPoint*nc))
flat_size = na * nz * ne
for p in range(nPoint):
    for c in range(nc):
        data2[:, :, :, counter] = \
            solar_raw2[7 * p + c][119:119 + flat_size].reshape((na,
                                                          nz,
                                                          ne))
        counter += 1



xt = np.zeros((flat_size, 3))
yt = np.zeros((flat_size, 1))
counter = 0
for i in range(elevation.shape[0]):
    for j in range(azimuth.shape[0]):
        for k in range(angle.shape[0]):
            xt[counter,:]= np.array([angle[k], azimuth[j], elevation[i]])
            yt[counter] = data2[k, j, i, 1]
            counter += 1

#print(xt)
#print(xt.shape)
#yt = data[:,1]
#print(yt.shape)
sm = RMTB(xlimits=xlimits, order=4, num_ctrl_pts=40, reg_dv=1e-15, reg_cons=1e-15)
sm.set_training_values(xt, yt)
sm.train()


# Test the model
Az, El = np.meshgrid(azimuth, elevation)
Z = np.zeros((Az.shape[0],Az.shape[1]))

for i in range(Az.shape[0]):
    for j in range(Az.shape[1]):
        #Z[i,j] = sm.predict_values(np.hstack((np.pi / 2, Az[i,j], El[i,j])).reshape((1,3)))
        #print("angle : ",np.hstack((np.pi/2, Az[i,j], El[i,j])).reshape((1,3)))
        Z[i, j] = data2[9, j, i, 1]

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Az, El, Z)

plt.show()