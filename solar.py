''' Solar discipline for CADRE '''



#from openmdao.main.api import Component
#from openmdao.lib.datatypes.api import Float, Array

#from CADRE.kinematics import fixangles

from smt.methods import RBF
import os
import numpy as np
from smt.methods import RBF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

class Solar_ExposedArea:

    '''Exposed area calculation for a given solar cell

       p: panel ID [0,11]

       c: cell ID [0,6]

       a: fin angle [0,90]

       z: azimuth [0,360]

       e: elevation [0,180]

       LOS: line of sight with the sun [0,1]
    '''

    # Inputs
    # finAngle = Float(0., iotype="in", units="rad", desc="Fin angle of solar panel", copy=None)

    def __init__(self):

        fpath = os.path.dirname(os.path.realpath(__file__))
        raw1 = np.genfromtxt(fpath + '/data/Solar/Area10.txt')
        fpath = os.path.dirname(os.path.realpath(__file__))
        raw2 = np.loadtxt(fpath + "/data/Solar/Area_all.txt")

        self.n = 1
        self.nc = 7
        self.np = 12

        # Inputs
        #self.add('azimuth', Array(np.zeros((n,)), size=(n,), dtype=np.float,
        #                          units='rad',
        #                          desc='Azimuth angle of the sun in the body-fixed frame over time',
        #                          iotype='in'))
        #self.add('elevation', Array(np.zeros((n,)), size=(n,), dtype=np.float,
        #                            units='rad',
        #                            desc='Elevation angle of the sun in the body-fixed frame over time',
        #                            iotype='in'))

        # Outputs
        #self.add('exposedArea', Array(np.zeros((self.nc, self.np, self.n)),
        #                              size=(self.nc, self.np, self.n),
        #                              dtype=np.float, iotype='out',
        #                              desc="Exposed area to sun for each solar cell over time",
        #                              units='m**2',
        #                              low=-5e-3, high=1.834e-1))

        self.na = 10
        self.nz = 73
        self.ne = 37
        angle = np.zeros(self.na)
        azimuth = np.zeros(self.nz)
        elevation = np.zeros(self.ne)

        index = 0
        for i in range(self.na):
            angle[i] = raw1[index]
            index += 1
        for i in range(self.nz):
            azimuth[i] = raw1[index]
            index += 1

        index -= 1
        azimuth[self.nz - 1] = 2.0 * np.pi
        for i in range(self.ne):
            elevation[i] = raw1[index]
            index += 1

        angle[0] = 0.0
        angle[-1] = np.pi / 2.0
        azimuth[0] = 0.0
        azimuth[-1] = 2 * np.pi
        elevation[0] = 0.0
        elevation[-1] = np.pi

        xlimits = np.array([[angle[0], angle[-1]],
                            [azimuth[0], azimuth[-1]],
                            [elevation[0], elevation[-1]]])

        counter = 0
        data = np.zeros((self.na, self.nz, self.ne, self.np * self.nc))
        flat_size = self.na * self.nz * self.ne
        for p in range(self.np):
            for c in range(self.nc):
                data[:, :, :, counter] = \
                    raw2[7 * p + c][119:119 + flat_size].reshape((self.na,
                                                                  self.nz,
                                                                  self.ne))
                counter += 1

        xt = np.zeros((flat_size, 3))
        yt = np.zeros((flat_size, 1))
        counter = 0
        for i in range(elevation.shape[0]):
            for j in range(azimuth.shape[0]):
                for k in range(angle.shape[0]):
                    xt[counter, :] = np.array([angle[k], azimuth[j], elevation[i]])
                    yt[counter] = data[k, j, i, 1]
                    counter += 1

        # print(xt)
        # print(xt.shape)
        # yt = data[:,1]
        # print(yt.shape)
        sm = RMTB(xlimits=xlimits, order=4, num_ctrl_pts=40, reg_dv=1e-15, reg_cons=1e-15)
        sm.set_training_values(xt, yt)
        sm.train()


        #self.MBI = MBI(data, [angle, azimuth, elevation],
        #                     [4, 10, 8],
        #                     [4, 4, 4])

        self.x = np.zeros((self.n, 3))
        self.Jfin = None
        self.Jaz = None
        self.Jel = None

        Az, El = np.meshgrid(azimuth, elevation)
        Z = np.zeros((Az.shape[0], Az.shape[1]))

        for i in range(Az.shape[0]):
            for j in range(Az.shape[1]):
                #Z[i,j] = sm.predict_values(np.hstack((np.pi / 2, Az[i,j], El[i,j])).reshape((1,3)))
                # print("angle : ",np.hstack((np.pi/2, Az[i,j], El[i,j])).reshape((1,3)))
                Z[i, j] = data[9, j, i, 1]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(Az, El, Z)

        plt.show()


'''
    def list_deriv_vars(self):
        input_keys = ('azimuth', 'elevation', 'finAngle',)
        output_keys = ('exposedArea',)
        return input_keys, output_keys

    def setx(self):
        """ Sets our state array"""

        result = fixangles(self.n, self.azimuth, self.elevation)
        self.x[:, 0] = self.finAngle
        self.x[:, 1] = result[0]
        self.x[:, 2] = result[1]

    def provideJ(self):
        """ Calculate and save derivatives (i.e., Jacobian). """

        self.Jfin = self.MBI.evaluate(self.x, 1).reshape(self.n, 7, 12,
                                                         order='F')
        self.Jaz = self.MBI.evaluate(self.x, 2).reshape(self.n, 7, 12,
                                                        order='F')
        self.Jel = self.MBI.evaluate(self.x, 3).reshape(self.n, 7, 12,
                                                        order='F')

    def execute(self):
        """ Calculate output. """

        self.setx()
        P = self.MBI.evaluate(self.x).T
        self.exposedArea = P.reshape(7, 12, self.n, order='F')

    def apply_deriv(self, arg, result):
        """ Matrix-vector product with the Jacobian. """

        if 'exposedArea' in result:
            for c in range(7):
                if 'finAngle' in arg:
                    result['exposedArea'][c, :, :] += \
                        self.Jfin[:, c, :].T * arg['finAngle']
                if 'azimuth' in arg:
                    result['exposedArea'][c, :, :] += \
                        self.Jaz[:, c, :].T * arg['azimuth']
                if 'elevation' in arg:
                    result['exposedArea'][c, :, :] += \
                        self.Jel[:, c, :].T * arg['elevation']

    def apply_derivT(self, arg, result):
        """ Matrix-vector product with the transpose of the Jacobian. """

        if 'exposedArea' in arg:
            for c in range(7):

                # incoming arg is often sparse, so check it first
                if len(np.nonzero(arg['exposedArea'][c, :, :])[0]) == 0:
                    continue

                if 'finAngle' in result:
                    result['finAngle'] += \
                        np.sum(
                            self.Jfin[:, c, :].T * arg['exposedArea'][c, :, :])

                if 'azimuth' in result:
                    result['azimuth'] += \
                        np.sum(
                            self.Jaz[:, c, :].T * arg['exposedArea'][c, :, :], 0)

                if 'elevation' in result:
                    result['elevation'] += \
                        np.sum(
                            self.Jel[:, c, :].T * arg['exposedArea'][c, :, :], 0)
'''

Solar_ExposedArea = Solar_ExposedArea()