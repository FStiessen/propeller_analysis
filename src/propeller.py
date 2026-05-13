import numpy as np
from scipy.interpolate import interp1d


class propeller:
    def __init__(self, R, N_b, filename):
        self.N_b = N_b
        self.R = R
        data = np.genfromtxt(filename,
                             delimiter=",",
                             skip_header=1,
                             dtype=[('r', float),
                                    ('c', float),
                                    ('pitch', float),
                                    ('airfoil', 'U50')
                                    ]
                             )
        self.r_raw = R*data['r']
        self.c_raw = R*data['c']
        self.pitch_raw = data['pitch']/360*2*np.pi
        self.af_raw = data['airfoil']

    def discretise(self, N_spanwise, spacing, type, N_chordwise, type_c):
        xi_s = np.linspace(0, 1, N_spanwise)
        r_1 = self.r_raw[0]
        r_2 = self.R
        if spacing == 'cosine':
            xi_s = (1 - np.cos(xi_s*np.pi))/2
            print('cosine')
        elif spacing == 'linear':
            print('linear')
        else:
            raise ValueError('wrong type of spacing')
        self.r = r_1 + (r_2 - r_1)*xi_s
        f = interp1d(self.r_raw, self.c_raw, kind=type)
        self.c = f(self.r)
        f = interp1d(self.r_raw, self.pitch_raw, kind=type)
        self.pitch = f(self.r)

        # airfoil interpolation
        xi_c = np.linspace(0, 1, N_chordwise)
        xi_c = (1 - np.cos(xi_c*np.pi))/2

        # interpolate all unique airfoils to common discretisation
        af_files = np.unique(self.af_raw)
        self.af_data = dict()
        for af in af_files:
            filename = 'lib/airfoils/' + af + '.dat'
            x, y = np.genfromtxt(filename,
                                 skip_header=1,
                                 unpack=True,
                                 dtype=[float,
                                        float
                                        ]
                                 )
            idx_le = np.argmin(x)
            x_u_raw = np.flip(x[:idx_le + 1])
            y_u_raw = np.flip(y[:idx_le + 1])
            x_l_raw = x[idx_le:]
            y_l_raw = y[idx_le:]

            f = interp1d(x_u_raw, y_u_raw, kind=type_c)
            y_u = f(xi_c)
            f = interp1d(x_l_raw, y_l_raw, kind=type_c)
            y_l = f(xi_c)

            y_new = np.concatenate((np.flip(y_u[1:]), y_l))
            x_new = np.concatenate((np.flip(xi_c[1:]), xi_c))
            self.af_data[af] = np.vstack((x_new, y_new))

        # blend airfoils
        self.af_r = []
        for r_val in self.r:
            idx_above = np.searchsorted(self.r_raw, r_val, side='left')
            idx_below = idx_above - 1

            r_above = self.r_raw[idx_above]
            r_below = self.r_raw[idx_below]
            af_above = self.af_data[self.af_raw[idx_above]]
            af_below = self.af_data[self.af_raw[idx_below]]

            if r_above == r_val:
                af_blended = af_above
            elif self.af_raw[idx_above] == self.af_raw[idx_below]:
                af_blended = af_above
            else:
                xi = (r_val - r_below)/(r_above - r_below)
                k = (1 + np.cos(xi*np.pi))/2
                af_blended = k*af_below + (1 - k)*af_above
            self.af_r.append(af_blended)
