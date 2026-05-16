import os
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
                                    ('airfoil', 'U50'),
                                    ('thickness', float)
                                    ]
                             )
        self.r_raw = R*data['r']
        self.c_raw = R*data['c']
        self.pitch_raw = data['pitch']/360*2*np.pi
        self.af_raw = data['airfoil']
        self.thickness_raw = R*data['thickness']

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
        f = interp1d(self.r_raw, self.thickness_raw, kind=type)
        self.thickness = f(self.r)

        # airfoil interpolation
        xi_c = np.linspace(0, 1, N_chordwise)
        xi_c = (1 - np.cos(xi_c*np.pi))/2

        # interpolate all unique airfoils to common discretisation
        af_files = np.unique(self.af_raw)
        self.af_data = dict()
        self.af_base_thickness = dict()
        for af in af_files:
            if af == 'blended':
                continue
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
            self.af_base_thickness[af] = np.max(y_u - y_l)
            print(f"Airfoil {af}: thickness = {self.af_base_thickness[af]:.3f}")

            y_new = np.concatenate((np.flip(y_u[1:]), y_l))
            x_new = np.concatenate((np.flip(xi_c[1:]), xi_c))
            self.af_data[af] = np.vstack((x_new, y_new))

        # blend airfoils
        airfoil_output_dir = os.path.join('output', 'airfoils')
        os.makedirs(airfoil_output_dir, exist_ok=True)
        for entry in os.listdir(airfoil_output_dir):
            path = os.path.join(airfoil_output_dir, entry)
            if os.path.isfile(path):
                os.remove(path)

        self.af_r = []
        self.areas = np.zeros_like(self.r)
        self.centroids = []
        for idx_r in range(N_spanwise):
            r_val = self.r[idx_r]
            idx_above = np.searchsorted(self.r_raw, r_val, side='left')
            if idx_above == len(self.r_raw):
                idx_above = len(self.r_raw) - 1
            idx_below = max(idx_above - 1, 0)

            def _find_non_blended(idx, step):
                while 0 <= idx < len(self.af_raw) and self.af_raw[idx] == 'blended':
                    idx += step
                return idx

            idx_above_nb = _find_non_blended(idx_above, 1)
            idx_below_nb = _find_non_blended(idx_below, -1)
            if idx_above_nb >= len(self.af_raw):
                idx_above_nb = idx_below_nb
            if idx_below_nb < 0:
                idx_below_nb = idx_above_nb

            r_above = self.r_raw[idx_above_nb]
            r_below = self.r_raw[idx_below_nb]
            af_above = self.af_data[self.af_raw[idx_above_nb]]
            af_below = self.af_data[self.af_raw[idx_below_nb]]

            if self.af_raw[idx_above_nb] == self.af_raw[idx_below_nb] or r_above == r_val:
                af_blended = af_above
            else:
                xi = (r_val - r_below)/(r_above - r_below)
                k = (1 + np.cos(xi*np.pi))/2
                af_blended = k*af_below + (1 - k)*af_above
            if self.thickness[idx_r] != 0:
                y_u = np.flip(af_blended[1, :N_chordwise - 1])
                y_l = af_blended[1, N_chordwise:]
                thickness = y_u - y_l
                max_thickness = np.max(thickness)
                thickness_new = thickness/max_thickness*self.thickness[idx_r]*self.R/self.c[idx_r]
                camber = (y_u + y_l)/2
                y_u_new = camber + thickness_new/2
                y_l_new = camber - thickness_new/2
                af_blended[1, :] = np.concatenate([np.flip(y_u_new), np.array([0]), y_l_new])
            self.af_r.append(af_blended)

            def airfoil_centroid(af_coords):
                # shoelace formula for polygon area and centroid
                x = af_coords[0, :]
                y = af_coords[1, :]
                x1 = np.roll(x, -1)
                y1 = np.roll(y, -1)

                cross = x*y1 - x1*y
                A = 0.5*np.sum(cross)         # signed polygon area
                if A == 0:
                    raise ValueError("Degenerate airfoil polygon")
                Cx = np.sum((x + x1)*cross)/(6*A)
                Cy = np.sum((y + y1)*cross)/(6*A)

                return abs(A), np.array([Cx, Cy])

            area, centroid = airfoil_centroid(af_blended)
            self.areas[idx_r] = area*self.c[idx_r]**2
            self.centroids.append(centroid*self.c[idx_r])

            filename = f"output/airfoils/af_{idx_r}.dat"
            np.savetxt(filename,
                       af_blended.T,
                       fmt="%.6f",
                       comments="")
