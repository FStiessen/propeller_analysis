import os
import numpy as np
from scipy.interpolate import CubicSpline, interp1d


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

    def discretise(
            self,
            r_0,
            N_spanwise,
            spacing, type,
            N_chordwise,
            type_c,
            enforce_te_thickness=False,
            min_thickness=4e-4
            ):
        xi_s = np.linspace(0, 1, N_spanwise)
        if r_0*self.R < self.r_raw[0]:
            r_1 = self.r_raw[0]
        else:
            r_1 = self.R*r_0
        r_2 = self.R
        if spacing == 'cosine':
            xi_s = (1 - np.cos(xi_s*np.pi))/2
            print('cosine')
        elif spacing == 'linear':
            print('linear')
        else:
            raise ValueError('wrong type of spacing')
        self.r = r_1 + (r_2 - r_1)*xi_s
        if type == 'cubic_spline':
            f_c = CubicSpline(self.r_raw, self.c_raw)
            f_p = CubicSpline(self.r_raw, self.pitch_raw)
            f_t = CubicSpline(self.r_raw, self.thickness_raw)
        elif type == 'linear':
            f_c = interp1d(self.r_raw, self.c_raw, kind=type)
            f_p = interp1d(self.r_raw, self.pitch_raw, kind=type)
            f_t = interp1d(self.r_raw, self.thickness_raw, kind=type)
        else:
            raise ValueError('wrong type of interpolation')
        self.c = f_c(self.r)
        self.pitch = f_p(self.r)
        self.thickness = f_t(self.r)

        # airfoil interpolation
        self.xi_c = np.linspace(0, 1, N_chordwise)
        self.xi_c = (1 - np.cos(self.xi_c*np.pi))/2

        # interpolate all unique airfoils to common discretisation
        af_files = np.unique(self.af_raw)
        self.af_data = dict()
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

            if type_c == 'cubic_spline':
                f_u = CubicSpline(x_u_raw, y_u_raw)
                f_l = CubicSpline(x_l_raw, y_l_raw)
            elif type_c == 'linear':
                f_u = interp1d(x_u_raw, y_u_raw, kind=type_c)
                f_l = interp1d(x_l_raw, y_l_raw, kind=type_c)
            else:
                raise ValueError('wrong type of interpolation')
            y_u = f_u(self.xi_c)
            y_l = f_l(self.xi_c)

            y_new = np.concatenate((np.flip(y_u[1:]), y_l))
            x_new = np.concatenate((np.flip(self.xi_c[1:]), self.xi_c))
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
        self.centroids = np.zeros((len(self.r), 2))
        for idx_r in range(N_spanwise):
            r_val = self.r[idx_r]
            idx_above = np.searchsorted(self.r_raw, r_val, side='left')
            if idx_above == len(self.r_raw):
                idx_above = len(self.r_raw) - 1
            idx_below = max(idx_above - 1, 0)

            def _find_non_blended(idx, step):
                while (0 <= idx < len(self.af_raw) and
                       self.af_raw[idx] == 'blended'):
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
            y_u = np.flip(af_blended[1, :N_chordwise - 1])
            y_l = af_blended[1, N_chordwise:]
            thickness = y_u - y_l
            max_thickness = np.max(thickness)
            camber = (y_u + y_l)/2
            if self.thickness[idx_r] != 0:
                thickness = (thickness/max_thickness*self.thickness[idx_r] /
                             self.c[idx_r])
            if enforce_te_thickness:
                x = af_blended[0, N_chordwise:]
                te_thickness = thickness[-1]
                mod = np.where(
                    x >= 0.5,
                    4*(x - 0.5)**2*(min_thickness/self.c[idx_r] -
                                    te_thickness), 0
                    )
                thickness = thickness + mod
            y_u_new = camber + thickness/2
            y_l_new = camber - thickness/2
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
            self.areas[idx_r] = area
            self.centroids[idx_r] = centroid

            filename = f"output/airfoils/af_{idx_r}.dat"
            np.savetxt(filename,
                       af_blended.T,
                       fmt="%.6f",
                       comments="")

    def naca_airfoil(self, name):
        length = len(name)
        x = self.xi_c
        if length == 4:
            m = float(name[0])/100
            p = float(name[1])/10
            t = float(name[2:4])/100
            print('maximum camber:', m)
            print('camber location:', p)
            print('thickness:', t)
            y_t = 5*t*(0.2969*x**0.5 -
                       0.1260*x -
                       0.3516*x**2 +
                       0.2843*x**3 -
                       0.1015*x**4)
            y_c = np.where(x <= p, m/p**2*(2*p*x - x**2),
                           m/(1 - p)**2*((1 - 2*p) + 2*p*x - x**2))
            dy_cdx = np.where(x <= p, 2*m/p**2*(p - x),
                              2*m/(1 - p)**2*(p - x))
            theta = np.atan(dy_cdx)
            print('camber:', y_c)
            print('theta:', np.rad2deg(theta))
            sin = np.sin(theta)
            cos = np.cos(theta)
            x_u = x - y_t*sin
            x_l = x + y_t*sin
            y_u = y_c + y_t*cos
            y_l = y_c - y_t*cos
            print('x_u:', x_u)
