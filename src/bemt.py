import os
import subprocess
import numpy as np
from scipy.optimize import brentq

"""
SOURCES:
[1] Ning, A. (2021). Using blade element momentum methods with gradient-based design optimization. Structural and Multidisciplinary Optimization, 64(2), 991-1014. https://doi.org/10.1007/s00158-021-02883-6
[2] Du, Z., & Selig, M. (1998). A 3-D stall-delay model for horizontal axis wind turbine performance prediction. In 1998 ASME Wind Energy Symposium. American Institute of Aeronautics and Astronautics. 1998 ASME Wind Energy Symposium. https://doi.org/10.2514/6.1998-21
[3] Viterna, L. A., & Janetzke, D. C. (1982). Theoretical and experimental power from large horizontal-axis wind turbines (NASA Technical Memorandum NASA-TM-82944). National Aeronautics and Space Administration.
[4] Drela, M. (1989). XFOIL: An analysis and design system for low Reynolds number airfoils [Computer software]. Massachusetts Institute of Technology. https://web.mit.edu/drela/Public/web/xfoil/
"""


class bemt:
    def __init__(self, prop, V, Omega, feather, mu, rho, c_sound):
        self.R = prop.R
        self.N_b = prop.N_b
        self.V = V
        self.r = prop.r
        self.pitch = prop.pitch + np.radians(feather)
        self.c_sound = c_sound
        self.Omega = Omega
        self.mu = mu
        self.rho = rho
        self.Re = (self.rho*(self.V**2 + (self.Omega*prop.r)**2)**0.5 *
                   prop.c/self.mu)
        self.M = (self.V**2 + (self.Omega*prop.r)**2)**0.5/c_sound
        self.sigma_prime = prop.N_b*prop.c/(2*np.pi*prop.r)
        self.chord = prop.c
        self.epsilon = 1e-6

    def generate_polars_simple(self):
        alpha = np.linspace(-180, 180, 361)
        alpha_rad = np.radians(alpha)
        A_1 = 0.5
        B_1 = 2

        cl = A_1*np.sin(2*alpha_rad)
        cd = B_1*np.sin(alpha_rad)**2

        extrapolated_polar_array = np.column_stack((alpha, cl, cd))
        self.af_polars_extrap = []
        for idx in range(len(self.r)):
            self.af_polars_extrap.append(extrapolated_polar_array)

    def generate_polars(
            self, alpha_min, alpha_max, dalpha, N_crit, turbine_airfoil
            ):
        # Usage of [4]
        airfoil_dir = os.path.join('output', 'airfoils')

        if not os.path.exists(airfoil_dir):
            print(f"Directory {airfoil_dir} not found")
            return

        airfoil_files = sorted([f for f in os.listdir(airfoil_dir) if
                                f.endswith('.dat')])

        if not airfoil_files:
            print(f"No .dat files found in {airfoil_dir}")
            return

        polars_dir = os.path.join('output', 'polars')
        if os.path.exists(polars_dir):
            for file in os.listdir(polars_dir):
                os.remove(os.path.join(polars_dir, file))
        os.makedirs(polars_dir, exist_ok=True)

        xfoil_exe = os.path.abspath(os.path.join('lib', 'xfoil', 'xfoil.exe'))

        if not os.path.exists(xfoil_exe):
            print(f"xfoil executable not found at {xfoil_exe}")
            return

        self.af_polars = [None]*len(self.r)
        self.af_polars_corrected = [None]*len(self.r)
        self.af_polars_extrap = [None]*len(self.r)

        for af_file in airfoil_files:
            af_path = os.path.abspath(os.path.join(airfoil_dir, af_file))
            af_name = os.path.splitext(af_file)[0]

            # Extract spanwise index from filename
            # (e.g., af_0.dat -> idx_r = 0)
            idx_r = int(af_name.split('_')[1])
            Re = int(round(self.Re[idx_r]))
            Mach = self.M[idx_r]

            output_file = os.path.abspath(os.path.join(polars_dir, af_name))
            af_path_xfoil = af_path.replace('\\', '/')
            output_file_xfoil = output_file.replace('\\', '/')

            # Remove any stale polar file before generation
            polar_path = f"{output_file}.polar"
            if os.path.exists(polar_path):
                os.remove(polar_path)

            input_script = f"""LOAD
{af_path_xfoil}

PANE
OPER
VISC
{Re}
MACH {Mach}
VPAR
N {N_crit}

PACC
{output_file_xfoil}.polar

ASEQ {alpha_min} {alpha_max} {dalpha}

QUIT
"""

            # Run xfoil
            try:
                process = subprocess.Popen(
                    [xfoil_exe],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.path.dirname(xfoil_exe)
                )
                stdout, stderr = process.communicate(input=input_script,
                                                     timeout=60)

                if stderr:
                    print(f"xfoil stderr for {af_file}: {stderr}")
                if stdout:
                    print(f"xfoil stdout for {af_file}: {stdout[:200]}")

                if os.path.exists(f"{output_file}.polar"):
                    print(f"✓ Generated polar for {af_file}")

                    # Parse polar file
                    try:
                        with open(f"{output_file}.polar", 'r') as pf:
                            lines = pf.readlines()

                        header_index = None
                        for idx, line in enumerate(lines):
                            if line.strip().lower().startswith('alpha'):
                                header_index = idx
                                break

                        if header_index is None:
                            raise ValueError('Could not find polar header line')

                        polar_data = np.loadtxt(lines[header_index + 2:])
                        if polar_data.size == 0:
                            raise ValueError('No polar data rows found')
                        if polar_data.ndim == 1:
                            polar_data = polar_data.reshape(1, -1)

                        alpha = polar_data[:, 0]
                        cl = polar_data[:, 1]
                        cd = polar_data[:, 2]

                        # Store as structured array with alpha, cl, cd columns
                        polar_array = np.column_stack((alpha, cl, cd))
                        self.af_polars[idx_r] = polar_array

                        print(f"  → Stored polar data for {af_name}")
                    except Exception as e:
                        print(f"  → Error parsing polar file: {e}")
                else:
                    print(f"✗ No polar file created for {af_file}")

            except subprocess.TimeoutExpired:
                process.kill()
                print(f"Timeout generating polar for {af_file}")
            except FileNotFoundError as e:
                print(f"Error running xfoil: {e}")
                return

            def polar_3d_correction(data_raw):
                # Implementation of [2]
                try:
                    alpha_raw = data_raw[:, 0]
                    cl_raw = data_raw[:, 1]
                    cd_raw = data_raw[:, 2]
                except TypeError:
                    return []
                cd_0 = np.interp(0, alpha_raw, cd_raw)
                f = lambda x: np.interp(x, alpha_raw, cl_raw)
                if f(alpha_raw[0])*f(alpha_raw[-1]) < 0:
                    alpha_0 = brentq(f, alpha_raw[0], alpha_raw[-1])
                else:
                    alpha_0 = alpha_raw[0]
                cl_p = 2*np.pi*np.radians(alpha_raw - alpha_0)

                a = 1
                b = 1
                d = 1
                c = self.chord[idx_r]
                r = self.r[idx_r]
                Omega = np.abs(self.Omega)
                R = self.R
                V_w = np.abs(self.V)

                Lambda = Omega*R/(V_w**2 + (Omega*R)**2)**0.5
                f_l = (
                    1/(2*np.pi) *
                    (1.6*(c/r)/0.1267 *
                     (a - (c/r)**(d/Lambda*R/r)) /
                     (b + (c/r)**(d/Lambda*R/r)) - 1)
                    )
                f_d = (
                    1/(2*np.pi) *
                    (1.6*(c/r)/0.1267 *
                     (a - (c/r)**(d/(2*Lambda)*R/r)) /
                     (b + (c/r)**(d/(2*Lambda)*R/r)) - 1)
                    )
                delta_cl = f_l*(cl_p - cl_raw)
                delta_cd = f_d*(cd - cd_0)
                cl_new = cl + delta_cl
                cd_new = cd + delta_cd
                corrected_polar_array = (
                    np.column_stack((alpha_raw, cl_new, cd_new))
                    )
                return corrected_polar_array

            def extrapolation_fun(data_raw):
                # Implementation of [3]
                alpha = np.linspace(-180, 180, 361)
                alpha_rad = np.radians(alpha)
                A_1 = 0.5
                B_1 = 2
                try:
                    alpha_raw = data_raw[:, 0]
                    cl_raw = data_raw[:, 1]
                    cd_raw = data_raw[:, 2]
                    a_min = alpha_raw[0]
                    a_max = alpha_raw[-1]
                    cl_min = cl_raw[0]
                    cl_max = cl_raw[-1]
                    A_2_min = ((cl_min - A_1*np.sin(2*np.radians(a_min))) *
                               np.sin(np.radians(a_min)) /
                               np.cos(np.radians(a_min))**2)
                    A_2_max = ((cl_max - A_1*np.sin(2*np.radians(a_max))) *
                               np.sin(np.radians(a_max)) /
                               np.cos(np.radians(a_max))**2)

                    cd_min = cd_raw[0]
                    cd_max = cd_raw[-1]
                    B_2_min = ((cd_min - B_1*np.sin(np.radians(a_min))**2) /
                               np.cos(np.radians(a_min)))
                    B_2_max = ((cd_max - B_1*np.sin(np.radians(a_max))**2) /
                               np.cos(np.radians(a_max)))

                    # Shared conditions for cl and cd
                    conditions = [
                        abs(alpha) >= 90,
                        (-90 < alpha) & (alpha < a_min),
                        (90 > alpha) & (alpha > a_max),
                        (alpha >= a_min) & (alpha <= a_max)
                    ]

                    # Choices for cl
                    cl_choices = [
                        A_1*np.sin(2*alpha_rad),
                        (A_1*np.sin(2*alpha_rad) +
                         A_2_min*np.cos(alpha_rad)**2/np.sin(alpha_rad)),
                        (A_1*np.sin(2*alpha_rad) +
                         A_2_max*np.cos(alpha_rad)**2/np.sin(alpha_rad)),
                        np.interp(alpha, alpha_raw, cl_raw)
                    ]
                    cl = np.select(conditions, cl_choices, default=0)

                    # Choices for cd
                    cd_choices = [
                        B_1*np.sin(alpha_rad)**2,
                        B_1*np.sin(alpha_rad)**2 + B_2_min*np.cos(alpha_rad),
                        B_1*np.sin(alpha_rad)**2 + B_2_max*np.cos(alpha_rad),
                        np.interp(alpha, alpha_raw, cd_raw)
                    ]
                    cd = np.select(conditions, cd_choices, default=0)
                except TypeError:
                    cl = A_1*np.sin(2*alpha_rad)
                    cd = B_1*np.sin(alpha_rad)**2
                    print(f"Error: Polar data for {af_file} is incomplete. Using simple extrapolation.")

                if turbine_airfoil:
                    alpha = -np.flip(alpha)
                    cd = np.flip(cd)
                    cl = -np.flip(cl)
                extrapolated_polar_array = np.column_stack((alpha, cl, cd))
                return extrapolated_polar_array

            self.af_polars_corrected[idx_r] = polar_3d_correction(
                self.af_polars[idx_r]
                )
            self.af_polars_extrap[idx_r] = extrapolation_fun(
                self.af_polars_corrected[idx_r]
                )

    def method_ning(self):
        # implementation of [1]
        def alpha_fun(phi, i):
            return self.pitch[i] - phi

        def c_n_fun(phi, i):
            alpha = alpha_fun(phi, i)
            alpha_raw = np.radians(self.af_polars_extrap[i][:, 0])
            cl_raw = self.af_polars_extrap[i][:, 1]
            cd_raw = self.af_polars_extrap[i][:, 2]

            cl = np.interp(alpha, alpha_raw, cl_raw)
            cd = np.interp(alpha, alpha_raw, cd_raw)

            return cl*np.cos(phi) - cd*np.sin(phi)

        def c_t_fun(phi, i):
            alpha = alpha_fun(phi, i)
            alpha_raw = np.radians(self.af_polars_extrap[i][:, 0])
            cl_raw = self.af_polars_extrap[i][:, 1]
            cd_raw = self.af_polars_extrap[i][:, 2]

            cl = np.interp(alpha, alpha_raw, cl_raw)
            cd = np.interp(alpha, alpha_raw, cd_raw)

            return cl*np.sin(phi) + cd*np.cos(phi)

        def F_fun(phi, i):
            f_tip = self.N_b/2*(self.R - self.r[i])/(self.r[i]*abs(np.sin(phi)))
            F_tip = 2/np.pi*np.arccos(np.exp(-f_tip))
            f_hub = self.N_b/2*(self.r[i] - self.r[0])/(self.r[i]*abs(np.sin(phi)))
            F_hub = 2/np.pi*np.arccos(np.exp(-f_hub))
            return F_tip

        def kappa(phi, i):
            c_n = c_n_fun(phi, i)
            F = F_fun(phi, i)
            return c_n*self.sigma_prime[i]/(4*F*np.sin(phi)**2)

        def kappa_p(phi, i):
            c_t = c_t_fun(phi, i)
            F = F_fun(phi, i)
            return c_t*self.sigma_prime[i]/(4*F*np.sin(phi)*np.cos(phi))

        def a_fun(phi, i):
            k = kappa(phi, i)
            F = F_fun(phi, i)
            if phi < 0:
                k = -k
            if k >= -2/3:
                a = k/(1 - k)
            else:
                g_1 = F*(2*k - 1) + 10/9
                g_2 = F*(F - 2*k - 4/3)
                g_3 = 2*F*(1 - k) - 25/9
                if g_3 == 0:
                    a = 1/(2*g_2**0.5) - 1
                else:
                    a = (g_1 + g_2**0.5)/g_3
            return a

        def a_p_fun(phi, i):
            V_x = self.V
            k_p = kappa_p(phi, i)
            if V_x < 0:
                k_p = -k_p
            a_p = k_p/(1 + k_p)
            return a_p

        def residual_fun(phi, i):
            V_x = self.V
            V_y = self.Omega*self.r[i]
            k = kappa(phi, i)
            k_p = kappa_p(phi, i)
            if V_x == 0:
                res = np.sign(phi) - k
            elif abs(k) == 1:
                return 1
            elif V_y == 0:
                res = np.sign(V_x) + k_p
            elif k_p == -1:
                return 1
            else:
                a = a_fun(phi, i)
                a_p = a_p_fun(phi, i)
                res = np.sin(phi)/(1 + a) - V_x/V_y*np.cos(phi)/(1 - a_p)
            return res

        # --- solver storage ---
        self.phi_sol = np.zeros_like(self.r)
        self.alpha_sol = np.zeros_like(self.r)
        self.p_n = np.zeros_like(self.r)
        self.p_t = np.zeros_like(self.r)

        for i in range(1, len(self.r) - 1):
            V_x = self.V
            V_y = self.Omega*self.r[i]
            theta = self.pitch[i]

            q_1 = [self.epsilon, np.pi/2]
            q_2 = [-np.pi/2, -self.epsilon]
            q_3 = [np.pi/2, np.pi - self.epsilon]
            q_4 = [-np.pi + self.epsilon, -np.pi/2]

            if V_x == 0:
                if (V_y > 0) & (theta > 0):
                    quadrants = [q_1, q_2]
                elif (V_y > 0) & (theta < 0):
                    quadrants = [q_2, q_1]
                elif (V_y < 0) & (theta > 0):
                    quadrants = [q_3, q_4]
                elif (V_y < 0) & (theta < 0):
                    quadrants = [q_4, q_3]
            elif V_y == 0:
                if (V_x > 0) & (np.abs(theta) < np.pi/2):
                    quadrants = [q_1, q_3]
                elif (V_x < 0) & (np.abs(theta) < np.pi/2):
                    quadrants = [q_2, q_4]
                elif (V_x > 0) & (np.abs(theta) > np.pi/2):
                    quadrants = [q_3, q_1]
                elif (V_x < 0) & (np.abs(theta) > np.pi/2):
                    quadrants = [q_4, q_2]
            elif (V_x > 0) & (V_y > 0):
                quadrants = [q_1, q_2, q_3, q_4]
            elif (V_x < 0) & (V_y > 0):
                quadrants = [q_2, q_1, q_4, q_3]
            elif (V_x > 0) & (V_y < 0):
                quadrants = [q_3, q_4, q_1, q_2]
            elif (V_x < 0) & (V_y < 0):
                quadrants = [q_4, q_3, q_2, q_1]

            g = lambda x: residual_fun(x, i)
            phi = np.nan
            for q in quadrants:
                phi_1 = q[0]
                phi_2 = q[1]
                if g(phi_1)*g(phi_2) < 0:
                    phi = brentq(g, phi_1, phi_2)
                    break

            self.phi_sol[i] = phi
            self.alpha_sol[i] = alpha_fun(phi, i)
            k_sol = kappa(phi, i)
            k_p_sol = kappa_p(phi, i)
            if V_x == 0:
                u = np.sign(phi)*k_sol*V_y*np.tan(phi)
                v = 0
                W = ((V_x + u)**2 + (V_y - v)**2)**0.5
            elif V_y == 0:
                u = 0
                v = k_p_sol*np.abs(V_x)/np.tan(phi)
                W = ((V_x + u)**2 + (V_y - v)**2)**0.5
            else:
                a_sol = a_fun(phi, i)
                a_p_sol = a_p_fun(phi, i)
                W = ((V_x*(1 + a_sol))**2 + (V_y*(1 - a_p_sol))**2)**0.5
            q = 0.5*self.rho*W**2
            self.p_n[i] = c_n_fun(phi, i)*q*self.chord[i]
            self.p_t[i] = c_t_fun(phi, i)*q*self.chord[i]

        self.dTdr = self.p_n*self.N_b
        self.dDdr = self.p_t*self.N_b
