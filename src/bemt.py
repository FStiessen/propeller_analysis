import os
import subprocess
import numpy as np
from scipy.optimize import brentq


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
        self.Re = self.rho*(self.V**2 + (self.Omega*prop.r)**2)**0.5*prop.c/self.mu
        self.M = (self.V**2 + (self.Omega*prop.r)**2)**0.5/c_sound
        self.lambda_r = self.Omega*prop.r/self.V
        self.sigma_prime = prop.N_b*prop.c/(2*np.pi*prop.r)
        self.chord = prop.c

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

    def generate_polars(self, alpha_min, alpha_max, dalpha, N_crit):
        """Generate lift and drag polars for all airfoils in output/airfoils using xfoil."""
        airfoil_dir = os.path.join('output', 'airfoils')

        if not os.path.exists(airfoil_dir):
            print(f"Directory {airfoil_dir} not found")
            return

        airfoil_files = sorted([f for f in os.listdir(airfoil_dir) if f.endswith('.dat')])

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
        self.af_polars_extrap = [None]*len(self.r)

        for af_file in airfoil_files:
            af_path = os.path.abspath(os.path.join(airfoil_dir, af_file))
            af_name = os.path.splitext(af_file)[0]

            # Extract spanwise index from filename (e.g., af_0.dat -> idx_r = 0)
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
                stdout, stderr = process.communicate(input=input_script, timeout=60)

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

            def extrapolation_fun(data_raw):
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
                    A_2_min = (cl_min - A_1*np.sin(2*np.radians(a_min)))*np.sin(np.radians(a_min))/np.cos(np.radians(a_min))**2
                    A_2_max = (cl_max - A_1*np.sin(2*np.radians(a_max)))*np.sin(np.radians(a_max))/np.cos(np.radians(a_max))**2

                    cd_min = cd_raw[0]
                    cd_max = cd_raw[-1]
                    B_2_min = (cd_min - B_1*np.sin(np.radians(a_min))**2)/np.cos(np.radians(a_min))
                    B_2_max = (cd_max - B_1*np.sin(np.radians(a_max))**2)/np.cos(np.radians(a_max))

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
                        A_1*np.sin(2*alpha_rad) + A_2_min*np.cos(alpha_rad)**2/np.sin(alpha_rad),
                        A_1*np.sin(2*alpha_rad) + A_2_max*np.cos(alpha_rad)**2/np.sin(alpha_rad),
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

                #cl = np.flip(cl)
                #cl = np.flip(cd)
                extrapolated_polar_array = np.column_stack((alpha, cl, cd))
                return extrapolated_polar_array

            self.af_polars_extrap[idx_r] = extrapolation_fun(self.af_polars[idx_r])

    def method_ning(self):
        self.epsilon = 1e-6

        def alpha_fun(phi, i):
            return (
                self.pitch[i] - phi
            )

        def c_n(phi, i):
            alpha = alpha_fun(phi, i)
            alpha_raw = np.radians(self.af_polars_extrap[i][:, 0])
            cl_raw = self.af_polars_extrap[i][:, 1]
            cd_raw = self.af_polars_extrap[i][:, 2]
            cl = np.interp(alpha, alpha_raw, cl_raw)
            cd = np.interp(alpha, alpha_raw, cd_raw)
            return cl*np.cos(phi) - cd*np.sin(phi)

        def c_t(phi, i):
            alpha = alpha_fun(phi, i)
            alpha_raw = np.radians(self.af_polars_extrap[i][:, 0])
            cl_raw = self.af_polars_extrap[i][:, 1]
            cd_raw = self.af_polars_extrap[i][:, 2]
            cl = np.interp(alpha, alpha_raw, cl_raw)
            cd = np.interp(alpha, alpha_raw, cd_raw)
            return cl*np.sin(phi) + cd*np.cos(phi)

        def F(phi, i):
            sin_phi = max(abs(np.sin(phi)), self.epsilon)
            f_tip = self.N_b/2*(self.R - self.r[i])/(self.r[i]*sin_phi)
            F_tip = 2/np.pi * np.arccos(np.exp(-f_tip))
            f_hub = self.N_b/2*(self.r[i] - self.r[0])/(self.r[i]*sin_phi)
            F_hub = 2/np.pi * np.arccos(np.exp(-f_hub))
            return np.clip(F_tip*F_hub, self.epsilon, 1.0)

        def kappa(phi, i):
            return (
                self.sigma_prime[i]*c_n(phi, i)/(4*F(phi, i)*np.sin(phi)**2)
            )

        def kappa_prime(phi, i):
            return (
                self.sigma_prime[i]*c_t(phi, i)/(4*F(phi, i)*np.sin(phi)*np.cos(phi))
            )

        def a(phi, i):
            k = kappa(phi, i)
            return k/(1 - k)

        def a_prime(phi, i):
            return (
                kappa_prime(phi, i)/(1 + kappa_prime(phi, i))
            )

        def f(phi, i):
            a_val = a(phi, i)
            ap_val = a_prime(phi, i)

            return (
                np.sin(phi)/(1 + a_val)
                - np.cos(phi)/(self.lambda_r[i]*(1 - ap_val))
            )

        def c_thrust(phi, i):
            return (
                ((1 + a(phi, i))/np.sin(phi))**2
                * c_n(phi, i)
                * self.sigma_prime[i]
            )

        def c_drag(phi, i):
            return (
                ((1 + a(phi, i))/np.sin(phi))**2
                * c_t(phi, i)
                * self.sigma_prime[i]
            )

        self.phi_sol = np.zeros_like(self.pitch)
        self.alpha_sol = np.zeros_like(self.pitch)
        self.c_thrust_sol = np.zeros_like(self.pitch)
        self.c_drag_sol = np.zeros_like(self.pitch)
        self.F_g = np.zeros_like(self.pitch)
        for idx in range(1, len(self.pitch) - 1):
            g = lambda x: f(x, idx)
            phi_sol_i = brentq(g, self.epsilon, np.pi/2 - self.epsilon)
            self.phi_sol[idx] = phi_sol_i
            self.alpha_sol[idx] = alpha_fun(phi_sol_i, idx)
            self.c_thrust_sol[idx] = c_thrust(phi_sol_i, idx)
            self.c_drag_sol[idx] = c_drag(phi_sol_i, idx)
            self.F_g[idx] = 0.25*self.sigma_prime[idx]*c_n(phi_sol_i, idx)/np.sin(phi_sol_i)**2

        self.dTdr = self.c_thrust_sol*self.r*self.rho*self.V**2*np.pi
        self.dDdr = self.c_drag_sol*self.r*self.rho*self.V**2*np.pi
        self.p_n = self.dTdr/self.N_b
        self.p_t = self.dDdr/self.N_b

    def method_ning_2(self):

        def alpha_fun(phi, i):
            return self.pitch[i] - phi

        # --- Aerodynamics (ONLY used for φ solve) ---
        def c_n(phi, i):
            alpha = alpha_fun(phi, i)
            alpha_raw = np.radians(self.af_polars_extrap[i][:, 0])
            cl_raw = self.af_polars_extrap[i][:, 1]
            cd_raw = self.af_polars_extrap[i][:, 2]

            cl = np.interp(alpha, alpha_raw, cl_raw)
            cd = np.interp(alpha, alpha_raw, cd_raw)

            return cl*np.cos(phi) - cd*np.sin(phi)

        def c_t(phi, i):
            alpha = alpha_fun(phi, i)
            alpha_raw = np.radians(self.af_polars_extrap[i][:, 0])
            cl_raw = self.af_polars_extrap[i][:, 1]
            cd_raw = self.af_polars_extrap[i][:, 2]

            cl = np.interp(alpha, alpha_raw, cl_raw)
            cd = np.interp(alpha, alpha_raw, cd_raw)

            return cl*np.sin(phi) + cd*np.cos(phi)

        # --- Prandtl factor ---
        def F(phi, i):
            f_tip = self.N_b/2*(self.R - self.r[i])/(self.r[i]*abs(np.sin(phi)))
            return (2/np.pi)*np.arccos(np.exp(-f_tip))

        # --- κ definitions ---
        def kappa(phi, i):
            return self.sigma_prime[i]*c_n(phi,i)/(4*F(phi,i)*np.sin(phi)**2)

        def kappa_p(phi, i):
            return self.sigma_prime[i]*c_t(phi,i)/(4*F(phi,i)*np.sin(phi)*np.cos(phi))

        # --- induction ---
        def a(phi, i):
            return kappa(phi,i)/(1 + kappa(phi,i))

        def a_p(phi, i):
            return kappa_p(phi,i)/(1 + kappa_p(phi,i))

        # --- Ning residual (propeller form) ---
        def f(phi, i):
            return (
                np.sin(phi)/(1 + a(phi,i))
                - np.cos(phi)*(1 - kappa_p(phi,i))/self.lambda_r[i]
            )

        # --- solver storage ---
        self.phi_sol = np.zeros_like(self.pitch)
        self.alpha_sol = np.zeros_like(self.pitch)

        self.dTdr = np.zeros_like(self.pitch)
        self.dQdr = np.zeros_like(self.pitch)

        epsilon = 1e-6

        for i in range(1, len(self.pitch)-1):

            # --- φ solve ---
            phi = brentq(lambda x: f(x,i),
                        epsilon,
                        np.pi/2 - epsilon)

            self.phi_sol[i] = phi
            self.alpha_sol[i] = alpha_fun(phi,i)

            # --- induced velocities ---
            a_i = a(phi,i)
            ap_i = a_p(phi,i)
            F_i = F(phi,i)

            r = self.r[i]

            # --- MOMENTUM thrust (stable, no blow-up) ---
            self.dTdr[i] = (
                4*np.pi*r*self.rho*self.V**2
                * a_i*(1 + a_i)
                * F_i
            )

            # --- MOMENTUM torque ---
            self.dQdr[i] = (
                4*np.pi*r**3*self.rho*self.V*self.lambda_r[i]*self.V
                * ap_i*(1 - ap_i)
                * F_i
            )
        self.p_n = self.dTdr/self.N_b
        self.p_t = self.dDdr/self.N_b
