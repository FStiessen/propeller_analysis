import os
import subprocess
import numpy as np
from scipy.interpolate import interp1d


class bemt:
    def __init__(self, prop, V, Omega, mu, rho):
        self.V = V
        self.Omega = Omega
        self.mu = mu
        self.rho = rho
        self.Re = self.rho*(self.V**2 + (self.Omega*prop.r)**2)**0.5*prop.c/self.mu

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

        self.af_polars = []
        self.af_polars_extrap = []

        for af_file in airfoil_files:
            af_path = os.path.abspath(os.path.join(airfoil_dir, af_file))
            af_name = os.path.splitext(af_file)[0]

            # Extract spanwise index from filename (e.g., af_0.dat -> idx_r = 0)
            idx_r = int(af_name.split('_')[1])
            Re = int(round(self.Re[idx_r]))

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
                        self.af_polars.append(polar_array)

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
                alpha_raw = data_raw[:, 0]
                cl_raw = data_raw[:, 1]
                cd_raw = data_raw[:, 2]
                a_min = alpha_raw[0]
                a_max = alpha_raw[-1]
                cl_min = cl_raw[0]
                cl_max = cl_raw[-1]
                A_1 = 0.5
                A_2_min = (cl_min - A_1*np.sin(2*np.radians(a_min)))*np.sin(np.radians(a_min))/np.cos(np.radians(a_min))**2
                A_2_max = (cl_max - A_1*np.sin(2*np.radians(a_max)))*np.sin(np.radians(a_max))/np.cos(np.radians(a_max))**2
                cl = np.zeros_like(alpha)
                cl = (np.where(abs(alpha) >= 90, A_1*np.sin(2*alpha_rad),
                      np.where((-90 < alpha) & (alpha < a_min), A_1*np.sin(2*alpha_rad) + A_2_min*np.cos(alpha_rad)**2/np.sin(alpha_rad),
                      np.where((90 > alpha) & (alpha > a_max), A_1*np.sin(2*alpha_rad) + A_2_max*np.cos(alpha_rad)**2/np.sin(alpha_rad),
                      np.where((alpha >= a_min) & (alpha <= a_max), np.interp(alpha, alpha_raw, cl_raw), cl)))))

                cd_min = cd_raw[0]
                cd_max = cd_raw[-1]
                B_1 = 2
                B_2_min = (cd_min - B_1*np.sin(np.radians(a_min))**2)/np.cos(np.radians(a_min))
                B_2_max = (cd_max - B_1*np.sin(np.radians(a_max))**2)/np.cos(np.radians(a_max))
                cd = np.zeros_like(alpha)
                cd = (np.where(abs(alpha) >= 90, B_1*np.sin(alpha_rad)**2,
                      np.where((-90 < alpha) & (alpha < a_min), B_1*np.sin(alpha_rad)**2 + B_2_min*np.cos(alpha_rad),
                      np.where((90 > alpha) & (alpha > a_max), B_1*np.sin(alpha_rad)**2 + B_2_max*np.cos(alpha_rad),
                      np.where((alpha >= a_min) & (alpha <= a_max), np.interp(alpha, alpha_raw, cd_raw), cd)))))

                polar_array = np.column_stack((alpha, cl, cd))
                return polar_array

            self.af_polars_extrap.append(extrapolation_fun(polar_array))
