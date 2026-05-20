import numpy as np


class propeller_optimiser:
    def __init__(self, prop, analysis, mode, value):
        V = analysis.V
        R = prop.R
        beta = prop.pitch
        r = prop.r
        zeta = np.zeros_like(r)
        B = prop.N_b
        rho = analysis.rho
        Omega = analysis.Omega
        if mode == 'power':
            P_c = 2*value/(rho*V**3*np.pi*R**2)
        elif mode == 'thrust':
            T_c = 2*value/(rho*V**2*np.pi*R**2)
        else:
            raise ValueError('Wrong mode')

        lambda_t = V/(Omega*R)
        xi = r/R
        for idx_r in range(100):
            phi_t = np.atan(lambda_t*(1 + zeta/2))
            f = B/2*(1 - xi)/np.sin(phi_t)
            F = 2/np.pi*np.acos(np.exp(-f))
            phi = np.atan(np.tan(phi_t)/xi)
            x = Omega*r/V
            G = F*x*np.cos(phi)*np.sin(phi)
            alpha = beta - phi
            alpha = np.zeros_like(r)
            C_l = np.zeros_like(r)
            C_d = np.zeros_like(r)
            for idx_r in range(len(r)):
                data = analysis.af_polars_extrap[idx_r]
                alpha_raw = data[:, 0]
                C_l_raw = data[:, 1]
                condition = (alpha_raw >= 1) & (alpha_raw < 20)
                C_l_raw = np.where(condition, C_l_raw, np.nan)
                C_d_raw = data[:, 2]
                C_d_raw = np.where(condition, C_d_raw, np.nan)
                epsilon_raw = C_d_raw/C_l_raw
                idx_opt = np.nanargmin(epsilon_raw)
                alpha[idx_r] = np.radians(alpha_raw[idx_opt])
                C_l[idx_r] = C_l_raw[idx_opt]
                C_d[idx_r] = C_d_raw[idx_opt]
            Wc = 4*np.pi*lambda_t*G*V*R*zeta/(C_l*B)
            epsilon = C_d/C_l
            a = zeta/2*np.cos(phi)**2*(1 - epsilon*np.tan(phi))
            a_p = zeta/(2*x)*np.cos(phi)*np.sin(phi)*(1 + epsilon*np.tan(phi))
            W = V*(1 + a)/np.sin(phi)
            c = Wc/W
            beta = alpha + phi
            print('W:', W)
            print('alpha_opt:', np.rad2deg(alpha))
            print('beta:', np.rad2deg(beta))
            print('c:', c)
            I_p_1 = 4*xi*G*(1 - epsilon*np.tan(phi))
            I_p_2 = lambda_t*I_p_1/(2*xi)*(1 + epsilon/np.tan(phi))*np.sin(phi)*np.cos(phi)
            J_p_1 = 4*xi*G*(1 + epsilon/np.tan(phi))
            J_p_2 = J_p_1/2*(1 - epsilon*np.tan(phi))*np.cos(phi)**2
            I_1 = np.trapezoid(I_p_1, xi)
            I_2 = np.trapezoid(I_p_2, xi)
            J_1 = np.trapezoid(J_p_1, xi)
            J_2 = np.trapezoid(J_p_2, xi)
            if mode == 'power':
                zeta = -J_1/(2*J_2) + ((J_1/(2*J_2))**2 + P_c/J_2)**0.5
                T_c = I_1*zeta - I_2*zeta**2
                T = T_c*rho*V**2*np.pi*R**2/2
                print('Thrust:', T)
            else:
                zeta = I_1/(2*I_2) - ((I_1/(2*I_2))**2 - T_c/I_2)**0.5
                P_c = J_1*zeta + J_2*zeta**2
                P = P_c*rho*V**3*np.pi*R**2/2
        c[-1] = 1e-3
        prop.c = c
        prop.pitch = beta
