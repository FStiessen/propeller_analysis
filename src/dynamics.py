import numpy as np
from scipy.integrate import cumulative_trapezoid


class dynamics:
    def __init__(self, propeller, analysis, rho_m):
        self.p_n = analysis.p_n
        self.p_t = analysis.p_t
        self.r = propeller.r
        self.Omega = analysis.Omega
        self.areas = propeller.areas
        self.m = propeller.areas*rho_m

        self.balance_blade()

    def balance_blade(self):
        f = self.m*self.Omega**2*self.r
        dphidr = self.p_t/(f*self.r)
        dzdr = self.p_n/f
        phi_blade = cumulative_trapezoid(dphidr, self.r, initial=0)
        self.z_blade = cumulative_trapezoid(dzdr, self.r, initial=0)
        self.tension = -cumulative_trapezoid(f[::-1], self.r[::-1], initial=0)[::-1]
        self.normal_stress = self.tension/self.areas
        self.shear_n = -cumulative_trapezoid(self.p_n[::-1], self.r[::-1], initial=0)[::-1]
        self.shear_t = -cumulative_trapezoid(self.p_t[::-1], self.r[::-1], initial=0)[::-1]
        self.shear = (self.shear_n**2 + self.shear_t**2)**0.5
        self.shear_stress = self.shear/self.areas

        self.x_blade = self.r*np.cos(phi_blade)
        self.y_blade = -self.r*np.sin(phi_blade)
