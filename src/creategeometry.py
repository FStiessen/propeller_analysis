# Required libraries
import openvsp as vsp
import numpy as np
import shutil


class create_geometry:
    def __init__(
            self,
            propeller,
            dynamics,
            N_U,
            N_W,
            left_handed=False,
            round_te=False,
            approximate_curves=True
            ):
        self.left_handed = left_handed
        self.prop = propeller
        self.vsp_default = "lib/propeller_default.vsp3"
        self.vsp_new = "output/geometry.vsp3"
        self.curve_type = 1
        self.approximate_curves = approximate_curves
        self.round_te = round_te
        self.axial_offset = dynamics.z_blade/self.prop.R
        self.sweep = -np.rad2deg(np.atan(dynamics.y_blade/dynamics.x_blade))
        self.N_U = N_U
        self.N_W = N_W
        shutil.copyfile(self.vsp_default, self.vsp_new)
        vsp.ReadVSPFile(self.vsp_new)
        self.create_blade_geometry()

    def create_blade_geometry(self):
        prop_id = vsp.FindGeom("propeller", 0)
        vsp.SetParmVal(
            prop_id, "RadiusFrac", "XSec_0",
            self.prop.r[0]/self.prop.R
            )
        if self.left_handed:
            vsp.SetParmVal(prop_id, "ReverseFlag", "Design", 1)
        vsp.SetParmVal(prop_id, "Diameter", "Design", 2*self.prop.R)
        vsp.SetParmVal(prop_id, "UseBeta34Flag", "Design", 0)
        vsp.SetParmVal(prop_id, "CylindricalSectionsFlag", "Design", 1)
        vsp.SetParmVal(prop_id, "Feather", "Design", 0)
        vsp.SetParmVal(prop_id, "ConstructXoC", "Design", 0)
        vsp.SetParmVal(prop_id, "FeatherAxisXoC", "Design", 0)
        vsp.Update()
        # vsp.SetParmVal(prop_id, "CapUMinOption", "EndCap", root_cap)
        # vsp.SetParmVal(prop_id, "CapUMaxOption", "EndCap", end_cap)
        # vsp.SetPCurve(geom_id, pcurveid, tvec, valvec, newtype)
        # geom_id: ID of geometry object
        # pcurveid: 0=chord, 1=twist, 2=rake, 3=skew, 4=sweep, 5=thickness,
        # 6=Cli, 7=axial, 8=tangential
        # newtype: 0=linear interpolation, 1=Spline PCHIP, 2=Cubic Bezier
        vsp.SetPCurve(
            prop_id, 0, self.prop.r/self.prop.R,
            self.prop.c/self.prop.R, self.curve_type
            )
        vsp.SetPCurve(
            prop_id, 1, self.prop.r/self.prop.R,
            np.rad2deg(self.prop.pitch), self.curve_type
            )
        Cx = self.prop.centroids[:, 0]*self.prop.c/self.prop.R
        Cy = self.prop.centroids[:, 1]*self.prop.c/self.prop.R
        tangential = -(
            Cx*np.cos(self.prop.pitch) + Cy*np.sin(self.prop.pitch)
            )
        axial = (
            Cy*np.cos(self.prop.pitch) - Cx*np.sin(self.prop.pitch) -
            self.axial_offset
            )
        vsp.SetPCurve(
            prop_id, 7, self.prop.r/self.prop.R, axial, self.curve_type
            )
        vsp.SetPCurve(
            prop_id, 8, self.prop.r/self.prop.R, tangential, self.curve_type
            )
        vsp.SetPCurve(
            prop_id, 4, self.prop.r/self.prop.R, self.sweep, self.curve_type
            )
        vsp.Update()
        # Approximate final blade characteritics curve with cubic splines
        if self.approximate_curves:
            vsp.ApproximateAllPropellerPCurves(prop_id)
        for idx_r in range(len(self.prop.r)):
            print(idx_r)
            vsp.Update()
            if idx_r not in [0, len(self.prop.r) - 1]:
                vsp.InsertXSec(prop_id, idx_r - 1, vsp.XS_FILE_AIRFOIL)
                xsec_surf_id = vsp.GetXSecSurf(prop_id, idx_r)
                vsp.SetParmVal(
                    prop_id, "RadiusFrac", "XSec_" + str(idx_r),
                    self.prop.r[idx_r]/self.prop.R
                    )
            else:
                xsec_surf_id = vsp.GetXSecSurf(prop_id, idx_r)
                vsp.ChangeXSecShape(xsec_surf_id, idx_r, vsp.XS_FILE_AIRFOIL)
            vsp.Update()
            filename = "output/airfoils/af_" + str(idx_r) + ".dat"
            print(filename)
            xsec = vsp.GetXSec(xsec_surf_id, idx_r)
            vsp.ReadFileAirfoil(xsec, filename)
            if self.round_te:
                vsp.SetParmVal(prop_id, "TE_Cap_Type", "Cap_" + str(idx_r), 2)
            vsp.Update()

        vsp.ResetPropellerThicknessCurve(prop_id)
        vsp.SetParmVal(prop_id, "Tess_U", "Shape", self.N_U)
        vsp.SetParmVal(prop_id, "Tess_W", "Shape", self.N_W)
        vsp.SetParmVal(prop_id, "NumBlade", "Design", self.prop.N_b)
        vsp.Update()
        vsp.WriteVSPFile(self.vsp_new)
