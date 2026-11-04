"""JFET builder for the course (DEVSIM, 2-D drift-diffusion).

n-channel JFET, half of a symmetric device (the bottom edge is the symmetry plane):

    source(n+) |  p+ gate on top of an n channel of half-thickness a  | drain(n+)

        y=0  ====S====        ======G======        ====D====
             n+ |gap| [ p+ gate, depth tg ] |gap| n+
                |   |   n channel (ND)      |   |
        y=tg+a  - - - - - - symmetry plane - - - - - - - - -

Usage:   dev = Jfet(L_um=5, a_um=0.6, ND=1e16);  dev.bias(gate=-0.5, drain=1.0)
Currents are per um of width for this HALF device (the full device carries twice as much).
"""
import math
import numpy as np
from common import ds, UM, Q, CreateSolution, CreateNodeModel, SetSiliconParameters, \
    CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact, CreateSiliconDriftDiffusion, \
    CreateSiliconDriftDiffusionAtContact, GetContactBiasName, Mosfet

_count = [0]


class Jfet:
    def __init__(self, L_um=5.0, a_um=0.6, ND=1e16, NG=1e19, NSD=1e19, tg_um=0.2,
                 gap_um=0.1, quiet=True):
        _count[0] += 1
        self.D = f"jfet{_count[0]}"
        self.L, self.a, self.ND, self.NG, self.NSD = L_um * UM, a_um * UM, ND, NG, NSD
        self.tg, self.gap = tg_um * UM, gap_um * UM
        self.applied = {"source": 0.0, "drain": 0.0, "gate": 0.0}
        if quiet:
            Mosfet._mute(self)
        self._mesh()
        self._physics()
        Mosfet._unmute(self)

    def _mesh(self):
        D, L, tg, a, gap = self.D, self.L, self.tg, self.a, self.gap
        SD = 0.5 * UM
        self.xg0 = SD + gap; self.xg1 = self.xg0 + L
        self.xd = self.xg1 + gap; self.xend = self.xd + SD
        self.H = H = tg + a
        m = D + "_mesh"; air = 1e-7
        ds.create_2d_mesh(mesh=m)
        for pos, ps in [(-air, 1.0), (0, 0.05 * UM), (SD, 0.01 * UM), (self.xg0, 0.01 * UM),
                        (self.xg0 + L / 2, 0.1 * UM), (self.xg1, 0.01 * UM), (self.xd, 0.01 * UM),
                        (self.xend, 0.05 * UM), (self.xend + air, 1.0)]:
            ds.add_2d_mesh_line(mesh=m, dir="x", pos=pos, ps=ps)
        for pos, ps in [(-air, 1.0), (0, 0.02 * UM), (tg, 0.005 * UM), (H, 0.015 * UM), (H + air, 1.0)]:
            ds.add_2d_mesh_line(mesh=m, dir="y", pos=pos, ps=ps)
        ds.add_2d_region(mesh=m, material="Air", region="air")
        ds.add_2d_region(mesh=m, material="Si", region="si", xl=0, xh=self.xend, yl=0, yh=H)
        b = 1e-10
        for name, xl, xh in (("source", 0, SD), ("gate", self.xg0, self.xg1), ("drain", self.xd, self.xend)):
            ds.add_2d_contact(mesh=m, name=name, material="metal", region="si",
                              xl=xl, xh=xh, yl=0, yh=0, bloat=b)
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device=D)

    def _physics(self):
        D = self.D
        SetSiliconParameters(D, "si", 300)
        CreateNodeModel(D, "si", "Donors",
                        f"{self.ND} + {self.NSD}*(step({0.5*UM}-x) + step(x-{self.xd}))")
        CreateNodeModel(D, "si", "Acceptors",
                        f"{self.NG}*step(x-{self.xg0})*step({self.xg1}-x)*step({self.tg}-y)")
        CreateNodeModel(D, "si", "NetDoping", "Donors-Acceptors")
        for c in self.applied:
            ds.set_parameter(device=D, name=GetContactBiasName(c), value=0.0)
        CreateSolution(D, "si", "Potential"); CreateSiliconPotentialOnly(D, "si")
        CreateNodeModel(D, "si", "Pinit", "V_t*asinh(NetDoping/(2*n_i))")
        ds.set_node_values(device=D, region="si", name="Potential", init_from="Pinit")
        for c in self.applied:
            CreateSiliconPotentialOnlyContact(D, "si", c)
        self._solve(1.0)
        CreateSolution(D, "si", "Electrons"); CreateSolution(D, "si", "Holes")
        ds.set_node_values(device=D, region="si", name="Electrons", init_from="IntrinsicElectrons")
        ds.set_node_values(device=D, region="si", name="Holes", init_from="IntrinsicHoles")
        CreateSiliconDriftDiffusion(D, "si")
        for c in self.applied:
            CreateSiliconDriftDiffusionAtContact(D, "si", c)
        self._solve()

    def _solve(self, abs_err=1e10):
        ds.solve(type="dc", absolute_error=abs_err, relative_error=1e-5, maximum_iterations=60)

    def bias(self, step=0.05, **volts):
        Mosfet._mute(self)
        try:
            for c, target in volts.items():
                v0 = self.applied[c]
                n = max(1, int(math.ceil(abs(target - v0) / step - 1e-9)))
                for k in range(1, n + 1):
                    v = v0 + (target - v0) * k / n
                    ds.set_parameter(device=self.D, name=GetContactBiasName(c), value=v)
                    self._solve()
                    self.applied[c] = v
        finally:
            Mosfet._unmute(self)

    def current(self, contact="drain"):
        i = sum(ds.get_contact_current(device=self.D, contact=contact, equation=e)
                for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))
        return abs(i) * UM

    def node(self, name):
        return np.array(ds.get_node_model_values(device=self.D, region="si", name=name))

    def opening(self, x_um=None):
        """Undepleted channel thickness (um) at x (default: gate centre): where n > ND/2."""
        x, y, n = self.node("x"), self.node("y"), self.node("Electrons")
        xm = 0.5 * (self.xg0 + self.xg1) if x_um is None else x_um * UM
        xs = np.unique(x); x0 = xs[np.argmin(abs(xs - xm))]
        k = np.where(np.isclose(x, x0) & (y >= self.tg))[0]
        k = k[np.argsort(y[k])]
        yy, nn = y[k], n[k] / self.ND
        above = np.where(nn >= 0.5)[0]
        if len(above) == 0:
            return 0.0
        j = above[0]
        if j == 0:
            return (self.H - yy[0]) / UM
        yc = np.interp(0.5, [nn[j - 1], nn[j]], [yy[j - 1], yy[j]])
        return (self.H - yc) / UM

    def theory(self):
        """Streetman 6.1 long-channel JFET formulas for this (half) device."""
        VT = ds.get_parameter(device=self.D, region="si", name="V_t")
        ni = ds.get_parameter(device=self.D, region="si", name="n_i")
        eps = ds.get_parameter(device=self.D, region="si", name="Permittivity")
        mu = ds.get_parameter(device=self.D, region="si", name="mu_n")
        Vbi = VT * math.log(self.NG * self.ND / ni ** 2)
        Vp = Q * self.ND * self.a ** 2 / (2 * eps)            # pinch-off voltage (internal)
        G0 = Q * mu * self.ND * self.a / self.L * UM           # S per um width, undepleted channel
        return {"VT": VT, "eps": eps, "mu": mu, "Vbi": Vbi, "Vp": Vp, "G0": G0,
                "Vg_off": Vbi - Vp, "a_um": self.a / UM, "ND": self.ND}


def id_formula(th, vg, vd):
    """Id (A/um, half device) from the gradual-channel JFET formula, flat after pinch-off."""
    Vbi, Vp, G0 = th["Vbi"], th["Vp"], th["G0"]
    if Vbi - vg >= Vp:
        return 0.0
    vd = min(vd, Vp - Vbi + vg)                                  # saturation: Vd_sat = Vp - Vbi + Vg
    return G0 * (vd - 2 / 3 / math.sqrt(Vp) * ((Vbi - vg + vd) ** 1.5 - (Vbi - vg) ** 1.5))


def opening_formula(th, vg):
    h = math.sqrt(2 * th["eps"] * max(th["Vbi"] - vg, 0) / (Q * th["ND"])) / UM
    return max(th["a_um"] - h, 0.0)
