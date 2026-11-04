"""Shared helpers for the MOSFET TCAD course (DEVSIM, 2-D drift-diffusion).

Every lesson builds a MOSFET with   dev = Mosfet(L_um=1.0, tox_nm=10, NA=1e17)
and then moves its terminals with   dev.bias(gate=1.5, drain=0.5)

Units inside DEVSIM are cm, V, A. Currents are reported per um of width.
"""
import math, os, sys
try:                                       # Windows / Apple-silicon Mac:
    import devsim_openblas                 #   pip install devsim-openblas
    devsim_openblas.configure()
except ImportError:                        # Linux: export DEVSIM_MATH_LIBS=<libopenblas.so>
    if "DEVSIM_MATH_LIBS" not in os.environ and os.path.exists("/usr/lib/x86_64-linux-gnu/libopenblas.so.0"):
        os.environ["DEVSIM_MATH_LIBS"] = "/usr/lib/x86_64-linux-gnu/libopenblas.so.0"
import numpy as np
import devsim as ds
from devsim.python_packages.simple_physics import (
    SetSiliconParameters, SetOxideParameters, CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact, CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact, CreateOxidePotentialOnly, CreateOxideContact,
    CreateSiliconOxideInterface, GetContactBiasName)
from devsim.python_packages.model_create import CreateSolution, CreateNodeModel

UM = 1e-4                 # 1 um in cm
Q = 1.602176634e-19
EG = 1.12                 # Si band gap (eV) used for drawing bands
_count = [0]


class Mosfet:
    """n-channel MOSFET:  n+ source | gate on SiO2 over p-substrate | n+ drain.

    L_um      channel (gate) length in um
    tox_nm    gate-oxide thickness in nm
    NA        substrate acceptor doping (cm^-3)
    NSD       source/drain donor doping (cm^-3)
    xj_um     source/drain junction depth (um)
    vfb_shift extra flat-band shift in V, used to model a gate work-function
              difference or a sheet of fixed oxide charge:  dVfb = phi_ms - Qf/Cox
              (0 = ideal mid-gap metal gate, no oxide charge)
    """

    def __init__(self, L_um=1.0, tox_nm=10.0, NA=1e17, NSD=1e20, xj_um=0.1,
                 vfb_shift=0.0, quiet=True):
        _count[0] += 1
        self.D = D = f"mos{_count[0]}"
        self.L, self.tox, self.NA, self.NSD = L_um * UM, tox_nm * 1e-7, NA, NSD
        self.xj, self.vfb_shift = xj_um * UM, vfb_shift
        self.applied = {"source": 0.0, "drain": 0.0, "gate": 0.0, "body": 0.0}
        if quiet:
            self._mute()
        self._mesh()
        self._physics()
        self._unmute()

    # ----------------------------------------------------------- set-up ----
    def _mute(self):
        """Hide DEVSIM's solver log while it works."""
        sys.stdout.flush()
        if os.name == "nt":       # Windows console: swap Python's stdout only (fd tricks crash there)
            self._saved = sys.stdout; sys.stdout = open(os.devnull, "w", encoding="utf-8")
            return
        self._fd = os.dup(1); self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)

    def _unmute(self):
        if hasattr(self, "_saved"):
            sys.stdout.close(); sys.stdout = self._saved; del self._saved
        if hasattr(self, "_fd"):
            sys.stdout.flush(); os.dup2(self._fd, 1)
            os.close(self._null); os.close(self._fd); del self._fd

    def _mesh(self):
        D, L, tox, xj = self.D, self.L, self.tox, self.xj
        SD = 0.5 * UM                                   # source/drain length
        self.xg0, self.xg1 = SD, SD + L                 # gate edges
        self.xend = 2 * SD + L
        ysub = max(1.0 * UM, 3 * L)
        fine = min(0.01 * UM, L / 40)
        m = D + "_mesh"; air = 1e-7
        ds.create_2d_mesh(mesh=m)
        xs = [(-air, 1.0), (0, 0.05 * UM), (SD - 0.1 * UM, 0.02 * UM), (self.xg0, fine),
              (self.xg0 + L / 2, max(fine, L / 20)), (self.xg1, fine),
              (self.xg1 + 0.1 * UM, 0.02 * UM), (self.xend, 0.05 * UM), (self.xend + air, 1.0)]
        for pos, ps in xs:
            ds.add_2d_mesh_line(mesh=m, dir="x", pos=pos, ps=ps)
        for pos, ps in [(-tox - air, 1.0), (-tox, tox / 5), (0.0, 1e-7), (xj, 0.01 * UM),
                        (ysub, 0.1 * UM), (ysub + air, 1.0)]:
            ds.add_2d_mesh_line(mesh=m, dir="y", pos=pos, ps=ps)
        ds.add_2d_region(mesh=m, material="Air", region="air")
        ds.add_2d_region(mesh=m, material="Si", region="si", xl=0, xh=self.xend, yl=0, yh=ysub)
        ds.add_2d_region(mesh=m, material="Ox", region="ox", xl=self.xg0, xh=self.xg1, yl=-tox, yh=0)
        b = 1e-10
        ds.add_2d_interface(mesh=m, name="si_ox", region0="si", region1="ox",
                            xl=self.xg0, xh=self.xg1, yl=0, yh=0, bloat=b)
        ds.add_2d_contact(mesh=m, name="source", material="metal", region="si",
                          xl=0, xh=SD - 0.1 * UM, yl=0, yh=0, bloat=b)
        ds.add_2d_contact(mesh=m, name="drain", material="metal", region="si",
                          xl=self.xg1 + 0.1 * UM, xh=self.xend, yl=0, yh=0, bloat=b)
        ds.add_2d_contact(mesh=m, name="body", material="metal", region="si",
                          xl=0, xh=self.xend, yl=ysub, yh=ysub, bloat=b)
        ds.add_2d_contact(mesh=m, name="gate", material="metal", region="ox",
                          xl=self.xg0, xh=self.xg1, yl=-tox, yh=-tox, bloat=b)
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device=D)

    def _physics(self):
        D = self.D
        SetSiliconParameters(D, "si", 300); SetOxideParameters(D, "ox", 300)
        sd = f"(step({self.xg0}-x) + step(x-{self.xg1}))*step({self.xj}-y)"
        CreateNodeModel(D, "si", "Donors", f"{self.NSD}*{sd}")
        CreateNodeModel(D, "si", "Acceptors", f"{self.NA}")
        CreateNodeModel(D, "si", "NetDoping", "Donors-Acceptors")
        for c in self.applied:
            ds.set_parameter(device=D, name=GetContactBiasName(c), value=0.0)
        ds.set_parameter(device=D, name=GetContactBiasName("gate"), value=-self.vfb_shift)
        CreateSolution(D, "si", "Potential"); CreateSiliconPotentialOnly(D, "si")
        # good starting guess: charge-neutral potential from the doping (helps large meshes converge)
        CreateNodeModel(D, "si", "Pinit", "V_t*asinh(NetDoping/(2*n_i))")
        ds.set_node_values(device=D, region="si", name="Potential", init_from="Pinit")
        CreateSolution(D, "ox", "Potential"); CreateOxidePotentialOnly(D, "ox", "log_damp")
        for c in ("source", "drain", "body"):
            CreateSiliconPotentialOnlyContact(D, "si", c)
        CreateOxideContact(D, "ox", "gate")
        CreateSiliconOxideInterface(D, "si_ox")
        self._solve(1.0)
        CreateSolution(D, "si", "Electrons"); CreateSolution(D, "si", "Holes")
        ds.set_node_values(device=D, region="si", name="Electrons", init_from="IntrinsicElectrons")
        ds.set_node_values(device=D, region="si", name="Holes", init_from="IntrinsicHoles")
        CreateSiliconDriftDiffusion(D, "si")
        for c in ("source", "drain", "body"):
            CreateSiliconDriftDiffusionAtContact(D, "si", c)
        self._solve()

    def _solve(self, abs_err=1e10):
        ds.solve(type="dc", absolute_error=abs_err, relative_error=1e-5, maximum_iterations=60)

    # ------------------------------------------------------------ biasing --
    def bias(self, step=0.05, **volts):
        """Ramp terminals to the given voltages, e.g. dev.bias(gate=2, drain=0.5)."""
        self._mute()
        try:
            for c, target in volts.items():
                name = GetContactBiasName(c)
                off = -self.vfb_shift if c == "gate" else 0.0
                v0 = self.applied[c]
                n = max(1, int(math.ceil(abs(target - v0) / step - 1e-9)))
                for k in range(1, n + 1):
                    v = v0 + (target - v0) * k / n
                    ds.set_parameter(device=self.D, name=name, value=v + off)
                    self._solve()
                    self.applied[c] = v
        finally:
            self._unmute()

    # --------------------------------------------------------- read-outs ---
    def current(self, contact="drain"):
        """Terminal current in A per um of gate width (positive into drain)."""
        i = sum(ds.get_contact_current(device=self.D, contact=contact, equation=e)
                for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))
        return abs(i) * UM

    def node(self, name, region="si"):
        return np.array(ds.get_node_model_values(device=self.D, region=region, name=name))

    def surface(self, depth_nm=0.0):
        """Profiles along the channel direction at a given depth below the oxide.
        Returns dict x_um, psi, n, p, Ec (eV, Fermi level of the source = 0)."""
        x, y = self.node("x"), self.node("y")
        ys = np.unique(y)
        y0 = ys[np.argmin(abs(ys - depth_nm * 1e-7))]
        k = np.where(np.isclose(y, y0))[0]
        k = k[np.argsort(x[k])]
        psi = self.node("Potential")[k]
        return {"x_um": x[k] / UM, "psi": psi, "n": self.node("Electrons")[k],
                "p": self.node("Holes")[k], "Ec": -psi + EG / 2}

    def field(self, name="Electrons"):
        """(x_um, y_um, values) of a node quantity over the silicon, for maps."""
        return self.node("x") / UM, self.node("y") / UM, self.node(name)

    # ------------------------------------------------------- textbook ------
    def theory(self):
        """Long-channel textbook numbers for this device (Streetman ch.6)."""
        VT = ds.get_parameter(device=self.D, region="si", name="V_t")
        ni = ds.get_parameter(device=self.D, region="si", name="n_i")
        eps_si = ds.get_parameter(device=self.D, region="si", name="Permittivity")
        eps_ox = ds.get_parameter(device=self.D, region="ox", name="Permittivity")
        mu = ds.get_parameter(device=self.D, region="si", name="mu_n")
        Cox = eps_ox / self.tox
        phiF = VT * math.log(self.NA / ni)
        Qd = math.sqrt(2 * eps_si * Q * self.NA * 2 * phiF)       # |Qd| at threshold
        Vfb = -phiF + self.vfb_shift                              # mid-gap gate + extra shift
        Vt = Vfb + 2 * phiF + Qd / Cox
        Wmax = math.sqrt(2 * eps_si * 2 * phiF / (Q * self.NA))
        S = 1000 * VT * math.log(10) * (1 + (eps_si / Wmax) / Cox)
        return {"Cox": Cox, "phiF": phiF, "Vfb": Vfb, "Vt": Vt, "Qd": Qd, "Wmax_nm": Wmax * 1e7,
                "S_mV_dec": S, "mu": mu, "VT": VT, "L": self.L, "eps_si": eps_si}


# ------------------------------------------------------------- plotting ----
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "figure.dpi": 200, "savefig.bbox": "tight", "axes.axisbelow": True,
        "legend.frameon": False})
    return plt


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
