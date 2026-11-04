"""J1강 - JFET: 채널을 '막아서' 끄는 트랜지스터  (Streetman 6.1)

MOSFET은 게이트 전압으로 없던 채널을 '만들어서' 켭니다.
JFET은 반대입니다. n형 채널이 처음부터 열려 있고(그냥 저항),
p+ 게이트와 채널 사이 p-n 접합에 역바이어스를 걸면 공핍층이 채널 안으로 자라서
전자가 지나갈 통로를 '좁히고', 결국 막아 버립니다(핀치오프).

교재 공식 (이 반쪽 소자 기준, a = 채널 반두께, ND = 채널 도핑):
    공핍층 두께        h  = sqrt( 2 eps (Vbi - Vg) / (q ND) )
    핀치오프 전압      Vp = q ND a^2 / (2 eps)          (h = a 가 되는 접합 전압)
    소자가 꺼지는 Vg   Vg_off = Vbi - Vp
    전류 (Vd 작을 때)  Id = G0 [ Vd - 2/(3 sqrt(Vp)) ((Vbi-Vg+Vd)^1.5 - (Vbi-Vg)^1.5) ],  G0 = q mu ND a W / L
    포화 시작          Vd_sat = Vp - Vbi + Vg    (드레인 쪽 끝에서 채널이 막히는 순간)

[먼저 예측해 보기]
  Q1. Vg = 0 V (아무것도 안 걸었을 때) 소자는 켜져 있을까, 꺼져 있을까?
  Q2. Vg를 음(-)으로 키우면 채널은 넓어질까 좁아질까? 왜 '음'일까? (게이트는 p+, 채널은 n)
  Q3. Vd를 키우면 채널은 소스 쪽과 드레인 쪽 중 어디가 먼저 막힐까?

실행:  python lessonJ1_jfet.py --quick  (빠른 모드, 몇 분)  <- 숫자 바꿔 볼 때 이걸로
       python lessonJ1_jfet.py          (전체, 20분 이상)
       python lessonJ1_jfet.py --plot   (그림만 다시)
"""
import json, os, sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
L_UM, A_UM, ND = 5.0, 0.6, 1e16            # <- 바꿔 보세요: 채널 길이, 채널 반두께, 채널 도핑
VG_IDVD = [0.0, -0.5, -1.0]
VD_MAP = 3.0
QUICK = "--quick" in sys.argv               # 빠른 모드: 점 수를 줄여 몇 분 안에 결과 확인
VG_STEP, VD_STEP = (0.2, 0.25) if QUICK else (0.1, 0.1)
if QUICK:
    VG_IDVD = [0.0]


def simulate():
    from jfet import Jfet
    dev = Jfet(L_um=L_UM, a_um=A_UM, ND=ND)
    th = dev.theory()
    R = {"th": th}
    x, y = dev.node("x") / 1e-4, dev.node("y") / 1e-4
    R["map0"] = {"x": x.tolist(), "y": y.tolist(), "n": dev.node("Electrons").tolist()}

    # 1) small Vd: Id-Vg and channel opening
    dev.bias(drain=0.05)
    vg_list, idg, opn = [], [], []
    for vg in np.round(np.arange(0.0, -2.41, -VG_STEP), 2):
        dev.bias(gate=vg)
        vg_list.append(float(vg)); idg.append(dev.current()); opn.append(dev.opening())
        print(f"Vd=0.05  Vg = {vg:5.2f} V   Id = {idg[-1]*1e6:9.4f} uA/um   opening = {opn[-1]:.3f} um")
    R["idvg"] = {"vg": vg_list, "Id": idg, "open": opn}

    # 2) Id-Vd family
    R["idvd"] = {}
    for vg in VG_IDVD:
        dev.bias(drain=0.0); dev.bias(gate=vg)
        vds, ids = [], []
        for vd in np.round(np.arange(0.0, 3.51, VD_STEP), 2):
            dev.bias(drain=vd)
            vds.append(float(vd)); ids.append(dev.current())
            if vg == 0.0 and abs(vd - VD_MAP) < 1e-9:
                R["map1"] = {"n": dev.node("Electrons").tolist()}
        R["idvd"][str(vg)] = {"vd": vds, "Id": ids}
        print(f"Id-Vd at Vg = {vg}: Id(3.5 V) = {ids[-1]*1e6:.3f} uA/um   gate current = {dev.current('gate'):.1e} A/um")
    json.dump(R, open("lessonJ1.json", "w"))
    return R


def analyse(R):
    from jfet import id_formula, opening_formula
    th = R["th"]
    print(f"\nVbi = {th['Vbi']:.3f} V   Vp = {th['Vp']:.3f} V   -> Vg_off (textbook) = {th['Vg_off']:.2f} V")
    g = R["idvg"]; vg = np.array(g["vg"]); I = np.array(g["Id"])
    for v in (0.0, -0.5, -1.0, -1.5):
        o_t = np.interp(v, vg[::-1], np.array(g["open"])[::-1]); i_t = np.interp(v, vg[::-1], I[::-1])
        print(f"Vg = {v:5.1f}  opening TCAD {o_t:.3f} um  formula {opening_formula(th, v):.3f} um   "
              f"Id TCAD {i_t*1e6:.4f}  formula {id_formula(th, v, 0.05)*1e6:.4f} uA/um")
    off = vg[np.argmax(I < 0.01 * I[0])]
    print(f"Vg where Id < 1% of Id(0): TCAD {off:.2f} V   textbook Vg_off {th['Vg_off']:.2f} V")
    out = {"off": float(off), "sat": {}}
    for s, c in R["idvd"].items():
        v = float(s); vds, ids = np.array(c["vd"]), np.array(c["Id"])
        vsat = th["Vp"] - th["Vbi"] + v
        isat_f = id_formula(th, v, 10)
        isat = np.interp(vsat, vds, ids)
        out["sat"][s] = (vsat, isat)
        print(f"Vg = {v:5.1f}: Vd_sat = {vsat:.2f} V   Id at Vd_sat: TCAD {isat*1e6:.3f}  formula {isat_f*1e6:.3f} uA/um"
              f"   Id(3.5)/Id(Vd_sat) = {ids[-1]/isat:.2f}")
    return out


def plot(R, A):
    from common import style, COLORS, INK, INK2, MUTED
    from jfet import id_formula, opening_formula
    import matplotlib.tri as mtri
    plt = style()
    th = R["th"]
    fig = plt.figure(figsize=(8.4, 5.6))
    gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=1.2)
    x, y = np.array(R["map0"]["x"]), np.array(R["map0"]["y"])
    tri = mtri.Triangulation(x, y)
    xg0 = 0.5 + 0.1; xg1 = xg0 + L_UM; tg = 0.2
    for j, (key, title) in enumerate((("map0", "Vg = 0, Vd = 0: channel open (white = no electrons)"),
                                      ("map1", f"Vg = 0, Vd = {VD_MAP:g} V: channel narrowest at the drain end"))):
        ax = fig.add_subplot(gs[0, 3 * j:3 * j + 3])
        n = np.log10(np.maximum(np.array(R[key]["n"]), 1e4))
        cs = ax.tricontourf(tri, n, levels=np.linspace(10, 17, 15), cmap="Blues", extend="both")
        ax.plot([xg0, xg1], [0, 0], color=COLORS[1], lw=4, solid_capstyle="butt")
        ax.plot([xg0, xg1, xg1, xg0, xg0], [0, 0, tg, tg, 0], color=COLORS[1], lw=0.8)
        ax.text((xg0 + xg1) / 2, 0.1, "p+ gate", ha="center", va="center", fontsize=7, color=COLORS[1])
        ax.text(0.25, 0.45, "S", ha="center", fontsize=8, color=INK); ax.text(x.max() - 0.25, 0.45, "D", ha="center", fontsize=8, color=INK)
        ax.set_ylim(y.max(), 0); ax.set_xlim(0, x.max())
        ax.set_xlabel("x along channel (µm)"); ax.set_ylabel("depth (µm)")
        ax.set_title(title, loc="left", fontsize=8.2); ax.grid(False)
    cb = fig.colorbar(cs, ax=fig.axes[:2], fraction=0.025, pad=0.02)
    cb.set_label("log10 n (cm⁻³)", fontsize=7)

    g = R["idvg"]; vg = np.array(g["vg"])
    vv = np.linspace(vg.min(), 0, 200)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(vv, [opening_formula(th, v) for v in vv], color=MUTED, lw=3, alpha=0.6, label="formula a − h")
    ax.plot(vg, g["open"], "o", color=COLORS[0], ms=3, label="TCAD")
    ax.axhline(th["a_um"], color=MUTED, ls=":", lw=1); ax.text(-0.05, th["a_um"] + 0.02, "a (full channel)", ha="right", fontsize=6.5, color=INK2)
    ax.set_xlabel("Vg (V)"); ax.set_title("Open channel (µm)", loc="left", fontsize=9)
    ax.set_ylim(0, th["a_um"] * 1.25); ax.legend(fontsize=6.5, loc="center left")

    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(vv, [id_formula(th, v, 0.05) * 1e6 for v in vv], color=MUTED, lw=3, alpha=0.6, label="formula")
    ax.plot(vg, np.array(g["Id"]) * 1e6, "o", color=COLORS[0], ms=3, label="TCAD")
    ax.axvline(th["Vg_off"], color=MUTED, ls="--", lw=1)
    ax.text(th["Vg_off"] - 0.05, 0.45 * max(g["Id"]) * 1e6, f"textbook\nVg_off\n{th['Vg_off']:.2f} V", ha="right", fontsize=6.5, color=INK2)
    ax.set_xlabel("Vg (V)"); ax.set_title("Id at Vd = 50 mV (µA/µm)", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")

    ax = fig.add_subplot(gs[1, 4:6])
    for i, (s, c) in enumerate(R["idvd"].items()):
        v = float(s); vd = np.array(c["vd"])
        ax.plot(vd, np.array(c["Id"]) * 1e6, color=COLORS[i], lw=2)
        ax.plot(vd, [id_formula(th, v, d) * 1e6 for d in vd], color=COLORS[i], lw=1, ls="--")
        vs, isat = A["sat"][s]
        ax.plot([vs], [isat * 1e6], "o", color=COLORS[i], ms=4)
        ax.text(3.55, c["Id"][-1] * 1e6, f"Vg={v:g}", fontsize=6.5, color=COLORS[i], va="center")
    ax.set_xlim(0, 4.3)
    ax.set_xlabel("Vd (V)   dashed: formula, dot: Vd_sat"); ax.set_title("Id-Vd (µA/µm)", loc="left", fontsize=9)
    fig.savefig("lessonJ1.png")
    print("saved lessonJ1.png")


if __name__ == "__main__":
    R = json.load(open("lessonJ1.json")) if "--plot" in sys.argv else simulate()
    plot(R, analyse(R))
