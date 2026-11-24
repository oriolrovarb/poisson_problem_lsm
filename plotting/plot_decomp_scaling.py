"""Compare slab / pencil / cube from the CSV files in results/. Plain Python, writes SVG.

    python3 plotting/plot_decomp_scaling.py

Reads (whichever exists) and writes to plots/:
  results/mpi_decomp.csv                         -> decomp_scaling.svg    speedup + Mlup/s vs ranks, one node
  results/mpi_placement_{core,socket}.csv        -> decomp_placement.svg  core vs socket mapping
  results/mpi_multinode.csv (+ mpi_decomp.csv)   -> decomp_multinode.svg  one node vs two nodes
  results/mpi_socket2n*.csv                      -> decomp_socket2n.svg   two nodes, ranks spread over sockets:
                                                    Mlup/s and compute/communication time per rank (N=200)
Points are the mean of the repetitions; vertical bars show min..max.
"""
import csv
import os
from collections import defaultdict

RESULTS, OUT = "results", "plots"
# matplotlib C0 (blue), C1 (orange), C2 (green), as in docs/plotting_rules.md
COLOR = {"slab": "#1f77b4", "pencil": "#ff7f0e", "cube": "#2ca02c"}
FONT = "font-family='Helvetica, Arial, sans-serif'"


def text(x, y, s, size=12, anchor="middle", weight="normal", fill="#222", extra=""):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' text-anchor='{anchor}' "
            f"font-weight='{weight}' fill='{fill}' {FONT} {extra}>{s}</text>")


def load(name):
    """Return {(kind, ranks): [mlups, ...]} from results/<name>.csv, or None if missing."""
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    data = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            kind = row["impl"].split("-")[1]           # mpi-pencil-4x3x1 -> pencil
            data[(kind, int(row["ranks"]))].append(float(row["mlups"]))
    return data


def series(data, kind, with_p1=None):
    """Sorted [(P, mean, min, max)] for one layout; optionally prepend the P=1 run of another."""
    pts = {p: v for (k, p), v in data.items() if k == kind}
    if with_p1 and 1 not in pts and (("slab", 1) in with_p1):
        pts[1] = with_p1[("slab", 1)]
    return [(p, sum(v) / len(v), min(v), max(v)) for p, v in sorted(pts.items())]


def panel(x0, y0, w, h, xs, ymax, xlabel, ylabel, curves, title="", ideal=None):
    """Linear-axis line chart. curves = [(label, colour, dash, [(x, mean, lo, hi)])]."""
    sx = lambda x: x0 + (x - xs[0]) / (xs[-1] - xs[0]) * w
    sy = lambda v: y0 + h - v / ymax * h
    s = [text(x0 + w / 2, y0 - 14, title, 14, weight="bold"),
         f"<rect x='{x0}' y='{y0}' width='{w}' height='{h}' fill='none' stroke='#888'/>"]
    step = next(t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if ymax / t <= 8)
    v = 0.0
    while v <= ymax + 1e-9:
        s.append(f"<line x1='{x0}' y1='{sy(v):.1f}' x2='{x0 + w}' y2='{sy(v):.1f}' stroke='#e5e5e5'/>")
        s.append(text(x0 - 8, sy(v) + 4, f"{v:g}", 11, anchor="end", fill="#555"))
        v += step
    for x in xs:
        s.append(f"<line x1='{sx(x):.1f}' y1='{y0}' x2='{sx(x):.1f}' y2='{y0 + h}' stroke='#f0f0f0'/>")
        s.append(text(sx(x), y0 + h + 18, str(x), 11, fill="#555"))
    s.append(text(x0 + w / 2, y0 + h + 40, xlabel))
    s.append(text(0, 0, ylabel, 12, extra=f"transform='translate({x0 - 52},{y0 + h / 2}) rotate(-90)'"))
    if ideal:
        a, b = ideal
        s.append(f"<line x1='{sx(a[0]):.1f}' y1='{sy(a[1]):.1f}' x2='{sx(b[0]):.1f}' y2='{sy(b[1]):.1f}' "
                 f"stroke='#444' stroke-dasharray='6 4'/>")
    for k, (label, col, dash, pts) in enumerate(curves):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        s.append("<polyline points='" + " ".join(f"{sx(p):.1f},{sy(m):.1f}" for p, m, _, _ in pts)
                 + f"' fill='none' stroke='{col}' stroke-width='2.4'{d}/>")
        for p, m, lo, hi in pts:
            s.append(f"<line x1='{sx(p):.1f}' y1='{sy(lo):.1f}' x2='{sx(p):.1f}' y2='{sy(hi):.1f}' "
                     f"stroke='{col}' stroke-width='1.5'/>")
            s.append(f"<circle cx='{sx(p):.1f}' cy='{sy(m):.1f}' r='3.8' fill='{col}'/>")
        s.append(f"<line x1='{x0 + 14}' y1='{y0 + 18 + k * 20}' x2='{x0 + 38}' y2='{y0 + 18 + k * 20}' "
                 f"stroke='{col}' stroke-width='2.4'{d}/>")
        s.append(text(x0 + 44, y0 + 22 + k * 20, label, 12, anchor="start"))
    if ideal:
        s.append(text(x0 + w - 8, sy(ideal[1][1]) + 14, "ideal", 11, anchor="end", fill="#444"))
    return s


def svg(W, H, title, body):
    """Title line 1 = what is measured, line 2 = setup (docs/plotting_rules.md); the body moves down for line 2."""
    what, setup = title.split("\n")
    H += 24
    return [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
            f"<rect width='{W}' height='{H}' fill='white'/>", text(W / 2, 30, what, 18, weight="bold"),
            text(W / 2, 52, setup, 14, fill="#444"),
            "<g transform='translate(0,24)'>", *body, "</g>", "</svg>"]


def write(name, parts):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("\n".join(parts))
    print("wrote", os.path.join(OUT, name))


def fig_scaling(d):
    base = sum(d[("slab", 1)]) / len(d[("slab", 1)])
    Ps = sorted({p for _, p in d})
    curves_t, curves_s = [], []
    for kind in ("slab", "pencil", "cube"):
        pts = series(d, kind, with_p1=d)
        if not pts:
            continue
        curves_t.append((kind, COLOR[kind], "", pts))
        curves_s.append((kind, COLOR[kind], "", [(p, m / base, lo / base, hi / base) for p, m, lo, hi in pts]))
    pmax = Ps[-1]
    body = panel(90, 80, 450, 330, Ps, pmax, "number of ranks P", "speedup  T(1) / T(P)", curves_s,
                 "Speedup", ideal=((1, 1), (pmax, pmax)))
    body += panel(640, 80, 450, 330, Ps, max(m for c in curves_t for *_, pts in [c] for _, _, _, m in pts) * 1.1,
                  "number of ranks P", "Mlup/s", curves_t, "Performance")
    write("decomp_scaling.svg", svg(1160, 470, "Strong scaling, NumPy MPI: slab vs pencil vs cube\nN = 200 · 1 node · 1–24 ranks", body))


def fig_placement(core, sock):
    Ps = sorted({p for _, p in core} | {p for _, p in sock})
    curves = []
    for kind in ("slab", "pencil", "cube"):
        for name, dat, dash in (("map-by core (fill socket 0 first)", core, ""),
                                ("map-by socket (alternate sockets)", sock, "7 4")):
            pts = series(dat, kind)
            if pts:
                curves.append((f"{kind}, {name.split(' (')[0]}", COLOR[kind], dash, pts))
    top = max(m for *_, pts in curves for _, _, _, m in pts) * 1.1
    body = panel(90, 80, 560, 340, Ps, top, "number of ranks P", "Mlup/s", curves,
                 "Rank placement: solid = core, dashed = socket")
    write("decomp_placement.svg", svg(740, 480, "Rank placement, NumPy MPI: map-by core vs map-by socket\nN = 200 · 1 node · 1–24 ranks · slab / pencil / cube", body))


def fig_multinode(one, multi):
    """Grouped bars: P=24 on one node, P=24 on two nodes (12+12), P=48 on two nodes (24+24)."""
    groups = [("P=24, 1 node", one, 24), ("P=24, 2 nodes (12+12)", multi, 24), ("P=48, 2 nodes (24+24)", multi, 48)]
    W, H, x0, y0, w, h = 780, 460, 90, 80, 640, 320
    vals = {(g, k): v[(k, p)] for g, (_, v, p) in enumerate(groups) for k in COLOR if (k, p) in v}
    top = max(max(v) for v in vals.values()) * 1.12
    s = [f"<rect x='{x0}' y='{y0}' width='{w}' height='{h}' fill='none' stroke='#888'/>"]
    step = next(t for t in (50, 100, 200, 250, 500, 1000) if top / t <= 8)
    v = 0
    while v <= top:
        yy = y0 + h - v / top * h
        s.append(f"<line x1='{x0}' y1='{yy:.1f}' x2='{x0 + w}' y2='{yy:.1f}' stroke='#e5e5e5'/>")
        s.append(text(x0 - 8, yy + 4, str(v), 11, anchor="end", fill="#555"))
        v += step
    s.append(text(0, 0, "Mlup/s", 12, extra=f"transform='translate({x0 - 52},{y0 + h / 2}) rotate(-90)'"))
    gw = w / len(groups)
    for g, (label, _, _) in enumerate(groups):
        s.append(text(x0 + gw * (g + 0.5), y0 + h + 22, label, 12))
        for j, kind in enumerate(COLOR):
            if (g, kind) not in vals:
                continue
            vv = vals[(g, kind)]
            m, lo, hi = sum(vv) / len(vv), min(vv), max(vv)
            bx = x0 + gw * g + gw * 0.12 + j * gw * 0.26
            bh = m / top * h
            s.append(f"<rect x='{bx:.1f}' y='{y0 + h - bh:.1f}' width='{gw * 0.22:.1f}' height='{bh:.1f}' fill='{COLOR[kind]}'/>")
            s.append(f"<line x1='{bx + gw * 0.11:.1f}' y1='{y0 + h - lo / top * h:.1f}' x2='{bx + gw * 0.11:.1f}' "
                     f"y2='{y0 + h - hi / top * h:.1f}' stroke='#222' stroke-width='1.5'/>")
            s.append(text(bx + gw * 0.11, y0 + h - bh - 8, f"{m:.0f}", 11))
    for j, kind in enumerate(COLOR):
        s.append(f"<rect x='{x0 + 14 + j * 90}' y='{y0 + 10}' width='14' height='14' fill='{COLOR[kind]}'/>")
        s.append(text(x0 + 34 + j * 90, y0 + 22, kind, 12, anchor="start"))
    write("decomp_multinode.svg", svg(W, H, "One node vs two nodes, NumPy MPI\nN = 200 · 1–2 nodes · 24–48 ranks · slab / pencil / cube", s))


def fig_socket2n(n=200):
    """Left: Mlup/s per layout at P=24 and 48. Right: mean per-rank compute and comm time (stacked)."""
    perf = defaultdict(list)
    path = os.path.join(RESULTS, "mpi_socket2n.csv")
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if int(row["N"]) == n:
                perf[(row["impl"].split("-")[1], int(row["ranks"]))].append(float(row["mlups"]))
    times = defaultdict(lambda: [0.0, 0.0, 0])
    for kind in COLOR:
        with open(os.path.join(RESULTS, f"mpi_socket2n_ranks_{kind}.csv")) as fh:
            for row in csv.DictReader(fh):
                if int(row["N"]) == n:
                    t = times[(kind, int(row["ranks"]))]
                    t[0] += float(row["compute_s"]); t[1] += float(row["comm_s"]); t[2] += 1
    Ps = [24, 48]
    W, H, y0, h = 1160, 470, 80, 320
    s = []
    for panel_i, (x0, w, ylab) in enumerate(((90, 450, "Mlup/s"), (640, 450, "mean time per rank [s] (100 iterations)"))):
        top = 1800 if panel_i == 0 else 1.0
        s.append(f"<rect x='{x0}' y='{y0}' width='{w}' height='{h}' fill='none' stroke='#888'/>")
        step = 300 if panel_i == 0 else 0.2
        v = 0.0
        while v <= top + 1e-9:
            yy = y0 + h - v / top * h
            s.append(f"<line x1='{x0}' y1='{yy:.1f}' x2='{x0 + w}' y2='{yy:.1f}' stroke='#e5e5e5'/>")
            s.append(text(x0 - 8, yy + 4, f"{v:g}", 11, anchor="end", fill="#555"))
            v += step
        s.append(text(0, 0, ylab, 12, extra=f"transform='translate({x0 - 56},{y0 + h / 2}) rotate(-90)'"))
        s.append(text(x0 + w / 2, y0 - 14, "Performance" if panel_i == 0 else "Where the time goes: compute (light) + communication (dark)", 14, weight="bold"))
        gw = w / len(Ps)
        for g, P in enumerate(Ps):
            s.append(text(x0 + gw * (g + 0.5), y0 + h + 22, f"P = {P}", 12))
            for j, kind in enumerate(COLOR):
                bx = x0 + gw * g + gw * 0.12 + j * gw * 0.26
                bw = gw * 0.22
                if panel_i == 0:
                    vv = perf.get((kind, P))
                    if not vv:
                        continue
                    m = sum(vv) / len(vv)
                    bh = m / top * h
                    s.append(f"<rect x='{bx:.1f}' y='{y0 + h - bh:.1f}' width='{bw:.1f}' height='{bh:.1f}' fill='{COLOR[kind]}'/>")
                    s.append(f"<line x1='{bx + bw / 2:.1f}' y1='{y0 + h - min(vv) / top * h:.1f}' x2='{bx + bw / 2:.1f}' "
                             f"y2='{y0 + h - max(vv) / top * h:.1f}' stroke='#222' stroke-width='1.5'/>")
                    s.append(text(bx + bw / 2, y0 + h - bh - 8, f"{m:.0f}", 11))
                else:
                    t = times.get((kind, P))
                    if not t:
                        continue
                    comp, comm = t[0] / t[2], t[1] / t[2]
                    hc, hm = comp / top * h, comm / top * h
                    s.append(f"<rect x='{bx:.1f}' y='{y0 + h - hc:.1f}' width='{bw:.1f}' height='{hc:.1f}' fill='{COLOR[kind]}' opacity='0.45'/>")
                    s.append(f"<rect x='{bx:.1f}' y='{y0 + h - hc - hm:.1f}' width='{bw:.1f}' height='{hm:.1f}' fill='{COLOR[kind]}'/>")
                    s.append(text(bx + bw / 2, y0 + h - hc - hm - 8, f"{comm / (comp + comm) * 100:.0f}% comm", 11))
        for j, kind in enumerate(COLOR):
            s.append(f"<rect x='{x0 + 14 + j * 90}' y='{y0 + 10}' width='14' height='14' fill='{COLOR[kind]}'/>")
            s.append(text(x0 + 34 + j * 90, y0 + 22, kind, 12, anchor="start"))
    write("decomp_socket2n.svg", svg(W, H, f"Two nodes, ranks spread over the four sockets, NumPy MPI\nN = {n} · 2 nodes · 24–48 ranks · slab / pencil / cube", s))


if __name__ == "__main__":
    one = load("mpi_decomp.csv")
    if one:
        fig_scaling(one)
    core, sock = load("mpi_placement_core.csv"), load("mpi_placement_socket.csv")
    if core and sock:
        fig_placement(core, sock)
    multi = load("mpi_multinode.csv")
    if one and multi:
        fig_multinode(one, multi)
    if os.path.exists(os.path.join(RESULTS, "mpi_socket2n.csv")):
        fig_socket2n()
