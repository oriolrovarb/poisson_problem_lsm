"""Draw the domain decompositions (slab / pencil / cube) as SVG files. Plain Python, no dependencies.

    python3 plotting/draw_decompositions.py

Writes to plots/:
  decomposition_layouts.svg      the same cube split into 8 blocks, three ways (isometric 3D)
  decomposition_halo.svg         what one halo exchange sends (2D cross-section)
  decomposition_halo_volume.svg  predicted halo data per rank vs number of ranks (N=200)
"""
import itertools
import math
import os

OUT = "plots"
# colour-blind friendly qualitative colours (Okabe-Ito) + extras
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
          "#F0E442", "#999999", "#44AA99", "#AA4499", "#882255", "#117733"]
FONT = "font-family='Helvetica, Arial, sans-serif'"


def shade(hex_color, f):
    """Multiply the RGB channels of a '#rrggbb' colour by f (f<1 darker, f>1 lighter)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (max(0, min(255, round(c * f))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def text(x, y, s, size=13, anchor="middle", weight="normal", fill="#222"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' text-anchor='{anchor}' "
            f"font-weight='{weight}' fill='{fill}' {FONT}>{s}</text>")


def split(N, parts):
    """Same even split as poisson.decomposition.slab_range, 0-based [a, b)."""
    base, extra = divmod(N, parts)
    out, a = [], 0
    for r in range(parts):
        n = base + (1 if r < extra else 0)
        out.append((a, a + n))
        a += n
    return out


# ---------------------------------------------------------------- 3D layouts
def layout_panel(dims, title, subtitle, ox, oy, N=8, gap=0.4, S=22):
    """Isometric drawing of the N^3 cube cut into blocks; (ox, oy) = panel centre."""
    c30, s30 = math.cos(math.radians(30)), 0.5

    def P(x, y, z):
        return (ox + (x - y) * c30 * S, oy + ((x + y) * s30 - z) * S)

    def poly(pts, fill):
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in (P(*p) for p in pts))
        return f"<polygon points='{d}' fill='{fill}' stroke='#222' stroke-width='1' stroke-linejoin='round'/>"

    ranges = [split(N, d) for d in dims]
    blocks = list(itertools.product(*[range(d) for d in dims]))   # = Cartesian rank order
    shift = (max(dims) - 1) * gap
    ox_shift = 0
    out = []
    # painter's algorithm: far blocks (small i, j, k) first
    for n, c in sorted(enumerate(blocks), key=lambda t: t[1]):
        lo = [ranges[d][c[d]][0] + gap * c[d] - (N + shift) / 2 for d in range(3)]
        hi = [ranges[d][c[d]][1] + gap * c[d] - (N + shift) / 2 for d in range(3)]
        col = COLORS[n % len(COLORS)]
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        out.append(poly([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], shade(col, 1.12)))  # top
        out.append(poly([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], shade(col, 0.92)))  # +i face
        out.append(poly([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], shade(col, 0.72)))  # +j face
    # axis glyph in a free corner (screen space, same projection directions)
    gx, gy = ox - 170, oy + 175
    for lab, (dx, dy), (tx, ty) in (("i", (c30, s30), (12, 8)), ("j", (-c30, s30), (-12, 8)), ("k", (0, -1), (0, -8))):
        ex, ey = gx + dx * 38, gy + dy * 38
        out.append(f"<line x1='{gx}' y1='{gy}' x2='{ex:.1f}' y2='{ey:.1f}' stroke='#555' "
                   f"stroke-width='1.6' marker-end='url(#arr)'/>")
        out.append(text(ex + tx, ey + ty, lab, 13, weight="bold", fill="#555"))
    out.append(text(ox, 100, title, 17, weight="bold"))
    out.append(text(ox, 120, subtitle, 12, fill="#555"))
    # colour key: rank numbers (hidden blocks can still be identified)
    ky = 600
    n_ranks = len(blocks)
    x_start = ox - (n_ranks - 1) * 15
    for n in range(n_ranks):
        out.append(f"<rect x='{x_start + n * 30 - 11:.1f}' y='{ky - 11:.1f}' width='22' height='22' rx='3' "
                   f"fill='{COLORS[n % len(COLORS)]}' stroke='#222'/>")
        out.append(text(x_start + n * 30, ky + 4, str(n), 11, weight="bold", fill="#111"))
    return "\n".join(out)


def fig_layouts():
    W, H = 1380, 640
    cases = [((8, 1, 1), "Slab", "--dims 8 1 1  |  cuts along i only  |  2 neighbours"),
             ((4, 2, 1), "Pencil", "--dims 4 2 1  |  cuts along i and j  |  up to 4 neighbours"),
             ((2, 2, 2), "Cube", "--dims 2 2 2  |  cuts along i, j and k  |  up to 6 neighbours")]
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
             "<defs><marker id='arr' markerWidth='8' markerHeight='8' refX='6' refY='4' orient='auto'>"
             "<path d='M0,0 L8,4 L0,8 z' fill='#555'/></marker></defs>",
             f"<rect width='{W}' height='{H}' fill='white'/>",
             text(W / 2, 32, "The same N&#179; grid on P = 8 ranks: three decompositions", 20, weight="bold"),
             text(W / 2, 54, "Each colour / number = one rank. Blocks are pulled apart for clarity; "
                  "blocks touching each other exchange one face of data per sweep.", 12, fill="#555")]
    for n, (dims, title, sub) in enumerate(cases):
        parts.append(layout_panel(dims, title, sub, ox=230 + n * 460, oy=345, S=22))
    parts.append("</svg>")
    write("decomposition_layouts.svg", parts)


# ---------------------------------------------------------------- halo sketch
def fig_halo():
    W, H, cell = 980, 560, 38
    n = 6
    x_a, x_b, y0 = 70, 70 + (n + 4) * cell, 130
    p = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
         "<defs><marker id='arr' markerWidth='9' markerHeight='9' refX='7' refY='4.5' orient='auto'>"
         "<path d='M0,0 L9,4.5 L0,9 z' fill='#222'/></marker>"
         "<pattern id='hatch' width='8' height='8' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
         "<line x1='0' y1='0' x2='0' y2='8' stroke='#222' stroke-width='1.5' opacity='0.5'/></pattern></defs>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         text(W / 2, 30, "Halo exchange between two neighbouring blocks (2D cross-section)", 19, weight="bold")]

    def cells(xs, ys, nx, ny, fill, hatch=False):
        s = []
        for i in range(nx):
            for j in range(ny):
                s.append(f"<rect x='{xs + i * cell}' y='{ys + j * cell}' width='{cell}' height='{cell}' "
                         f"fill='{fill}' stroke='white' stroke-width='1.5'/>")
                if hatch:
                    s.append(f"<rect x='{xs + i * cell}' y='{ys + j * cell}' width='{cell}' height='{cell}' "
                             f"fill='url(#hatch)' stroke='#222' stroke-width='1'/>")
        return s

    A, B = "#0072B2", "#E69F00"
    p += cells(x_a, y0, n, n, A)                                  # A own
    p += cells(x_a + n * cell, y0, 1, n, B, hatch=True)           # A ghost (copy of B's edge)
    p += cells(x_b - cell, y0, 1, n, A, hatch=True)               # B ghost (copy of A's edge)
    p += cells(x_b, y0, n, n, B)                                  # B own
    p.append(text(x_a + n * cell / 2, y0 - 14, "rank A: own points", 14, weight="bold"))
    p.append(text(x_b + n * cell / 2, y0 - 14, "rank B: own points", 14, weight="bold"))
    # arrows between the edge columns and the ghost columns
    ya = y0 + n * cell + 26
    p.append(f"<path d='M{x_a + (n - 0.5) * cell},{y0 + n * cell + 4} V{ya + 18} H{x_b - 0.5 * cell} V{y0 + n * cell + 4}' "
             f"fill='none' stroke='#222' stroke-width='2' marker-end='url(#arr)'/>")
    p.append(f"<path d='M{x_b + 0.5 * cell},{y0 + n * cell + 4} V{ya + 44} H{x_a + (n + 0.5) * cell} V{y0 + n * cell + 4}' "
             f"fill='none' stroke='#222' stroke-width='2' marker-end='url(#arr)'/>")
    p.append(text(W / 2, ya + 75,
                  "Before every Jacobi sweep: A's last own column goes into B's ghost column, "
                  "B's first own column goes into A's ghost column.", 13))
    p.append(text(W / 2, ya + 95,
                  "(one MPI_Sendrecv per direction; hatched cells = ghost copies, never updated locally)", 13, fill="#555"))
    p.append(text(x_a + (n + 0.5) * cell, y0 - 40, "A's ghost", 12, fill="#333"))
    p.append(text(x_b - 0.5 * cell, y0 - 40, "B's ghost", 12, fill="#333"))
    p.append("</svg>")
    write("decomposition_halo.svg", p)


# ---------------------------------------------------------------- halo volume chart
def balanced_dims(P, ndim):
    """Factorisation of P into ndim factors with the smallest total cut area (like MPI_Dims_create)."""
    best = None
    for c in itertools.product(range(1, P + 1), repeat=ndim - 1):
        rest, ok = P, True
        for f in c:
            if rest % f:
                ok = False
                break
            rest //= f
        if not ok:
            continue
        d = tuple(sorted(c + (rest,), reverse=True)) + (1,) * (3 - ndim)
        score = d[0] * d[1] + d[0] * d[2] + d[1] * d[2]
        if best is None or score < best[0]:
            best = (score, d)
    return best[1]


def halo_points(N, dims):
    """Largest number of values a rank sends per sweep (own faces that have a neighbour)."""
    sizes = [[b - a for a, b in split(N, d)] for d in dims]
    worst = 0
    for c in itertools.product(*[range(d) for d in dims]):
        n = [sizes[d][c[d]] for d in range(3)]
        tot = 0
        for d in range(3):
            area = n[(d + 1) % 3] * n[(d + 2) % 3]
            tot += area * ((c[d] > 0) + (c[d] < dims[d] - 1))
        worst = max(worst, tot)
    return worst


def chart(series, x0, y0, w, h, ylabel, ymin, ymax, Ps):
    """Log-log line chart in the box (x0,y0,w,h); series = [(name, colour, [y...])]."""
    lx = lambda P: x0 + (math.log2(P) - math.log2(Ps[0])) / (math.log2(Ps[-1]) - math.log2(Ps[0])) * w
    ly = lambda v: y0 + h - (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * h
    s = [f"<rect x='{x0}' y='{y0}' width='{w}' height='{h}' fill='none' stroke='#888'/>"]
    e = math.floor(math.log10(ymin))
    while 10 ** e <= ymax:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if ymin <= v <= ymax:
                s.append(f"<line x1='{x0}' y1='{ly(v):.1f}' x2='{x0 + w}' y2='{ly(v):.1f}' stroke='#ddd'/>")
                if m == 1:
                    s.append(text(x0 - 8, ly(v) + 4, f"{v:g}", 11, anchor="end", fill="#555"))
        e += 1
    for P in Ps:
        s.append(f"<line x1='{lx(P):.1f}' y1='{y0}' x2='{lx(P):.1f}' y2='{y0 + h}' stroke='#eee'/>")
        s.append(text(lx(P), y0 + h + 18, str(P), 11, fill="#555"))
    s.append(text(x0 + w / 2, y0 + h + 40, "number of ranks P", 12))
    s.append(f"<text transform='translate({x0 - 62},{y0 + h / 2}) rotate(-90)' font-size='12' text-anchor='middle' "
             f"fill='#222' {FONT}>{ylabel}</text>")
    for k, (name, col, ys) in enumerate(series):
        pts = " ".join(f"{lx(P):.1f},{ly(y):.1f}" for P, y in zip(Ps, ys))
        s.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2.5'/>")
        for P, y in zip(Ps, ys):
            s.append(f"<circle cx='{lx(P):.1f}' cy='{ly(y):.1f}' r='3.8' fill='{col}'/>")
        s.append(f"<rect x='{x0 + 14}' y='{y0 + 12 + k * 22}' width='16' height='4' fill='{col}'/>")
        s.append(text(x0 + 38, y0 + 18 + k * 22, name, 12, anchor="start"))
    return s


def fig_halo_volume(N=200):
    Ps = [2, 4, 8, 12, 16, 24, 32, 48, 64]   # P=1 sends nothing
    kinds = (("slab", 1, COLORS[0]), ("pencil", 2, COLORS[1]), ("cube", 3, COLORS[2]))
    halo = {name: [halo_points(N, balanced_dims(P, nd)) for P in Ps] for name, nd, _ in kinds}
    ratio = {name: [h / (N ** 3 / P) for h, P in zip(halo[name], Ps)] for name, _, _ in kinds}
    W, H = 1100, 470
    p = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         text(W / 2, 30, f"Predicted halo traffic for N = {N} (balanced process grids, exact uneven block sizes)",
              18, weight="bold")]
    p += chart([(n, c, halo[n]) for n, _, c in kinds], 100, 70, 400, 300,
               "values sent per rank per sweep (worst rank)", 1e3, 2e5, Ps)
    p += chart([(n, c, [max(v, 1e-3) for v in ratio[n]]) for n, _, c in kinds], 640, 70, 400, 300,
               "halo values per owned point (comm / compute)", 1e-2, 1, Ps)
    p.append("</svg>")
    write("decomposition_halo_volume.svg", p)


def write(name, parts):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    fig_layouts()
    fig_halo()
    fig_halo_volume()
    print("wrote plots/decomposition_*.svg")
