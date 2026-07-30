"""Extract the plotted data from the original lasso_intro.pdf.

Parses the PDF content streams of the original figure (Matplotlib 3.10.8
output) and recovers, for each panel (in-distribution / out-of-distribution)
and each series (L2O, DR-L2O, OPT-PEP):

  - the mean curve vertices (stroked polylines, lw 1.5), and
  - the 10th/90th-quantile band edges (filled closed paths, alpha 0.2),

converted from PDF points to data coordinates using the major-gridline
anchors (10^-1/10^1 left, 10^0/10^2 right).

Output: data/pdf_extracted.csv with columns
  panel, series, K, mean, q10, q90
(mean is NaN where the original path was clipped out of the axes, i.e. the
out-of-distribution L2O curve beyond K ~ 10).

Usage: .venv/bin/python extract_pdf_curves.py [path/to/lasso_intro.pdf]
"""

import re
import sys
import zlib

import numpy as np
import pandas as pd

PDF_DEFAULT = (
    "/Users/bs37/Library/CloudStorage/Dropbox/work/research/papers/2026/"
    "dr-l2o/figures/lasso_intro.pdf"
)

SERIES_BY_COLOR = {
    "0.862745 0.196078 0.125490": "L2O",
    "0.000000 0.352941 0.709804": "DR-L2O",
    "0.000000 0.701961 0.176471": "OPT-PEP",
}

# Panel geometry measured from the PDF (axes rects and major gridlines).
PANELS = {
    "in_dist": dict(
        x0=48.9873489551, x1=247.073388,
        xK1=57.99126, dxK=12.8627297941,
        y_anchor=103.891702, v_anchor=1e-1,
        pt_per_decade=(159.213684 - 103.891702) / 2.0,
    ),
    "ood": dict(
        x0=287.193961171, x1=485.28,
        xK1=296.1978720269, dxK=12.8627297941,
        y_anchor=99.450116, v_anchor=1e0,
        pt_per_decade=(151.062515 - 99.450116) / 2.0,
    ),
}


def decompress_content(pdf_path):
    raw = open(pdf_path, "rb").read()
    content = b""
    for s in re.findall(rb"stream\r?\n(.*?)endstream", raw, re.S):
        try:
            content += zlib.decompress(s)
        except zlib.error:
            pass
    return content.decode("latin1")


def collect_paths(txt):
    """Sequentially track fill/stroke color and collect painted paths."""
    token_re = re.compile(
        r"([\d.\-]+ [\d.\-]+ (?:m|l))"
        r"|([\d.\-]+ [\d.\-]+ [\d.\-]+ (?:RG|rg))"
        r"|(\bS\b|\bf\b|\bB\b)"
        r"|(h)"
    )
    paths = []
    cur, stroke, fill, closed = [], None, None, False
    for mv, col, paint, close in token_re.findall(txt):
        if mv:
            x, y, op = mv.split()
            if op == "m":
                cur = [(float(x), float(y))]
                closed = False
            else:
                cur.append((float(x), float(y)))
        elif col:
            r, g, b, op = col.split()
            key = f"{float(r):.6f} {float(g):.6f} {float(b):.6f}"
            if op == "RG":
                stroke = key
            else:
                fill = key
        elif close:
            closed = True
        elif paint:
            if cur:
                paths.append((paint, stroke, fill, closed, cur))
            cur = []
    return paths


def to_data(pts, geo):
    """Convert PDF-point vertices to (K, value) data coordinates."""
    out = []
    for x, y in pts:
        K = 1.0 + (x - geo["xK1"]) / geo["dxK"]
        v = geo["v_anchor"] * 10.0 ** ((y - geo["y_anchor"]) / geo["pt_per_decade"])
        out.append((K, v))
    return out


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else PDF_DEFAULT
    txt = decompress_content(pdf_path)
    paths = collect_paths(txt)

    records = {}  # (panel, series) -> dict K -> {mean, q10, q90}
    for paint, stroke, fill, closed, pts in paths:
        key = stroke if paint == "S" else fill
        series = SERIES_BY_COLOR.get(key)
        if series is None or len(pts) < 10:
            continue
        xmid = np.mean([p[0] for p in pts])
        panel = "in_dist" if xmid < 260 else "ood"
        geo = PANELS[panel]
        data = to_data(pts, geo)
        rec = records.setdefault((panel, series), {})
        if paint == "S":  # mean curve
            for K, v in data:
                Kr = round(K)
                # skip clip-interpolated endpoints (non-integer K)
                if abs(K - Kr) < 1e-3:
                    rec.setdefault(Kr, {})["mean"] = v
        else:  # quantile band: split vertices into lower/upper edge per K
            byK = {}
            for K, v in data:
                Kr = round(K)
                if abs(K - Kr) < 1e-3:
                    byK.setdefault(Kr, []).append(v)
            for Kr, vals in byK.items():
                rec.setdefault(Kr, {})["q10"] = min(vals)
                rec.setdefault(Kr, {})["q90"] = max(vals)

    rows = []
    for (panel, series), rec in sorted(records.items()):
        for K in sorted(rec):
            r = rec[K]
            rows.append(dict(
                panel=panel, series=series, K=K,
                mean=r.get("mean", np.nan),
                q10=r.get("q10", np.nan),
                q90=r.get("q90", np.nan),
            ))
    df = pd.DataFrame(rows)
    df.to_csv("data/pdf_extracted.csv", index=False)
    print(df.to_string(float_format=lambda v: f"{v:.6g}"))


if __name__ == "__main__":
    main()
