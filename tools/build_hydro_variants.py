"""
Variantes de la couche « vulnérabilité hydrogéologique » sur la grille IVS 0,05°.

Constat (vérifié) : la couche hydro de la carte IVS publiée est 1 − (classe − 1)/3 de
`potentiel_hydro_classes_v2.tif` (classification par quartiles du score GWPI v2, qui
inclut la densité de linéaments), moyennée à 0,05° : r = −0,999.

Ce script :
  1. retrouve les poids du score v2 par régression non négative sur les 5 couches
     normalisées 1 km (géologie BGS, pluie CHIRPS, pente, TPI, linéaments) ;
  2. recompose un score continu AVEC pluie (contrôle) et SANS pluie (poids renormalisés) ;
  3. agrège à 0,05° (moyenne), normalise min-max sur le masque IVS, inverse → vulnérabilité ;
  4. exporte data/derived/hydro_*.tif et tools/hydro.json (grilles 8 bits + poids + stats).

Usage : python -I -X utf8 tools/build_hydro_variants.py
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
PROC = Path("D:/Python/Niger_Hydrogéologie/python/data/processed")
OUT_TIF = ROOT / "data" / "derived"
OUT_JSON = ROOT / "tools" / "hydro.json"
BBOX = (0.2, 11.7, 15.9, 23.5); W, H = 314, 236
T = from_bounds(*BBOX, W, H)
CRIT = ["geology", "rainfall", "slope", "tpi", "lineaments"]

def read(name):
    with rasterio.open(PROC / f"{name}.tif") as s:
        a = s.read(1).astype(np.float32); a[a == s.nodata] = np.nan
        return a, s.crs, s.transform

layers = {k: read("layer_" + k)[0] for k in CRIT}
layers["geology"] = layers["geology"] / 4.0          # scores BGS 0-4 -> 0-1 (comme le notebook)
score_v2, crs, tr = read("score_composite_v2")
classes_v2 = read("potentiel_hydro_classes_v2")[0]
valid = ~np.isnan(score_v2) & np.all([~np.isnan(layers[k]) for k in CRIT], axis=0)

# ---------------------------------------------------------------- 1. poids v2 (NNLS)
X = np.column_stack([layers[k][valid] for k in CRIT]); y = score_v2[valid]
w, _ = nnls(X, y); pred = X @ w
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
W2 = dict(zip(CRIT, w))
print("Poids v2 retrouvés :", {k: round(v, 3) for k, v in W2.items()}, f"somme {w.sum():.3f}  R² {r2:.4f}  MAE {np.abs(y - pred).mean():.4f}")
print("Seuils quartiles v2 :", [round(float(np.nanpercentile(score_v2, p)), 3) for p in (25, 50, 75)])

# ---------------------------------------------------------------- 2. scores continus
def composite(weights):
    tot = sum(weights.values()); out = np.zeros_like(score_v2)
    for k, wk in weights.items(): out += wk / tot * layers[k]
    return np.where(valid, out, np.nan).astype(np.float32)
W_norain = {k: v for k, v in W2.items() if k != "rainfall"}
scores = {"cont": composite(W2), "norain": composite(W_norain)}
print("Poids sans pluie (renormalisés) :", {k: round(v / sum(W_norain.values()), 3) for k, v in W_norain.items()})

# ---------------------------------------------------------------- 2b. classification Fisher-Jenks du score v2
# Reproduit la carte de potentiel telle que le pipeline la prévoit (method='jenks'), au lieu
# des quartiles du notebook racine. Pour l'usage comme facteur, chaque classe reçoit la
# MÉDIANE du score de son intervalle (et non 0 / 1/3 / 2/3 / 1, équidistance arbitraire).
import jenkspy
rng = np.random.default_rng(42)
vals = scores["cont"][valid]
breaks = jenkspy.jenks_breaks(rng.choice(vals, 25000, replace=False), n_classes=4)
jenks_t = [float(b) for b in breaks[1:-1]]
classes_j = np.full_like(score_v2, np.nan); classes_j[valid] = 1
for i, t in enumerate(jenks_t): classes_j[valid & (scores["cont"] > t)] = i + 2
areas = [round(float((classes_j[valid] == k).mean() * 100), 1) for k in (1, 2, 3, 4)]
medians = [float(np.median(vals[classes_j[valid] == k])) for k in (1, 2, 3, 4)]
score_jenks = np.full_like(score_v2, np.nan)
for k, m in zip((1, 2, 3, 4), medians): score_jenks[classes_j == k] = m
scores["jenks"] = score_jenks.astype(np.float32)
print(f"Jenks : seuils {np.round(jenks_t, 3)} ; surfaces % {areas} ; médianes de classe {np.round(medians, 3)}")
with rasterio.open(OUT_TIF / "gwpi_v2_classes_jenks.tif", "w", driver="GTiff", dtype="float32", width=score_v2.shape[1], height=score_v2.shape[0],
                   count=1, crs=crs, transform=tr, nodata=-9999.0, compress="deflate") as d:
    d.write(np.where(np.isnan(classes_j), -9999, classes_j).astype(np.float32), 1)

# ---------------------------------------------------------------- 3. agrégation 0,05° + inversion
an = json.load(open(ROOT / "tools" / "analysis.json", encoding="utf-8"))
def png(uri): return np.array(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))
mask = png(an["grids"]["ivs"]) != 255
hydro_pub = png(an["grids"]["hydro"]).astype(float); hydro_pub = np.where(hydro_pub == 255, np.nan, hydro_pub / 254)
region = png(an["grids"]["region"]); names = an["region_names"]

def to_grid(a, rs=Resampling.average):
    dst = np.full((H, W), np.nan, np.float32)
    reproject(np.where(np.isnan(a), -9999, a).astype(np.float32), dst, src_transform=tr, src_crs=crs, src_nodata=-9999,
              dst_transform=T, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=rs)
    return dst
def vuln(a):
    g = to_grid(a); ok = mask & ~np.isnan(g)
    lo, hi = np.nanmin(g[ok]), np.nanmax(g[ok])
    return np.where(ok, 1 - (g - lo) / (hi - lo), np.nan).astype(np.float32)
out = {"cont": vuln(scores["cont"]), "norain": vuln(scores["norain"]), "jenks": vuln(scores["jenks"])}
# contrôle : reproduction de la couche publiée à partir des classes v2
repro = to_grid(1 - (classes_v2 - 1) / 3.0); ok = mask & ~np.isnan(repro) & ~np.isnan(hydro_pub)
print(f"Contrôle couche publiée vs 1-(classes_v2-1)/3 moyennée : r = {np.corrcoef(hydro_pub[ok], repro[ok])[0, 1]:.4f}, MAE {np.abs(hydro_pub[ok] - repro[ok]).mean():.4f}")
for k, g in out.items():
    ok = mask & ~np.isnan(g) & ~np.isnan(hydro_pub)
    print(f"  {k:7s} : r avec couche publiée = {np.corrcoef(hydro_pub[ok], g[ok])[0, 1]:+.3f}, moyenne {np.nanmean(g):.3f}")

# ---------------------------------------------------------------- stats régionales
stats = {}
for i, n in enumerate(names):
    sel = (region == i + 1) & mask
    stats[n] = {"pub": round(float(np.nanmean(hydro_pub[sel])), 3), "cont": round(float(np.nanmean(out["cont"][sel])), 3),
                "norain": round(float(np.nanmean(out["norain"][sel])), 3), "jenks": round(float(np.nanmean(out["jenks"][sel])), 3)}
    print(f"  {n:10s} publiée {stats[n]['pub']:.2f}  continue {stats[n]['cont']:.2f}  sans pluie {stats[n]['norain']:.2f}  Jenks {stats[n]['jenks']:.2f}")

# ---------------------------------------------------------------- exports
OUT_TIF.mkdir(parents=True, exist_ok=True)
prof = dict(driver="GTiff", dtype="float32", width=W, height=H, count=1, crs="EPSG:4326", transform=T, nodata=-9999.0, compress="deflate")
for k, g in out.items():
    with rasterio.open(OUT_TIF / f"hydro_vuln_{k}.tif", "w", **prof) as d: d.write(np.where(np.isnan(g), -9999, g), 1)
def to_png8(a):
    q = np.where(np.isnan(a), 255, np.clip(np.round(a * 254), 0, 254)).astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(q, "L").save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
json.dump({
    "weights_v2": {k: round(float(v), 4) for k, v in W2.items()}, "r2_v2": round(float(r2), 4),
    "weights_norain": {k: round(float(v / sum(W_norain.values())), 4) for k, v in W_norain.items()},
    "published": "1 - (classes_v2 - 1)/3, classification par quartiles du score v2, moyennée à 0,05°",
    "jenks": {"thresholds": [round(t, 4) for t in jenks_t], "areas_pct": areas, "class_medians": [round(m, 4) for m in medians]},
    "grids": {k: to_png8(g) for k, g in out.items()}, "stats": stats,
}, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False)
print(f"{OUT_JSON.name} écrit ({OUT_JSON.stat().st_size // 1024} Ko) ; GeoTIFF dans {OUT_TIF}")
