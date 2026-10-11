"""
Aléa sécheresse par anomalies (SPI) sur la grille IVS 0,05° — Niger, 1990-2024.

Reprend les fonctions SPI du module sécheresse de geoai-niger (ajustement gamma,
McKee et al. 1993 / Thom 1958) et les applique à l'archive CHIRPS mensuelle locale :
    - référence climatologique : 1981-2024 (44 ans, recommandation OMM >= 30 ans)
    - années analysées        : 1990-2024 (35 saisons)
    - fenêtres                : SPI-3 (JAS, coeur de l'hivernage), SPI-6 (avr-sept,
                                saison complète), SPI-12 (année civile)

Couches produites (grille 314 x 236, identique à l'IVS, masque IVS) :
    freq{3,6,12}      fréquence des saisons avec SPI <= -1   (modérée ou pire)
    freqsev{3,6,12}   fréquence des saisons avec SPI <= -1.5 (sévère ou pire)
    sev{3,6,12}       sévérité ordinale moyenne (0-3 McKee) ramenée sur 0-1
    trend6            tendance linéaire du SPI-6 (unités SPI par décennie)
    worst6            année du SPI-6 minimal
    spi6_YYYY         SPI-6 de chaque saison (pile temporelle pour le curseur)

Sorties : data/derived/*.tif (GeoTIFF WGS84) et tools/drought.json (grilles 8 bits
en PNG base64 + séries régionales), consommé par tools/build.py.

Usage : python -I -X utf8 tools/build_drought_hazard.py
"""
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
GEOAI = Path("D:/Python/geoai-niger")
CHIRPS_DIR = Path("D:/Python/Niger_Hydrogéologie/python/data/raw/chirps_monthly")
OUT_TIF = ROOT / "data" / "derived"
OUT_JSON = ROOT / "tools" / "drought.json"

sys.path.insert(0, str(GEOAI))
from modules.drought.indicators import compute_spi_gamma, spi_severity  # noqa: E402

BBOX = (0.2, 11.7, 15.9, 23.5)  # W, S, E, N — emprise de la grille IVS
W, H = 314, 236
REF_YEARS = list(range(1981, 2025))
AN_YEARS = list(range(1990, 2025))
WINDOWS = {"spi3": (7, 8, 9), "spi6": (4, 5, 6, 7, 8, 9), "spi12": tuple(range(1, 13))}
SEV_T = (-1.0, -1.5, -2.0)

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

# ---------------------------------------------------------------- masque IVS
analysis = json.load(open(ROOT / "tools" / "analysis.json", encoding="utf-8"))
def png_to_array(uri):
    return np.array(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))
mask = png_to_array(analysis["grids"]["ivs"]) != 255
region_grid = png_to_array(analysis["grids"]["region"])
region_names = analysis["region_names"]
assert mask.shape == (H, W)

# ---------------------------------------------------------------- lecture CHIRPS
with rasterio.open(CHIRPS_DIR / "chirps-v2.0.2000.01.tif") as src:
    win = src.window(*BBOX)
    win = Window(round(win.col_off), round(win.row_off), W, H)
    log(f"Fenêtre CHIRPS : {win} ; grille source {src.width}x{src.height} @ {src.res[0]:.3f}°")

monthly = {}
for y in REF_YEARS:
    for m in range(1, 13):
        p = CHIRPS_DIR / f"chirps-v2.0.{y}.{m:02d}.tif"
        if not p.exists():
            raise FileNotFoundError(p)
        with rasterio.open(p) as src:
            a = src.read(1, window=win).astype(np.float32)
        monthly[(y, m)] = np.where(a < 0, np.nan, a)
log(f"{len(monthly)} rasters mensuels chargés ({REF_YEARS[0]}-{REF_YEARS[-1]})")

def accum(y, months):
    return np.nansum(np.stack([monthly[(y, m)] for m in months]), axis=0).astype(np.float32)

# ---------------------------------------------------------------- fiabilité du SPI en zone hyper-aride
# Le SPI n'est pas interprétable là où la saison des pluies est quasi inexistante :
# l'ajustement gamma porte sur une poignée de millimètres et la fréquence de cumuls
# nuls domine (Wu et al. 2007 ; Stagge et al. 2015). Critère retenu : pluie moyenne
# avril-septembre >= 50 mm ET moins de 30 % de saisons à cumul nul sur 1981-2024.
PMIN_MM, QMAX = 50.0, 0.30
ref6 = np.stack([accum(y, WINDOWS["spi6"]) for y in REF_YEARS])
pmean6 = np.where(mask, np.nanmean(ref6, axis=0), np.nan).astype(np.float32)
qzero6 = np.where(mask, np.nanmean(ref6 <= 0, axis=0), np.nan).astype(np.float32)
reliable = mask & (pmean6 >= PMIN_MM) & (qzero6 <= QMAX)
log(f"Fiabilité SPI : {reliable.sum()} / {mask.sum()} cellules ({reliable.sum() / mask.sum() * 100:.1f} %) ; "
    f"exclues : pluie avr-sept < {PMIN_MM:.0f} mm ou > {QMAX:.0%} de saisons nulles")

# ---------------------------------------------------------------- SPI par fenêtre
results = {"pmean6": pmean6}
yearly = {}
for key, months in WINDOWS.items():
    ref_stack = ref6 if key == "spi6" else np.stack([accum(y, months) for y in REF_YEARS])  # (44, H, W)
    spi_years = {}
    for y in AN_YEARS:
        spi_years[y] = compute_spi_gamma(ref_stack[REF_YEARS.index(y)], ref_stack, mask=reliable)
    stack = np.stack([spi_years[y] for y in AN_YEARS])  # (35, H, W)
    sev = np.stack([spi_severity(spi_years[y], SEV_T) for y in AN_YEARS])
    with np.errstate(invalid="ignore"):
        results[f"freq{key[3:]}"] = np.nanmean(stack <= -1.0, axis=0)
        results[f"freqsev{key[3:]}"] = np.nanmean(stack <= -1.5, axis=0)
        results[f"sev{key[3:]}"] = np.nanmean(sev, axis=0) / 3.0
    yearly[key] = stack
    nat = [float(np.nanmean(spi_years[y][mask])) for y in AN_YEARS]
    log(f"{key}: moyenne nationale min {min(nat):+.2f} ({AN_YEARS[int(np.argmin(nat))]}), "
        f"max {max(nat):+.2f} ({AN_YEARS[int(np.argmax(nat))]}) ; freq<=-1 moyenne {np.nanmean(results[f'freq{key[3:]}'][mask]):.3f}")

# tendance SPI-6 (pente OLS par pixel, en SPI/décennie) et année la plus sèche
s6 = yearly["spi6"]
t = np.array(AN_YEARS, dtype=float)
tc = t - t.mean()
with np.errstate(invalid="ignore"):
    slope = np.nansum(tc[:, None, None] * (s6 - np.nanmean(s6, axis=0)), axis=0) / (tc ** 2).sum()
results["trend6"] = np.where(reliable, slope * 10.0, np.nan)
worst_idx = np.nanargmin(np.where(np.isnan(s6), np.inf, s6), axis=0)
results["worst6"] = np.where(reliable, np.array(AN_YEARS)[worst_idx], np.nan)
with np.errstate(invalid="ignore"):
    results["min6"] = np.where(reliable, np.nanmin(np.where(np.isnan(s6), np.inf, s6), axis=0), np.nan)
for k in list(results):
    results[k] = np.where(mask if k == "pmean6" else reliable, results[k], np.nan).astype(np.float32)
log(f"trend6 : médiane {np.nanmedian(results['trend6']):+.3f} SPI/décennie ; "
    f"années les plus sèches dominantes : "
    + ", ".join(f"{int(v)} ({int(c)} px)" for v, c in zip(*np.unique(results['worst6'][reliable], return_counts=True)) if c > 800))

# ---------------------------------------------------------------- séries régionales
series = {"years": AN_YEARS, "national": {}, "regions": {}}
def summarize(sel):
    out = {}
    for key in WINDOWS:
        st = yearly[key][:, sel]
        out[key] = [round(float(np.nanmean(st[i])), 3) for i in range(len(AN_YEARS))]
        out[key + "_pct_dry"] = [round(float(np.nanmean(st[i] <= -1.0) * 100), 1) for i in range(len(AN_YEARS))]
    out["freq6"] = round(float(np.nanmean(results["freq6"][sel])), 4)
    out["freqsev6"] = round(float(np.nanmean(results["freqsev6"][sel])), 4)
    out["sev6"] = round(float(np.nanmean(results["sev6"][sel])), 4)
    out["trend6"] = round(float(np.nanmean(results["trend6"][sel])), 4)
    wv = results["worst6"][sel]; wv = wv[~np.isnan(wv)]
    out["worst_years"] = [int(y) for y, c in sorted(zip(*np.unique(wv, return_counts=True)), key=lambda x: -x[1])[:3]] if wv.size else []
    out["reliable_pct"] = round(float(reliable[sel].mean() * 100), 1)
    out["pmean6_mm"] = round(float(np.nanmean(pmean6[sel])), 0)
    return out
series["national"] = summarize(mask)
for i, name in enumerate(region_names):
    series["regions"][name] = summarize((region_grid == i + 1) & mask)
    r = series["regions"][name]
    log(f"  {name:10s} fiable {r['reliable_pct']:5.1f}% pluie {r['pmean6_mm']:4.0f} mm | freq6 {r['freq6']:.2f} sev6 {r['sev6']:.2f} "
        f"trend {r['trend6']:+.2f}/déc. pires années {r['worst_years']}")

# ---------------------------------------------------------------- exports
OUT_TIF.mkdir(parents=True, exist_ok=True)
transform = from_bounds(*BBOX, W, H)
prof = dict(driver="GTiff", dtype="float32", width=W, height=H, crs="EPSG:4326",
            transform=transform, nodata=-9999.0, compress="deflate")
for k, arr in results.items():
    with rasterio.open(OUT_TIF / f"{k}.tif", "w", count=1, **prof) as dst:
        dst.write(np.where(np.isnan(arr), -9999.0, arr).astype(np.float32), 1)
for key in WINDOWS:
    with rasterio.open(OUT_TIF / f"{key}_1990_2024.tif", "w", count=len(AN_YEARS), **prof) as dst:
        for i, y in enumerate(AN_YEARS):
            dst.write(np.where(np.isnan(yearly[key][i]), -9999.0, yearly[key][i]).astype(np.float32), i + 1)
            dst.set_band_description(i + 1, str(y))
log(f"GeoTIFF écrits dans {OUT_TIF}")

def to_png8(arr, lo, hi):
    """Quantifie [lo, hi] sur 0..254, 255 = nodata ; PNG niveaux de gris en base64."""
    q = np.where(np.isnan(arr), 255, np.clip(np.round((arr - lo) / (hi - lo) * 254), 0, 254)).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(q, "L").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

grids, ranges = {}, {}
for k in ["freq3", "freqsev3", "sev3", "freq6", "freqsev6", "sev6", "freq12", "freqsev12", "sev12"]:
    grids[k] = to_png8(results[k], 0, 1); ranges[k] = [0, 1]
grids["trend6"] = to_png8(results["trend6"], -1, 1); ranges["trend6"] = [-1, 1]
grids["pmean6"] = to_png8(results["pmean6"], 0, 800); ranges["pmean6"] = [0, 800]
grids["min6"] = to_png8(results["min6"], -3, 0); ranges["min6"] = [-3, 0]
grids["worst6"] = to_png8(results["worst6"], AN_YEARS[0], AN_YEARS[0] + 254); ranges["worst6"] = [AN_YEARS[0], AN_YEARS[0] + 254]
for i, y in enumerate(AN_YEARS):
    grids[f"spi6_{y}"] = to_png8(yearly["spi6"][i], -3, 3)
ranges["spi6"] = [-3, 3]

payload = {
    "method": {
        "reference_years": [REF_YEARS[0], REF_YEARS[-1]], "analysis_years": [AN_YEARS[0], AN_YEARS[-1]],
        "windows": {k: list(v) for k, v in WINDOWS.items()}, "severity_thresholds": list(SEV_T),
        "source": "CHIRPS v2.0 mensuel (UCSB/CHC), grille native 0,05°, fenêtre col 404 / ligne 330",
        "spi": "ajustement gamma (Thom 1958) + mélange avec la fréquence de mois secs (McKee 1993), code geoai-niger/modules/drought/indicators.py",
        "reliability": {"pmin_mm": PMIN_MM, "qmax": QMAX, "reliable_cells": int(reliable.sum()), "total_cells": int(mask.sum())},
    },
    "ranges": ranges, "grids": grids, "series": series,
}
json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False)
log(f"{OUT_JSON.name} écrit : {OUT_JSON.stat().st_size / 1024:.0f} Ko, {len(grids)} grilles")
