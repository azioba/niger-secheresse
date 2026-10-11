"""Decode the colour-mapped PNGs back to values, recover WIO weights, compute regional statistics."""
import json, io, base64, itertools, sys
import numpy as np
from PIL import Image
from matplotlib import colormaps
from scipy.optimize import nnls
import shapely
from shapely.geometry import shape

D = sys.argv[1]
LAYERS = {
    "ivs":   ("ivs_vulnrabilit_scheresse.png", "RdYlBu_r"),
    "chirps":("dficit_chirps.png", "YlOrRd"),
    "ndvi":  ("stress_ndvi.png", "RdYlGn"),
    "hydro": ("vulnrabilit_hydrogo.png", "Blues"),
    "pop":   ("exposition_population.png", "YlOrRd"),
}
BOUNDS = [[11.7, 0.2], [23.5, 15.9]]  # [[south, west], [north, east]]

def lut(name):
    return (colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255)

def decode(png, cmap):
    a = np.array(Image.open(f"{D}/{png}").convert("RGBA"))
    mask = a[..., 3] > 0
    px = a[..., :3].astype(float)
    L = lut(cmap)
    # nearest colour index -> value in [0,1]
    flat = px[mask]
    idx = np.empty(len(flat), dtype=int)
    for s in range(0, len(flat), 20000):
        chunk = flat[s:s+20000]
        d = ((chunk[:, None, :] - L[None, :, :]) ** 2).sum(-1)
        idx[s:s+20000] = d.argmin(1)
    v = np.full(mask.shape, np.nan)
    v[mask] = idx / 255.0
    return v, mask

vals, masks = {}, {}
for k, (png, cm) in LAYERS.items():
    vals[k], masks[k] = decode(png, cm)
    print(k, "mean %.3f min %.3f max %.3f" % (np.nanmean(vals[k]), np.nanmin(vals[k]), np.nanmax(vals[k])))

# --- recover weights: IVS ~ sum(w_i * x_i), test both orientations of each sub-layer
common = masks["ivs"] & masks["chirps"] & masks["ndvi"] & masks["hydro"] & masks["pop"]
y = vals["ivs"][common]
best = None
for flips in itertools.product([False, True], repeat=4):
    X = np.column_stack([ (1 - vals[k][common]) if f else vals[k][common]
                          for k, f in zip(["chirps", "ndvi", "hydro", "pop"], flips)])
    w, rnorm = nnls(X, y)
    pred = X @ w
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    if best is None or r2 > best[0]:
        best = (r2, flips, w, np.abs(y - pred).mean())
r2, flips, w, mae = best
print("\nBest orientation (flip?)", dict(zip(["chirps", "ndvi", "hydro", "pop"], flips)))
print("Recovered weights:", dict(zip(["chirps", "ndvi", "hydro", "pop"], w.round(3))), "sum", w.sum().round(3), "R2 %.3f MAE %.3f" % (r2, mae))
corr = {k: float(np.corrcoef(vals[k][common], y)[0, 1]) for k in ["chirps", "ndvi", "hydro", "pop"]}
print("Pearson r with IVS:", {k: round(v, 3) for k, v in corr.items()})

# apply flips so that every layer is oriented "higher = worse"
for k, f in zip(["chirps", "ndvi", "hydro", "pop"], flips):
    if f:
        vals[k] = 1 - vals[k]

# --- geometry / pixel centres
H, W = masks["ivs"].shape
lat = np.linspace(BOUNDS[1][0], BOUNDS[0][0], H, endpoint=False) - (BOUNDS[1][0] - BOUNDS[0][0]) / H / 2
lon = np.linspace(BOUNDS[0][1], BOUNDS[1][1], W, endpoint=False) + (BOUNDS[1][1] - BOUNDS[0][1]) / W / 2
LON, LAT = np.meshgrid(lon, lat)
dlat = (BOUNDS[1][0] - BOUNDS[0][0]) / H
dlon = (BOUNDS[1][1] - BOUNDS[0][1]) / W
pix_km2 = (dlat * 111.32) * (dlon * 111.32 * np.cos(np.radians(LAT)))

regions = json.load(open(f"{D}/regions.geojson", encoding="utf-8"))
CLASSES = [(0, .2, "Faible"), (.2, .4, "Modérée"), (.4, .6, "Élevée"), (.6, .8, "Très élevée"), (.8, 1.01, "Extrême")]
rows = []
wmap = dict(zip(["chirps", "ndvi", "hydro", "pop"], w))
for f in regions["features"]:
    g = shape(f["geometry"])
    inside = shapely.contains_xy(g, LON, LAT) & masks["ivs"]
    name = f["properties"]["NAME_1"]
    iv = vals["ivs"][inside]
    r = {"region": name, "n_pixels": int(inside.sum()), "area_km2": float(pix_km2[inside].sum()),
         "ivs_mean": float(iv.mean()), "ivs_median": float(np.median(iv)), "ivs_p90": float(np.percentile(iv, 90)),
         "ivs_max": float(iv.max()), "ivs_std": float(iv.std())}
    for k in ["chirps", "ndvi", "hydro", "pop"]:
        vv = vals[k][inside]
        r[k + "_mean"] = float(np.nanmean(vv))
    contrib = {k: wmap[k] * r[k + "_mean"] for k in wmap}
    tot = sum(contrib.values())
    r["contrib"] = {k: float(v / tot) for k, v in contrib.items()}
    r["driver"] = max(contrib, key=contrib.get)
    r["classes_pct"] = {c[2]: float(((iv >= c[0]) & (iv < c[1])).mean() * 100) for c in CLASSES}
    r["pct_gt_0_6"] = float((iv >= 0.6).mean() * 100)
    r["pct_gt_0_8"] = float((iv >= 0.8).mean() * 100)
    r["centroid"] = [float(g.representative_point().y), float(g.representative_point().x)]
    rows.append(r)
rows.sort(key=lambda r: -r["ivs_mean"])
for i, r in enumerate(rows):
    r["rank"] = i + 1
    print(f'{r["rank"]} {r["region"]:10s} IVS {r["ivs_mean"]:.3f} p90 {r["ivs_p90"]:.3f} >0.6 {r["pct_gt_0_6"]:5.1f}% driver {r["driver"]:7s} '
          f'chirps {r["chirps_mean"]:.2f} ndvi {r["ndvi_mean"]:.2f} hydro {r["hydro_mean"]:.2f} pop {r["pop_mean"]:.2f} area {r["area_km2"]:.0f}')

# national
iv = vals["ivs"][masks["ivs"]]
national = {"ivs_mean": float(iv.mean()), "ivs_median": float(np.median(iv)),
            "classes_pct": {c[2]: float(((iv >= c[0]) & (iv < c[1])).mean() * 100) for c in CLASSES},
            "area_km2": float(pix_km2[masks["ivs"]].sum()),
            "hist": np.histogram(iv, bins=20, range=(0, 1))[0].tolist()}
print("national", national["ivs_mean"], national["classes_pct"])

# --- value grids as 8-bit grayscale PNGs (255 = nodata)
grids = {}
for k in LAYERS:
    v = vals[k]
    g = np.where(np.isnan(v), 255, np.clip(np.round(v * 254), 0, 254)).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(g, "L").save(buf, "PNG", optimize=True)
    grids[k] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    print(k, "value grid", len(buf.getvalue()), "bytes")

# --- region id grid (0 = outside, 1..8 = region index in regions.geojson order)
rid = np.zeros((H, W), np.uint8)
for i, f in enumerate(regions["features"]):
    rid[shapely.contains_xy(shape(f["geometry"]), LON, LAT) & masks["ivs"]] = i + 1
buf = io.BytesIO(); Image.fromarray(rid, "L").save(buf, "PNG", optimize=True)
grids["region"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
print("region grid", len(buf.getvalue()), "bytes; unassigned IVS pixels:", int((masks["ivs"] & (rid == 0)).sum()))
region_names = [f["properties"]["NAME_1"] for f in regions["features"]]

# --- classified IVS raster (5 classes) for display
palette = {"Faible": (44, 123, 182), "Modérée": (171, 217, 233), "Élevée": (255, 255, 191), "Très élevée": (253, 174, 97), "Extrême": (215, 25, 28)}
rgba = np.zeros((H, W, 4), np.uint8)
for lo, hi, nm in CLASSES:
    sel = masks["ivs"] & (vals["ivs"] >= lo) & (vals["ivs"] < hi)
    rgba[sel, :3] = palette[nm]; rgba[sel, 3] = 215
buf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
classified = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
print("classified png", len(buf.getvalue()))

# --- simplified geojson
def simplify(fc, tol):
    out = {"type": "FeatureCollection", "features": []}
    for f in fc["features"]:
        g = shape(f["geometry"]).simplify(tol, preserve_topology=True)
        gj = json.loads(shapely.to_geojson(g))
        # round coords
        def rnd(c):
            return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 4), round(c[1], 4)]
        gj["coordinates"] = rnd(gj["coordinates"])
        props = {k: f["properties"].get(k) for k in ("NAME_1", "VARNAME_1", "TYPE_1", "HASC_1") if f["properties"].get(k)}
        out["features"].append({"type": "Feature", "properties": props, "geometry": gj})
    return out
regions_s = simplify(regions, 0.01)
niger_s = simplify(json.load(open(f"{D}/niger.geojson", encoding="utf-8")), 0.01)
print("regions simplified", len(json.dumps(regions_s)), "niger", len(json.dumps(niger_s)))

json.dump({"weights": {k: float(v) for k, v in wmap.items()}, "r2": float(r2), "mae": float(mae), "corr": corr,
           "flips": dict(zip(["chirps", "ndvi", "hydro", "pop"], flips)),
           "regions": rows, "region_names": region_names, "national": national, "grids": grids,
           "regions_geojson": regions_s, "niger_geojson": niger_s, "bounds": BOUNDS, "size": [W, H]},
          open(f"{D}/analysis.json", "w", encoding="utf-8"), ensure_ascii=False)
