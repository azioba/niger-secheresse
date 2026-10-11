import re, json, base64, io, sys
from PIL import Image
import numpy as np

src = open(sys.argv[1], encoding="utf-8").read()
out = sys.argv[2]

# --- image overlays ---
imgs = re.findall(r'var (image_overlay_\w+) = L\.imageOverlay\(\s*"(data:image/png;base64,[^"]+)",\s*(\[\[[^\]]+\],\s*\[[^\]]+\]\])', src)
names = {}
for m in re.finditer(r'"([^"]+)" : (image_overlay_\w+),', src):
    names[m.group(2)] = json.loads('"' + m.group(1) + '"')
layers = []
for var, uri, bounds in imgs:
    raw = base64.b64decode(uri.split(",", 1)[1])
    im = Image.open(io.BytesIO(raw))
    a = np.array(im.convert("RGBA"))
    label = names.get(var, var)
    fname = re.sub(r'[^a-z0-9]+', '_', label.lower().encode('ascii', 'ignore').decode()).strip('_') + ".png"
    open(f"{out}/{fname}", "wb").write(raw)
    opaque = a[..., 3] > 0
    uniq = np.unique(a[opaque][:, :3], axis=0)
    print(var, "|", label, "|", im.size, im.mode, "| opaque px:", int(opaque.sum()), "| unique colors:", len(uniq), "| bounds", bounds)
    layers.append({"var": var, "label": label, "file": fname, "bounds": json.loads(bounds), "size": im.size})
json.dump(layers, open(f"{out}/layers.json", "w"), ensure_ascii=False, indent=1)

# --- geojson ---
for i, m in enumerate(re.finditer(r'geo_json_\w+_add\((\{"bbox".*?\})\);\s*\n', src)):
    gj = json.loads(m.group(1))
    nm = "regions.geojson" if len(gj["features"]) > 1 else "niger.geojson"
    json.dump(gj, open(f"{out}/{nm}", "w"))
    print(nm, len(gj["features"]), "features", [f["properties"].get("NAME_1") for f in gj["features"]])

# --- colormap ---
dom = re.search(r'\.domain\((\[[^\]]+\])\)\s*\.range\((\[[^\]]+\])\)', src)
domain = json.loads(dom.group(1)); rng = json.loads(dom.group(2).replace("'", '"'))
print("colormap:", len(domain), "thresholds,", len(rng), "colors; first/last", rng[0], rng[-1])
json.dump({"domain": domain, "range": rng}, open(f"{out}/colormap.json", "w"))

# --- stats ---
stats = {}
for h4, body in re.findall(r'<h4[^>]*>([^<]+)</h4>(.*?)</table>', src):
    stats[h4] = dict(re.findall(r'<tr><td><b>([^<]+)</b></td><td>([^<]+)</td></tr>', body))
json.dump(stats, open(f"{out}/stats.json", "w"), ensure_ascii=False, indent=1)
print("stats:", list(stats))
# markers
mk = re.findall(r'L\.marker\(\s*\[([\d.]+), ([\d.]+)\]', src)
print("markers:", mk)
