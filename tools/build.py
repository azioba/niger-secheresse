"""
Assemble index.html à partir de tools/template.html et des données de tools/ :
    analysis.json  grilles IVS + facteurs, régions simplifiées (produit par analyze.py)
    luts.json      tables de couleurs 256 pas
    drought.json   aléa SPI (produit par build_drought_hazard.py) — optionnel

Usage : python -I -X utf8 tools/build.py [sortie]   (défaut : index.html à la racine)
"""
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else TOOLS.parent / "index.html"

a = json.load(open(TOOLS / "analysis.json", encoding="utf-8"))
data = {k: a[k] for k in ("weights", "r2", "corr", "regions", "region_names", "grids", "regions_geojson", "niger_geojson", "bounds", "size")}
data["regions"] = [{k: r[k] for k in ("region", "area_km2", "centroid", "ivs_mean")} for r in data["regions"]]

luts = json.load(open(TOOLS / "luts.json"))
luts = {k: v for k, v in luts.items() if k != "ivs"}  # la rampe IVS est construite côté client (ColorBrewer RdYlBu 5 classes)

opt = lambda name: json.load(open(TOOLS / name, encoding="utf-8")) if (TOOLS / name).exists() else {}
drought, hydro = opt("drought.json"), opt("hydro.json")

tpl = open(TOOLS / "template.html", encoding="utf-8").read()
js = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
html = (tpl.replace("/*__DATA__*/{}", js(data)).replace("/*__LUTS__*/{}", js(luts))
           .replace("/*__DROUGHT__*/{}", js(drought)).replace("/*__HYDRO__*/{}", js(hydro)))
OUT.write_text(html, encoding="utf-8")
print(f"written {OUT} ({len(html.encode('utf-8')) / 1024:.0f} Ko ; aléa SPI : {'oui' if drought else 'non'} ; variantes hydro : {'oui' if hydro else 'non'})")
