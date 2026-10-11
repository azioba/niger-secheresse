# Niger — Indice de Vulnérabilité à la Sécheresse (IVS)

Carte interactive : **https://azioba.github.io/niger-secheresse/**

Analyse multicritère (Weighted Index Overlay) à 0,05° (~5,5 km), recalculable dans le navigateur :
poids ajustables, scénarios, aléa sécheresse par anomalies (SPI CHIRPS 1990-2024), variantes de la
sensibilité hydrogéologique (publiée, Jenks, continue, sans pluie), statistiques régionales, export CSV.

## Données (ouvertes)
CHIRPS v2.0 · MODIS MOD13 (NDVI) · WorldPop · BGS Africa Groundwater Atlas · HydroSHEDS · GADM 4.1
(pipeline hydrogéologique : [hydrogeological-potential-mapping](https://github.com/azioba/hydrogeological-potential-mapping)).

## Contenu
- `index.html` — carte (autonome, aucune dépendance locale)
- `archive_v1_folium.html` — version initiale (Folium)
- `tools/` — gabarit, scripts de reconstruction et données embarquées dans la carte (JSON)

## Régénérer
```bash
python -I -X utf8 tools/build_drought_hazard.py   # SPI (nécessite l'archive CHIRPS mensuelle locale)
python -I -X utf8 tools/build_hydro_variants.py   # variantes hydro (nécessite les couches du pipeline)
python -I -X utf8 tools/build.py                  # assemble index.html depuis tools/
```
Les deux premiers scripts lisent des chemins locaux (`D:/Python/...`) à adapter et écrivent des GeoTIFF dans `data/derived/` (non publié) ; `build.py` ne dépend que de `tools/`.

## Limites
Voir l'onglet « Limites » de la carte : aridité vs sécheresse, double comptage de la pluie (corrigé par
la variante « sans pluie »), résolution 5,5 km, absence de validation externe (Cadre Harmonisé à faire).
