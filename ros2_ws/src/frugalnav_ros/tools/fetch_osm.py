#!/usr/bin/env python3
"""
Fetches real building footprints of an Indian city from OpenStreetMap (the same data
source terraforge uses) and saves them as local-metre polygons for gen_maps.build_india().

    py fetch_osm.py

No heavy geo toolchain: one Overpass API query, a simple local projection, JSON out.
"""
import json, math, urllib.request, urllib.parse, os

CITY = "Jaipur"
LAT0, LON0 = 26.9239, 75.8267      # near Badi Chaupar, walled city, wide bazaar streets
R = 300.0                          # half size of the square area, metres

dlat = R / 110540.0
dlon = R / (111320.0 * math.cos(math.radians(LAT0)))
S, W, N, E = LAT0 - dlat, LON0 - dlon, LAT0 + dlat, LON0 + dlon
q = (f"[out:json][timeout:50];(way[\"building\"]({S:.6f},{W:.6f},{N:.6f},{E:.6f}););"
     f"out body;>;out skel qt;")

ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
raw = None
for ep in ENDPOINTS:
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                ep, data=("data=" + urllib.parse.quote(q)).encode(),
                headers={"User-Agent": "frugalnav-map-builder/1.0"})
            raw = json.load(urllib.request.urlopen(req, timeout=90))
            print("fetched from", ep)
            break
        except Exception as ex:
            print("  ", ep, "failed:", type(ex).__name__, str(ex)[:60])
    if raw is not None:
        break
if raw is None:
    raise SystemExit("all Overpass endpoints failed")

nodes = {e["id"]: e for e in raw["elements"] if e["type"] == "node"}
buildings = []
for e in raw["elements"]:
    if e.get("type") != "way" or "nodes" not in e:
        continue
    pts = []
    for nid in e["nodes"]:
        n = nodes.get(nid)
        if not n:
            continue
        x = (n["lon"] - LON0) * 111320.0 * math.cos(math.radians(LAT0))
        y = (n["lat"] - LAT0) * 110540.0
        pts.append([round(x, 2), round(y, 2)])
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) >= 3:
        lv = e.get("tags", {}).get("building:levels")
        h = None
        try:
            h = float(lv) * 3.2 if lv else None
        except ValueError:
            h = None
        buildings.append({"pts": pts, "h": h})

xs = [p[0] for b in buildings for p in b["pts"]]
ys = [p[1] for b in buildings for p in b["pts"]]
out = {"city": CITY, "lat": LAT0, "lon": LON0, "half_m": R, "buildings": buildings}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osm_city.json")
json.dump(out, open(path, "w"))
print(f"{CITY}: {len(buildings)} buildings")
if xs:
    print(f"extent x[{min(xs):.0f},{max(xs):.0f}] y[{min(ys):.0f},{max(ys):.0f}] metres")
print("saved", path)
