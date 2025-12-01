#!/usr/bin/env python3
from __future__ import annotations
import json, sys, zipfile
from pathlib import Path

ASSETS = Path("assets")
ASSETS.mkdir(parents=True, exist_ok=True)
OUT_GEOJSON = ASSETS / "taxi_zones.geojson"


def _ensure_unzipped(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(ASSETS)
        shp = next(iter(ASSETS.glob("*.shp")), None)
        if not shp:
            raise FileNotFoundError("No .shp found after unzip.")
        print(f"[convert] Unzipped into {ASSETS}")
        return shp
    elif input_path.suffix.lower() == ".shp":
        return input_path
    else:
        raise ValueError("Input must be a .zip or .shp")


def _to_geojson_wgs84(shp_path: Path, out_geojson: Path) -> int:
    import shapefile  # pyshp
    from pyproj import CRS, Transformer

    base = shp_path.with_suffix("")
    prj_path = base.with_suffix(".prj")

    # Source CRS from .prj if available; default to EPSG:2263 (TLC zones are in StatePlane feet)
    if prj_path.exists():
        src_crs = CRS.from_wkt(prj_path.read_text())
    else:
        src_crs = CRS.from_epsg(2263)

    dst_crs = CRS.from_epsg(4326)  # WGS84 lon/lat
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    shp = shapefile.Reader(str(base))
    fields = [f[0] for f in shp.fields[1:]]  # skip DeletionFlag
    features = []

    for sr in shp.iterShapeRecords():
        props = dict(zip(fields, sr.record))
        # Normalize LocationID
        loc = (props.get("LocationID") or props.get("locationid")
               or props.get("location_id") or props.get("LOCATIONID"))
        try:
            props["LocationID"] = int(loc)
        except Exception:
            # best effort
            try:
                props["LocationID"] = int(float(str(loc)))
            except Exception:
                props["LocationID"] = loc

        # Reproject coordinates to lon/lat
        pts = sr.shape.points
        parts = list(sr.shape.parts) + [len(pts)]

        rings_wgs84 = []
        for i in range(len(parts) - 1):
            ring = pts[parts[i]:parts[i + 1]]
            if not ring:
                continue
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            lons, lats = transformer.transform(xs, ys)
            rings_wgs84.append([[float(lon), float(lat)] for lon, lat in zip(lons, lats)])

        if not rings_wgs84:
            geom = {"type": "GeometryCollection", "geometries": []}
        elif len(rings_wgs84) == 1:
            geom = {"type": "Polygon", "coordinates": [rings_wgs84[0]]}
        else:
            geom = {"type": "MultiPolygon", "coordinates": [[r] for r in rings_wgs84]}

        features.append({"type": "Feature", "properties": props, "geometry": geom})

    gj = {"type": "FeatureCollection", "features": features}
    out_geojson.write_text(json.dumps(gj))
    print(f"[convert] Wrote {out_geojson} with {len(features)} features (WGS84)")
    return len(features)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python convert_taxi_zones_wgs84.py assets/taxi_zones.zip|.shp")
        return 2
    src = Path(argv[1])
    shp_path = _ensure_unzipped(src)
    _ = _to_geojson_wgs84(shp_path, OUT_GEOJSON)
    # sanity: count unique LocationIDs
    gj = json.loads(OUT_GEOJSON.read_text())
    ids = {f.get("properties", {}).get("LocationID") for f in gj.get("features", [])}
    ids.discard(None)
    print(f"[convert] Unique LocationIDs: {len(ids)} (expected ~263–265)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
