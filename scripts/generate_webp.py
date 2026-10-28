#!/usr/bin/env python3
"""HymecNG + HX-Hybrid im NATIVEN DWD-Raster (1100 x 1200, stereographisch).

Kein Warp nach Web Mercator. Ausgabe: rohe uint8-Bytes, keine Dateiendung.
Layout: 1200 Zeilen x 1100 Spalten, row-major, Zeile 0 = Norden (wie im HDF5).
Dateigroesse immer exakt xsize * ysize = 1.320.000 Bytes.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import requests
from pyproj import Transformer
from scipy import ndimage

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
SRC_DIR = Path("data/hymecng")
HX_SRC_DIR = Path("data/hx")
OUT_DIR = Path("output/hymecng")

FILENAME_RE = re.compile(r"composite_HymecNG_(\d{8})_(\d{4})_(\d{3})-hd5")

# Code-Bedeutung in der Ausgabedatei:
#   255 = kein Daten / ausserhalb      1 = Regen ohne HX-Wert (unsichtbar)
#   30 Niesel, 31 leicht, 32 maessig, 33 stark
#   4/5 gefrierender Regen             61/62 Schneeregen leicht / maessig+
#   71/72/73 Schnee leicht/maessig/stark
#   8 Graupel, 9/10 Hagel
#   200 Blitz, 201 Blitz in Hagelzone
HAIL_CLASSES = {9, 10}

RAIN_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.08, 0.1, 30),
    (0.1, 1.25, 31),
    (1.25, 10.0, 32),
    (10.0, float("inf"), 33),
]
SNOW_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.08, 1.0, 71),
    (1.0, 4.0, 72),
    (4.0, float("inf"), 73),
]
SLEET_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.08, 1.0, 61),
    (1.0, float("inf"), 62),
]

SNOW_TYPE_CODES = {7, 71, 72, 73}
SLEET_TYPE_CODES = {6, 61, 62}
FREEZING_RAIN_TYPE_CODES = {4, 5}
HAIL_TYPE_CODES = {9, 10}

MIN_PRECIP_RATE_MMH = 0.08

NODATA_CLASS = -1
NODATA_BYTE = 255
THUNDER_CODE = 200
STRONG_THUNDER_CODE = 201

LIGHTNING_BASE_URL = "https://radar.wetterstation-neustadt.de/blitze/archive/"
LIGHTNING_BACKUP_URL = "https://nowsky.vercel.app/api/lightning"
LIGHTNING_WINDOW_MINUTES = 5
# Native Pixel sind 1 km. Frueher 8 px im Web-Mercator-Bild (~0,65 km/px bei 50 N)
# -> ca. 5 km Radius -> hier 5 Pixel.
LIGHTNING_MARKER_RADIUS_PX = 5
LIGHTNING_PRECIP_MMH_THRESHOLD = 0.1

BERLIN = ZoneInfo("Europe/Berlin")

# HX (250 m, dBZ) -> mm/h ueber Z-R-Beziehung, danach 4x4-Mittelung auf 1 km
ZR_A = 256.0            # DWD/RADOLAN: Z = a * R^b
ZR_B = 1.42
HX_FACTOR = 4           # 250 m -> 1 km
HX_MAX_DBZ: float | None = 55.0   # Kappung (Hagel-Kontamination); None = aus
HX_DEFAULT_GAIN = 0.5            # nur Fallback
HX_DEFAULT_OFFSET = -32.5        # nur Fallback
HX_DEFAULT_NODATA = 65535.0

PRECIP_SOURCE_CODES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
REFINED_PRECIP_CODES = [30, 31, 32, 33, 61, 62, 71, 72, 73]
ALL_PRECIP_CODES = PRECIP_SOURCE_CODES + REFINED_PRECIP_CODES


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def parse_timestamp(filename: str) -> datetime:
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Dateiname passt nicht zum Schema: {filename}")
    date_str, time_str, _ = m.groups()
    naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return naive.replace(tzinfo=timezone.utc)


def hx_path_for(ts: datetime) -> Path:
    return HX_SRC_DIR / f"composite_hx_{ts:%Y%m%d}_{ts:%H%M}-hd5"


def dbz_to_rain(dbz):
    return (10.0 ** (dbz / 10.0) / ZR_A) ** (1.0 / ZR_B)


_DATA_PATH_RE = re.compile(r"(^|/)dataset\d+/data\d+/data$")


def _find_2d_dataset_by_quantity(h5file: h5py.File, keywords: tuple[str, ...], error_msg: str) -> h5py.Dataset:
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and _DATA_PATH_RE.search(name):
            candidates.append(obj)

    h5file.visititems(visitor)

    if not candidates:
        def loose_visitor(name, obj):
            if (
                isinstance(obj, h5py.Dataset)
                and name.endswith("/data")
                and obj.ndim == 2
                and "quality" not in name
            ):
                candidates.append(obj)

        h5file.visititems(loose_visitor)

    if not candidates:
        raise RuntimeError(error_msg)

    matches: list[h5py.Dataset] = []
    for ds in candidates:
        what = ds.parent.get("what")
        if what is not None and "quantity" in what.attrs:
            quantity = what.attrs["quantity"]
            if isinstance(quantity, bytes):
                quantity = quantity.decode(errors="ignore")
            if any(k in str(quantity).upper() for k in keywords):
                matches.append(ds)

    if matches:
        if len(matches) > 1:
            print(f"Warnung: {len(matches)} Datasets passen auf {keywords} — nehme {matches[0].name}.", file=sys.stderr)
        return matches[0]

    candidates.sort(key=lambda ds: ds.name)
    chosen = candidates[0]
    print(
        f"Warnung: keine Quantity passte auf {keywords} — nehme Fallback-Dataset {chosen.name}. "
        f"Bitte pruefen!",
        file=sys.stderr,
    )
    return chosen


def find_classification_dataset(h5file: h5py.File) -> h5py.Dataset:
    return _find_2d_dataset_by_quantity(
        h5file,
        keywords=("CLASS", "PRECIP", "HCLASS", "TYPE"),
        error_msg="Kein 2D-Datensatz in der HymecNG-Datei gefunden.",
    )


def find_hx_dataset(h5file: h5py.File) -> h5py.Dataset:
    return _find_2d_dataset_by_quantity(
        h5file,
        keywords=("DBZ", "TH", "REFL"),
        error_msg="Kein 2D-Datensatz in der HX-Datei gefunden.",
    )


def _attr_float(attrs, key: str, default: float | None = None) -> float | None:
    v = attrs.get(key)
    if v is None:
        return default
    return float(np.asarray(v).ravel()[0])


def _attr_str(attrs, key: str, default: str = "") -> str:
    v = attrs.get(key)
    if v is None:
        return default
    if isinstance(v, bytes):
        return v.decode(errors="ignore")
    return str(v)


def load_classification_array(dataset: h5py.Dataset, nodata_class: int) -> np.ndarray:
    """Liest die HymecNG-Klassifikation (nativ) und maskiert nodata/undetect."""
    raw = dataset[()]
    what = dataset.parent.get("what")
    attrs = what.attrs if what is not None else {}

    nodata = _attr_float(attrs, "nodata")
    undetect = _attr_float(attrs, "undetect")

    class_arr = raw.astype(np.int32)
    mask_invalid = np.zeros(class_arr.shape, dtype=bool)
    if nodata is not None:
        mask_invalid |= raw == nodata
    if undetect is not None:
        mask_invalid |= raw == undetect
    class_arr[mask_invalid] = nodata_class
    return class_arr


def find_where_group(h5file: h5py.File) -> h5py.Group | None:
    required = ("projdef", "xsize", "ysize", "xscale", "yscale", "LL_lon", "LL_lat")

    def complete(grp) -> bool:
        return all(k in grp.attrs for k in required)

    root_where = h5file.get("where")
    if root_where is not None and complete(root_where):
        return root_where

    found: list[h5py.Group] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where" and complete(obj):
            found.append(obj)

    h5file.visititems(visitor)
    return found[0] if found else None


def extract_grid_info(where: h5py.Group) -> dict:
    def as_str(key: str) -> str:
        v = where.attrs[key]
        return v.decode() if isinstance(v, bytes) else str(v)

    def as_float(key: str) -> float:
        return float(where.attrs[key])

    return {
        "projdef": as_str("projdef"),
        "xsize": int(as_float("xsize")),
        "ysize": int(as_float("ysize")),
        "xscale": as_float("xscale"),
        "yscale": as_float("yscale"),
        "ll_lon": as_float("LL_lon"),
        "ll_lat": as_float("LL_lat"),
    }


# --------------------------------------------------------------------------- #
# Raster-Geometrie (nativ)
# --------------------------------------------------------------------------- #
def lonlat_to_native_rowcol(lon, lat, grid: dict, to_proj: Transformer):
    """lon/lat (Skalar oder Array) -> (row, col) im nativen Raster.
    Zeile 0 = Norden. Rueckgabe als float-Pixelindizes (floor fuer Pixel)."""
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    x, y = to_proj.transform(lon, lat)
    col = (np.asarray(x) - ll_x) / grid["xscale"]
    row = grid["ysize"] - 1 - (np.asarray(y) - ll_y) / grid["yscale"]
    return row, col


# --------------------------------------------------------------------------- #
# HX laden (250 m, dBZ) -> mm/h auf dem 1-km-Raster
# --------------------------------------------------------------------------- #
def load_hx_rate_native(hx_path: Path, class_grid: dict) -> np.ndarray:
    """HX-Reflektivitaet -> Regenrate in mm/h auf dem HymecNG-Raster.

    Reihenfolge: dBZ -> mm/h (pro 250-m-Pixel), dann Mittel ueber 4x4 Pixel.
    undetect = 0 mm/h (zaehlt im Mittel mit), nodata = NaN (zaehlt nicht mit).
    """
    with h5py.File(hx_path, "r") as f:
        ds = find_hx_dataset(f)
        raw = ds[()]
        what = None
        for grp in (ds.parent, ds.parent.parent, ds.file):
            w_ = grp.get("what")
            if w_ is not None and "gain" in w_.attrs:
                what = w_
                break
        attrs = what.attrs if what is not None else {}
        gain = _attr_float(attrs, "gain", HX_DEFAULT_GAIN)
        offset = _attr_float(attrs, "offset", HX_DEFAULT_OFFSET)
        nodata = _attr_float(attrs, "nodata", HX_DEFAULT_NODATA)
        undetect = _attr_float(attrs, "undetect")

        where = find_where_group(f)
        if where is None:
            raise RuntimeError("Keine 'where'-Projektionsinfo in der HX-Datei gefunden.")
        hx_grid = extract_grid_info(where)

    h, w = raw.shape
    fct = HX_FACTOR
    if (w, h) != (class_grid["xsize"] * fct, class_grid["ysize"] * fct):
        raise RuntimeError(
            f"HX-Raster {w}x{h} passt nicht zum {fct}x{fct}-Faktor auf "
            f"{class_grid['xsize']}x{class_grid['ysize']}."
        )
    if abs(hx_grid["xscale"] * fct - class_grid["xscale"]) > 1e-6:
        raise RuntimeError("HX-Pixelgroesse x 4 entspricht nicht der HymecNG-Pixelgroesse.")

    dbz = raw.astype(np.float32) * np.float32(gain) + np.float32(offset)
    if HX_MAX_DBZ is not None:
        np.minimum(dbz, np.float32(HX_MAX_DBZ), out=dbz)
    rate = dbz_to_rain(dbz).astype(np.float32)
    if undetect is not None:
        rate[raw == undetect] = 0.0
    if nodata is not None:
        rate[raw == nodata] = np.nan
    del dbz

    blocks = rate.reshape(h // fct, fct, w // fct, fct)
    valid = np.isfinite(blocks)
    cnt = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
    out = np.full(cnt.shape, np.nan, dtype=np.float64)
    ok = cnt > 0
    out[ok] = total[ok] / cnt[ok]

    print(f"HX-Composite: {hx_path.name} ({w}x{h} -> {out.shape[1]}x{out.shape[0]}, mm/h)")
    return out


# --------------------------------------------------------------------------- #
# HYBRID-STRATEGIE: HX fuer Detektion + Intensitaet, HymecNG fuer Art
# --------------------------------------------------------------------------- #
def refine_with_hybrid_strategy(
    class_arr: np.ndarray,
    rate: np.ndarray | None,
) -> np.ndarray:
    refined = class_arr.copy()

    if rate is None:
        mask_rain = class_arr == 3
        mask_snow = class_arr == 7
        mask_sleet = class_arr == 6
        refined[mask_rain] = 31
        refined[mask_snow] = SNOW_MMH_THRESHOLDS[0][2]
        refined[mask_sleet] = SLEET_MMH_THRESHOLDS[0][2]
        if mask_rain.any() or mask_snow.any() or mask_sleet.any():
            print("Warnung: HX nicht verfuegbar, verwende niedrigste Intensitaetsstufe als Fallback.", file=sys.stderr)
        return refined

    has_rain = ~np.isnan(rate) & (rate >= MIN_PRECIP_RATE_MMH)

    # Regen (3)
    mask_refine_rain = (class_arr == 3) & has_rain
    for lower, upper, new_code in RAIN_MMH_THRESHOLDS:
        m = mask_refine_rain & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    m = (class_arr == 3) & has_rain & (refined == 3)
    refined[m] = RAIN_MMH_THRESHOLDS[0][2]

    # Schnee (7)
    mask_refine_snow = (class_arr == 7) & has_rain
    for lower, upper, new_code in SNOW_MMH_THRESHOLDS:
        m = mask_refine_snow & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    m = (class_arr == 7) & has_rain & (refined == 7)
    refined[m] = SNOW_MMH_THRESHOLDS[0][2]

    # Schneeregen (6)
    mask_refine_sleet = (class_arr == 6) & has_rain
    for lower, upper, new_code in SLEET_MMH_THRESHOLDS:
        m = mask_refine_sleet & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    m = (class_arr == 6) & has_rain & (refined == 6)
    refined[m] = SLEET_MMH_THRESHOLDS[0][2]

    # Naechstgelegenen bekannten Niederschlagstyp bestimmen
    known_type = np.isin(class_arr, PRECIP_SOURCE_CODES)
    if known_type.any():
        _, (ir, ic) = ndimage.distance_transform_edt(~known_type, return_indices=True)
        nearest_type = class_arr[ir, ic]
    else:
        nearest_type = np.full(class_arr.shape, 3, dtype=class_arr.dtype)

    is_snow_type = np.isin(nearest_type, tuple(SNOW_TYPE_CODES))
    is_sleet_type = np.isin(nearest_type, tuple(SLEET_TYPE_CODES))
    is_preserved_type = np.isin(nearest_type, tuple(FREEZING_RAIN_TYPE_CODES | {8} | HAIL_TYPE_CODES))
    is_rain_type = ~is_snow_type & ~is_sleet_type & ~is_preserved_type

    # Luecken fuellen (HX hat Niederschlag, HymecNG nichts Konkretes)
    gap = has_rain & ~np.isin(class_arr, PRECIP_SOURCE_CODES)

    for lower, upper, new_code in RAIN_MMH_THRESHOLDS:
        m = gap & is_rain_type & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    for lower, upper, new_code in SNOW_MMH_THRESHOLDS:
        m = gap & is_snow_type & (rate >= lower) & (rate < upper)
        refined[m] = new_code
    for lower, upper, new_code in SLEET_MMH_THRESHOLDS:
        m = gap & is_sleet_type & (rate >= lower) & (rate < upper)
        refined[m] = new_code

    m = gap & is_preserved_type
    refined[m] = nearest_type[m]

    return refined


# --------------------------------------------------------------------------- #
# Blitze (nativ)
# --------------------------------------------------------------------------- #
def _window_ms(ts: datetime, minutes: int) -> tuple[int, int]:
    end = int(ts.timestamp() * 1000)
    return end - minutes * 60_000, end


def _fetch_strikes_primary(ts: datetime, minutes: int) -> list[tuple[float, float]]:
    ts_local = ts.astimezone(BERLIN)
    url = f"{LIGHTNING_BASE_URL}{ts_local:%Y-%m-%d-%H%M}.json"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        print(f"Warnung: Primaerer Blitz-Feed liefert 404 ({url}) - nutze Backup-API.", file=sys.stderr)
        raise FileNotFoundError(url)
    resp.raise_for_status()

    start_ms, end_ms = _window_ms(ts, minutes)
    return [
        (s["lat"], s["lon"])
        for s in resp.json().get("strikes", [])
        if start_ms <= s.get("t", 0) <= end_ms
    ]


def _fetch_strikes_backup(ts: datetime, minutes: int) -> list[tuple[float, float]]:
    resp = requests.get(LIGHTNING_BACKUP_URL, timeout=30)
    resp.raise_for_status()

    start_ms, end_ms = _window_ms(ts, minutes)
    strikes = []
    for s in resp.json().get("strikes", []):
        t_ms = int(datetime.fromisoformat(s["time"].replace("Z", "+00:00")).timestamp() * 1000)
        if start_ms <= t_ms <= end_ms:
            strikes.append((s["lat"], s["lon"]))
    return strikes


def fetch_recent_strikes(ts: datetime, minutes: int = LIGHTNING_WINDOW_MINUTES) -> list[tuple[float, float]]:
    try:
        return _fetch_strikes_primary(ts, minutes)
    except FileNotFoundError:
        return _fetch_strikes_backup(ts, minutes)


def apply_lightning_overlay(
    out_codes: np.ndarray,        # uint8, nativ, wird in-place veraendert
    class_native: np.ndarray,     # verfeinerte Klassen (int32), nativ
    rate_native: np.ndarray | None,
    ts: datetime,
    grid: dict,
    to_proj: Transformer,
) -> int:
    strikes = fetch_recent_strikes(ts)
    print(f"{len(strikes)} Blitze in den letzten {LIGHTNING_WINDOW_MINUTES} Minuten geladen.")

    h, w = class_native.shape
    radius = LIGHTNING_MARKER_RADIUS_PX

    precip_mask = np.isin(class_native, ALL_PRECIP_CODES)
    if rate_native is not None:
        rate_ok = np.isnan(rate_native) | (rate_native >= LIGHTNING_PRECIP_MMH_THRESHOLD)
        precip_mask &= rate_ok

    offsets = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    circle = dr * dr + dc * dc <= radius * radius

    hits = 0
    for lat, lon in strikes:
        row_f, col_f = lonlat_to_native_rowcol(lon, lat, grid, to_proj)
        row = int(np.floor(row_f)) + 1
        col = int(np.floor(col_f))
        if not (0 <= col < w and 0 <= row < h):
            continue
        if not precip_mask[row, col]:
            continue

        r0, r1 = max(0, row - radius), min(h, row + radius + 1)
        c0, c1 = max(0, col - radius), min(w, col + radius + 1)
        circ = circle[r0 - (row - radius): r1 - (row - radius),
                      c0 - (col - radius): c1 - (col - radius)]

        area = precip_mask[r0:r1, c0:c1] & circ
        hail = np.isin(class_native[r0:r1, c0:c1], tuple(HAIL_CLASSES))
        out_codes[r0:r1, c0:c1][area & ~hail] = THUNDER_CODE
        out_codes[r0:r1, c0:c1][area & hail] = STRONG_THUNDER_CODE
        hits += 1
    return hits


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    candidates = sorted(p for p in SRC_DIR.glob("composite_HymecNG_*-hd5") if FILENAME_RE.match(p.name))
    if not candidates:
        sys.exit(f"Keine HymecNG-Datei in {SRC_DIR} gefunden.")
    src_path = candidates[-1]
    ts = parse_timestamp(src_path.name)

    with h5py.File(src_path, "r") as f:
        ds = find_classification_dataset(f)
        class_arr = load_classification_array(ds, NODATA_CLASS)
        class_arr[class_arr == 2] = 3   # Code 2 wie Code 3 behandeln
        where = find_where_group(f)
        if where is None:
            sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden.")
        grid = extract_grid_info(where)

    h, w = class_arr.shape
    if (w, h) != (grid["xsize"], grid["ysize"]):
        sys.exit(f"Rastergroesse passt nicht: Array {w}x{h}, where {grid['xsize']}x{grid['ysize']}")
    print(f"Natives Raster: {w} x {h} px, {grid['xscale']} m / px")

    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)

    # Regenrate aus HX laden (nativ, 1 km)
    rate = None
    try:
        hx_path = hx_path_for(ts)
        if not hx_path.exists():
            raise FileNotFoundError(f"{hx_path} nicht gefunden")
        rate = load_hx_rate_native(hx_path, grid)
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        print(f"Warnung: HX nicht verfuegbar ({e}). Nutze HymecNG-Fallback.", file=sys.stderr)

    # Hybrid-Verfeinerung
    print("Wende Hybrid-Strategie an (HX + HymecNG)...")
    class_arr = refine_with_hybrid_strategy(class_arr, rate)

    # uint8
    out_codes = class_arr.copy()
    out_codes[out_codes < 0] = NODATA_BYTE
    out_codes = out_codes.astype(np.uint8)

    # Blitze
    try:
        hits = apply_lightning_overlay(out_codes, class_arr, rate, ts, grid, to_proj)
        print(f"{hits} Blitz-Treffer eingetragen.")
    except requests.RequestException as e:
        print(f"Warnung: Blitzdaten konnten nicht geladen werden ({e}). Ueberspringe Overlay.", file=sys.stderr)

    # Speichern: rohe Bytes, keine Dateiendung, Zeile 0 = Norden
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"liveanalyse_latest"
    np.ascontiguousarray(out_codes).tofile(out_path)
    print(f"Gespeichert: {out_path} ({w}x{h}, {out_codes.nbytes} Bytes)")

    # Metadaten fuer den Client (einmalig kopieren, aendern sich nicht)
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    meta = {
        "width": w,
        "height": h,
        "projdef": grid["projdef"],
        "xscale": grid["xscale"],
        "yscale": grid["yscale"],
        "ll_x": ll_x,
        "ll_y": ll_y,
        "rowOrder": "north-to-south",
    }
    print("Client-Metadaten:", json.dumps(meta))


if __name__ == "__main__":
    main()