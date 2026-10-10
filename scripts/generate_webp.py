#!/usr/bin/env python3
import re
import struct
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
RV_SRC_DIR = Path("data/rv")          # ANPASSEN: Ablageort der RV-Composites
OUT_DIR = Path("output/hymecng")
BIN_FILENAME = "hg_latest.bin"        # unkomprimierte Pixeldatei fuer den Worker (R2-Range-Reads)

FILENAME_RE = re.compile(r"composite_HymecNG_(\d{8})_(\d{4})_(\d{3})-hd5")
RV_FILENAME_RE = re.compile(r"composite_rv_(\d{8})_(\d{4})_(\d{3})-hd5")

# Wettercodes fuer Gewitter / Hagel-Umwandlung
THUNDER_CODE = 11
STRONG_THUNDER_CODE = 12          # Blitz in Hagelzone
HAIL_NO_THUNDER_CODE = 32         # Hagel ohne Blitz -> Regen

# Ausgabewerte
MM_QUANTUM = 0.01                 # int16-Wert * 0.01 = mm/h
NO_DATA_IN_CHUNK = 0              # "kein Wert" in der .bin (niemals -1 oder 1)

# Header der .bin-Datei (muss zu HG_HEADER_SIZE / loadHgHeader im Worker passen)
#   magic(4s) width(I) height(I) 11 x double  epoch(q)  + 20 Byte Padding = 128 Byte
BIN_MAGIC = b"HGB2"
BIN_HEADER_FMT = "<4sII11dq20x"
BIN_HEADER_SIZE = 128

# Alle Codes, die als echte Daten in die .bin geschrieben werden
VALID_OUTPUT_CODES = np.array(
    sorted([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 31, 32, 33, 61, 62, 71, 72, 73]),
    dtype=np.int32,
)
HAIL_CLASSES = {9, 10}

# SCHWELLWERTE
RAIN_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.1, 1.25, 31),
    (1.25, 10.0, 32),
    (10.0, float("inf"), 33),
]

SNOW_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.1, 1.0, 71),
    (1.0, 4.0, 72),
    (4.0, float("inf"), 73),
]

SLEET_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.1, 1.0, 61),
    (1.0, float("inf"), 62),
]

PRESERVE_PRECIP_TYPE_CODES = {8, 9, 10}

RAIN_TYPE_CODES = {2, 3, 31, 32, 33}
SNOW_TYPE_CODES = {7, 71, 72, 73}
MIXED_TYPE_CODES = {8}
SLEET_TYPE_CODES = {6, 61, 62}
FREEZING_RAIN_TYPE_CODES = {4, 5}
HAIL_TYPE_CODES = {9, 10}

MIN_PRECIP_RATE_MMH = 0.1

# Blitze
LIGHTNING_BASE_URL = "https://radar.wetterstation-neustadt.de/blitze/archive/"
LIGHTNING_BACKUP_URL = "https://nowsky.vercel.app/api/lightning"
LIGHTNING_WINDOW_MINUTES = 5
LIGHTNING_MARKER_RADIUS_CELLS = 4   # Radius der Gewitter-Markierung in 1-km-Zellen (~4 km)
LIGHTNING_PRECIP_MMH_THRESHOLD = 0.1

# Geometrie / Ausgabe
BERLIN = ZoneInfo("Europe/Berlin")
NODATA_CLASS = -1
INVISIBLE_CLASS = 1   # Regen ohne RV-Wert -> nicht dargestellt

# RV Meta (nur Fallback)
RV_DEFAULT_GAIN = 0.0009999999317806213
RV_DEFAULT_OFFSET = -0.0009999999317806213
RV_DEFAULT_NODATA = 4294967295.0
RV_DEFAULT_UNDETECT = 0.0
RV_STEP_MINUTES = 5.0

PRECIP_SOURCE_CODES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
REFINED_PRECIP_CODES = [31, 32, 33, 61, 62, 71, 72, 73]
ALL_PRECIP_CODES = PRECIP_SOURCE_CODES + REFINED_PRECIP_CODES


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def parse_timestamp(filename: str, pattern: re.Pattern = FILENAME_RE) -> datetime:
    """Zeitstempel aus dem Dateinamen (UTC)."""
    m = pattern.match(filename)
    if not m:
        raise ValueError(f"Dateiname passt nicht zum Schema {pattern.pattern}: {filename}")
    date_str, time_str, _ = m.groups()
    naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return naive.replace(tzinfo=timezone.utc)


def rv_path_for(ts: datetime) -> Path:
    """Pfad der RV-Analysedatei (Vorhersageschritt 000) zum Zeitstempel."""
    return RV_SRC_DIR / f"composite_rv_{ts:%Y%m%d}_{ts:%H%M}_000-hd5"


# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
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
            print(
                f"Warnung: {len(matches)} Datasets passen auf Quantity-Keywords "
                f"{keywords} — nehme das erste ({matches[0].name}).",
                file=sys.stderr,
            )
        return matches[0]

    candidates.sort(key=lambda ds: ds.name)
    chosen = candidates[0]
    print(
        f"Warnung: keine Quantity passte auf {keywords} in dieser Datei — "
        f"nehme Fallback-Dataset {chosen.name} (Quantity unbekannt/nicht gesetzt). "
        f"Bitte pruefen, ob das die richtigen Daten sind!",
        file=sys.stderr,
    )
    return chosen


def find_classification_dataset(h5file: h5py.File) -> h5py.Dataset:
    return _find_2d_dataset_by_quantity(
        h5file,
        keywords=("CLASS", "PRECIP", "HCLASS", "TYPE"),
        error_msg="Kein 2D-Datensatz in der HymecNG-Datei gefunden.",
    )


def find_rate_dataset(h5file: h5py.File) -> h5py.Dataset:
    return _find_2d_dataset_by_quantity(
        h5file,
        keywords=("RATE", "ACRR"),
        error_msg="Kein 2D-Datensatz in der RV-Datei gefunden.",
    )


def load_classification_array(h5file: h5py.File, dataset: h5py.Dataset, nodata_class: int) -> np.ndarray:
    raw = dataset[()]
    what = dataset.parent.get("what")
    attrs = what.attrs if what is not None else {}

    def attr_float(key: str):
        v = attrs.get(key)
        if v is None:
            return None
        return float(np.asarray(v).ravel()[0])

    nodata = attr_float("nodata")
    undetect = attr_float("undetect")
    gain = attr_float("gain")
    offset = attr_float("offset")

    class_arr = raw.astype(np.int32).copy()

    if (gain is not None and gain != 1.0) or (offset is not None and offset != 0.0):
        print(
            f"Warnung: Klassifikations-Dataset hat gain={gain}, offset={offset} "
            f"— fuer Klassencodes ungewoehnlich, bitte pruefen.",
            file=sys.stderr,
        )

    mask_invalid = np.zeros_like(class_arr, dtype=bool)
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


def grid_differences(class_grid: dict, rv_grid: dict, tol: float = 1e-6) -> list[str]:
    """Liste der Unterschiede zwischen HymecNG- und RV-Raster (leer = identisch)."""
    problems = []
    if class_grid["projdef"] != rv_grid["projdef"]:
        problems.append(f"projdef: {class_grid['projdef']!r} != {rv_grid['projdef']!r}")
    for key in ("xsize", "ysize"):
        if class_grid[key] != rv_grid[key]:
            problems.append(f"{key}: {class_grid[key]} != {rv_grid[key]}")
    for key in ("xscale", "yscale", "ll_lon", "ll_lat"):
        if abs(class_grid[key] - rv_grid[key]) > tol:
            problems.append(f"{key}: {class_grid[key]} != {rv_grid[key]}")
    return problems


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


# --------------------------------------------------------------------------- #
# Projektion / natives Raster (kein Warp mehr: alles bleibt im 1-km-Radarraster)
# --------------------------------------------------------------------------- #
def parse_stere_params(projdef: str) -> dict:
    """Parameter der polar-stereografischen Projektion (Nordpol) aus dem PROJ-String.
    Genau diese Variante ist im Worker implementiert; alles andere wird abgelehnt."""
    kv: dict[str, str] = {}
    for tok in projdef.split():
        if tok.startswith("+"):
            k, _, v = tok[1:].partition("=")
            kv[k] = v

    def num(key: str, default: float | None = None) -> float:
        if key in kv:
            return float(kv[key])
        if default is None:
            raise ValueError(f"'+{key}' fehlt in projdef: {projdef!r}")
        return default

    if kv.get("proj") != "stere" or abs(num("lat_0", 90.0) - 90.0) > 1e-9:
        raise ValueError(f"Nur '+proj=stere +lat_0=90' wird unterstuetzt, bekommen: {projdef!r}")
    if kv.get("units", "m") != "m":
        raise ValueError(f"Nur units=m wird unterstuetzt: {projdef!r}")

    return {
        "a": num("a"),
        "b": num("b"),
        "lat_ts": num("lat_ts"),
        "lon_0": num("lon_0", 0.0),
        "x_0": num("x_0", 0.0),
        "y_0": num("y_0", 0.0),
    }


def resample_to_grid(data: np.ndarray, src_grid: dict, dst_grid: dict, fill_value: float) -> np.ndarray:
    """Nearest-Neighbor-Resampling von src_grid auf dst_grid (nur noetig, falls RV- und
    HymecNG-Raster NICHT identisch sind; sonst wird nichts umgerechnet)."""
    to_wgs84_dst = Transformer.from_crs(dst_grid["projdef"], "EPSG:4326", always_xy=True)
    to_src = Transformer.from_crs("EPSG:4326", src_grid["projdef"], always_xy=True)
    dst_to_proj = Transformer.from_crs("EPSG:4326", dst_grid["projdef"], always_xy=True)

    d_llx, d_lly = dst_to_proj.transform(dst_grid["ll_lon"], dst_grid["ll_lat"])
    s_llx, s_lly = to_src.transform(src_grid["ll_lon"], src_grid["ll_lat"])

    cols = np.arange(dst_grid["xsize"])
    rows = np.arange(dst_grid["ysize"])
    cx = d_llx + (cols + 0.5) * dst_grid["xscale"]
    cy = d_lly + (dst_grid["ysize"] - 1 - rows + 0.5) * dst_grid["yscale"]
    xx, yy = np.meshgrid(cx, cy)

    lon, lat = to_wgs84_dst.transform(xx.ravel(), yy.ravel())
    sx, sy = to_src.transform(lon, lat)
    sx = np.asarray(sx).reshape(xx.shape)
    sy = np.asarray(sy).reshape(xx.shape)

    scol = np.floor((sx - s_llx) / src_grid["xscale"]).astype(np.int64)
    srow = (src_grid["ysize"] - 1 - np.floor((sy - s_lly) / src_grid["yscale"])).astype(np.int64)
    valid = (scol >= 0) & (scol < src_grid["xsize"]) & (srow >= 0) & (srow < src_grid["ysize"])

    out = np.full(xx.shape, fill_value, dtype=np.float64)
    out[valid] = data[srow[valid], scol[valid]]
    return out


# --------------------------------------------------------------------------- #
# HYBRID-STRATEGIE: RV fuer Detektion + Intensitaet, HymecNG fuer Art
# --------------------------------------------------------------------------- #
def refine_with_hybrid_strategy(
    class_arr: np.ndarray,
    rate_arr: np.ndarray | None,
) -> np.ndarray:
    refined = class_arr.copy()

    if rate_arr is None:
        mask_rain = class_arr == 3
        mask_snow = class_arr == 7
        mask_sleet = class_arr == 6
        if mask_rain.any():
            refined[mask_rain] = RAIN_MMH_THRESHOLDS[0][2]
        if mask_snow.any():
            refined[mask_snow] = SNOW_MMH_THRESHOLDS[0][2]
        if mask_sleet.any():
            refined[mask_sleet] = SLEET_MMH_THRESHOLDS[0][2]
        if mask_rain.any() or mask_snow.any() or mask_sleet.any():
            print("Warnung: RV nicht verfügbar, verwende niedrigste Intensitätsstufe als Fallback.", file=sys.stderr)
        return refined

    # === SCHRITT 1: Pixel mit echtem Niederschlag (RV) ===
    has_rain = ~np.isnan(rate_arr) & (rate_arr >= MIN_PRECIP_RATE_MMH)

    # === SCHRITT 2: REGEN (Code 3) ===
    mask_refine_rain = (class_arr == 3) & has_rain
    for lower, upper, new_code in RAIN_MMH_THRESHOLDS:
        m = mask_refine_rain & (rate_arr >= lower) & (rate_arr < upper)
        refined[m] = new_code

    mask_rain_below_min = (class_arr == 3) & has_rain & (refined == 3)
    if mask_rain_below_min.any():
        refined[mask_rain_below_min] = RAIN_MMH_THRESHOLDS[0][2]

    # === SCHRITT 2b: SCHNEE (Code 7) ===
    mask_refine_snow = (class_arr == 7) & has_rain
    for lower, upper, new_code in SNOW_MMH_THRESHOLDS:
        m = mask_refine_snow & (rate_arr >= lower) & (rate_arr < upper)
        refined[m] = new_code

    mask_snow_below_min = (class_arr == 7) & has_rain & (refined == 7)
    if mask_snow_below_min.any():
        refined[mask_snow_below_min] = SNOW_MMH_THRESHOLDS[0][2]

    # === SCHRITT 2c: SCHNEEREGEN (Code 6) ===
    mask_refine_sleet = (class_arr == 6) & has_rain
    for lower, upper, new_code in SLEET_MMH_THRESHOLDS:
        m = mask_refine_sleet & (rate_arr >= lower) & (rate_arr < upper)
        refined[m] = new_code

    mask_sleet_below_min = (class_arr == 6) & has_rain & (refined == 6)
    if mask_sleet_below_min.any():
        refined[mask_sleet_below_min] = SLEET_MMH_THRESHOLDS[0][2]

    # === SCHRITT 3: Naechstgelegenen bekannten Niederschlagstyp bestimmen ===
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

    # === SCHRITT 4: Luecken fuellen ===
    gap = has_rain & ~np.isin(class_arr, PRECIP_SOURCE_CODES)

    for lower, upper, new_code in RAIN_MMH_THRESHOLDS:
        m = gap & is_rain_type & (rate_arr >= lower) & (rate_arr < upper)
        refined[m] = new_code
    for lower, upper, new_code in SNOW_MMH_THRESHOLDS:
        m = gap & is_snow_type & (rate_arr >= lower) & (rate_arr < upper)
        refined[m] = new_code
    for lower, upper, new_code in SLEET_MMH_THRESHOLDS:
        m = gap & is_sleet_type & (rate_arr >= lower) & (rate_arr < upper)
        refined[m] = new_code

    m = gap & is_preserved_type
    refined[m] = nearest_type[m]

    return refined


# --------------------------------------------------------------------------- #
# Hagel -> Regen
# --------------------------------------------------------------------------- #
def convert_hail_to_rain(class_arr: np.ndarray) -> np.ndarray:
    """Hagel -> Regen (Code 32), in place. Gibt die Hagel-Maske zurueck,
    die spaeter fuer 'Blitz in Hagelzone' (Code 12) gebraucht wird."""
    hail_mask = np.isin(class_arr, tuple(HAIL_CLASSES))
    class_arr[hail_mask] = HAIL_NO_THUNDER_CODE
    return hail_mask


# --------------------------------------------------------------------------- #
# Blitze
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
    class_arr: np.ndarray,           # wird IN PLACE veraendert
    hail_mask: np.ndarray,
    rate_arr: np.ndarray | None,
    ts: datetime,
    grid: dict,
    to_proj: Transformer,
    ll_x: float,
    ll_y: float,
) -> int:
    """Schreibt Gewittercodes (11 / 12) in class_arr und ersetzt damit den
    darunterliegenden Wert. Gibt die Trefferzahl zurueck."""
    strikes = fetch_recent_strikes(ts)
    print(f"{len(strikes)} Blitze in den letzten {LIGHTNING_WINDOW_MINUTES} Minuten geladen.")

    n_rows, n_cols = class_arr.shape
    radius = LIGHTNING_MARKER_RADIUS_CELLS

    precip_mask = np.isin(class_arr, ALL_PRECIP_CODES)
    if rate_arr is not None:
        rate_ok = np.isnan(rate_arr) | (rate_arr >= LIGHTNING_PRECIP_MMH_THRESHOLD)
        precip_mask &= rate_ok

    offsets = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    circle = dr * dr + dc * dc <= radius * radius

    thunder_mask = np.zeros((n_rows, n_cols), dtype=bool)
    hits = 0
    for lat, lon in strikes:
        sx, sy = to_proj.transform(lon, lat)
        col = int(np.floor((sx - ll_x) / grid["xscale"]))
        row = n_rows - 1 - int(np.floor((sy - ll_y) / grid["yscale"]))
        if not (0 <= col < n_cols and 0 <= row < n_rows):
            continue
        if not precip_mask[row, col]:
            continue

        r0, r1 = max(0, row - radius), min(n_rows, row + radius + 1)
        c0, c1 = max(0, col - radius), min(n_cols, col + radius + 1)
        circ = circle[r0 - (row - radius): r1 - (row - radius),
                      c0 - (col - radius): c1 - (col - radius)]

        thunder_mask[r0:r1, c0:c1] |= precip_mask[r0:r1, c0:c1] & circ
        hits += 1

    # Erst normaler Blitz, dann starker Blitz (Hagelzone) darueber
    class_arr[thunder_mask] = THUNDER_CODE
    class_arr[thunder_mask & hail_mask] = STRONG_THUNDER_CODE
    return hits


# --------------------------------------------------------------------------- #
# RV-Composite laden (natives Raster)
# --------------------------------------------------------------------------- #
def load_rv_rate(rv_path: Path) -> tuple[np.ndarray, dict]:
    with h5py.File(rv_path, "r") as f:
        ds = find_rate_dataset(f)
        raw = ds[()]
        what = ds.parent.get("what")
        attrs = what.attrs if what is not None else {}
        quantity = _attr_str(attrs, "quantity").upper()
        gain = _attr_float(attrs, "gain", RV_DEFAULT_GAIN)
        offset = _attr_float(attrs, "offset", RV_DEFAULT_OFFSET)
        nodata = _attr_float(attrs, "nodata", RV_DEFAULT_NODATA)
        undetect = _attr_float(attrs, "undetect", RV_DEFAULT_UNDETECT)

        where = find_where_group(f)
        if where is None:
            raise RuntimeError("Keine 'where'-Projektionsinfo in der RV-Datei gefunden.")
        grid = extract_grid_info(where)

    rate = raw.astype(np.float64) * gain + offset
    if undetect is not None:
        rate[raw == undetect] = 0.0
    if nodata is not None:
        rate[raw == nodata] = np.nan

    # ACRR (mm pro Zeitschritt) -> mm/h; RATE ist bereits mm/h
    if "RATE" not in quantity:
        rate *= 60.0 / RV_STEP_MINUTES

    print(
        f"RV-Composite: {rv_path.name} (quantity={quantity or '?'}, gain={gain}, offset={offset}, "
        f"nodata={nodata}, undetect={undetect}, Raster {grid['xsize']}x{grid['ysize']})"
    )
    return rate, grid


# --------------------------------------------------------------------------- #
# Pixel-Arrays + .bin
# --------------------------------------------------------------------------- #
def make_pixel_arrays(class_arr: np.ndarray, rate_arr: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt (mm_int16, code_int16). Wo keine echten Daten vorliegen, steht
    NO_DATA_IN_CHUNK (0) -- niemals -1 oder 1.

    - Wettercode: nur gueltige Ausgabecodes (VALID_OUTPUT_CODES).
    - mm/h: nur dort, wo ein Wettercode UND eine gueltige RV-Rate existiert."""
    has_code = np.isin(class_arr, VALID_OUTPUT_CODES)
    code_int = np.where(has_code, class_arr, NO_DATA_IN_CHUNK).astype(np.int16)

    if rate_arr is None:
        mm_int = np.full(class_arr.shape, NO_DATA_IN_CHUNK, dtype=np.int16)
    else:
        has_mm = has_code & ~np.isnan(rate_arr)
        scaled = np.round(np.nan_to_num(rate_arr, nan=0.0) / MM_QUANTUM)
        mm_int = np.where(has_mm, np.clip(scaled, 0, 32767), NO_DATA_IN_CHUNK).astype(np.int16)

    return mm_int, code_int


def write_pixel_bin(
    path: Path,
    mm_int: np.ndarray,
    code_int: np.ndarray,
    grid: dict,
    proj: dict,
    ll_x: float,
    ll_y: float,
    ts: datetime,
) -> None:
    """Layout (little endian), Header = 128 Byte:
        0    4s   Magic "HGB2"
        4    u32  width  (= xsize)
        8    u32  height (= ysize)
        12   f64  xscale  [m]
        20   f64  yscale  [m]
        28   f64  ll_x    Projektions-x der linken unteren ECKE des Rasters [m]
        36   f64  ll_y    Projektions-y der linken unteren ECKE des Rasters [m]
        44   f64  a       Ellipsoid grosse Halbachse [m]
        52   f64  b       Ellipsoid kleine Halbachse [m]
        60   f64  lat_ts  Standardparallele [Grad]
        68   f64  lon_0   Zentralmeridian [Grad]
        76   f64  x_0     False Easting [m]
        84   f64  y_0     False Northing [m]
        92   f64  mm-Quantum
        100  i64  epoch (UTC, Sekunden)
        108  20x  Padding
        128  int16[w*h]          mm/h        (Zeile 0 = Norden, wie im HDF5)
        128 + w*h*2  int16[w*h]  Wettercode

    Projektion: polar-stereografisch (Nordpol), Zelle = floor((x - ll_x)/xscale),
    Zeile von oben = height - 1 - floor((y - ll_y)/yscale)."""
    height, width = code_int.shape
    if (width, height) != (grid["xsize"], grid["ysize"]):
        raise ValueError(f"Array {width}x{height} passt nicht zum Raster {grid['xsize']}x{grid['ysize']}")
    header = struct.pack(
        BIN_HEADER_FMT, BIN_MAGIC, width, height,
        grid["xscale"], grid["yscale"], float(ll_x), float(ll_y),
        proj["a"], proj["b"], proj["lat_ts"], proj["lon_0"], proj["x_0"], proj["y_0"],
        MM_QUANTUM, int(ts.timestamp()),
    )
    assert len(header) == BIN_HEADER_SIZE, f"Header hat {len(header)} Byte, erwartet {BIN_HEADER_SIZE}"
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(mm_int, dtype="<i2").tobytes())
        f.write(np.ascontiguousarray(code_int, dtype="<i2").tobytes())


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
        class_array = load_classification_array(f, ds, NODATA_CLASS)
        class_array[class_array == 2] = 3   # Code 2 wie Code 3 behandeln
        where = find_where_group(f)
        if where is None:
            sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden.")
        grid = extract_grid_info(where)

    if class_array.shape != (grid["ysize"], grid["xsize"]):
        sys.exit(f"Array {class_array.shape} passt nicht zu ysize/xsize {grid['ysize']}x{grid['xsize']}.")

    try:
        proj = parse_stere_params(grid["projdef"])
    except ValueError as e:
        sys.exit(str(e))

    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    print(f"Raster: {grid['xsize']} x {grid['ysize']} Zellen a {grid['xscale']:.0f} m, ll_x={ll_x:.1f}, ll_y={ll_y:.1f}")

    # === Schritt 1: RV laden (liegt normalerweise auf demselben Raster) ===
    rate_arr = None
    rv_path = rv_path_for(ts)
    try:
        if not rv_path.exists():
            raise FileNotFoundError(f"{rv_path} nicht gefunden")
        rate_arr, rv_grid = load_rv_rate(rv_path)
        problems = grid_differences(grid, rv_grid)
        if problems:
            print(
                "Warnung: HymecNG- und RV-Raster unterscheiden sich, RV wird umgerechnet:\n  "
                + "\n  ".join(problems),
                file=sys.stderr,
            )
            rate_arr = resample_to_grid(rate_arr, rv_grid, grid, fill_value=np.nan)
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        rate_arr = None
        print(f"Warnung: RV-Composite nicht verfuegbar ({e}). Nutze HymecNG-Fallback.", file=sys.stderr)

    # === Schritt 2: Hybrid-Verfeinerung ===
    print("Wende Hybrid-Strategie an (RV + HymecNG)...")
    class_arr = refine_with_hybrid_strategy(class_array, rate_arr)

    # === Schritt 3: Hagel -> Regen (Code 32) ===
    hail_mask = convert_hail_to_rain(class_arr)

    # === Schritt 4: Blitze (schreibt Gewittercodes in class_arr) ===
    try:
        hits = apply_lightning_overlay(class_arr, hail_mask, rate_arr, ts, grid, to_proj, ll_x, ll_y)
        print(f"{hits} Blitz-Treffer als Gewitter markiert.")
    except requests.RequestException as e:
        print(f"Warnung: Blitzdaten konnten nicht geladen werden ({e}). Ueberspringe Overlay.", file=sys.stderr)

    # === Speichern: nur die .bin ===
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mm_int, code_int = make_pixel_arrays(class_arr, rate_arr)
    bin_path = OUT_DIR / BIN_FILENAME
    write_pixel_bin(bin_path, mm_int, code_int, grid, proj, ll_x, ll_y, ts)
    print(
        f"{int((mm_int > 0).sum())} Zellen mit mm, {int((code_int != 0).sum())} Zellen mit Code "
        f"({ts:%Y-%m-%d %H:%M} UTC)."
    )
    print(f"Gespeichert: {bin_path} ({bin_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()