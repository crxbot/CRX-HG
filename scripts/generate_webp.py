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
from scipy.ndimage import map_coordinates

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
#   magic(4s) width(I) height(I) extent(4d) mm_quantum(d) epoch(q) + 4 Byte Padding = 64 Byte
BIN_MAGIC = b"HGB1"
BIN_HEADER_FMT = "<4sII4ddqB3x"

# Alle Codes, die als echte Daten in die .bin geschrieben werden
VALID_OUTPUT_CODES = np.array(
    sorted([4, 5, 8, 9, 10, 11, 12, 30, 31, 32, 33, 61, 62, 71, 72, 73]),
    dtype=np.int32,
)
HAIL_CLASSES = {9, 10}

# SCHWELLWERTE
RAIN_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.06, 0.09, 30),          
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
LIGHTNING_MARKER_RADIUS_PX = 8
LIGHTNING_PRECIP_MMH_THRESHOLD = 0.1

# Geometrie / Ausgabe
BERLIN = ZoneInfo("Europe/Berlin")
WEBMERCATOR_OUT_WIDTH = 1927
EDGE_SAMPLES = 200
BBOX_MARGIN_DEG = 0.02
EARTH_RADIUS = 6378137.0
NODATA_CLASS = -1
INVISIBLE_CLASS = 1   # Regen ohne RV-Wert -> nicht dargestellt

# RV Meta (nur Fallback)
RV_DEFAULT_GAIN = 0.0009999999317806213
RV_DEFAULT_OFFSET = -0.0009999999317806213
RV_DEFAULT_NODATA = 4294967295.0
RV_DEFAULT_UNDETECT = 0.0
RV_STEP_MINUTES = 5.0

PRECIP_SOURCE_CODES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
REFINED_PRECIP_CODES = [30, 31, 32, 33, 61, 62, 71, 72, 73]
ALL_PRECIP_CODES = PRECIP_SOURCE_CODES + REFINED_PRECIP_CODES


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_to_lonlat(x, y):
    lon = np.degrees(x / EARTH_RADIUS)
    lat = np.degrees(2 * np.arctan(np.exp(y / EARTH_RADIUS)) - np.pi / 2)
    return lon, lat


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


def assert_compatible_grids(class_grid: dict, rv_grid: dict, tol: float = 1e-6) -> None:
    problems = []
    if class_grid["projdef"] != rv_grid["projdef"]:
        problems.append(f"projdef: {class_grid['projdef']!r} != {rv_grid['projdef']!r}")
    for key in ("xscale", "yscale", "ll_lon", "ll_lat"):
        if abs(class_grid[key] - rv_grid[key]) > tol:
            problems.append(f"{key}: {class_grid[key]} != {rv_grid[key]}")

    if problems:
        print(
            "Warnung: HymecNG- und RV-Grid unterscheiden sich:\n  "
            + "\n  ".join(problems)
            + "\nTyp und Intensitaet werden trotzdem unabhaengig gewarpt, "
              "koennen aber leicht gegeneinander verschoben sein.",
            file=sys.stderr,
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


# --------------------------------------------------------------------------- #
# Geometrie / Warp
# --------------------------------------------------------------------------- #
def native_origin_and_extent(grid: dict, to_proj: Transformer):
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    x_max = ll_x + grid["xsize"] * grid["xscale"]
    y_max = ll_y + grid["ysize"] * grid["yscale"]
    return ll_x, ll_y, x_max, y_max


def wgs84_bbox_from_perimeter(ll_x, ll_y, x_max, y_max, to_wgs84: Transformer):
    t = np.linspace(0.0, 1.0, EDGE_SAMPLES)
    xs_span = ll_x + t * (x_max - ll_x)
    ys_span = ll_y + t * (y_max - ll_y)
    xs = np.concatenate([xs_span, xs_span, np.full_like(ys_span, ll_x), np.full_like(ys_span, x_max)])
    ys = np.concatenate([np.full_like(xs_span, ll_y), np.full_like(xs_span, y_max), ys_span, ys_span])
    lons, lats = (np.asarray(a) for a in to_wgs84.transform(xs, ys))
    return (
        float(lons.min()) - BBOX_MARGIN_DEG,
        float(lons.max()) + BBOX_MARGIN_DEG,
        float(lats.min()) - BBOX_MARGIN_DEG,
        float(lats.max()) + BBOX_MARGIN_DEG,
    )


def webmercator_target_grid(lon_min, lon_max, lat_min, lat_max):
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_h = max(int(round(WEBMERCATOR_OUT_WIDTH * aspect)), 1)
    dx = (x_max - x_min) / WEBMERCATOR_OUT_WIDTH
    dy = (y_max - y_min) / out_h
    x_new = x_min + (np.arange(WEBMERCATOR_OUT_WIDTH) + 0.5) * dx
    y_new = y_min + (np.arange(out_h) + 0.5) * dy
    return x_new, y_new, [x_min, y_min, x_max, y_max]


def nearest_neighbor_warp(
    data: np.ndarray,
    grid: dict,
    to_proj: Transformer,
    x_new: np.ndarray,
    y_new: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Nearest-Neighbor-Warp (fuer diskrete Klassencodes)."""
    xx, yy = np.meshgrid(x_new, y_new)
    lon, lat = webmercator_to_lonlat(xx, yy)
    x_nat, y_nat = to_proj.transform(lon.ravel(), lat.ravel())
    x_nat = np.asarray(x_nat).reshape(xx.shape)
    y_nat = np.asarray(y_nat).reshape(xx.shape)

    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])

    col = np.floor((x_nat - ll_x) / grid["xscale"]).astype(np.int64)
    row = (grid["ysize"] - 1 - np.floor((y_nat - ll_y) / grid["yscale"])).astype(np.int64)
    valid = (col >= 0) & (col < grid["xsize"]) & (row >= 0) & (row < grid["ysize"])

    out = np.full(xx.shape, fill_value, dtype=np.float64)
    out[valid] = data[row[valid], col[valid]]
    return out


def bilinear_warp(
    data: np.ndarray,
    grid: dict,
    to_proj: Transformer,
    x_new: np.ndarray,
    y_new: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Bilinearer Warp fuer kontinuierliche Messwerte (aktuell ungenutzt)."""
    xx, yy = np.meshgrid(x_new, y_new)
    lon, lat = webmercator_to_lonlat(xx, yy)
    x_nat, y_nat = to_proj.transform(lon.ravel(), lat.ravel())
    x_nat = np.asarray(x_nat).reshape(xx.shape)
    y_nat = np.asarray(y_nat).reshape(xx.shape)

    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])

    fcol = (x_nat - ll_x) / grid["xscale"]
    frow = grid["ysize"] - 1 - (y_nat - ll_y) / grid["yscale"]

    is_nan = np.isnan(data)
    src = np.where(is_nan, 0.0, data)
    valid_src = (~is_nan).astype(np.float64)

    coords = [frow.ravel(), fcol.ravel()]
    val_interp = map_coordinates(src, coords, order=1, mode="constant", cval=0.0)
    valid_interp = map_coordinates(valid_src, coords, order=1, mode="constant", cval=0.0)

    out = np.where(
        valid_interp > 0.5,
        val_interp / np.maximum(valid_interp, 1e-9),
        fill_value,
    )
    return out.reshape(xx.shape)


def warp_classification_to_webmercator(
    class_array: np.ndarray,
    grid: dict,
    to_proj: Transformer,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    warped = nearest_neighbor_warp(
        class_array.astype(np.float64), grid, to_proj, x_new, y_new, fill_value=float(NODATA_CLASS)
    )
    return np.round(warped).astype(np.int32)


# --------------------------------------------------------------------------- #
# HYBRID-STRATEGIE: RV fuer Detektion + Intensitaet, HymecNG fuer Art
# --------------------------------------------------------------------------- #
def refine_with_hybrid_strategy(
    class_merc: np.ndarray,
    rate_merc: np.ndarray | None,
) -> np.ndarray:
    refined = class_merc.copy()

    if rate_merc is None:
        mask_rain = class_merc == 3
        mask_snow = class_merc == 7
        mask_sleet = class_merc == 6
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
    has_rain = ~np.isnan(rate_merc) & (rate_merc >= MIN_PRECIP_RATE_MMH)

    # === SCHRITT 2: REGEN (Code 3) ===
    mask_refine_rain = (class_merc == 3) & has_rain
    for lower, upper, new_code in RAIN_MMH_THRESHOLDS:
        m = mask_refine_rain & (rate_merc >= lower) & (rate_merc < upper)
        refined[m] = new_code

    mask_rain_below_min = (class_merc == 3) & has_rain & (refined == 3)
    if mask_rain_below_min.any():
        refined[mask_rain_below_min] = RAIN_MMH_THRESHOLDS[0][2]

    # === SCHRITT 2b: SCHNEE (Code 7) ===
    mask_refine_snow = (class_merc == 7) & has_rain
    for lower, upper, new_code in SNOW_MMH_THRESHOLDS:
        m = mask_refine_snow & (rate_merc >= lower) & (rate_merc < upper)
        refined[m] = new_code

    mask_snow_below_min = (class_merc == 7) & has_rain & (refined == 7)
    if mask_snow_below_min.any():
        refined[mask_snow_below_min] = SNOW_MMH_THRESHOLDS[0][2]

    # === SCHRITT 2c: SCHNEEREGEN (Code 6) ===
    mask_refine_sleet = (class_merc == 6) & has_rain
    for lower, upper, new_code in SLEET_MMH_THRESHOLDS:
        m = mask_refine_sleet & (rate_merc >= lower) & (rate_merc < upper)
        refined[m] = new_code

    mask_sleet_below_min = (class_merc == 6) & has_rain & (refined == 6)
    if mask_sleet_below_min.any():
        refined[mask_sleet_below_min] = SLEET_MMH_THRESHOLDS[0][2]

    # === SCHRITT 3: Naechstgelegenen bekannten Niederschlagstyp bestimmen ===
    known_type = np.isin(class_merc, PRECIP_SOURCE_CODES)
    if known_type.any():
        _, (ir, ic) = ndimage.distance_transform_edt(~known_type, return_indices=True)
        nearest_type = class_merc[ir, ic]
    else:
        nearest_type = np.full(class_merc.shape, 3, dtype=class_merc.dtype)

    is_snow_type = np.isin(nearest_type, tuple(SNOW_TYPE_CODES))
    is_sleet_type = np.isin(nearest_type, tuple(SLEET_TYPE_CODES))
    is_preserved_type = np.isin(nearest_type, tuple(FREEZING_RAIN_TYPE_CODES | {8} | HAIL_TYPE_CODES))
    is_rain_type = ~is_snow_type & ~is_sleet_type & ~is_preserved_type

    # === SCHRITT 4: Luecken fuellen ===
    gap = has_rain & ~np.isin(class_merc, PRECIP_SOURCE_CODES)

    for lower, upper, new_code in RAIN_MMH_THRESHOLDS:
        m = gap & is_rain_type & (rate_merc >= lower) & (rate_merc < upper)
        refined[m] = new_code
    for lower, upper, new_code in SNOW_MMH_THRESHOLDS:
        m = gap & is_snow_type & (rate_merc >= lower) & (rate_merc < upper)
        refined[m] = new_code
    for lower, upper, new_code in SLEET_MMH_THRESHOLDS:
        m = gap & is_sleet_type & (rate_merc >= lower) & (rate_merc < upper)
        refined[m] = new_code

    m = gap & is_preserved_type
    refined[m] = nearest_type[m]

    return refined


# --------------------------------------------------------------------------- #
# Hagel -> Regen
# --------------------------------------------------------------------------- #
def convert_hail_to_rain(class_merc: np.ndarray) -> np.ndarray:
    """Hagel -> Regen (Code 32), in place. Gibt die Hagel-Maske zurueck,
    die spaeter fuer 'Blitz in Hagelzone' (Code 12) gebraucht wird."""
    hail_mask = np.isin(class_merc, tuple(HAIL_CLASSES))
    class_merc[hail_mask] = HAIL_NO_THUNDER_CODE
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
    class_merc: np.ndarray,          # wird IN PLACE veraendert
    hail_mask: np.ndarray,
    rate_merc: np.ndarray | None,
    ts: datetime,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> int:
    """Schreibt Gewittercodes (11 / 12) in class_merc und ersetzt damit den
    darunterliegenden Wert. Gibt die Trefferzahl zurueck."""
    strikes = fetch_recent_strikes(ts)
    print(f"{len(strikes)} Blitze in den letzten {LIGHTNING_WINDOW_MINUTES} Minuten geladen.")

    out_h, out_w = class_merc.shape
    radius = LIGHTNING_MARKER_RADIUS_PX

    precip_mask = np.isin(class_merc, ALL_PRECIP_CODES)
    if rate_merc is not None:
        rate_ok = np.isnan(rate_merc) | (rate_merc >= LIGHTNING_PRECIP_MMH_THRESHOLD)
        precip_mask &= rate_ok

    offsets = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    circle = dr * dr + dc * dc <= radius * radius

    dx = x_new[1] - x_new[0]
    dy = y_new[1] - y_new[0]
    x_min = x_new[0] - dx / 2
    y_min = y_new[0] - dy / 2

    thunder_mask = np.zeros((out_h, out_w), dtype=bool)
    hits = 0
    for lat, lon in strikes:
        sx, sy = lonlat_to_webmercator(lon, lat)
        col = int(np.floor((sx - x_min) / dx))
        row = int(np.floor((sy - y_min) / dy))
        if not (0 <= col < out_w and 0 <= row < out_h):
            continue
        if not precip_mask[row, col]:
            continue

        r0, r1 = max(0, row - radius), min(out_h, row + radius + 1)
        c0, c1 = max(0, col - radius), min(out_w, col + radius + 1)
        circ = circle[r0 - (row - radius): r1 - (row - radius),
                      c0 - (col - radius): c1 - (col - radius)]

        thunder_mask[r0:r1, c0:c1] |= precip_mask[r0:r1, c0:c1] & circ
        hits += 1

    # Erst normaler Blitz, dann starker Blitz (Hagelzone) darueber
    class_merc[thunder_mask] = THUNDER_CODE
    class_merc[thunder_mask & hail_mask] = STRONG_THUNDER_CODE
    return hits


# --------------------------------------------------------------------------- #
# RV-Composite laden + auf Zielraster warpen
# --------------------------------------------------------------------------- #
def load_rv_rate_on_target_grid(rv_path: Path, x_new: np.ndarray, y_new: np.ndarray) -> tuple[np.ndarray, dict]:
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

    to_proj_rv = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
    rate_merc = nearest_neighbor_warp(rate, grid, to_proj_rv, x_new, y_new, fill_value=np.nan)
    return rate_merc, grid


# --------------------------------------------------------------------------- #
# Pixel-Arrays + .bin
# --------------------------------------------------------------------------- #
def make_pixel_arrays(class_merc: np.ndarray, rate_merc: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt (mm_int16, code_int16). Wo keine echten Daten vorliegen, steht
    NO_DATA_IN_CHUNK (0) -- niemals -1 oder 1.

    - Wettercode: nur gueltige Ausgabecodes (VALID_OUTPUT_CODES).
    - mm/h: nur dort, wo ein Wettercode UND eine gueltige RV-Rate existiert."""
    has_code = np.isin(class_merc, VALID_OUTPUT_CODES)
    code_int = np.where(has_code, class_merc, NO_DATA_IN_CHUNK).astype(np.int16)

    if rate_merc is None:
        mm_int = np.full(class_merc.shape, NO_DATA_IN_CHUNK, dtype=np.int16)
    else:
        has_mm = ~np.isnan(rate_merc)
        scaled = np.round(np.nan_to_num(rate_merc, nan=0.0) / MM_QUANTUM)
        mm_int = np.where(has_mm, np.clip(scaled, 0, 32767), NO_DATA_IN_CHUNK).astype(np.int16)

    return mm_int, code_int


def write_pixel_bin(
    path: Path,
    mm_int: np.ndarray,
    code_int: np.ndarray,
    extent: list[float],
    ts: datetime,
) -> None:
    """Layout (little endian):
        0   4s   Magic "HGB1"
        4   u32  width
        8   u32  height
        12  4*f64 extent [xmin, ymin, xmax, ymax] (EPSG:3857)
        44  f64  mm-Quantum
        52  i64  epoch (UTC, Sekunden)
        56  4x   Padding  -> Header = 64 Bytea
        64  int16[w*h]          mm/h   (Zeile 0 = Norden)
        64 + w*h*2  int16[w*h]  Wettercode

    mm_int / code_int kommen in Arbeitsorientierung (Sued -> Nord) und werden
    hier vertikal gespiegelt (Zeile 0 = Norden)."""
    height, width = code_int.shape
    header = struct.pack(
        BIN_HEADER_FMT, BIN_MAGIC, width, height, *extent, MM_QUANTUM, int(ts.timestamp()), 1
    )
    assert len(header) == 64, f"Header hat {len(header)} Byte, erwartet 64"
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(mm_int[::-1], dtype="<i2").tobytes())
        f.write(np.ascontiguousarray(code_int[::-1], dtype="<i2").tobytes())


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
            sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden - Warp nicht moeglich.")
        grid = extract_grid_info(where)

    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
    to_wgs84 = Transformer.from_crs(grid["projdef"], "EPSG:4326", always_xy=True)

    ll_x, ll_y, x_max, y_max = native_origin_and_extent(grid, to_proj)
    lon_min, lon_max, lat_min, lat_max = wgs84_bbox_from_perimeter(ll_x, ll_y, x_max, y_max, to_wgs84)
    x_new, y_new, extent = webmercator_target_grid(lon_min, lon_max, lat_min, lat_max)
    print(f"WGS84-BBox: lon [{lon_min:.4f}, {lon_max:.4f}], lat [{lat_min:.4f}, {lat_max:.4f}]")
    print(f"EPSG:3857-Extent [xmin, ymin, xmax, ymax]: {extent}")
    print(f"Zielraster: {len(x_new)} x {len(y_new)} px")

    # === Schritt 1: Warpen ===
    class_merc = warp_classification_to_webmercator(class_array, grid, to_proj, x_new, y_new)

    # === Schritt 2: RV laden ===
    rate_merc = None
    rv_path = rv_path_for(ts)
    try:
        if not rv_path.exists():
            raise FileNotFoundError(f"{rv_path} nicht gefunden")
        rate_merc, rv_grid = load_rv_rate_on_target_grid(rv_path, x_new, y_new)
        assert_compatible_grids(grid, rv_grid)
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        print(f"Warnung: RV-Composite nicht verfuegbar ({e}). Nutze HymecNG-Fallback.", file=sys.stderr)

    # === Schritt 3: Hybrid-Verfeinerung ===
    print("Wende Hybrid-Strategie an (RV + HymecNG)...")
    class_merc = refine_with_hybrid_strategy(class_merc, rate_merc)

    # === Schritt 3b: Hagel -> Regen (Code 32) ===
    hail_mask = convert_hail_to_rain(class_merc)

    # === Schritt 4: Blitze (schreibt Gewittercodes in class_merc) ===
    try:
        hits = apply_lightning_overlay(class_merc, hail_mask, rate_merc, ts, x_new, y_new)
        print(f"{hits} Blitz-Treffer als Gewitter markiert.")
    except requests.RequestException as e:
        print(f"Warnung: Blitzdaten konnten nicht geladen werden ({e}). Ueberspringe Overlay.", file=sys.stderr)

    # === Speichern: nur noch die .bin ===
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mm_int, code_int = make_pixel_arrays(class_merc, rate_merc)
    bin_path = OUT_DIR / BIN_FILENAME
    write_pixel_bin(bin_path, mm_int, code_int, extent, ts)
    print(
        f"{int((mm_int > 0).sum())} Pixel mit mm, {int((code_int != 0).sum())} Pixel mit Code "
        f"({ts:%Y-%m-%d %H:%M} UTC)."
    )
    print(f"Gespeichert: {bin_path} ({bin_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
