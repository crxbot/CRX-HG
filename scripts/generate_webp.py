#!/usr/bin/env python3
import argparse
import re
import struct
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import h5py
import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
from scipy.signal import fftconvolve

# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------

# Dateinamensschema laut DWD: composite_rv_yyyymmdd_HHMM_000-hd5
FILENAME_RE = re.compile(r"composite_rv_(\d{8})_(\d{4})_(\d{3})-hd5")

SRC_DIR = Path("data/rv")
OUT_FILENAME = "gewitter_latest.webp"

# --------------------------------------------------------------------------
# RV-Composite: quantitatives Produkt (quantity=ACRR, mm pro 5 Minuten)
# --------------------------------------------------------------------------

RV_QUANTITY_KEYWORDS = ("ACRR", "RATE", "RR")

# Fallback-Werte, falls die what-Gruppe im HD5 selbst keine Angaben macht
# (Werte laut Aufgabenstellung fuer den RV-Composite)
RV_DEFAULT_GAIN = 0.0009999999317806213
RV_DEFAULT_OFFSET = -0.0009999999317806213
RV_DEFAULT_NODATA = 4294967295.0
RV_DEFAULT_UNDETECT = 0.0

# Ab dieser Rate (mm/5min - Aufloesung des RV-Produkts) gilt ein Pixel als
# "hat messbaren Niederschlag" und kommt fuer das Gewitter-/Blitz-Overlay
# in Frage. Es wird KEINE eigene Niederschlagskarte gerendert - nur die
# Umkreise um Blitztreffer werden eingefaerbt (siehe apply_thunderstorm_overlay).
PRECIP_VISIBLE_THRESHOLD_MM = 0.01

# Fuellen von nodata-Luecken (auf dem NATIVEN Raster, vor dem Warp):
# Umkreis-Mittelwert aus validen Nachbarpixeln
FILL_RADIUS_PX = 8
FILL_MIN_VALID_NEIGHBORS = 4

# --------------------------------------------------------------------------
# Gewitter-/Blitz-Overlay (Klasse 11, kommt NICHT aus der HD5-Datei)
# --------------------------------------------------------------------------

THUNDER_COLOR_HEX = "#FD5FFF"

LIGHTNING_BASE_URL = "https://radar.wetterstation-neustadt.de/blitze/archive/"
LIGHTNING_BACKUP_URL = "https://nowsky.vercel.app/api/lightning"
LIGHTNING_WINDOW_MINUTES = 5  # nur Blitze der letzten 5 Minuten vor dem Radar-Zeitstempel

LIGHTNING_ARCHIVE_TZ = ZoneInfo("Europe/Berlin")

# Radius in Pixeln AUF DEM WEBMERCATOR-ZIELRASTER (siehe WEBMERCATOR_OUT_WIDTH).
# Identisch zu Dokument 1, damit beide Overlays bei vergleichbarem
# Kartenausschnitt visuell gleich groß erscheinen.
LIGHTNING_MARKER_RADIUS_PX = 8

GEWITTER_FOURCC = b"GWTR"

# --------------------------------------------------------------------------
# Geometrie / Ausgabe (identisch zu Dokument 1, fuer gemeinsames Zielraster)
# --------------------------------------------------------------------------
WEBMERCATOR_OUT_WIDTH = 1927
EDGE_SAMPLES = 200
BBOX_MARGIN_DEG = 0.02
EARTH_RADIUS = 6378137.0


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def parse_timestamp(filename: str) -> datetime:
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            f"Dateiname passt nicht zum erwarteten RV-Composite-Schema "
            f"'composite_rv_yyyymmdd_HHMM_000-hd5': {filename}"
        )
    date_str, time_str, _step = m.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M")


def find_data_dataset(h5file: h5py.File) -> h5py.Dataset:
    """Sucht den 2D-Rohdatensatz mit quantity=ACRR (bzw. aehnlich)."""

    candidates = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            candidates.append(obj)

    h5file.visititems(visitor)

    if not candidates:
        raise RuntimeError(
            "Kein 2D-Datensatz in der HDF5-Datei gefunden. "
            "Mit --inspect die Dateistruktur pruefen."
        )

    for ds in candidates:
        what_grp = ds.parent.get("what")
        if what_grp is not None and "quantity" in what_grp.attrs:
            quantity = what_grp.attrs["quantity"]
            if isinstance(quantity, bytes):
                quantity = quantity.decode(errors="ignore")
            quantity = str(quantity).upper()
            if any(k in quantity for k in RV_QUANTITY_KEYWORDS):
                return ds

    # Fallback: ersten gefundenen 2D-Datensatz verwenden
    return candidates[0]


def read_precip_mm(ds: h5py.Dataset) -> np.ndarray:
    """Wandelt die Rohwerte des RV-Datensatzes in mm/5min um.

    nodata -> NaN (keine Messung), undetect -> 0.0 (gemessen, kein Niederschlag).
    gain/offset/nodata/undetect werden bevorzugt aus der what-Gruppe des
    Datensatzes gelesen, sonst greifen die RV_DEFAULT_*-Konstanten."""
    raw = ds[()].astype(np.float64)

    what_grp = ds.parent.get("what")

    def attr_or_default(name, default):
        if what_grp is not None and name in what_grp.attrs:
            return float(what_grp.attrs[name])
        return default

    gain = attr_or_default("gain", RV_DEFAULT_GAIN)
    offset = attr_or_default("offset", RV_DEFAULT_OFFSET)
    nodata = attr_or_default("nodata", RV_DEFAULT_NODATA)
    undetect = attr_or_default("undetect", RV_DEFAULT_UNDETECT)

    nodata_mask = raw == nodata
    undetect_mask = raw == undetect

    precip = raw * gain + offset
    precip[undetect_mask] = 0.0
    precip[nodata_mask] = np.nan

    # Rundungsbedingtes leichtes Minus (offset == -gain) auf 0 klemmen
    negative_valid = (~nodata_mask) & (precip < 0)
    precip[negative_valid] = 0.0

    return precip


def lightning_url_for_timestamp(ts: datetime) -> str:
    ts_utc = ts.replace(tzinfo=timezone.utc)
    ts_local = ts_utc.astimezone(LIGHTNING_ARCHIVE_TZ)
    return f"{LIGHTNING_BASE_URL}{ts_local:%Y-%m-%d-%H%M}.json"


def _parse_iso_to_ms(iso_str: str) -> int:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def fetch_recent_strikes_backup(ts: datetime, minutes: int = LIGHTNING_WINDOW_MINUTES) -> list[tuple[float, float]]:
    """Fallback-Quelle, falls der primaere Archiv-Feed (noch) kein 404-freies
    JSON fuer den angefragten Zeitstempel liefert. Die Backup-API liefert ein
    flaches 'strikes'-Array mit lat/lon/time - wir filtern daraus das
    gewuenschte Fenster [ts - minutes, ts]."""
    resp = requests.get(LIGHTNING_BACKUP_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    ts_utc = ts.replace(tzinfo=timezone.utc)
    end_ms = int(ts_utc.timestamp() * 1000)
    start_ms = end_ms - minutes * 60 * 1000

    strikes = []
    for s in data.get("strikes", []):
        t_ms = _parse_iso_to_ms(s["time"])
        if start_ms <= t_ms <= end_ms:
            strikes.append((s["lat"], s["lon"]))
    return strikes


def fetch_recent_strikes(ts: datetime, minutes: int = LIGHTNING_WINDOW_MINUTES) -> list[tuple[float, float]]:
    url = lightning_url_for_timestamp(ts)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(
                f"Warnung: Primaerer Blitz-Feed liefert 404 ({url}). "
                "Weiche auf Backup-API aus.",
                file=sys.stderr,
            )
            return fetch_recent_strikes_backup(ts, minutes=minutes)
        raise

    data = resp.json()

    ts_utc = ts.replace(tzinfo=timezone.utc)
    end_ms = int(ts_utc.timestamp() * 1000)
    start_ms = end_ms - minutes * 60 * 1000

    strikes = data.get("strikes", [])
    return [
        (s["lat"], s["lon"])
        for s in strikes
        if start_ms <= s.get("t", 0) <= end_ms
    ]


def find_where_group(h5file: h5py.File):
    required = ("projdef", "xsize", "ysize", "xscale", "yscale", "LL_lon", "LL_lat")

    root_where = h5file.get("where")
    if root_where is not None and all(k in root_where.attrs for k in required):
        return root_where

    candidates = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where":
            if all(k in obj.attrs for k in required):
                candidates.append(obj)

    h5file.visititems(visitor)
    return candidates[0] if candidates else None


def extract_grid_info(where_grp: h5py.Group) -> dict:
    def attr_str(name):
        v = where_grp.attrs[name]
        return v.decode() if isinstance(v, bytes) else str(v)

    def attr_num(name):
        return float(where_grp.attrs[name])

    return {
        "projdef": attr_str("projdef"),
        "xsize": int(attr_num("xsize")),
        "ysize": int(attr_num("ysize")),
        "xscale": attr_num("xscale"),
        "yscale": attr_num("yscale"),
        "ll_lon": attr_num("LL_lon"),
        "ll_lat": attr_num("LL_lat"),
    }


# --------------------------------------------------------------------------
# Geometrie / Warp auf EPSG:3857 (uebernommen aus Dokument 1, fuer ein
# gemeinsames Zielraster beider Skripte)
# --------------------------------------------------------------------------
def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_to_lonlat(x, y):
    lon = np.degrees(x / EARTH_RADIUS)
    lat = np.degrees(2 * np.arctan(np.exp(y / EARTH_RADIUS)) - np.pi / 2)
    return lon, lat


def native_origin_and_extent(grid: dict, to_proj: Transformer):
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    x_max = ll_x + grid["xsize"] * grid["xscale"]
    y_max = ll_y + grid["ysize"] * grid["yscale"]
    return ll_x, ll_y, x_max, y_max


def wgs84_bbox_from_perimeter(ll_x, ll_y, x_max, y_max, to_wgs84: Transformer):
    """WGS84-Bounding-Box durch Abtasten des nativen Rasterrands."""
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
    x_new = np.linspace(x_min, x_max, WEBMERCATOR_OUT_WIDTH)
    y_new = np.linspace(y_min, y_max, out_h)
    return x_new, y_new, [x_min, y_min, x_max, y_max]


def nearest_neighbor_warp(
    data: np.ndarray,
    grid: dict,
    to_proj: Transformer,
    x_new: np.ndarray,
    y_new: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Nearest-Neighbor-Warp eines nativen Rasters auf EPSG:3857."""
    xx, yy = np.meshgrid(x_new, y_new)
    lon, lat = webmercator_to_lonlat(xx, yy)
    x_nat, y_nat = to_proj.transform(lon.ravel(), lat.ravel())
    x_nat = np.asarray(x_nat).reshape(xx.shape)
    y_nat = np.asarray(y_nat).reshape(xx.shape)

    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])

    col = np.round((x_nat - ll_x) / grid["xscale"]).astype(np.int64)
    row = np.round(grid["ysize"] - 1 - (y_nat - ll_y) / grid["yscale"]).astype(np.int64)
    valid = (col >= 0) & (col < grid["xsize"]) & (row >= 0) & (row < grid["ysize"])

    out = np.full(xx.shape, fill_value, dtype=np.float64)
    out[valid] = data[row[valid], col[valid]]
    return out


# --------------------------------------------------------------------------
# Gewitter-/Blitz-Overlay auf dem Webmercator-Zielraster
# --------------------------------------------------------------------------
def apply_thunderstorm_overlay(
    rgba: np.ndarray,
    precip_merc: np.ndarray,
    ts: datetime,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Faerbt Pixel mit Blitztreffer UND messbarem Niederschlag
    (>= PRECIP_VISIBLE_THRESHOLD_MM) als Gewitter (Klasse 11) ein.
    Arbeitet auf dem bereits gewarpten Webmercator-Raster."""

    strikes = fetch_recent_strikes(ts, minutes=LIGHTNING_WINDOW_MINUTES)
    print(f"{len(strikes)} Blitze in den letzten {LIGHTNING_WINDOW_MINUTES} Minuten geladen.")

    out_h, out_w = precip_merc.shape
    r, g, b = hex_to_rgb(THUNDER_COLOR_HEX)
    radius = LIGHTNING_MARKER_RADIUS_PX

    safe_precip = np.nan_to_num(precip_merc, nan=-1.0)
    combined_mask = safe_precip >= PRECIP_VISIBLE_THRESHOLD_MM
    thunder_mask = np.zeros((out_h, out_w), dtype=bool)

    offsets = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    circle_mask_full = (dr * dr + dc * dc) <= radius * radius

    x_min, x_max = x_new[0], x_new[-1]
    y_min, y_max = y_new[0], y_new[-1]

    hit_count = 0

    for lat, lon in strikes:
        sx, sy = lonlat_to_webmercator(lon, lat)
        if not (x_min <= sx <= x_max and y_min <= sy <= y_max):
            continue
        col = int(round((sx - x_min) / (x_max - x_min) * (out_w - 1)))
        row = int(round((sy - y_min) / (y_max - y_min) * (out_h - 1)))
        if safe_precip[row, col] < PRECIP_VISIBLE_THRESHOLD_MM:
            continue

        row_start = max(0, row - radius)
        row_end = min(out_h, row + radius + 1)
        col_start = max(0, col - radius)
        col_end = min(out_w, col + radius + 1)

        mask_row_start = row_start - (row - radius)
        mask_row_end = mask_row_start + (row_end - row_start)
        mask_col_start = col_start - (col - radius)
        mask_col_end = mask_col_start + (col_end - col_start)
        circle_mask = circle_mask_full[
            mask_row_start:mask_row_end, mask_col_start:mask_col_end
        ]

        area = combined_mask[row_start:row_end, col_start:col_end]
        final_mask = circle_mask & area
        rgba[row_start:row_end, col_start:col_end][final_mask] = (r, g, b, 255)
        thunder_mask[row_start:row_end, col_start:col_end] |= final_mask

        hit_count += 1

    return hit_count, thunder_mask


def embed_thunderstorm_chunk(
    webp_path: Path, thunder_mask: np.ndarray, extent: list[float]
) -> None:
    height, width = thunder_mask.shape
    # 0 = kein Gewitter, 1 = Gewitter. Kein Quantisierungsverlust moeglich,
    # da die Maske ohnehin nur 0/1 kennt - das Quantum-Feld wird trotzdem mit
    # eingebettet, damit das Chunk-Format zu anderen DVAL-artigen Chunks
    # kompatibel bleibt.
    quant = thunder_mask.astype(np.int16)

    header = struct.pack("<BBII", 2, 1, width, height)
    header += struct.pack("<4d", *extent)
    header += struct.pack("<d", 1.0)  # Quantum (bei 0/1-Maske irrelevant)
    compressed = zlib.compress(np.ascontiguousarray(quant, dtype="<i2").tobytes(), level=9)
    payload = header + compressed

    size = len(payload)
    chunk = GEWITTER_FOURCC + struct.pack("<I", size) + payload
    if size % 2 == 1:
        chunk += b"\x00"

    with open(webp_path, "rb") as f:
        content = f.read()

    if content[0:4] != b"RIFF" or content[8:12] != b"WEBP":
        raise ValueError(f"{webp_path} ist keine gueltige WebP-Datei (RIFF/WEBP-Header fehlt)")

    riff_size = struct.unpack("<I", content[4:8])[0]
    new_riff_size = riff_size + len(chunk)

    with open(webp_path, "wb") as f:
        f.write(content[:4])
        f.write(struct.pack("<I", new_riff_size))
        f.write(content[8:])
        f.write(chunk)


def save_webp(rgba_array: np.ndarray, out_path: Path) -> None:
    img = Image.fromarray(rgba_array, mode="RGBA")
    img.save(out_path, format="WEBP", lossless=True)


# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------

def main() -> None:
    candidates = sorted(
        p for p in SRC_DIR.glob("composite_rv_*-hd5")
        if FILENAME_RE.match(p.name)
    )
    if not candidates:
        sys.exit(f"Keine RV-Composite-Datei in {SRC_DIR} gefunden.")

    src_path = candidates[-1]
    filename = src_path.name
    ts = parse_timestamp(filename)

    with h5py.File(src_path, "r") as f:
        ds = find_data_dataset(f)
        precip = read_precip_mm(ds)
        where_grp = find_where_group(f)
        grid_info = extract_grid_info(where_grp) if where_grp is not None else None

    if grid_info is None:
        sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden - Warp nicht moeglich.")

    # Fehlende Messungen (nodata) aus der Nachbarschaft auffuellen (auf dem
    # nativen Raster), damit Blitze am Rand von Messluecken den
    # Niederschlag in der Naehe trotzdem erkennen

    to_proj = Transformer.from_crs("EPSG:4326", grid_info["projdef"], always_xy=True)
    to_wgs84 = Transformer.from_crs(grid_info["projdef"], "EPSG:4326", always_xy=True)

    ll_x, ll_y, x_max, y_max = native_origin_and_extent(grid_info, to_proj)
    lon_min, lon_max, lat_min, lat_max = wgs84_bbox_from_perimeter(ll_x, ll_y, x_max, y_max, to_wgs84)
    x_new, y_new, extent = webmercator_target_grid(lon_min, lon_max, lat_min, lat_max)
    print(f"WGS84-BBox: lon [{lon_min:.4f}, {lon_max:.4f}], lat [{lat_min:.4f}, {lat_max:.4f}]")
    print(f"EPSG:3857-Extent [xmin, ymin, xmax, ymax]: {extent}")
    print(f"Zielraster: {len(x_new)} x {len(y_new)} px")

    # Niederschlag unabhaengig auf das gemeinsame Webmercator-Zielraster warpen
    precip_merc = nearest_neighbor_warp(precip, grid_info, to_proj, x_new, y_new, fill_value=np.nan)

    out_h, out_w = len(y_new), len(x_new)
    # Vollstaendig transparentes Bild - es wird NUR der Gewitter-/Blitz-
    # Umkreis eingefaerbt, keine eigene Niederschlagskarte gerendert.
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    thunder_mask = None
    try:
        hits, thunder_mask = apply_thunderstorm_overlay(rgba, precip_merc, ts, x_new, y_new)
        print(f"{hits} Blitz-Treffer als Gewitter (Klasse 11, {THUNDER_COLOR_HEX}) eingefaerbt.")
    except requests.RequestException as e:
        print(
            f"Warnung: Blitzdaten konnten nicht geladen werden ({e}). "
            "Ueberspringe Gewitter-Overlay.",
            file=sys.stderr,
        )
        thunder_mask = np.zeros((out_h, out_w), dtype=bool)

    out_path = SRC_DIR / OUT_FILENAME
    # Zeilen umdrehen: y_new laeuft von Sued nach Nord, Bilder von oben nach unten
    save_webp(rgba[::-1], out_path)

    embed_thunderstorm_chunk(out_path, thunder_mask[::-1], extent)
    print(
        f"Gewitter-Datenchunk ({GEWITTER_FOURCC.decode()}) eingebettet: "
        f"{int(thunder_mask.sum())} Pixel als Gewitter markiert."
    )

    print(f"Gespeichert: {out_path}")


if __name__ == "__main__":
    main()
