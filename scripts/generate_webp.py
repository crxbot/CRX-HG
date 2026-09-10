#!/usr/bin/env python3
import argparse
import re
import struct
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo
import h5py
import numpy as np
import requests
from PIL import Image
from pyproj import Transformer

# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------

BASE_URL = "https://opendata.dwd.de/weather/radar/composite/hymecng/"

# Dateinamensschema laut DWD: composite_HymecNG_yyyymmdd_HHMM_000-hd5
FILENAME_RE = re.compile(r"composite_HymecNG_(\d{8})_(\d{4})_(\d{3})-hd5")

# Klassen, ueber die sich das Gewitter-/Blitz-Overlay legen darf. Bewusst
# NICHT einfach "= set(COLOR_MAP_HEX.keys())", sondern eine eigene,
# unabhaengige Liste: Hagel/Graupelkoerner (9/10) sind zwar eingefaerbt,
# sollen aber NIE vom Gewitter-Kreis ueberschrieben werden, liegen also quasi
# "ueber" dem Gewitter. Nur an Pixeln mit einer dieser Klassen wird das
# Gewitter-Overlay ueberhaupt SICHTBAR gesetzt.
PRECIP_CLASSES = {2, 3, 5, 6, 7, 1, 2}

# Klassen, die fuer das EINGEBETTETE Gewitter-Datenfeld (siehe
# embed_thunderstorm_chunk) als "hier ist Gewitter" zaehlen. Bewusst
# groesser als PRECIP_CLASSES: hier zaehlen auch Hagel/Graupelkorn (9/10)
# dazu, weil ein Blitz unter Hagel im DATENFELD trotzdem als Gewitter
# gemeldet werden soll - auch wenn im BILD an dieser Stelle weiterhin die
# Hagel-Farbe zu sehen ist (Hagel liegt visuell einfach "oben drauf").
THUNDER_DATA_CLASSES = PRECIP_CLASSES | {9, 10}

# --------------------------------------------------------------------------
# Gewitter-/Blitz-Overlay (Klasse 11, kommt NICHT aus der HD5-Datei)
# --------------------------------------------------------------------------

THUNDER_COLOR_HEX = "#FD5FFF"

LIGHTNING_BASE_URL = "https://radar.wetterstation-neustadt.de/blitze/archive/"
LIGHTNING_WINDOW_MINUTES = 5  # nur Blitze der letzten 5 Minuten vor dem Radar-Zeitstempel

# WICHTIG: Die DWD-HD5-Dateien sind nach UTC benannt, die Blitz-Archivdateien
# auf radar.wetterstation-neustadt.de aber nach LOKALER Zeit (Europe/Berlin,
# CET im Winter / CEST im Sommer). Ein HD5-Zeitstempel "1440" (=14:40 UTC)
# entspricht im Sommer also der Archivdatei "..._1640.json" (16:40 MESZ).
# Ohne diese Umrechnung wird die falsche Archivdatei abgerufen und die
# 5-Minuten-Filterung liefert faelschlicherweise 0 Treffer.
LIGHTNING_ARCHIVE_TZ = ZoneInfo("Europe/Berlin")

# Radius (in Pixeln) fuer jeden Blitz-Marker (Kreis um den Treffer-Pixel).
# Ein einzelner Pixel geht auf einem grossen Radarbild optisch unter - mit
# einem kleinen Radius wird jeder Blitz als deutlich sichtbarer, runder Fleck
# gezeichnet, der klar UEBER den anderen Niederschlags-Codes liegt (wird als
# allerletzter Schritt in rgba geschrieben und ueberschreibt daher jede
# darunterliegende Farbe).
LIGHTNING_MARKER_RADIUS_PX = 6

# FourCC des privaten RIFF-Chunks, in dem die Gewitter-Maske eingebettet
# wird (analog zum DVAL-Chunk in anderen Skripten, aber eigener Name, damit
# ein Frontend zwischen "Wetterdatenfeld" (DVAL) und "Gewitter-Maske"
# (GWTR) unterscheiden kann).
GEWITTER_FOURCC = b"GWTR"


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def list_remote_files() -> list[str]:
    """Listet alle HymecNG-Dateien im DWD-Open-Data-Verzeichnis auf."""
    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()
    names = sorted(set(m.group(0) for m in FILENAME_RE.finditer(resp.text)))
    return names


def latest_remote_file() -> str:
    names = list_remote_files()
    if not names:
        raise RuntimeError(f"Keine HymecNG-Dateien unter {BASE_URL} gefunden.")

    def sort_key(name: str):
        m = FILENAME_RE.match(name)
        return m.group(1) + m.group(2) + m.group(3)  # yyyymmdd + HHMM + PPP

    return sorted(names, key=sort_key)[-1]


def download_file(filename: str, dest_dir: Path) -> Path:
    url = urljoin(BASE_URL, filename)
    dest = dest_dir / filename
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def parse_timestamp(filename: str) -> datetime:
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            f"Dateiname passt nicht zum erwarteten HymecNG-Schema "
            f"'composite_HymecNG_yyyymmdd_HHMM_000-hd5': {filename}"
        )
    date_str, time_str, _step = m.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M")


def find_classification_dataset(h5file: h5py.File) -> h5py.Dataset:
    """
    Sucht den 2D-Datensatz mit der Niederschlagsart-Klassifikation in der
    ODIM_H5-Datei. Bevorzugt einen Datensatz, dessen 'what/quantity'-Attribut
    auf eine Klassifikation hindeutet (z.B. enthaelt CLASS/PRECIP/TYPE),
    faellt sonst auf den ersten passenden 2D-Datensatz zurueck.
    """
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
            if any(k in quantity for k in ("CLASS", "PRECIP", "HCLASS", "TYPE")):
                return ds

    # Fallback: ersten gefundenen 2D-Datensatz verwenden
    return candidates[0]


def lightning_url_for_timestamp(ts: datetime) -> str:
    """
    'ts' ist der (naive) Zeitstempel aus dem HD5-Dateinamen und wird als UTC
    interpretiert (DWD-Konvention). Die Blitz-Archivdateien sind jedoch nach
    lokaler Zeit (Europe/Berlin) benannt, deshalb hier zuerst umrechnen.
    """
    ts_utc = ts.replace(tzinfo=timezone.utc)
    ts_local = ts_utc.astimezone(LIGHTNING_ARCHIVE_TZ)
    return f"{LIGHTNING_BASE_URL}{ts_local:%Y-%m-%d-%H%M}.json"


def fetch_recent_strikes(ts: datetime, minutes: int = LIGHTNING_WINDOW_MINUTES) -> list[tuple[float, float]]:
    """
    Laedt die Blitzdaten-Archivdatei fuer den Zeitstempel 'ts' (z.B. 16:40) und
    gibt alle Blitze (lat, lon) zurueck, die im Zeitfenster
    [ts - minutes, ts] aufgetreten sind (z.B. 16:35-16:40).
    """
    url = lightning_url_for_timestamp(ts)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
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
    """
    Sucht die ODIM 'where'-Gruppe mit den Projektions-/Gitterinformationen
    (projdef, xsize, ysize, xscale, yscale, LL_lon, LL_lat). Bevorzugt die
    Gruppe auf Root-Ebene ('/where'), wie im ODIM-Composite-Standard ueblich.
    """
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
    """
    Liest die benoetigten Projektions-/Gitterattribute aus der ODIM
    'where'-Gruppe in ein einfaches dict aus purem Python-Werten aus.
    Muss aufgerufen werden, WAEHREND die HDF5-Datei noch geoeffnet ist -
    h5py-Objekte werden ungueltig, sobald die Datei geschlossen wird.
    """

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


def build_pixel_mapper(grid_info: dict):
    """
    Baut aus den (bereits als reine Python-Werte extrahierten) Gitterdaten
    eine Funktion lat/lon -> (row, col), die eine geografische Position auf
    die passende Pixelposition im Klassifikations-Array abbildet (oder None,
    falls ausserhalb des Gitters).
    """
    projdef = grid_info["projdef"]
    xsize = grid_info["xsize"]
    ysize = grid_info["ysize"]
    xscale = grid_info["xscale"]
    yscale = grid_info["yscale"]
    ll_lon = grid_info["ll_lon"]
    ll_lat = grid_info["ll_lat"]

    transformer = Transformer.from_crs("EPSG:4326", projdef, always_xy=True)
    ll_x, ll_y = transformer.transform(ll_lon, ll_lat)

    def latlon_to_pixel(lat: float, lon: float):
        x, y = transformer.transform(lon, lat)
        col = int((x - ll_x) / xscale)
        # Zeile 0 liegt (ODIM-Konvention) am Nordrand (UR), daher von oben zaehlen
        row = ysize - 1 - int((y - ll_y) / yscale)
        if 0 <= row < ysize and 0 <= col < xsize:
            return row, col
        return None

    return latlon_to_pixel


def apply_thunderstorm_overlay(
    rgba: np.ndarray, class_array: np.ndarray, ts: datetime, grid_info: dict
) -> tuple[int, np.ndarray]:
    """
    Faerbt fuer jeden Blitz der letzten LIGHTNING_WINDOW_MINUTES Minuten einen
    kreisfoermigen Bereich (Radius LIGHTNING_MARKER_RADIUS_PX) um den
    Treffer-Pixel (mittig) in #FD5FFF ein - aber NUR an den Pixeln
    INNERHALB dieses Kreises, an denen laut HymecNG tatsaechlich eine der
    sichtbar eingefaerbten Niederschlagsklassen vorliegt (PRECIP_CLASSES).
    Hagel/Graupelkorn (9/10) werden dabei NIE ueberschrieben und bleiben
    sichtbar "ueber" dem Gewitter.

    Zusaetzlich wird eine zweite, unabhaengige Maske (thunder_mask)
    berechnet: True ueberall dort, wo ein Blitz mit einer beliebigen
    Niederschlags- ODER Hagelklasse (THUNDER_DATA_CLASSES) zusammenfaellt -
    also auch unter Hagel/Graupelkorn, wo im BILD keine sichtbare Aenderung
    passiert. Diese Maske ist NICHT fuers Bild gedacht, sondern zum
    Einbetten als eigener Datenchunk (siehe embed_thunderstorm_chunk), damit
    z.B. ein Frontend "ist hier Gewitter?" auch dort korrekt beantworten
    kann, wo visuell die Hagel-Farbe zu sehen ist.

    Der Treffer-Pixel selbst muss mindestens THUNDER_DATA_CLASSES angehoeren,
    damit der Blitz ueberhaupt als Treffer zaehlt (sonst z.B. Blitz ueber
    trockenem Boden -> kein Treffer).

    Gibt (hit_count, thunder_mask) zurueck.
    """
    mapper = build_pixel_mapper(grid_info)
    strikes = fetch_recent_strikes(ts, minutes=LIGHTNING_WINDOW_MINUTES)
    print(f"{len(strikes)} Blitze in den letzten {LIGHTNING_WINDOW_MINUTES} Minuten geladen.")

    ysize, xsize = class_array.shape
    r, g, b = hex_to_rgb(THUNDER_COLOR_HEX)
    radius = LIGHTNING_MARKER_RADIUS_PX

    # Zwei getrennte Masken: paint_mask fuers sichtbare Bild (schmal, ohne
    # Hagel), data_mask fuers eingebettete Datenfeld (breiter, inkl. Hagel).
    combined_mask = np.isin(class_array, list(THUNDER_DATA_CLASSES))
    thunder_mask = np.zeros((ysize, xsize), dtype=bool)

    # Vorgefertigte kreisfoermige Maske (relativ zum Mittelpunkt) einmal
    # ausserhalb der Schleife berechnen, um sie fuer jeden Treffer wieder-
    # zuverwenden.
    offsets = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    circle_mask_full = (dr * dr + dc * dc) <= radius * radius

    hit_count = 0

    for lat, lon in strikes:
        pixel = mapper(lat, lon)
        if pixel is None:
            continue
        row, col = pixel
        if class_array[row, col] not in THUNDER_DATA_CLASSES:
            continue

        # Ausschnitts-Grenzen um den Treffer-Pixel, an den Bildraendern
        # abgeschnitten - dabei auch die passenden Teile der kreisfoermigen
        # Maske mit abschneiden, damit sie zum (evtl. verkleinerten)
        # Ausschnitt passt.
        row_start = max(0, row - radius)
        row_end = min(ysize, row + radius + 1)
        col_start = max(0, col - radius)
        col_end = min(xsize, col + radius + 1)

        mask_row_start = row_start - (row - radius)
        mask_row_end = mask_row_start + (row_end - row_start)
        mask_col_start = col_start - (col - radius)
        mask_col_end = mask_col_start + (col_end - col_start)
        circle_mask = circle_mask_full[
            mask_row_start:mask_row_end, mask_col_start:mask_col_end
        ]

        # Sichtbares Bild: nur echte Niederschlagsklassen einfaerben - Hagel
        # bleibt sichtbar oben, wird hier also bewusst NICHT ueberschrieben.
        paint_area = combined_mask[row_start:row_end, col_start:col_end]
        final_paint_mask = circle_mask & paint_area
        rgba[row_start:row_end, col_start:col_end][final_paint_mask] = (r, g, b, 255)

        # Eingebettetes Datenfeld: hier zaehlt auch Hagel/Graupelkorn als
        # "Gewitter", unabhaengig davon, was im Bild sichtbar ist.
        data_area = combined_mask[row_start:row_end, col_start:col_end]
        final_data_mask = circle_mask & data_area
        thunder_mask[row_start:row_end, col_start:col_end] |= final_data_mask

        hit_count += 1

    return hit_count, thunder_mask


def compute_projection_extent(grid_info: dict) -> list[float]:
    """
    Berechnet die Ausdehnung des Klassifikations-Gitters in den Metern der
    ODIM-Projektion (grid_info['projdef']): [x_min, y_min, x_max, y_max].
    Passend zur row0=Norden-Konvention von class_array/rgba/thunder_mask.
    Wird beim Einbetten des Gewitter-Datenchunks gebraucht, damit ein
    Frontend die eingebetteten Pixel geografisch verorten kann.
    """
    transformer = Transformer.from_crs("EPSG:4326", grid_info["projdef"], always_xy=True)
    ll_x, ll_y = transformer.transform(grid_info["ll_lon"], grid_info["ll_lat"])
    x_max = ll_x + grid_info["xsize"] * grid_info["xscale"]
    y_max = ll_y + grid_info["ysize"] * grid_info["yscale"]
    return [ll_x, ll_y, x_max, y_max]


def embed_thunderstorm_chunk(
    webp_path: Path, thunder_mask: np.ndarray, extent: list[float]
) -> None:
    """
    Haengt die Gewitter-Maske (thunder_mask, siehe apply_thunderstorm_overlay)
    als eigenen, privaten RIFF-Chunk (FourCC GEWITTER_FOURCC = "GWTR") an die
    bereits gespeicherte WEBP-Datei an. Bewusst getrennt vom sichtbaren
    Bildinhalt: so kann z.B. ein Frontend beim Hovern abfragen "ist hier
    Gewitter?" und bekommt dort auch dann "ja", wo im Bild wegen Hagel eine
    andere Farbe zu sehen ist - Hagel liegt visuell ueber dem Gewitter, im
    Datenfeld bleibt Gewitter aber True.

    thunder_mask: 2D bool-Array, row0 = Norden (wie class_array/rgba).
    extent: [x_min, y_min, x_max, y_max] in den Metern der ODIM-Projektion
            (siehe compute_projection_extent).
    """
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
        chunk += b"\x00"  # RIFF-Padding auf gerade Laenge, zaehlt nicht zu size

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
    src_dir = Path("data/hg")

    candidates = sorted(
        p for p in src_dir.glob("composite_HymecNG_*-hd5")
        if FILENAME_RE.match(p.name)
    )
    if not candidates:
        sys.exit(f"Keine HymecNG-Datei in {src_dir} gefunden.")

    src_path = candidates[-1]
    filename = src_path.name
    ts = parse_timestamp(filename)

    with h5py.File(src_path, "r") as f:
        ds = find_classification_dataset(f)
        class_array = ds[()]
        where_grp = find_where_group(f)
        grid_info = extract_grid_info(where_grp) if where_grp is not None else None

    height, width = class_array.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    thunder_mask = None
    thunder_extent = None
    if grid_info is None:
        print(
            "Warnung: Keine 'where'-Projektionsinfo in der HD5-Datei gefunden "
            "- Gewitter/Blitz-Overlay wird uebersprungen.",
            file=sys.stderr,
        )
    else:
        try:
            hits, thunder_mask = apply_thunderstorm_overlay(rgba, class_array, ts, grid_info)
            thunder_extent = compute_projection_extent(grid_info)
            print(f"{hits} Blitz-Treffer als Gewitter (Klasse 11, {THUNDER_COLOR_HEX}) eingefaerbt.")
        except requests.RequestException as e:
            print(
                f"Warnung: Blitzdaten konnten nicht geladen werden ({e}). "
                "Ueberspringe Gewitter-Overlay.",
                file=sys.stderr,
            )

    out_name = f"niederschlagsart_{ts:%Y%m%d}_{ts:%H%M}.webp"
    out_path = src_dir / out_name
    save_webp(rgba, out_path)

    if thunder_mask is not None:
        embed_thunderstorm_chunk(out_path, thunder_mask, thunder_extent)
        print(
            f"Gewitter-Datenchunk ({GEWITTER_FOURCC.decode()}) eingebettet: "
            f"{int(thunder_mask.sum())} Pixel als Gewitter markiert."
        )

    print(f"Gespeichert: {out_path}")


if __name__ == "__main__":
    main()
