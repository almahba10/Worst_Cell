"""
wc_engine.py — Drive-Test "Worst Cells" generation engine (v4).

Design:
  * Same (eNB, Cell) from CS and PS kept as SEPARATE records.
  * Service Rate kept per source file.
  * RF ← PS+CS · Throughput ← PS · MOS ← CS · CST VoLTE ← VoLTE Export
  * Blank DB matches remain eligible as Worst Cells.
  * Column mapping is supplied by the UI (per file-type, including 4G DB).
  * Worst-cell definition and classification thresholds are user-editable.
  * Extra definition / classification rules (from known KPIs) are ANDed / applied first.
  * PCI-EARFCN Serving Sector rows match the 4G DB on
    PCI + EARFCN DL + eNB_Part + Cell_Part (unique hits only).
  * A contiguous PCI-EARFCN run (no new Serving Sector between rows) is one
    serving cell even when TEMS writes conflicting eNB/Cell identities; the
    valid DB match and neighboring named sectors decide the cell, and the
    sample count follows the resolved CELLName.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical metric keys used internally after mapping is applied
# ---------------------------------------------------------------------------
CANON = {
    "enb": "Cell Identity (eNB Part)",
    "cell": "Cell Identity (Cell Part)",
    "rsrp": "Serving Cell RSRP (dBm)",
    "rsrq": "Serving Cell RSRQ (dB)",
    "sinr": "Serving Cell RS SINR (dB)",
    "rssi": "Serving Cell Channel RSSI (dBm)",
    "dist": "Serving Sector Distance-to-cell (km)",
    "angle": "Serving Sector Relative Angle vs Planned Antenna Azimuth",
    "sector": "Serving Sector",
    "pdsch": "PDSCH Phy Throughput (kbps)",
    "ftp_dl": "FTP Download Throughput Mean (kbps)",
    "ftp_ul": "FTP Upload Throughput Mean (kbps)",
    "mcs": "PDSCH MCS C1 TB0",
    "bler": "PDSCH BLER Carrier 1 (%)",
    "rb": "PDSCH Resource Block Allocation Count Carrier 1",
    "tm": "Transmission Mode Carrier 1",
    "polqa": "POLQA SWB Score DL",
    "cs_polqa": "CS POLQA SWB Score DL",
    "aqm": "AQM Score DL",
    "setup": "Voice_Call_Setup_Time_sec",
    # 4G Database (join keys + site attributes)
    "db_enb": "eNodBID",
    "db_cell": "CellID",
    "site": "ENODEBNAME",
    "cellname": "CELLName",
    "lon": "LONgitude",
    "lat": "LATitude",
    "pci": "PCI",
    "height": "Height",
    "az": "Azimuth",
    "mtilt": "Mechanical_Downtilt",
    "etilt": "Electrical_Downtilt",
    "earfcn": "DLEARFCN",
    "bw": "Downlink_bandwidth",
    "rspow": "RS_Power",
    "lb": "In Load Balance Report",
    "pwr": "In Power Report",
}

# (canonical_key, display_label, mandatory)
MAPPING_SPECS: dict[str, list[tuple[str, str, bool]]] = {
    "PS": [
        ("enb", "Cell Identity (eNB Part)", True),
        ("cell", "Cell Identity (Cell Part)", True),
        ("sector", "Serving Sector", False),
        ("rsrp", "Serving Cell RSRP (dBm)", True),
        ("sinr", "Serving Cell RS SINR (dB)", True),
        ("dist", "Serving Sector Distance-to-cell (km)", True),
        ("pdsch", "PDSCH Phy Throughput (kbps)", True),
        ("rsrq", "Serving Cell RSRQ (dB)", False),
        ("rssi", "Serving Cell Channel RSSI (dBm)", False),
        ("angle", "Serving Sector Relative Angle vs Planned Antenna Azimuth", False),
        ("ftp_dl", "FTP Download Throughput Mean (kbps)", False),
        ("ftp_ul", "FTP Upload Throughput Mean (kbps)", False),
        ("mcs", "PDSCH MCS C1 TB0", False),
        ("bler", "PDSCH BLER Carrier 1 (%)", False),
        ("rb", "PDSCH Resource Block Allocation Count Carrier 1", False),
        ("tm", "Transmission Mode Carrier 1", False),
    ],
    "CS": [
        ("enb", "Cell Identity (eNB Part)", True),
        ("cell", "Cell Identity (Cell Part)", True),
        ("sector", "Serving Sector", False),
        ("rsrp", "Serving Cell RSRP (dBm)", True),
        ("sinr", "Serving Cell RS SINR (dB)", True),
        ("dist", "Serving Sector Distance-to-cell (km)", True),
        ("polqa", "POLQA SWB Score DL", False),
        ("cs_polqa", "CS POLQA SWB Score DL", False),
        ("aqm", "AQM Score DL", False),
        ("rsrq", "Serving Cell RSRQ (dB)", False),
        ("rssi", "Serving Cell Channel RSSI (dBm)", False),
        ("angle", "Serving Sector Relative Angle vs Planned Antenna Azimuth", False),
        ("tm", "Transmission Mode Carrier 1", False),
        ("setup", "Voice_Call_Setup_Time_sec", False),
    ],
    "VoLTE": [
        ("enb", "Cell Identity (eNB Part)", True),
        ("cell", "Cell Identity (Cell Part)", True),
        ("sector", "Serving Sector", False),
        ("rsrp", "Serving Cell RSRP (dBm)", True),
        ("sinr", "Serving Cell RS SINR (dB)", True),
        ("dist", "Serving Sector Distance-to-cell (km)", True),
        ("setup", "Voice_Call_Setup_Time_sec", True),
        ("rsrq", "Serving Cell RSRQ (dB)", False),
        ("rssi", "Serving Cell Channel RSSI (dBm)", False),
        ("angle", "Serving Sector Relative Angle vs Planned Antenna Azimuth", False),
    ],
    "DB": [
        ("db_enb", "eNodeB ID (join key)", True),
        ("db_cell", "Cell ID (join key)", True),
        ("site", "Site Name / ENODEBNAME", True),
        ("cellname", "Cell Name / CELLName", True),
        ("lon", "Longitude", False),
        ("lat", "Latitude", False),
        ("pci", "PCI", False),
        ("height", "Antenna Height", False),
        ("az", "Azimuth", False),
        ("mtilt", "Mechanical Downtilt", False),
        ("etilt", "Electrical Downtilt", False),
        ("earfcn", "DL EARFCN", False),
        ("bw", "Downlink bandwidth / MAX_TX_POWER", False),
        ("rspow", "RS Power", False),
        ("lb", "In Load Balance Report", False),
        ("pwr", "In Power Report", False),
    ],
}

# CS: at least one of these MOS columns is required (special-cased in validate)
CS_MOS_KEYS = ("polqa", "cs_polqa", "aqm")
# Auto-mapped even when optional — needed to count rows and extract CELLName
ALWAYS_AUTO_MAP = {"sector"}

_ALIASES: dict[str, list[str]] = {
    "enb": [
        "cell identity (enb part)", "enb part", "enodeb id", "enodbid", "enb",
        "cell identity(enb part)", "serving cell identity (enb part)",
    ],
    "cell": [
        "cell identity (cell part)", "cell part", "cellid", "cell id",
        "cell identity(cell part)", "serving cell identity (cell part)",
    ],
    "rsrp": [
        "serving cell rsrp (dbm)", "serving cell rsrp", "rsrp (dbm)", "rsrp",
        "serving rsrp", "cell rsrp",
    ],
    "rsrq": ["serving cell rsrq (db)", "serving cell rsrq", "rsrq (db)", "rsrq"],
    "sinr": [
        "serving cell rs sinr (db)", "serving cell rs sinr", "serving cell sinr (db)",
        "serving cell sinr", "rs sinr (db)", "sinr (db)", "sinr",
    ],
    "rssi": [
        "serving cell channel rssi (dbm)", "serving cell channel rssi",
        "serving cell rssi (dbm)", "rssi (dbm)", "rssi",
    ],
    "dist": [
        "serving sector distance-to-cell (km)", "serving sector distance-to-cell",
        "distance-to-cell (km)", "distance to cell (km)", "distance (km)",
        "serving distance", "serv_dist",
    ],
    "angle": [
        "serving sector relative angle vs planned antenna azimuth",
        "relative angle", "azimuth delta", "azimuth orientation delta",
    ],
    "sector": ["serving sector", "serving cell name"],
    "pdsch": [
        "pdsch phy throughput (kbps)", "pdsch phy throughput",
        "netpdschthrpt", "schpdschthrpt", "pdsch throughput",
    ],
    "ftp_dl": [
        "ftp download throughput mean (kbps)", "ftp download throughput mean",
        "ftp download throughput", "dl throughput",
    ],
    "ftp_ul": [
        "ftp upload throughput mean (kbps)", "ftp upload throughput mean",
        "ftp upload throughput", "ul throughput",
    ],
    "mcs": ["pdsch mcs c1 tb0", "pdsch mcs", "avgmcs", "mcs"],
    "bler": ["pdsch bler carrier 1 (%)", "pdsch bler carrier 1", "bler", "pdsch bler"],
    "rb": [
        "pdsch resource block allocation count carrier 1",
        "number of pdsch resource blocks", "numrbs", "resource blocks",
    ],
    "tm": ["transmission mode carrier 1", "transmission mode", "tm"],
    "polqa": ["polqa swb score dl", "polqa score dl", "polqa"],
    "cs_polqa": ["cs polqa swb score dl", "cs polqa"],
    "aqm": ["aqm score dl", "aqm"],
    "setup": [
        "voice_call_setup_time_sec", "voice call setup time sec",
        "voice call setup time", "call setup time", "setup time",
    ],
    "db_enb": [
        "enodbid", "enodeb id", "enbid", "enb_part", "enb id", "enodebid",
        "cell identity (enb part)", "site id", "eid",
    ],
    "db_cell": [
        "cellid", "cell id", "cell_part", "cid", "cell identity (cell part)",
        "local cell id",
    ],
    "site": ["site name", "enodebname", "enodeb name", "site", "site_name"],
    "cellname": ["cell id", "cellname", "cell name", "cell_id", "sector name"],
    "lon": ["longtude", "longitude", "long", "lon", "x"],
    "lat": ["latitude", "lat", "y"],
    "pci": ["pci", "physical cell id", "phy cell id"],
    "height": ["ant_height", "antenna height", "height", "ant height"],
    "az": ["azimuth", "az"],
    "mtilt": ["m_tilt", "mechanical_downtilt", "mechanical downtilt", "m tilt"],
    "etilt": ["e_tilt", "electrical_downtilt", "electrical downtilt", "e tilt"],
    "earfcn": ["uarfcn downlink", "earfcn downlink", "dlearfcn", "dl earfcn", "earfcn"],
    "bw": ["max_tx_power", "downlink_bandwidth", "dl bandwidth", "bandwidth"],
    "rspow": ["ref_power", "rs_power", "rs power", "reference power"],
    "lb": ["in load balance report", "load balance"],
    "pwr": ["in power report", "power report"],
}


def _norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", str(h).strip().strip("'").lstrip("\ufeff")).lower()


def auto_map_headers(
    headers: list[str], file_type: str, auto_map_optional: bool = False
) -> dict[str, str | None]:
    """Return {canonical_key: matched_header_or_None}. Strict 1:1.
    When auto_map_optional is False, only mandatory metrics (+ CS MOS group
    + Serving Sector) are auto-mapped.
    """
    specs = MAPPING_SPECS.get(file_type, [])
    remaining_norm = {_norm_header(h): h for h in headers}
    used: set[str] = set()
    result: dict[str, str | None] = {}

    for key, _label, _mand in specs:
        is_required = (
            _mand
            or (file_type == "CS" and key in CS_MOS_KEYS)
            or key in ALWAYS_AUTO_MAP
        )
        if not is_required and not auto_map_optional:
            result[key] = None
            continue

        matched = None
        canon_name = CANON.get(key, key)
        cn = _norm_header(canon_name)
        if cn in remaining_norm and remaining_norm[cn] not in used:
            matched = remaining_norm[cn]
        if matched is None:
            for alias in _ALIASES.get(key, []):
                if alias in remaining_norm and remaining_norm[alias] not in used:
                    matched = remaining_norm[alias]
                    break
        if matched is None:
            tokens = set(re.findall(r"[a-z0-9]+", key + " " + _norm_header(canon_name)))
            best, best_score = None, 0.0
            for nh, orig in remaining_norm.items():
                if orig in used:
                    continue
                ht = set(re.findall(r"[a-z0-9]+", nh))
                if not ht:
                    continue
                overlap = len(tokens & ht) / max(len(tokens | ht), 1)
                if key in nh or (key == "pdsch" and "pdsch" in nh) or (
                    key == "dist" and "distance" in nh
                ):
                    overlap += 0.25
                if overlap > best_score and overlap >= 0.45:
                    best_score, best = overlap, orig
            matched = best
        if matched is not None:
            used.add(matched)
        result[key] = matched
    return result


def validate_mapping(mapping: dict[str, str | None], file_type: str) -> list[str]:
    """Return list of missing mandatory metric labels."""
    missing = []
    specs = MAPPING_SPECS.get(file_type, [])
    for key, label, mand in specs:
        if file_type == "CS" and key in CS_MOS_KEYS:
            continue  # group-checked below
        if mand and not mapping.get(key):
            missing.append(label)
    if file_type == "CS":
        if not any(mapping.get(k) for k in CS_MOS_KEYS):
            missing.append("POLQA / CS POLQA / AQM Score DL (at least one)")
    return missing


def display_specs(file_type: str, include_optional: bool = False) -> list[tuple[str, str, bool]]:
    """Rows shown in the mapping UI. Mandatory (+ CS MOS group + Serving Sector) by default."""
    specs = MAPPING_SPECS.get(file_type, [])
    if include_optional:
        return specs
    out = []
    for key, label, mand in specs:
        if mand:
            out.append((key, label, mand))
        elif file_type == "CS" and key in CS_MOS_KEYS:
            out.append((key, label, False))
        elif key in ALWAYS_AUTO_MAP:
            out.append((key, label, False))
    return out


# Fallback header→output map when the user has not supplied a DB mapping
DB_MAP = {
    "Site Name": "ENODEBNAME",
    "Cell ID": "CELLName",
    "Longtude": "LONgitude",
    "Longitude": "LONgitude",
    "Latitude": "LATitude",
    "PCI": "PCI",
    "ANT_Height": "Height",
    "Azimuth": "Azimuth",
    "M_Tilt": "Mechanical_Downtilt",
    "E_Tilt": "Electrical_Downtilt",
    "UARFCN DOWNLINK": "DLEARFCN",
    "EARFCN DOWNLINK": "DLEARFCN",
    "REF_POWER": "RS_Power",
    "MAX_TX_POWER": "Downlink_bandwidth",
    "In Load Balance Report": "In Load Balance Report",
    "In Power Report": "In Power Report",
    "eNodBID": "eNodBID",
    "CellID": "CellID",
}

# KPIs users may add as extra definition / classification conditions
EDITABLE_METRICS: dict[str, str] = {
    # Core RF
    "RSRP": "RSRP",
    "RSRQ": "RSRQ",
    "SINR": "SINR",
    "RSSI": "RSSI",
    # Distance / Geometry
    "Distance (m)": "Serv_Dist_avg",
    "Distance Min (m)": "Serv_Dist_min",
    "Distance Max (m)": "Serv_Dist_max",
    "Azimuth Delta": "Azimuth_Orientation_Delta",
    # Service / Samples
    "Service Rate": "Service Rate",
    "Sample Count": "count",
    # Throughput
    "DL FTP Throughput (kbps)": "dl_throughput",
    "UL FTP Throughput (kbps)": "ul_throughput",
    "Sch PDSCH Thrpt (kbps)": "SchPDSCHThrpt",
    "Net PDSCH Thrpt (kbps)": "NetPDSCHThrpt",
    "Avg MCS": "AvgMCS",
    "BLER (%)": "BLER",
    "Num RBs": "NumRBs",
    "Transmission Mode": "TransmissionMode",
    # MOS / Voice Quality
    "MOS": "p863LQ",
    "MOS <=4%": "MOS=<4%",
    "MOS <=3.5%": "MOS=<3.5%",
    "MOS <=2.6%": "MOS=<2.6%",
    "MOS <=2%": "MOS=<2%",
    # Call Setup
    "Call Setup Time (s)": "Voice_Call_Setup_Time_sec",
    # RF Scores (Worst-Cell only)
    "Normalized RSRP": "Normalized RSRP",
    "Normalized RSRQ": "Normalized RSRQ",
    "Normalized SINR": "Normalized SINR",
    "RSRP Score": "RSRP Score",
    "RSRQ Score": "RSRQ Score",
    "SINR Score": "SINR Score",
    # Throughput Scores (Worst-Cell only)
    "Normalized Thput": "Normalized Thput",
    "Thput Score": "Thput Score",
    # MOS Scores (Worst-Cell only)
    "MOS Normalize": "MOS Normalize",
    "MOS Score": "MOS Score",
}

OPS: dict[str, Callable] = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
}

DEFAULT_CRITERIA = {
    "definition": {
        "rsrp": -95.0,
        "sinr": 0.0,
        "rf_service_rate": 0.005,
        "thput_dl_mbps": 10.0,
        "mos": 3.5,
        "cst": 5.0,
    },
    "classification": {
        "rsrp": -95.0,
        "sinr": 0.0,
        "dist": 1500.0,
        "rf_score_const": 0.6,
        "rf_sinr_const": 0.8333333333,
        "thput_dl_mbps": 30.0,
        "mos": 3.5,
        "cst": 5.0,
        "cst_dist_over": 1500,
        "cst_rsrp_non_dom": -95,
        "cst_q_lo": 1000,
        "cst_q_hi": 1300,
        "cst_rsrp_poorcov": -90,
        "cst_sinr_poorq": 3,
    },
    # Extra AND-gates on worst-cell inclusion: {category, metric, op, value}
    "extra_definition": [],
    # Extra first-match classification rules:
    #   {category, label, conditions: [{metric, op, value}, ...]}
    "extra_classification": [],
}

CATS = ["RF", "Throughput", "MOS", "CST"]


# ---------------------------------------------------------------------------
# Serving-sector helpers
# ---------------------------------------------------------------------------

def _is_null_or_empty(val: Any) -> bool:
    if pd.isna(val):
        return True
    s = str(val).strip()
    return s in ("", "nan", "None", "NaN", "<NA>", "*", "-", "NULL", "null")


def _clean_identity_series(s: pd.Series) -> pd.Series:
    """Coerce an identity column to numeric, treating TEMS placeholders as missing."""
    cleaned = s.replace(["*", "-", "NULL", "null", ""], np.nan)
    if cleaned.dtype == object:
        cleaned = cleaned.infer_objects(copy=False)
    return pd.to_numeric(cleaned, errors="coerce")


def carry_forward_cell_identity_raw(
    df: pd.DataFrame,
    enb_src_col: str | None,
    cell_src_col: str | None,
    sector_src_col: str | None = None,
) -> pd.DataFrame:
    """Python equivalent of the VBA CarryForwardCellIdentities macro, plus
    blank-sector row removal (Option A).

    Step 1 — Carry-forward (VBA equivalent):
      Operates on the **raw** source column names (before apply_mapping renames
      them to canonical names), exactly as the macro operates on columns M and N
      of the Excel sheet.

      Rules (matching VBA IsIdentityBlank / carry-forward logic):
        - If the cell value is blank / placeholder, fill it from the last valid
          value seen above.
        - If the cell already holds a valid value, update the running "last seen"
          and leave the cell unchanged.
        - eNB Part and Cell Part are carried forward independently.

    Step 2 — Drop blank-sector rows:
      After eNB/Cell identity has been propagated downward, any row whose
      Serving Sector is empty or null is deleted.  These rows were purely
      identity-carrier rows whose only purpose was to propagate eNB/Cell onto
      the following Serving-Sector rows; now that propagation is done they
      serve no further purpose and must be removed so they do not act as
      implicit group-boundary markers inside _resolve_occurrences, which is
      the root cause of inflated counts for PCI-EARFCN cells.

    This must be called on the raw DataFrame (load_excel output) before
    apply_mapping.
    """
    df = df.copy()

    # ── Step 1: carry eNB / Cell identity forward row-by-row ─────────────────
    for src_col in (enb_src_col, cell_src_col):
        if not src_col or src_col not in df.columns:
            continue

        series = _clean_identity_series(df[src_col])
        last_valid = np.nan

        new_vals = series.tolist()
        for i, val in enumerate(new_vals):
            if pd.isna(val):
                # Blank → fill from last valid (mirrors: cENB.Value2 = lastENB)
                if not pd.isna(last_valid):
                    new_vals[i] = last_valid
            else:
                # Valid value → update running tracker
                last_valid = val

        df[src_col] = new_vals

    # ── Step 2: blank Serving Sector rows are kept until PCI-group tagging
    # so they can separate isolated PCI/EARFCN occurrences from earlier runs.
    # They are dropped later in _resolve_occurrences (not counted).

    return df


def _propagate_cell_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Carry Cell Identity (eNB Part) and Cell Identity (Cell Part) forward in row order.

    Empty identity cells inherit the most recently seen valid value. A row that
    already holds a valid identity is never overwritten. The two fields are
    filled independently, so a new eNB Part updates eNB going forward without
    clearing a still-valid Cell Part (and vice versa).

    This operates on the **canonical** column names (after apply_mapping) and
    serves as a second-pass safety net. The primary carry-forward is done by
    carry_forward_cell_identity_raw() on the raw source columns before mapping.
    """
    df = df.copy()
    for key in ("enb", "cell"):
        col = CANON[key]
        if col in df.columns:
            df[col] = _clean_identity_series(df[col]).ffill()
    return df


def _first_non_empty(s: pd.Series) -> Any:
    for v in s:
        if not _is_null_or_empty(v):
            return str(v).strip()
    return np.nan


def _mode_or_first(s: pd.Series) -> Any:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return np.nan
    m = v.mode()
    return m.iloc[0] if len(m) else v.iloc[0]


def _extract_cell_name(sector_val: Any) -> str | None:
    """Format A: 'DOD289-L18C1-C -- DOD289_UL' or '4A16X180_211 -- A16X180' → cell name."""
    if _is_null_or_empty(sector_val):
        return None
    s = str(sector_val).strip()
    if "--" in s:
        parts = re.split(r"\s*--\s*", s)
        if parts and parts[0].strip():
            return parts[0].strip()
    return None


def _extract_pci(sector_val: Any) -> int | None:
    """Format B: 'PCI-314_EARFCN_DL-1499' → 314."""
    if _is_null_or_empty(sector_val):
        return None
    s = str(sector_val).strip()
    m = re.search(r"PCI[-_:\s]*(\d+)", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_earfcn(sector_val: Any) -> int | None:
    """Format B: 'PCI-314_EARFCN_DL-1499' or 'PCI-183_EARFCN_DL-1300' → 1499 / 1300."""
    if _is_null_or_empty(sector_val):
        return None
    s = str(sector_val).strip()
    m = re.search(
        r"EARFCN[_\s]*DL[-_:\s]*(\d+)|EARFCN[-_:\s]*(\d+)|DL[-_:\s]*EARFCN[-_:\s]*(\d+)",
        s,
        re.IGNORECASE,
    )
    if m:
        for g in m.groups():
            if g is not None:
                return int(g)
    return None


def _is_pci_earfcn_format(sector_val: Any) -> bool:
    """True when Serving Sector looks like PCI-…_EARFCN_DL-…"""
    if _is_null_or_empty(sector_val):
        return False
    s = str(sector_val).strip()
    return bool(
        re.search(r"PCI[-_:\s]*\d+", s, re.IGNORECASE)
        and re.search(r"EARFCN", s, re.IGNORECASE)
    )


def _tag_and_attach_pci_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Attach isolated PCI-EARFCN rows to the most recent same-PCI/EARFCN group.

    Blank Serving Sector rows (identity-only) *break* a PCI run, so a single
    ``PCI-238_EARFCN_DL-1300`` row after identity-only rows is isolated even
    though the previous PCI-238 run had the same string.

    Isolated runs (1 sample) inherit the eNB/Cell of the most recent
    multi-row run with the same PCI+EARFCN. They must not keep the identity
    that TEMS happened to write next to them. If no earlier group exists they
    attach to the next multi-row run of the same PCI+EARFCN.
    """
    sector_col = CANON["sector"]
    enb_col = CANON["enb"]
    cell_col = CANON["cell"]
    if df.empty or sector_col not in df.columns:
        return df

    df = df.copy()
    if "_pci_group_id" not in df.columns:
        df["_pci_group_id"] = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    if "_resolve_reason" not in df.columns:
        df["_resolve_reason"] = pd.Series([np.nan] * len(df), index=df.index, dtype=object)

    runs: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    def _flush() -> None:
        nonlocal cur
        if cur is not None:
            runs.append(cur)
            cur = None

    for idx, sector in zip(df.index, df[sector_col]):
        if _is_null_or_empty(sector):
            _flush()
            continue
        if not _is_pci_earfcn_format(sector):
            _flush()
            continue
        pci = _extract_pci(sector)
        ear = _extract_earfcn(sector)
        key = (pci, ear)
        if cur is None or cur["key"] != key:
            _flush()
            cur = {"key": key, "idxs": [idx], "pci": pci, "ear": ear}
        else:
            cur["idxs"].append(idx)
    _flush()

    next_gid = 1
    last_anchor: dict[tuple, dict[str, Any]] = {}
    pending: dict[tuple, list[dict[str, Any]]] = {}

    def _anchor_identity(idxs: list) -> tuple[Any, Any]:
        enb = np.nan
        cell = np.nan
        if enb_col in df.columns:
            enb = _mode_or_first(df.loc[idxs, enb_col])
        if cell_col in df.columns:
            cell = _mode_or_first(df.loc[idxs, cell_col])
        return enb, cell

    def _apply_group(idxs: list, gid: int, enb: Any, cell: Any, reason: str | None) -> None:
        df.loc[idxs, "_pci_group_id"] = gid
        if pd.notna(enb) and enb_col in df.columns:
            df.loc[idxs, enb_col] = enb
        if pd.notna(cell) and cell_col in df.columns:
            df.loc[idxs, cell_col] = cell
        if reason:
            df.loc[idxs, "_resolve_reason"] = reason

    for run in runs:
        key = run["key"]
        idxs = run["idxs"]
        n = len(idxs)
        if n >= 2:
            gid = next_gid
            next_gid += 1
            enb, cell = _anchor_identity(idxs)
            _apply_group(idxs, gid, enb, cell, None)
            last_anchor[key] = {"gid": gid, "enb": enb, "cell": cell}
            for iso in pending.pop(key, []):
                reason = (
                    f"Isolated PCI-EARFCN occurrence attached to following "
                    f"group PCI={iso['pci']} EARFCN={iso['ear']}"
                )
                _apply_group(iso["idxs"], gid, enb, cell, reason)
        else:
            if key in last_anchor:
                anc = last_anchor[key]
                reason = (
                    f"Isolated PCI-EARFCN occurrence attached to most recent "
                    f"group PCI={run['pci']} EARFCN={run['ear']}"
                )
                _apply_group(idxs, anc["gid"], anc["enb"], anc["cell"], reason)
            else:
                pending.setdefault(key, []).append(run)

    # Isolates with no multi-row group anywhere: each stays its own group
    for key, isolist in pending.items():
        for iso in isolist:
            gid = next_gid
            next_gid += 1
            enb, cell = _anchor_identity(iso["idxs"])
            _apply_group(iso["idxs"], gid, enb, cell, None)

    return df


def _resolve_occurrences(df: pd.DataFrame) -> pd.DataFrame:
    """Attach grouping keys from Serving Sector and drop empty-sector rows.

    Identity (eNB / Cell Part) is first carried forward in original row order.
    Rows with an empty Serving Sector are then excluded from counts and from
    all worst-cell calculations — even if they hold a cell identity. Those
    identity-only rows exist only to propagate eNB/Cell onto the following
    Serving Sector rows.

    Cell-name extraction from Serving Sector is unchanged:
      * Format A ('NAME -- SITE') → group by extracted CELLName
      * Format B (PCI/EARFCN) → group by `_pci_group_id` (isolated singles
        already attached to the most recent same-PCI/EARFCN multi-row run).
      * Anything else → group by the propagated (eNB, Cell) pair.
    """
    sector_col = CANON["sector"]

    df = _propagate_cell_identity(df)

    occurrence_ids = []
    occurrence_id = 0
    prev_valid_sector = None

    for s in df[sector_col]:
        if _is_null_or_empty(s):
            # Empty Serving Sector rows are never counted
            occurrence_ids.append(0)
        else:
            s_clean = str(s).strip()
            if s_clean != prev_valid_sector:
                occurrence_id += 1
                prev_valid_sector = s_clean
            occurrence_ids.append(occurrence_id)

    df["_occurrence_id"] = occurrence_ids

    # Keep only rows that belong to a valid serving sector occurrence
    df = df[df["_occurrence_id"] > 0].copy()
    if df.empty:
        return df

    # Extract CELLName / PCI / EARFCN from this row's Serving Sector
    df["_occurrence_sector"] = df[sector_col].map(
        lambda v: str(v).strip() if not _is_null_or_empty(v) else None
    )
    df["_cell_name"] = df[sector_col].map(_extract_cell_name)
    df["_pci_from_sector"] = df[sector_col].map(_extract_pci)
    df["_earfcn_from_sector"] = df[sector_col].map(_extract_earfcn)

    name_mask = df["_cell_name"].notna()
    pci_mask = (~name_mask) & df["_occurrence_sector"].map(_is_pci_earfcn_format)

    # Format A: group by extracted CELLName.
    # Format B (PCI/EARFCN): group by contiguous occurrence — NOT by the
    # raw (eNB, Cell) pair, which TEMS often gets wrong inside a run.
    # Everything else: group by the propagated (eNB, Cell) pair.
    df["_group_type"] = np.where(name_mask, "name", np.where(pci_mask, "pci", "numeric"))
    df["_group_key"] = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    if name_mask.any():
        df.loc[name_mask, "_group_key"] = df.loc[name_mask, "_cell_name"]
    if pci_mask.any():
        if "_pci_group_id" in df.columns:
            df.loc[pci_mask, "_group_key"] = df.loc[pci_mask, "_pci_group_id"]
        else:
            df.loc[pci_mask, "_group_key"] = df.loc[pci_mask, "_occurrence_id"]


    # Distance / angle: invalidate for PCI/EARFCN-format rows
    pci_occ_mask = df["_occurrence_sector"].astype(str).str.contains(
        "PCI|EARFCN", case=False, na=False
    )
    if CANON["dist"] in df.columns:
        df.loc[pci_occ_mask, CANON["dist"]] = np.nan
    if CANON["angle"] in df.columns:
        df.loc[pci_occ_mask, CANON["angle"]] = np.nan

    return df


def _lookup_cell_identity_from_db(
    df: pd.DataFrame,
    db_path: str | None,
    db_mapping: dict[str, str | None] | None,
) -> pd.DataFrame:
    """For rows with _cell_name but missing eNB/Cell, look up eNB/Cell from DB.
    For rows with eNB + PCI but missing Cell, look up Cell from DB by
    (eNB, PCI, EARFCN) when available (unique hits only).
    """
    if not db_path or not os.path.isfile(db_path):
        return df
    db = load_excel(db_path)
    if db_mapping and any(db_mapping.values()):
        dbk = apply_mapping(db, db_mapping)
    else:
        present = {k: v for k, v in DB_MAP.items() if k in db.columns}
        dbk = db.rename(columns=present)
        for candidate in ("eNodBID", "eNB_Part", "eNodeB ID", "eNodeBID", "ENBID"):
            if candidate in dbk.columns:
                if candidate != "eNodBID":
                    dbk = dbk.rename(columns={candidate: "eNodBID"})
                break
        for candidate in ("CellID", "Cell_Part", "Cell Id", "CID"):
            if candidate in dbk.columns:
                if candidate != "CellID":
                    dbk = dbk.rename(columns={candidate: "CellID"})
                break

    enb_col = CANON["enb"]
    cell_col = CANON["cell"]

    # --- Cell-name lookup ---
    if "_cell_name" in df.columns and "CELLName" in dbk.columns:
        missing = df[enb_col].isna() | df[cell_col].isna()
        has_name = df["_cell_name"].notna()
        to_fix = df[missing & has_name]
        if not to_fix.empty:
            lookup = (
                dbk[dbk["CELLName"].notna()][["CELLName", "eNodBID", "CellID"]]
                .drop_duplicates("CELLName")
                .set_index("CELLName")
            )
            for idx in to_fix.index:
                cn = df.at[idx, "_cell_name"]
                if cn in lookup.index:
                    if pd.isna(df.at[idx, enb_col]):
                        df.at[idx, enb_col] = lookup.at[cn, "eNodBID"]
                    if pd.isna(df.at[idx, cell_col]):
                        df.at[idx, cell_col] = lookup.at[cn, "CellID"]

    # --- (eNB + PCI [+ EARFCN]) lookup when Cell Part is missing ---
    if "_pci_from_sector" in df.columns and "PCI" in dbk.columns:
        missing_cell = df[cell_col].isna() & df[enb_col].notna() & df["_pci_from_sector"].notna()
        to_fix = df[missing_cell]
        if not to_fix.empty:
            earfcn_db_col = None
            for cand in ("DLEARFCN", "EARFCN", "EARFCN DOWNLINK", "UARFCN DOWNLINK"):
                if cand in dbk.columns:
                    earfcn_db_col = cand
                    dbk[cand] = pd.to_numeric(dbk[cand], errors="coerce")
                    break

            # Prefer unique (eNB, PCI, EARFCN) when EARFCN is available
            if (
                earfcn_db_col
                and "_earfcn_from_sector" in df.columns
                and df["_earfcn_from_sector"].notna().any()
            ):
                cols = ["eNodBID", "PCI", earfcn_db_col, "CellID"]
                lookup = dbk[dbk["PCI"].notna()].dropna(subset=cols[:3])[cols].copy()
                lookup["eNodBID"] = pd.to_numeric(lookup["eNodBID"], errors="coerce")
                lookup["PCI"] = pd.to_numeric(lookup["PCI"], errors="coerce")
                lookup[earfcn_db_col] = pd.to_numeric(lookup[earfcn_db_col], errors="coerce")
                uniq = (
                    lookup.groupby(["eNodBID", "PCI", earfcn_db_col])["CellID"]
                    .nunique()
                    .reset_index(name="n")
                )
                uniq = uniq[uniq["n"] == 1]
                lookup = lookup.merge(
                    uniq[["eNodBID", "PCI", earfcn_db_col]],
                    on=["eNodBID", "PCI", earfcn_db_col],
                    how="inner",
                ).drop_duplicates(["eNodBID", "PCI", earfcn_db_col])
                lookup = lookup.set_index(["eNodBID", "PCI", earfcn_db_col])
                for idx in to_fix.index:
                    enb = float(df.at[idx, enb_col])
                    pci = float(df.at[idx, "_pci_from_sector"])
                    ear = df.at[idx, "_earfcn_from_sector"]
                    if pd.isna(ear):
                        continue
                    key = (enb, pci, float(ear))
                    if key in lookup.index:
                        df.at[idx, cell_col] = lookup.at[key, "CellID"]
            else:
                lookup = (
                    dbk[dbk["PCI"].notna()][["eNodBID", "PCI", "CellID"]]
                    .drop_duplicates(["eNodBID", "PCI"])
                )
                lookup["eNodBID"] = pd.to_numeric(lookup["eNodBID"], errors="coerce")
                lookup["PCI"] = pd.to_numeric(lookup["PCI"], errors="coerce")
                uniq = (
                    lookup.groupby(["eNodBID", "PCI"])["CellID"]
                    .nunique()
                    .reset_index(name="n")
                )
                uniq = uniq[uniq["n"] == 1]
                lookup = lookup.merge(
                    uniq[["eNodBID", "PCI"]], on=["eNodBID", "PCI"], how="inner"
                ).drop_duplicates(["eNodBID", "PCI"])
                lookup = lookup.set_index(["eNodBID", "PCI"])
                for idx in to_fix.index:
                    enb = float(df.at[idx, enb_col])
                    pci = float(df.at[idx, "_pci_from_sector"])
                    if (enb, pci) in lookup.index:
                        df.at[idx, cell_col] = lookup.at[(enb, pci), "CellID"]

    df = _resolve_pci_earfcn_groups(df, dbk)
    return df


def _db_earfcn_col(dbk: pd.DataFrame) -> str | None:
    for cand in ("DLEARFCN", "EARFCN", "EARFCN DOWNLINK", "UARFCN DOWNLINK"):
        if cand in dbk.columns:
            return cand
    return None


def _resolve_pci_earfcn_groups(
    df: pd.DataFrame,
    dbk: pd.DataFrame | None,
) -> pd.DataFrame:
    """Collapse each contiguous PCI-EARFCN occurrence onto one serving cell.

    TEMS often writes conflicting (eNB, Cell) values inside a run of the same
    ``PCI-{n}_EARFCN_DL-{n}`` Serving Sector, with no new sector string (only
    blank rows) between them. Those rows belong to one serving cell:

      1. Neighboring named Serving Sector whose DB PCI+EARFCN matches the run
         (e.g. a ``NAME -- SITE`` row immediately before the PCI block).
      2. Otherwise the unique 4-tuple hit (PCI, EARFCN, eNB, Cell) among the
         identities present in the run. Identities that do not match the DB
         are absorbed into that cell — they do not start a new group.
      3. If several identities match different cells and the neighbor does
         not break the tie, the run stays together but is left unmatched.

    Resolved rows become name-groups so their samples count with that CELLName.
    """
    if df.empty or "_group_type" not in df.columns:
        return df

    pci_mask = df["_group_type"] == "pci"
    if not pci_mask.any():
        return df

    df = df.copy()
    enb_col = CANON["enb"]
    cell_col = CANON["cell"]

    db_ready = (
        dbk is not None
        and not dbk.empty
        and "PCI" in dbk.columns
        and "CELLName" in dbk.columns
    )
    earfcn_col = _db_earfcn_col(dbk) if db_ready else None
    db_ready = bool(db_ready and earfcn_col)

    db_by_name = None
    db_combo = None
    if db_ready:
        work = dbk.copy()
        work["PCI"] = pd.to_numeric(work["PCI"], errors="coerce")
        work[earfcn_col] = pd.to_numeric(work[earfcn_col], errors="coerce")
        if "eNodBID" in work.columns:
            work["eNodBID"] = pd.to_numeric(work["eNodBID"], errors="coerce")
        if "CellID" in work.columns:
            work["CellID"] = pd.to_numeric(work["CellID"], errors="coerce")
        db_by_name = (
            work[work["CELLName"].notna()]
            .drop_duplicates("CELLName")
            .set_index("CELLName")
        )
        need = ["eNodBID", "CellID", "PCI", earfcn_col, "CELLName"]
        have = [c for c in need if c in work.columns]
        db_combo = work.dropna(subset=["PCI", earfcn_col]).copy()
        if "eNodBID" in db_combo.columns and "CellID" in db_combo.columns:
            db_combo = db_combo.dropna(subset=["eNodBID", "CellID"])
        db_combo = db_combo[have] if have else db_combo

    # Immediate previous / next named CELLName in file order
    names = df["_cell_name"].tolist()
    prev_name: list[Any] = []
    last = None
    for n in names:
        prev_name.append(last)
        if not _is_null_or_empty(n):
            last = str(n).strip()
    next_name: list[Any] = [None] * len(names)
    nxt = None
    for i in range(len(names) - 1, -1, -1):
        next_name[i] = nxt
        if not _is_null_or_empty(names[i]):
            nxt = str(names[i]).strip()

    pos = {idx: i for i, idx in enumerate(df.index)}

    def _neighbor_row(cellname: str | None) -> pd.Series | None:
        if not cellname or db_by_name is None or cellname not in db_by_name.index:
            return None
        row = db_by_name.loc[cellname]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _neighbor_matches(row: pd.Series | None, pci: float, ear: float) -> bool:
        if row is None:
            return False
        p = pd.to_numeric(row.get("PCI"), errors="coerce")
        e = pd.to_numeric(row.get(earfcn_col), errors="coerce") if earfcn_col else np.nan
        return pd.notna(p) and pd.notna(e) and float(p) == float(pci) and float(e) == float(ear)

    if "_resolve_reason" not in df.columns:
        df["_resolve_reason"] = pd.Series([np.nan] * len(df), index=df.index, dtype=object)

    for occ_id, gidx in df.loc[pci_mask].groupby("_group_key", sort=False).groups.items():

        gidx = list(gidx)
        pci_v = pd.to_numeric(df.loc[gidx, "_pci_from_sector"], errors="coerce").dropna()
        ear_v = pd.to_numeric(df.loc[gidx, "_earfcn_from_sector"], errors="coerce").dropna()
        pci = float(pci_v.iloc[0]) if not pci_v.empty else np.nan
        ear = float(ear_v.iloc[0]) if not ear_v.empty else np.nan

        # Previous named Serving Sector only (drive-test continuation).
        # Do not use the *next* named sector — it is often a different cell
        # that merely follows an isolated PCI row.
        first_pos = pos[gidx[0]]
        neighbors: list[str] = []
        prev = prev_name[first_pos]
        if prev:
            neighbors.append(prev)


        resolved_name = None
        resolved_enb = np.nan
        resolved_cell = np.nan
        reason = None

        neighbor_hits: list[tuple[str, pd.Series]] = []
        if db_ready and pd.notna(pci) and pd.notna(ear):
            for nb in neighbors:
                nrow = _neighbor_row(nb)
                if _neighbor_matches(nrow, pci, ear):
                    neighbor_hits.append((nb, nrow))

        combo_hits: list[tuple[float, float, str, pd.Series, int]] = []
        if db_ready and db_combo is not None and pd.notna(pci) and pd.notna(ear):
            seen_keys: set[tuple[float, float]] = set()
            for idx in gidx:
                enb = pd.to_numeric(df.at[idx, enb_col], errors="coerce") if enb_col in df.columns else np.nan
                cid = pd.to_numeric(df.at[idx, cell_col], errors="coerce") if cell_col in df.columns else np.nan
                if pd.isna(enb) or pd.isna(cid):
                    continue
                key = (float(enb), float(cid))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                hits = db_combo[
                    (db_combo["PCI"] == pci)
                    & (db_combo[earfcn_col] == ear)
                    & (db_combo["eNodBID"] == key[0])
                    & (db_combo["CellID"] == key[1])
                ]
                if len(hits) == 1:
                    n_rows = int(
                        (
                            (pd.to_numeric(df.loc[gidx, enb_col], errors="coerce") == key[0])
                            & (pd.to_numeric(df.loc[gidx, cell_col], errors="coerce") == key[1])
                        ).sum()
                    )
                    combo_hits.append(
                        (key[0], key[1], str(hits.iloc[0]["CELLName"]), hits.iloc[0], n_rows)
                    )
                elif len(hits) > 1:
                    # Unique-cell requirement: ignore ambiguous 4-tuples
                    pass

        unique_neighbor_names = list(dict.fromkeys(h[0] for h in neighbor_hits))
        unique_combo_names = list(dict.fromkeys(h[2] for h in combo_hits))

        # 1. Neighbor named sector with matching PCI+EARFCN
        if len(unique_neighbor_names) == 1:
            nb, nrow = neighbor_hits[0]
            resolved_name = nb
            resolved_enb = pd.to_numeric(nrow.get("eNodBID"), errors="coerce")
            resolved_cell = pd.to_numeric(nrow.get("CellID"), errors="coerce")
            reason = (
                f"Resolved PCI-EARFCN group via neighboring sector {nb} "
                f"(PCI={int(pci)}, EARFCN={int(ear)}; absorbed {len(gidx)} samples)"
            )
        # 2. Unique DB match among identities in the run
        elif len(unique_combo_names) == 1:
            enb_h, cid_h, cname, hit, _n = combo_hits[0]
            resolved_name = cname
            resolved_enb = enb_h
            resolved_cell = cid_h
            reason = (
                f"Resolved PCI-EARFCN group via unique DB match "
                f"(PCI={int(pci)}, EARFCN={int(ear)}, eNB={int(enb_h)}, Cell={int(cid_h)} "
                f"→ {cname}; absorbed {len(gidx)} samples)"
            )
        elif len(unique_combo_names) > 1 and unique_neighbor_names:
            # Tie-break: combo that equals a matching neighbor
            nb_set = set(unique_neighbor_names)
            tied = [h for h in combo_hits if h[2] in nb_set]
            tied_names = list(dict.fromkeys(h[2] for h in tied))
            if len(tied_names) == 1:
                enb_h, cid_h, cname, hit, _n = tied[0]
                resolved_name = cname
                resolved_enb = enb_h
                resolved_cell = cid_h
                reason = (
                    f"Resolved PCI-EARFCN group via neighbor+DB match "
                    f"→ {cname} (PCI={int(pci)}, EARFCN={int(ear)}; "
                    f"absorbed {len(gidx)} samples)"
                )
            else:
                reason = (
                    f"Ambiguous PCI-EARFCN group: multiple DB cells "
                    f"{unique_combo_names} for PCI={int(pci)}, EARFCN={int(ear)}"
                )
        elif len(unique_combo_names) > 1:
            reason = (
                f"Ambiguous PCI-EARFCN group: multiple DB cells "
                f"{unique_combo_names} for PCI={int(pci)}, EARFCN={int(ear)}"
            )
        else:
            reason = None  # keep as pci occurrence group (mode identity)

        if resolved_name:
            df.loc[gidx, "_cell_name"] = resolved_name
            df.loc[gidx, "_group_type"] = "name"
            df.loc[gidx, "_group_key"] = resolved_name
            df.loc[gidx, "_resolve_reason"] = reason
            if pd.notna(resolved_enb) and enb_col in df.columns:
                df.loc[gidx, enb_col] = resolved_enb
            if pd.notna(resolved_cell) and cell_col in df.columns:
                df.loc[gidx, cell_col] = resolved_cell
        elif reason:
            df.loc[gidx, "_resolve_reason"] = reason

    return df


# ---------------------------------------------------------------------------
# Load / preprocess
# ---------------------------------------------------------------------------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().strip("'").strip().lstrip("\ufeff") for c in df.columns
    ]
    drop = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop:
        df = df.drop(columns=drop, errors="ignore")
    return df


def load_excel(path: str) -> pd.DataFrame:
    return norm_cols(pd.read_excel(path, sheet_name=0))


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    rename = {}
    for key, src in (mapping or {}).items():
        if src and src in df.columns and key in CANON:
            rename[src] = CANON[key]
    return df.rename(columns=rename)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    sector_col = CANON["sector"]

    # Always carry identity forward in original row order first, so Serving
    # Sector rows inherit eNB/Cell from preceding identity-only rows.
    df = _propagate_cell_identity(df)

    if sector_col in df.columns and df[sector_col].apply(lambda s: not _is_null_or_empty(s)).any():
        # Tag isolated PCI/EARFCN rows *before* blank-sector rows are dropped
        # so identity-only gaps still separate runs.
        df = _tag_and_attach_pci_groups(df)
        df = _resolve_occurrences(df)
    return df


# ---------------------------------------------------------------------------
# Aggregate per serving cell
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per serving cell using occurrence-resolved grouping keys.

    When _group_type == "name", group by the extracted cell name.
    Otherwise, fall back to eNB Part + Cell Part (PCI format included).
    """
    enb_col = CANON["enb"]
    cell_col = CANON["cell"]

    if df.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    def _agg_group(gdf: pd.DataFrame, gcols: Any, group_type: str) -> pd.DataFrame:
        g = gdf.groupby(gcols, dropna=False)
        d: dict[str, Any] = {}

        def mean_col(raw: str, out: str) -> None:
            if raw in gdf.columns:
                d[out] = g[raw].mean()

        mean_col(CANON["rsrp"], "RSRP")
        mean_col(CANON["rsrq"], "RSRQ")
        mean_col(CANON["sinr"], "SINR")
        mean_col(CANON["rssi"], "RSSI")
        mean_col(CANON["ftp_dl"], "dl_throughput")
        mean_col(CANON["ftp_ul"], "ul_throughput")
        mean_col(CANON["pdsch"], "SchPDSCHThrpt")
        mean_col(CANON["pdsch"], "NetPDSCHThrpt")
        mean_col(CANON["mcs"], "AvgMCS")
        mean_col(CANON["bler"], "BLER")
        mean_col(CANON["rb"], "NumRBs")
        mean_col(CANON["setup"], "Voice_Call_Setup_Time_sec")

        d["count"] = g.size()

        if CANON["sector"] in gdf.columns:
            d["Serving Sector"] = g[CANON["sector"]].agg(_first_non_empty)
        elif "_occurrence_sector" in gdf.columns:
            d["Serving Sector"] = g["_occurrence_sector"].agg(_first_non_empty)

        if CANON["dist"] in gdf.columns:
            d["Serv_Dist_avg"] = g[CANON["dist"]].mean() * 1000.0
            d["Serv_Dist_min"] = g[CANON["dist"]].min() * 1000.0
            d["Serv_Dist_max"] = g[CANON["dist"]].max() * 1000.0
        if CANON["angle"] in gdf.columns:
            d["Azimuth_Orientation_Delta"] = g[CANON["angle"]].mean()
        if CANON["tm"] in gdf.columns:
            d["TransmissionMode"] = g[CANON["tm"]].agg(
                lambda s: s.dropna().mode().iloc[0] if len(s.dropna().mode()) else None
            )

        mcol = next(
            (m for m in (CANON["polqa"], CANON["cs_polqa"], CANON["aqm"]) if m in gdf.columns),
            None,
        )
        if mcol:
            d["p863LQ"] = g[mcol].mean()
            for thr, cname in [
                (4, "MOS=<4%"),
                (3.5, "MOS=<3.5%"),
                (2.6, "MOS=<2.6%"),
                (2, "MOS=<2%"),
            ]:
                d[cname] = g[mcol].apply(lambda s, t=thr: float((s <= t).mean()) * 100)

        if group_type == "name":
            if enb_col in gdf.columns:
                d["eNB_Part"] = g[enb_col].agg(_mode_or_first)
            if cell_col in gdf.columns:
                d["Cell_Part"] = g[cell_col].agg(_mode_or_first)
        elif group_type == "pci":
            if enb_col in gdf.columns:
                d["eNB_Part"] = g[enb_col].agg(_mode_or_first)
            if cell_col in gdf.columns:
                d["Cell_Part"] = g[cell_col].agg(_mode_or_first)
        if "_pci_from_sector" in gdf.columns:
            d["PCI"] = g["_pci_from_sector"].agg(_mode_or_first)
            d["Match_PCI"] = g["_pci_from_sector"].agg(_mode_or_first)
        if "_earfcn_from_sector" in gdf.columns:
            d["Match_EARFCN"] = g["_earfcn_from_sector"].agg(_mode_or_first)
        if "_resolve_reason" in gdf.columns:
            d["Match_Reason"] = g["_resolve_reason"].agg(_first_non_empty)

        out = pd.DataFrame(d)
        if isinstance(gcols, list):
            out.index = out.index.set_names(gcols)
        else:
            out.index = out.index.set_names(gcols)
        out = out.reset_index()

        if group_type == "name":
            out = out.rename(columns={gcols: "CELLName"})
            if "eNB_Part" not in out.columns:
                out["eNB_Part"] = np.nan
            if "Cell_Part" not in out.columns:
                out["Cell_Part"] = np.nan
            out["ENODEBNAME"] = pd.Series([np.nan] * len(out), dtype=object)
        elif group_type == "pci":
            # Group key is the occurrence id — not an eNB identity.
            if gcols in out.columns:
                out = out.drop(columns=[gcols])
            if "eNB_Part" not in out.columns:
                out["eNB_Part"] = np.nan
            if "Cell_Part" not in out.columns:
                out["Cell_Part"] = np.nan
            out["CELLName"] = pd.Series([np.nan] * len(out), dtype=object)
            out["ENODEBNAME"] = pd.Series([np.nan] * len(out), dtype=object)
        else:  # numeric
            out = out.rename(columns={enb_col: "eNB_Part", cell_col: "Cell_Part"})
            out["CELLName"] = pd.Series([np.nan] * len(out), dtype=object)
            out["ENODEBNAME"] = pd.Series([np.nan] * len(out), dtype=object)

        return out

    if "_group_type" in df.columns:
        name_mask = df["_group_type"] == "name"
        pci_mask = df["_group_type"] == "pci"
        num_mask = df["_group_type"] == "numeric"

        if name_mask.any():
            ndf = df[name_mask].copy()
            ndf["_group_key_str"] = ndf["_group_key"].astype(str)
            frames.append(_agg_group(ndf, "_group_key_str", "name"))

        if pci_mask.any():
            pdf = df[pci_mask].copy()
            frames.append(_agg_group(pdf, "_group_key", "pci"))

        if num_mask.any():
            nmdf = df[num_mask].copy()
            keys = [enb_col, cell_col]
            if all(k in nmdf.columns for k in keys):
                frames.append(_agg_group(nmdf.dropna(subset=keys), keys, "numeric"))

        if frames:
            res = pd.concat(frames, ignore_index=True)
            for c in ("ENODEBNAME", "CELLName"):
                if c not in res.columns:
                    res[c] = pd.Series([np.nan] * len(res), dtype=object)
            return res

    # Fallback: standard numeric grouping
    keys = [enb_col, cell_col]
    for k in keys:
        if k not in df.columns:
            return pd.DataFrame()
    sub = df.dropna(subset=keys).copy()
    if sub.empty:
        return pd.DataFrame()
    res = _agg_group(sub, keys, "numeric")
    for c in ("ENODEBNAME", "CELLName"):
        if c not in res.columns:
            res[c] = pd.Series([np.nan] * len(res), dtype=object)
    return res


def _merge_duplicate_cells(agg: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that share eNB + Cell + Serving Sector (+ source).

    Those three fields identify the same serving cell, so they must appear
    once in the worst-cell output with combined sample counts.
    CS and PS remain separate (Source Type is part of the key).
    """
    if agg is None or agg.empty:
        return agg

    keys = [
        c
        for c in ("eNB_Part", "Cell_Part", "Serving Sector", "Source File", "Source Type")
        if c in agg.columns
    ]
    if "eNB_Part" not in keys or "Cell_Part" not in keys or "Serving Sector" not in keys:
        return agg

    work = agg.copy()
    work["_k_enb"] = pd.to_numeric(work["eNB_Part"], errors="coerce")
    work["_k_cell"] = pd.to_numeric(work["Cell_Part"], errors="coerce")
    gcols = ["_k_enb", "_k_cell", "Serving Sector"]
    for extra in ("Source File", "Source Type"):
        if extra in work.columns:
            gcols.append(extra)

    ng = work.groupby(gcols, dropna=False).ngroups
    if ng == len(work):
        return agg

    count = pd.to_numeric(work.get("count", pd.Series(1, index=work.index)), errors="coerce").fillna(1.0)
    work["_w"] = count

    num_minmax = {
        "Serv_Dist_min": "min",
        "Serv_Dist_max": "max",
        "count": "sum",
        "Service Rate": "sum",
    }
    skip = set(gcols) | {"_k_enb", "_k_cell", "_w", "eNB_Part", "Cell_Part"}
    status_rank = {"Matched": 0, "Ambiguous": 1, "Unmatched": 2}

    rows = []
    for _, g in work.groupby(gcols, dropna=False):
        w = g["_w"].astype(float)
        wsum = float(w.sum()) if float(w.sum()) else float(len(g))
        out: dict[str, Any] = {}
        for c in agg.columns:
            if c in skip:
                continue
            series = g[c]
            if c in num_minmax:
                nums = pd.to_numeric(series, errors="coerce")
                if num_minmax[c] == "sum":
                    out[c] = nums.sum()
                elif num_minmax[c] == "min":
                    out[c] = nums.min()
                else:
                    out[c] = nums.max()
            elif c == "Match_Status":
                ranked = sorted(
                    (status_rank.get(str(v), 9), v)
                    for v in series
                    if not _is_null_or_empty(v)
                )
                out[c] = ranked[0][1] if ranked else series.iloc[0]
            elif pd.api.types.is_numeric_dtype(series) or c in (
                "RSRP", "RSRQ", "SINR", "RSSI", "Serv_Dist_avg",
                "dl_throughput", "ul_throughput", "SchPDSCHThrpt", "NetPDSCHThrpt",
                "AvgMCS", "BLER", "NumRBs", "p863LQ", "Azimuth_Orientation_Delta",
                "Match_PCI", "Match_EARFCN", "MOS=<4%", "MOS=<3.5%", "MOS=<2.6%", "MOS=<2%",
            ):
                nums = pd.to_numeric(series, errors="coerce")
                mask = nums.notna()
                if mask.any():
                    ww = w[mask]
                    out[c] = float((nums[mask] * ww).sum() / ww.sum()) if float(ww.sum()) else float(nums[mask].mean())
                else:
                    out[c] = np.nan
            else:
                out[c] = _first_non_empty(series)
        out["eNB_Part"] = g["_k_enb"].dropna().iloc[0] if g["_k_enb"].notna().any() else np.nan
        out["Cell_Part"] = g["_k_cell"].dropna().iloc[0] if g["_k_cell"].notna().any() else np.nan
        if "Serving Sector" in g.columns:
            out["Serving Sector"] = _first_non_empty(g["Serving Sector"])
        if "Source File" in g.columns and "Source File" not in out:
            out["Source File"] = _first_non_empty(g["Source File"])
        if "Source Type" in g.columns and "Source Type" not in out:
            out["Source Type"] = _first_non_empty(g["Source Type"])
        rows.append(out)

    merged = pd.DataFrame(rows)
    # Keep original column order, plus any new
    ordered = [c for c in agg.columns if c in merged.columns]
    extra = [c for c in merged.columns if c not in ordered]
    return merged[ordered + extra]


def join_db(
    agg: pd.DataFrame,
    db_path: str | None,
    mapping: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """Join site attributes from the 4G database onto aggregated cells.

    Matching priority
    -----------------
    1. CELLName already present (name-based Serving Sector groups).
    2. **PCI + EARFCN + eNB + Cell** (preferred for PCI-EARFCN format sectors).
       - Only applied when Serving Sector is PCI-…_EARFCN_DL-… format.
       - CELLName is filled only when the combined key identifies a **unique**
         database row. Multiple hits → ambiguous (CELLName left empty).
       - Missing any of the four keys → clear reason recorded.
    3. (eNB_Part, Cell_Part) fallback for non-PCI-format rows.
    4. (eNB_Part, PCI) tertiary fallback when Cell_Part is still missing
       (non-PCI-format rows only).

    Diagnostic columns always written when a DB is present:
      Match_PCI, Match_EARFCN, Match_Status, Match_Reason
    """
    for col in ("ENODEBNAME", "CELLName", "Match_Status", "Match_Reason"):
        if col not in agg.columns:
            agg[col] = pd.Series([np.nan] * len(agg), dtype=object)
    for col in ("Match_PCI", "Match_EARFCN"):
        if col not in agg.columns:
            agg[col] = pd.Series([np.nan] * len(agg), dtype=float)

    if not db_path or not os.path.isfile(db_path):
        agg["Match_Status"] = "No database"
        agg["Match_Reason"] = "No database file provided"
        return agg

    db = load_excel(db_path)
    if mapping and any(mapping.values()):
        dbk = apply_mapping(db, mapping)
    else:
        present = {k: v for k, v in DB_MAP.items() if k in db.columns}
        dbk = db.rename(columns=present)
        for candidate in ("eNodBID", "eNB_Part", "eNodeB ID", "eNodeBID", "ENBID"):
            if candidate in dbk.columns:
                if candidate != "eNodBID":
                    dbk = dbk.rename(columns={candidate: "eNodBID"})
                break
        for candidate in ("CellID", "Cell_Part", "Cell ID", "CID"):
            if candidate in dbk.columns:
                if candidate != "CellID":
                    dbk = dbk.rename(columns={candidate: "CellID"})
                break

    if "eNodBID" not in dbk.columns or "CellID" not in dbk.columns:
        agg["Match_Status"] = "No database keys"
        agg["Match_Reason"] = "Database missing eNodBID / CellID columns"
        return agg

    dbk["eNodBID"] = pd.to_numeric(dbk["eNodBID"], errors="coerce")
    dbk["CellID"] = pd.to_numeric(dbk["CellID"], errors="coerce")
    if "PCI" in dbk.columns:
        dbk["PCI"] = pd.to_numeric(dbk["PCI"], errors="coerce")
    earfcn_db_col = None
    for cand in ("DLEARFCN", "EARFCN", "EARFCN DOWNLINK", "UARFCN DOWNLINK"):
        if cand in dbk.columns:
            earfcn_db_col = cand
            dbk[cand] = pd.to_numeric(dbk[cand], errors="coerce")
            break

    attr_cols = [
        c for c in (["ENODEBNAME", "CELLName"] + DB_COLS) if c in dbk.columns
    ]

    res = agg.copy()
    for col in ("ENODEBNAME", "CELLName", "Match_Status", "Match_Reason"):
        if col not in res.columns:
            res[col] = pd.Series([np.nan] * len(res), dtype=object)
        else:
            res[col] = res[col].astype(object)
    for col in ("Match_PCI", "Match_EARFCN"):
        if col not in res.columns:
            res[col] = pd.Series([np.nan] * len(res), dtype=float)

    def _apply_db_row(idx: Any, row_data: pd.Series) -> None:
        for col in attr_cols:
            if col in row_data.index and pd.notna(row_data[col]):
                if col not in res.columns or pd.isna(res.at[idx, col]) or col in (
                    "ENODEBNAME",
                    "CELLName",
                ):
                    res.at[idx, col] = row_data[col]
        if pd.isna(res.at[idx, "eNB_Part"]) and "eNodBID" in row_data.index:
            res.at[idx, "eNB_Part"] = row_data["eNodBID"]
        if pd.isna(res.at[idx, "Cell_Part"]) and "CellID" in row_data.index:
            res.at[idx, "Cell_Part"] = row_data["CellID"]

    # ── Strategy 1: Match on CELLName (name-based groups) ───────────────────
    if "CELLName" in res.columns and "CELLName" in dbk.columns:
        name_mask = res["CELLName"].notna()
        if name_mask.any():
            db_by_name = (
                dbk[dbk["CELLName"].notna()]
                .drop_duplicates("CELLName")
                .set_index("CELLName")
            )
            for idx in res[name_mask].index:
                cn = res.at[idx, "CELLName"]
                if cn in db_by_name.index:
                    _apply_db_row(idx, db_by_name.loc[cn])
                    res.at[idx, "Match_Status"] = "Matched"
                    existing = res.at[idx, "Match_Reason"]
                    if _is_null_or_empty(existing):
                        res.at[idx, "Match_Reason"] = "Matched by CELLName"


    # Ensure Match_PCI / Match_EARFCN are populated from sector extraction
    if "Serving Sector" in res.columns:
        for idx in res.index:
            if pd.isna(res.at[idx, "Match_PCI"]):
                pci_v = _extract_pci(res.at[idx, "Serving Sector"])
                if pci_v is not None:
                    res.at[idx, "Match_PCI"] = pci_v
            if pd.isna(res.at[idx, "Match_EARFCN"]):
                ear_v = _extract_earfcn(res.at[idx, "Serving Sector"])
                if ear_v is not None:
                    res.at[idx, "Match_EARFCN"] = ear_v
            if "PCI" in res.columns and pd.isna(res.at[idx, "PCI"]):
                if pd.notna(res.at[idx, "Match_PCI"]):
                    res.at[idx, "PCI"] = res.at[idx, "Match_PCI"]

    # ── Strategy 2 (priority for PCI-EARFCN format): combined key ───────────
    unmatched = res["ENODEBNAME"].isna() | (
        res["Match_Status"].isna() | (res["Match_Status"] != "Matched")
    )
    pci_format_mask = (
        res["Serving Sector"].map(_is_pci_earfcn_format)
        if "Serving Sector" in res.columns
        else pd.Series(False, index=res.index)
    )
    candidates = res[unmatched & pci_format_mask]

    if not candidates.empty and "PCI" in dbk.columns and earfcn_db_col is not None:
        db_combo = dbk.dropna(subset=["eNodBID", "CellID", "PCI", earfcn_db_col]).copy()
        for idx in candidates.index:
            pci_v = pd.to_numeric(res.at[idx, "Match_PCI"], errors="coerce")
            ear_v = pd.to_numeric(res.at[idx, "Match_EARFCN"], errors="coerce")
            enb_v = pd.to_numeric(res.at[idx, "eNB_Part"], errors="coerce")
            cell_v = pd.to_numeric(res.at[idx, "Cell_Part"], errors="coerce")

            res.at[idx, "Match_PCI"] = pci_v if pd.notna(pci_v) else np.nan
            res.at[idx, "Match_EARFCN"] = ear_v if pd.notna(ear_v) else np.nan

            missing_parts = []
            if pd.isna(pci_v):
                missing_parts.append("Missing PCI")
            if pd.isna(ear_v):
                missing_parts.append("Missing EARFCN DL")
            if pd.isna(enb_v):
                missing_parts.append("Missing eNB_Part")
            if pd.isna(cell_v):
                missing_parts.append("Missing Cell_Part")

            if missing_parts:
                res.at[idx, "Match_Status"] = "Unmatched"
                res.at[idx, "Match_Reason"] = "; ".join(missing_parts)
                continue

            hits = db_combo[
                (db_combo["PCI"] == pci_v)
                & (db_combo[earfcn_db_col] == ear_v)
                & (db_combo["eNodBID"] == enb_v)
                & (db_combo["CellID"] == cell_v)
            ]
            n = len(hits)
            if n == 1:
                _apply_db_row(idx, hits.iloc[0])
                res.at[idx, "Match_Status"] = "Matched"
                res.at[idx, "Match_Reason"] = (
                    f"Matched by PCI+EARFCN+eNB+Cell "
                    f"(PCI={int(pci_v)}, EARFCN={int(ear_v)}, "
                    f"eNB={int(enb_v)}, Cell={int(cell_v)})"
                )
            elif n > 1:
                res.at[idx, "Match_Status"] = "Ambiguous"
                res.at[idx, "Match_Reason"] = (
                    f"Multiple matching records ({n}) for "
                    f"PCI={int(pci_v)}, EARFCN={int(ear_v)}, "
                    f"eNB={int(enb_v)}, Cell={int(cell_v)}"
                )
            else:
                res.at[idx, "Match_Status"] = "Unmatched"
                res.at[idx, "Match_Reason"] = (
                    f"No matching database record for "
                    f"PCI={int(pci_v)}, EARFCN={int(ear_v)}, "
                    f"eNB={int(enb_v)}, Cell={int(cell_v)}"
                )

    # ── Strategy 3: (eNB_Part, Cell_Part) ONLY for non-PCI-format rows ──────
    still_open = (
        res["Match_Status"].isna()
        | (res["Match_Status"] == "")
        | (
            (res["Match_Status"] == "Unmatched")
            & ~pci_format_mask
            & res["ENODEBNAME"].isna()
        )
    )
    non_pci_open = still_open & ~pci_format_mask
    if non_pci_open.any():
        db_by_num = (
            dbk.dropna(subset=["eNodBID", "CellID"])
            .drop_duplicates(["eNodBID", "CellID"])
            .set_index(["eNodBID", "CellID"])
        )
        for idx in res[non_pci_open].index:
            enb_v = pd.to_numeric(res.at[idx, "eNB_Part"], errors="coerce")
            cell_v = pd.to_numeric(res.at[idx, "Cell_Part"], errors="coerce")
            if pd.isna(enb_v) or pd.isna(cell_v):
                if pd.isna(res.at[idx, "Match_Status"]) or res.at[idx, "Match_Status"] == "":
                    res.at[idx, "Match_Status"] = "Unmatched"
                    res.at[idx, "Match_Reason"] = "Missing eNB_Part or Cell_Part"
                continue
            key = (enb_v, cell_v)
            if key in db_by_num.index:
                row_data = db_by_num.loc[key]
                if isinstance(row_data, pd.DataFrame):
                    if len(row_data) > 1:
                        res.at[idx, "Match_Status"] = "Ambiguous"
                        res.at[idx, "Match_Reason"] = (
                            f"Multiple matching records ({len(row_data)}) for "
                            f"eNB={int(enb_v)}, Cell={int(cell_v)}"
                        )
                        continue
                    row_data = row_data.iloc[0]
                _apply_db_row(idx, row_data)
                res.at[idx, "Match_Status"] = "Matched"
                res.at[idx, "Match_Reason"] = (
                    f"Matched by eNB+Cell (eNB={int(enb_v)}, Cell={int(cell_v)})"
                )
            else:
                if pd.isna(res.at[idx, "Match_Status"]) or res.at[idx, "Match_Status"] == "":
                    res.at[idx, "Match_Status"] = "Unmatched"
                    res.at[idx, "Match_Reason"] = (
                        f"No matching database record for "
                        f"eNB={int(enb_v)}, Cell={int(cell_v)}"
                    )

    # ── Strategy 4: (eNB_Part, PCI) ONLY for non-PCI-format rows still open ─
    still = (
        res["ENODEBNAME"].isna()
        & res["eNB_Part"].notna()
        & ~pci_format_mask
        & (
            res["Match_Status"].isna()
            | (res["Match_Status"] == "")
            | (res["Match_Status"] == "Unmatched")
        )
    )
    pci_col = "Match_PCI" if "Match_PCI" in res.columns else ("PCI" if "PCI" in res.columns else None)
    if pci_col and "PCI" in dbk.columns and still.any():
        need_pci = still & res[pci_col].notna()
        if need_pci.any():
            db_by_pci = (
                dbk.dropna(subset=["eNodBID", "PCI"])
                .drop_duplicates(["eNodBID", "PCI"])
                .set_index(["eNodBID", "PCI"])
            )
            for idx in res[need_pci].index:
                enb_v = float(pd.to_numeric(res.at[idx, "eNB_Part"], errors="coerce"))
                pci_v = float(pd.to_numeric(res.at[idx, pci_col], errors="coerce"))
                key = (enb_v, pci_v)
                if key in db_by_pci.index:
                    row_data = db_by_pci.loc[key]
                    if isinstance(row_data, pd.DataFrame):
                        if len(row_data) > 1:
                            res.at[idx, "Match_Status"] = "Ambiguous"
                            res.at[idx, "Match_Reason"] = (
                                f"Multiple matching records ({len(row_data)}) for "
                                f"eNB={int(enb_v)}, PCI={int(pci_v)}"
                            )
                            continue
                        row_data = row_data.iloc[0]
                    _apply_db_row(idx, row_data)
                    res.at[idx, "Match_Status"] = "Matched"
                    res.at[idx, "Match_Reason"] = (
                        f"Matched by eNB+PCI (eNB={int(enb_v)}, PCI={int(pci_v)})"
                    )

    blank_status = res["Match_Status"].isna() | (res["Match_Status"] == "")
    if blank_status.any():
        res.loc[blank_status, "Match_Status"] = "Unmatched"
        res.loc[blank_status, "Match_Reason"] = "No matching database record"

    return res


# ---------------------------------------------------------------------------
# Scores & classification (Worst Cells only)
# ---------------------------------------------------------------------------
def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Service Rate" not in df.columns:
        df["Service Rate"] = 0.0
    sr = df["Service Rate"]
    if "RSRP" in df.columns:
        df["Normalized RSRP"] = ((df["RSRP"] + 105) / 25).clip(0, 1)
        df["RSRP Score"] = (sr + (1 - df["Normalized RSRP"])) / 2
    if "RSRQ" in df.columns:
        df["Normalized RSRQ"] = ((df["RSRQ"] + 16) / 6).clip(0, 1)
        df["RSRQ Score"] = (sr + (1 - df["Normalized RSRQ"])) / 2
    if "SINR" in df.columns:
        df["Normalized SINR"] = ((df["SINR"] + 2) / 12).clip(0, 1)
        df["SINR Score"] = (sr + (1 - df["Normalized SINR"])) / 2
    if "SchPDSCHThrpt" in df.columns:
        df["Normalized Thput"] = ((df["SchPDSCHThrpt"] - 20000) / 40000).clip(0, 1)
        df["Thput Score"] = (sr + (1 - df["Normalized Thput"])) / 2
    if "p863LQ" in df.columns:
        df["MOS Normalize"] = ((df["p863LQ"] - 3.0) / 1.5).clip(0, 1)
        df["MOS Score"] = (sr + (1 - df["MOS Normalize"])) / 2
    return df


def _rule_mask(df: pd.DataFrame, metric_label: str, op: str, value: float) -> pd.Series:
    col = EDITABLE_METRICS.get(metric_label)
    if not col or col not in df.columns or op not in OPS:
        return pd.Series(False, index=df.index)
    series = pd.to_numeric(df[col], errors="coerce")
    return OPS[op](series, float(value))


def apply_extra_classification(
    df: pd.DataFrame, extra: list[dict], category: str, target_col: str
) -> pd.DataFrame:
    """First-match extra rules overwrite the built-in label when they hit.

    Each rule may contain a list of *conditions* (AND-ed).  Legacy single-condition
    rules are still accepted for backward compatibility.
    """
    if df.empty or not extra:
        return df
    if target_col not in df.columns:
        df[target_col] = ""
    assigned = pd.Series(False, index=df.index)
    cat = df[target_col].copy()
    for rule in extra:
        if rule.get("category") != category:
            continue
        conditions = rule.get("conditions")
        if not conditions:
            # Legacy fallback: single flat condition
            conditions = [
                {
                    "metric": rule.get("metric", ""),
                    "op": rule.get("op", "<="),
                    "value": rule.get("value", 0),
                }
            ]
        mask = pd.Series(True, index=df.index)
        for cond in conditions:
            m = _rule_mask(
                df, cond.get("metric", ""), cond.get("op", "<="), cond.get("value", 0)
            )
            mask = mask & m.fillna(False)
        mask = mask & ~assigned
        label = str(rule.get("label") or "").strip()
        if label:
            cat = cat.where(~mask, label)
            assigned = assigned | mask
    df[target_col] = cat
    return df


def classify_rf_kpi(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    y, s, d = df.get("RSRP"), df.get("SINR"), df.get("Serv_Dist_avg")
    if y is None or s is None:
        df["Criteria Check Based on KPI"] = ""
        return df
    cat = pd.Series("", index=df.index)
    if d is not None:
        cat[(y <= c["rsrp"]) & (s <= c["sinr"]) & (d >= c["dist"])] = "Overshooter"
        cat[(y <= c["rsrp"]) & (s <= c["sinr"]) & (d < c["dist"])] = "NonDominant"
    cat[(y > c["rsrp"]) & (s <= c["sinr"])] = "PoorQuality"
    cat[(y > c["rsrp"]) & (s > c["sinr"])] = "GoodCoverage"
    df["Criteria Check Based on KPI"] = cat
    return df


def classify_rf_score(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    if "SINR Score" not in df.columns or "RSRP Score" not in df.columns:
        df["Criteria Check Based on Score"] = ""
        return df
    sr = df["Service Rate"]
    rg, si = df["RSRP Score"], df["SINR Score"]
    d = df.get("Serv_Dist_avg", pd.Series(np.nan, index=df.index))
    good = si < ((sr + c["rf_sinr_const"]) / 2)
    cat = pd.Series("", index=df.index)
    cat[good] = "GoodCoverage"
    nd = (~good) & (rg >= ((sr + c["rf_score_const"]) / 2)) & (d < c["dist"])
    ov = (~good) & (rg >= ((sr + c["rf_score_const"]) / 2)) & (d >= c["dist"])
    pr = (~good) & (rg < ((sr + c["rf_score_const"]) / 2))
    cat[nd], cat[ov], cat[pr] = "NonDominant", "Overshooter", "PoorQuality"
    df["Criteria Check Based on Score"] = cat
    return df


def classify_thput(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    y, s = df.get("RSRP"), df.get("SINR")
    d = df.get("Serv_Dist_avg", pd.Series(np.nan, index=df.index))
    net = pd.to_numeric(df.get("NetPDSCHThrpt"), errors="coerce")
    gate = net < c["thput_dl_mbps"] * 1000
    cat = pd.Series("", index=df.index)
    if y is not None and s is not None:
        cat[gate & (y <= c["rsrp"]) & (s <= c["sinr"]) & (d >= c["dist"])] = "OvershooterPoorThput"
        cat[gate & (y <= c["rsrp"]) & (s <= c["sinr"]) & (d < c["dist"])] = "NonDominantPoorThput"
        cat[gate & (y > c["rsrp"]) & (s <= c["sinr"]) & (d < c["dist"])] = "PoorQualityPoorThput"
        cat[gate & (y > c["rsrp"]) & (s > c["sinr"]) & (d < c["dist"])] = "GoodCoveragePoorThput"
        cat[gate & (y > c["rsrp"]) & (s > c["sinr"]) & (d >= c["dist"])] = "OvershooterGoodCoveragePoorThput"
    df["Criteria Check Based on KPI"] = cat
    return df


def classify_mos(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    y, s = df.get("RSRP"), df.get("SINR")
    d = df.get("Serv_Dist_avg", pd.Series(np.nan, index=df.index))
    mos = pd.to_numeric(df.get("p863LQ"), errors="coerce")
    gate = mos <= c["mos"]
    cat = pd.Series("", index=df.index)
    if y is not None and s is not None:
        cat[gate & (y <= c["rsrp"]) & (s <= c["sinr"]) & (d >= c["dist"])] = "OvershooterPoorMOS"
        cat[gate & (y <= c["rsrp"]) & (s <= c["sinr"]) & (d < c["dist"])] = "NonDominantPoorMOS"
        cat[gate & (y > c["rsrp"]) & (s <= c["sinr"]) & (d < c["dist"])] = "PoorQualPoorMOS"
        cat[gate & (y > c["rsrp"]) & (s > c["sinr"]) & (d < c["dist"])] = "GoodCoveragePoorMOS"
        cat[gate & (y > c["rsrp"]) & (s > c["sinr"]) & (d >= c["dist"])] = "OvershooterGoodCoveragePoorMOS"
    df["Criteria Check Based on KPI"] = cat
    return df


def classify_cst(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    q = df.get("Serv_Dist_avg", pd.Series(np.nan, index=df.index))
    x = df.get("RSRP", pd.Series(np.nan, index=df.index))
    z = df.get("SINR", pd.Series(np.nan, index=df.index))

    def f(qv, xv, zv):
        if pd.notna(qv) and qv >= c["cst_dist_over"]:
            return "Overshooter"
        if pd.isna(xv) or xv in ("", "NULL"):
            return "Poor Quality"
        if (
            pd.notna(xv)
            and pd.notna(qv)
            and xv <= c["cst_rsrp_non_dom"]
            and c["cst_q_lo"] <= qv <= c["cst_q_hi"]
        ):
            return "Non Dominant"
        if pd.notna(xv) and xv <= c["cst_rsrp_poorcov"]:
            return "Poor Coverage"
        if pd.notna(zv) and zv < c["cst_sinr_poorq"]:
            return "Poor Quality"
        return "RF is good"

    df["Status"] = [f(qv, xv, zv) for qv, xv, zv in zip(q, x, z)]
    return df


# ---------------------------------------------------------------------------
# Worst-Cell definition filter
# ---------------------------------------------------------------------------
def worst_mask(df: pd.DataFrame, category: str, cdef: dict, extra: list[dict] | None = None) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)

    if category == "RF":
        r = df.get("RSRP", pd.Series(np.nan, index=df.index))
        s = df.get("SINR", pd.Series(np.nan, index=df.index))
        sr = df.get("Service Rate", pd.Series(0.0, index=df.index))
        mask = ((r <= cdef["rsrp"]) | (s <= cdef["sinr"])) & (sr >= cdef["rf_service_rate"])
    elif category == "Throughput":
        net = pd.to_numeric(df.get("NetPDSCHThrpt"), errors="coerce")
        mask = net <= cdef["thput_dl_mbps"] * 1000
    elif category == "MOS":
        mos = pd.to_numeric(df.get("p863LQ"), errors="coerce")
        mask = mos <= cdef["mos"]
    elif category == "CST":
        cst = pd.to_numeric(df.get("Voice_Call_Setup_Time_sec"), errors="coerce")
        mask = cst >= cdef["cst"]
    else:
        mask = pd.Series(False, index=df.index)

    for rule in extra or []:
        if rule.get("category") != category:
            continue
        mask = mask & _rule_mask(
            df, rule.get("metric", ""), rule.get("op", "<="), rule.get("value", 0)
        )
    return mask


# ---------------------------------------------------------------------------
# Column selection for output sheets
# ---------------------------------------------------------------------------
ID_COLS = [
    "eNB_Part", "Cell_Part", "ENODEBNAME", "CELLName", "Serving Sector",
    "Match_PCI", "Match_EARFCN", "Match_Status", "Match_Reason",
    "Source File", "Source Type",
]
DB_COLS = [
    "LONgitude", "LATitude", "PCI", "Height", "Azimuth",
    "Mechanical_Downtilt", "Electrical_Downtilt", "DLEARFCN",
    "Downlink_bandwidth", "RS_Power", "In Load Balance Report", "In Power Report",
]
RF_RAW = [
    "RSRP", "RSRQ", "SINR", "RSSI", "Serv_Dist_avg", "Serv_Dist_min", "Serv_Dist_max",
    "Azimuth_Orientation_Delta", "count", "Service Rate",
]
RF_SCORE = [
    "Normalized RSRP", "Normalized RSRQ", "Normalized SINR",
    "RSRP Score", "RSRQ Score", "SINR Score",
]
RF_CHECK = ["Criteria Check Based on KPI", "Criteria Check Based on Score"]
THP_RAW = RF_RAW + [
    "dl_throughput", "ul_throughput", "SchPDSCHThrpt", "NetPDSCHThrpt",
    "AvgMCS", "BLER", "NumRBs", "TransmissionMode",
]
THP_SCORE = ["Normalized Thput", "Thput Score"]
THP_CHECK = ["Criteria Check Based on KPI"]
MOS_RAW = RF_RAW + ["p863LQ", "MOS=<4%", "MOS=<3.5%", "MOS=<2.6%", "MOS=<2%"]
MOS_SCORE = ["MOS Normalize", "MOS Score"]
MOS_CHECK = ["Criteria Check Based on KPI"]
CST_RAW = RF_RAW + ["Voice_Call_Setup_Time_sec"]
CST_CHECK = ["Status"]

_SPEC = {
    "RF": (RF_RAW, RF_SCORE, RF_CHECK),
    "Throughput": (THP_RAW, THP_SCORE, THP_CHECK),
    "MOS": (MOS_RAW, MOS_SCORE, MOS_CHECK),
    "CST": (CST_RAW, [], CST_CHECK),
}


def select_columns(df: pd.DataFrame, category: str, is_worst: bool) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for c in ("ENODEBNAME", "CELLName"):
        if c not in df.columns:
            df[c] = pd.Series([np.nan] * len(df), dtype=object)
    raw, score, check = _SPEC[category]
    cols = ID_COLS + raw + (score if is_worst else []) + (check if is_worst else []) + DB_COLS
    seen = set()
    ordered = []
    for c in cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            ordered.append(c)
    return df[ordered].copy()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _process_one_file(
    path: str,
    ftype: str,
    mapping: dict[str, str | None],
    db_path: str | None,
    db_mapping: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    raw = load_excel(path)

    # ── VBA CarryForwardCellIdentities equivalent + blank-sector row removal ──
    # 1. Carry eNB / Cell identity forward on the raw source columns (before
    #    renaming) so every Serving-Sector row inherits the correct identity
    #    from the preceding identity-only rows.
    # 2. Then drop any row whose Serving Sector is blank/null — these rows were
    #    only needed to propagate identity and now act as spurious group-boundary
    #    markers inside _resolve_occurrences, inflating counts for PCI-EARFCN
    #    cells.
    enb_src    = mapping.get("enb")    if mapping else None
    cell_src   = mapping.get("cell")   if mapping else None
    sector_src = mapping.get("sector") if mapping else None
    raw = carry_forward_cell_identity_raw(raw, enb_src, cell_src, sector_src)
    # ─────────────────────────────────────────────────────────────────────────

    mapped = apply_mapping(raw, mapping)
    mapped = preprocess(mapped)

    # Look up missing eNB/Cell from DB before aggregation
    mapped = _lookup_cell_identity_from_db(mapped, db_path, db_mapping)

    # Service-rate denominator: rows with a non-empty Serving Sector
    # (empty-sector rows are dropped in preprocess and must not inflate counts)
    total = len(mapped)

    agg = aggregate(mapped)
    if agg.empty:
        return agg
    agg["Source File"] = os.path.basename(path)
    agg["Source Type"] = ftype
    agg["Service Rate"] = (agg["count"] / total) if total else 0.0
    agg = _merge_duplicate_cells(agg)
    agg = join_db(agg, db_path, db_mapping)
    return agg


def run_pipeline(
    ps_paths: list[str],
    cs_paths: list[str],
    db_path: str | None,
    mappings: dict[str, dict[str, str | None]],
    criteria: dict | None = None,
    enabled_cats: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    criteria = criteria or DEFAULT_CRITERIA
    cdef = criteria["definition"]
    cclass = criteria["classification"]
    extra_def = criteria.get("extra_definition") or []
    extra_cls = criteria.get("extra_classification") or []
    db_mapping = mappings.get("DB") or {}
    enabled = set(enabled_cats or CATS)

    rf_frames: list[pd.DataFrame] = []
    thp_frames: list[pd.DataFrame] = []
    mos_frames: list[pd.DataFrame] = []
    cst_frames: list[pd.DataFrame] = []

    for p in ps_paths:
        agg = _process_one_file(p, "PS", mappings.get("PS", {}), db_path, db_mapping)
        if agg.empty:
            continue
        rf_frames.append(agg.copy())
        thp_frames.append(agg.copy())

    for p in cs_paths:
        agg = _process_one_file(p, "CS", mappings.get("CS", {}), db_path, db_mapping)
        if agg.empty:
            continue
        rf_frames.append(agg.copy())
        if "MOS" in enabled:
            mos_frames.append(agg.copy())
        if "CST" in enabled:
            cst_frames.append(agg.copy())

    cat_data: dict[str, pd.DataFrame] = {
        "RF": pd.concat(rf_frames, ignore_index=True) if rf_frames else pd.DataFrame(),
        "Throughput": pd.concat(thp_frames, ignore_index=True) if thp_frames else pd.DataFrame(),
        "MOS": pd.concat(mos_frames, ignore_index=True) if mos_frames else pd.DataFrame(),
        "CST": pd.concat(cst_frames, ignore_index=True) if cst_frames else pd.DataFrame(),
    }

    if "CST" in enabled and not cat_data["CST"].empty:
        cat_data["CST"] = classify_cst(cat_data["CST"], cclass)
        cat_data["CST"] = apply_extra_classification(
            cat_data["CST"], extra_cls, "CST", "Status"
        )

    results: dict[str, pd.DataFrame] = {}
    for cat in CATS:
        if cat not in enabled:
            continue
        af = cat_data.get(cat, pd.DataFrame())
        if af.empty:
            results[f"{cat} Worst Cells"] = pd.DataFrame()
            results[f"{cat} All Cells"] = pd.DataFrame()
            continue
        allc = select_columns(af, cat, is_worst=False)
        mask = worst_mask(af, cat, cdef, extra_def)
        worst = af[mask].copy()
        if not worst.empty:
            worst = add_scores(worst)
            if cat == "RF":
                worst = classify_rf_score(classify_rf_kpi(worst, cclass), cclass)
                worst = apply_extra_classification(
                    worst, extra_cls, "RF", "Criteria Check Based on KPI"
                )
            elif cat == "Throughput":
                worst = classify_thput(worst, cclass)
                worst = apply_extra_classification(
                    worst, extra_cls, "Throughput", "Criteria Check Based on KPI"
                )
            elif cat == "MOS":
                worst = classify_mos(worst, cclass)
                worst = apply_extra_classification(
                    worst, extra_cls, "MOS", "Criteria Check Based on KPI"
                )
            elif cat == "CST":
                worst = classify_cst(worst, cclass)
                worst = apply_extra_classification(worst, extra_cls, "CST", "Status")
        results[f"{cat} Worst Cells"] = select_columns(worst, cat, is_worst=True)
        results[f"{cat} All Cells"] = allc
    return results


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c == "Service Rate":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif df[c].dtype == object:
            df[c] = df[c].apply(
                lambda v: (
                    ""
                    if (v is None or (isinstance(v, float) and pd.isna(v)))
                    else (
                        str(int(v))
                        if isinstance(v, float) and v.is_integer()
                        else str(v)
                    )
                )
            )
    return df


def export_excel(results: dict, path: str) -> None:
    from openpyxl.styles import Font, PatternFill

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="0099CC", end_color="0099CC", fill_type="solid")

    # Only export sheets that actually exist in results
    all_sheets = [f"{c} Worst Cells" for c in CATS] + [f"{c} All Cells" for c in CATS]
    sheets = [s for s in all_sheets if s in results]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for sheet in sheets:
            df = results.get(sheet)
            sheet_title = sheet[:31]
            if df is None or df.empty:
                pd.DataFrame({"Note": ["No qualifying cells"]}).to_excel(
                    xw, sheet_name=sheet_title, index=False
                )
            else:
                _sanitize(df).to_excel(xw, sheet_name=sheet_title, index=False)

            ws = xw.book[sheet_title]

            # Style header row: bold font, #0099CC fill
            for col_idx in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.font = header_font
                cell.fill = header_fill

            # Format Service Rate column as percentage (0.00%)
            sr_col_idx = None
            for col_idx in range(1, ws.max_column + 1):
                if ws.cell(row=1, column=col_idx).value == "Service Rate":
                    sr_col_idx = col_idx
                    break

            if sr_col_idx is not None:
                for row_idx in range(2, ws.max_row + 1):
                    cell = ws.cell(row=row_idx, column=sr_col_idx)
                    if cell.value is not None and isinstance(cell.value, (int, float)):
                        cell.number_format = "0.00%"