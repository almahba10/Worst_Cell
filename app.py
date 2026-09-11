"""
app.py — Streamlit Worst Cells dashboard (v4).

- Four input types: PS, CS, 4G DB, VoLTE
- Mapping UI shows mandatory metrics (optional auto-mapped, editable in expander)
- 4G Database has its own mapping tab
# Criteria live in the sidebar (Definition + Classification).
"""
from __future__ import annotations

import copy
import os

import pandas as pd
import streamlit as st

from wc_engine import (
    MAPPING_SPECS,
    DEFAULT_CRITERIA,
    EDITABLE_METRICS,
    CS_MOS_KEYS,
    CATS,
    auto_map_headers,
    validate_mapping,
    display_specs,
    run_pipeline,
    export_excel,
    norm_cols,
)

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Worst Cells Generator", layout="wide")
st.title("Drive-Test Worst Cells Generator")
# st.caption(
#     "RF ← PS + CS  ·  Throughput ← PS only  ·  MOS ← CS  ·  CST ← CS"
# )

# Compact red minus for rule removal buttons
st.markdown(
    """
    <style>
    button[kind="secondary"] {
        color: #ff4b4b !important;
        font-weight: 700;
        font-size: 1.2rem;
        line-height: 1;
        padding: 0.25rem 0.5rem;
        min-height: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "criteria" not in st.session_state:
    st.session_state.criteria = copy.deepcopy(DEFAULT_CRITERIA)

# Migrate legacy extra_classification rules (single condition) to new format
for rule in st.session_state.criteria.get("extra_classification", []):
    if "conditions" not in rule:
        rule["conditions"] = [
            {
                "metric": rule.pop("metric", "RSRP"),
                "op": rule.pop("op", "<="),
                "value": rule.pop("value", 0.0),
            }
        ]

for ftype in ("PS", "CS", "DB"):
    st.session_state.setdefault(f"mapping_{ftype}", {})
    st.session_state.setdefault(f"headers_{ftype}", [])

# Optional category toggles (persisted in session state)
st.session_state.setdefault("enable_rf", True)
st.session_state.setdefault("enable_thput", True)
st.session_state.setdefault("enable_mos", True)
st.session_state.setdefault("enable_cst", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_headers(uploaded) -> list[str]:
    try:
        df = pd.read_excel(uploaded, nrows=0)
        df = norm_cols(df)
        return [str(c) for c in df.columns]
    except Exception as e:
        st.warning(f"Could not read headers from {getattr(uploaded, 'name', '?')}: {e}")
        return []


def _union_headers(files) -> list[str]:
    seen: dict[str, None] = {}
    for f in files or []:
        for h in _read_headers(f):
            if h not in seen:
                seen[h] = None
        try:
            f.seek(0)
        except Exception:
            pass
    return list(seen.keys())


def render_mapping_table(ftype: str, headers: list[str]) -> dict[str, str | None]:
    """Mandatory rows in the main table; optional auto-mapped rows in an expander."""
    stored = st.session_state.get(f"mapping_{ftype}", {})

    if headers and (not stored or st.session_state.get(f"headers_{ftype}") != headers):
        stored = auto_map_headers(headers, ftype, auto_map_optional=False)
        st.session_state[f"mapping_{ftype}"] = stored
        st.session_state[f"headers_{ftype}"] = headers

    labels = {
        "PS": "PS Metric Mapping",
        "CS": "CS Metric Mapping",
        "DB": "4G Database Mapping",
    }
    st.markdown(f"#### {labels.get(ftype, ftype)}")
    if ftype == "CS":
        st.caption("MOS requires at least one of POLQA / CS POLQA / AQM.")
    if ftype == "DB":
        st.caption("Join keys (eNodeB ID + Cell ID) are mandatory. Site attributes fill output columns.")

    if not headers:
        st.info(f"Upload a {ftype} file to detect columns.")
        return stored

    used_by_others: dict[str, str] = {hdr: key for key, hdr in stored.items() if hdr}

    def _row(key: str, label: str, mandatory: bool, mapping: dict) -> None:
        c0, c1, c2 = st.columns([3, 4, 2])
        c0.write(label + (" *" if mandatory else ""))
        current = stored.get(key)
        options = [""]
        if current and current in headers:
            options.append(current)
        for h in headers:
            owner = used_by_others.get(h)
            if h != current and owner not in (None, key):
                continue
            if h not in options:
                options.append(h)
        idx = options.index(current) if current in options else 0
        choice = c1.selectbox(
            label,
            options=options,
            index=idx,
            key=f"map_{ftype}_{key}",
            label_visibility="collapsed",
        )
        chosen = choice if choice else None
        mapping[key] = chosen
        if chosen:
            c2.success("Mapped")
            used_by_others[chosen] = key
        elif mandatory:
            c2.error("Not mapped")
        else:
            c2.caption("Optional")

    new_mapping: dict[str, str | None] = dict(stored)

    h0, h1, h2 = st.columns([3, 4, 2])
    h0.markdown("**Required metric**")
    h1.markdown("**Excel column**")
    h2.markdown("**Status**")

    for key, label, mandatory in display_specs(ftype, include_optional=False):
        # CS MOS group: POLQA is not individually mandatory
        mand_flag = mandatory if not (ftype == "CS" and key in CS_MOS_KEYS) else False
        _row(key, label, mand_flag, new_mapping)

    shown_keys = {s[0] for s in display_specs(ftype, include_optional=False)}
    optional = [s for s in MAPPING_SPECS[ftype] if s[0] not in shown_keys]
    if optional:
        with st.expander("Optional columns (unmapped by default — assign if needed)", expanded=False):
            for key, label, _ in optional:
                _row(key, label, False, new_mapping)

    st.session_state[f"mapping_{ftype}"] = new_mapping
    return new_mapping


def save_uploads(files, prefix: str) -> list[str]:
    paths = []
    for f in files or []:
        p = os.path.join(HERE, f"_upl_{prefix}_{f.name}")
        with open(p, "wb") as fh:
            fh.write(f.getbuffer())
        paths.append(p)
    return paths


def _extra_editor(kind: str, categories: list[str]) -> None:
    """Single-condition rule editor used for Extra Definition gates."""
    rules: list[dict] = st.session_state.criteria.setdefault(kind, [])
    metrics = list(EDITABLE_METRICS.keys())
    ops = ["<=", "<", ">=", ">"]

    if not rules:
        st.caption("No extra rules yet.")

    remove_idx = None
    for i, rule in enumerate(rules):
        cols = st.columns([2, 3, 2, 2, 1])
        rule["category"] = cols[0].selectbox(
            "Category",
            categories,
            index=max(
                0,
                categories.index(rule.get("category", categories[0]))
                if rule.get("category") in categories
                else 0,
            ),
            key=f"{kind}_{i}_cat",
            label_visibility="collapsed",
        )
        m0 = rule.get("metric", metrics[0])
        rule["metric"] = cols[1].selectbox(
            "KPI",
            metrics,
            index=metrics.index(m0) if m0 in metrics else 0,
            key=f"{kind}_{i}_met",
            label_visibility="collapsed",
        )
        o0 = rule.get("op", "<=")
        rule["op"] = cols[2].selectbox(
            "Op",
            ops,
            index=ops.index(o0) if o0 in ops else 0,
            key=f"{kind}_{i}_op",
            label_visibility="collapsed",
        )
        rule["value"] = cols[3].number_input(
            "Value",
            value=float(rule.get("value", 0.0)),
            key=f"{kind}_{i}_val",
            label_visibility="collapsed",
        )
        if cols[4].button("−", key=f"{kind}_{i}_rm", help="Remove rule", type="secondary"):
            remove_idx = i

    if remove_idx is not None:
        rules.pop(remove_idx)
        st.rerun()

    if st.button("Add rule", key=f"{kind}_add"):
        rules.append(
            {"category": categories[0], "metric": metrics[0], "op": "<=", "value": 0.0}
        )
        st.rerun()


def render_classification_rules() -> None:
    """Multi-condition classification-rules editor.

    Each rule = one Category + one Class Label + multiple KPI conditions (AND-ed).
    """
    rules: list[dict] = st.session_state.criteria.setdefault("extra_classification", [])
    metrics = list(EDITABLE_METRICS.keys())
    categories = ["RF", "Throughput", "MOS", "CST"]
    ops = ["<=", "<", ">=", ">"]

    if not rules:
        st.caption("No extra classification rules yet.")

    remove_rule_idx = None
    for i, rule in enumerate(rules):
        with st.container(border=True):
            # ---- Rule header: Category | Label | Remove ----
            hdr = st.columns([2, 4, 1])
            cat0 = rule.get("category", categories[0])
            rule["category"] = hdr[0].selectbox(
                "Category",
                categories,
                index=categories.index(cat0) if cat0 in categories else 0,
                key=f"xcls_{i}_cat",
                label_visibility="collapsed",
            )
            rule["label"] = hdr[1].text_input(
                "Label",
                value=str(rule.get("label", "")),
                key=f"xcls_{i}_lab",
                label_visibility="collapsed",
                placeholder="Class label (e.g. Overshooter)",
            )
            if hdr[2].button(
                "−", key=f"xcls_{i}_rm_rule", help="Remove rule", type="secondary"
            ):
                remove_rule_idx = i

            # ---- Conditions ----
            conditions = rule.setdefault("conditions", [])
            remove_cond_idx = None
            for j, cond in enumerate(conditions):
                ccols = st.columns([3, 2, 2, 1])
                m0 = cond.get("metric", metrics[0])
                cond["metric"] = ccols[0].selectbox(
                    "KPI",
                    metrics,
                    index=metrics.index(m0) if m0 in metrics else 0,
                    key=f"xcls_{i}_{j}_met",
                    label_visibility="collapsed",
                )
                o0 = cond.get("op", "<=")
                cond["op"] = ccols[1].selectbox(
                    "Op",
                    ops,
                    index=ops.index(o0) if o0 in ops else 0,
                    key=f"xcls_{i}_{j}_op",
                    label_visibility="collapsed",
                )
                cond["value"] = ccols[2].number_input(
                    "Value",
                    value=float(cond.get("value", 0.0)),
                    key=f"xcls_{i}_{j}_val",
                    label_visibility="collapsed",
                )
                if ccols[3].button(
                    "−",
                    key=f"xcls_{i}_{j}_rm_cond",
                    help="Remove condition",
                    type="secondary",
                ):
                    remove_cond_idx = j

            if remove_cond_idx is not None:
                conditions.pop(remove_cond_idx)
                st.rerun()

            if st.button("Add condition", key=f"xcls_{i}_add_cond"):
                conditions.append({"metric": metrics[0], "op": "<=", "value": 0.0})
                st.rerun()

    if remove_rule_idx is not None:
        rules.pop(remove_rule_idx)
        st.rerun()

    if st.button("Add classification rule", key="xcls_add_rule"):
        rules.append({"category": categories[0], "label": "", "conditions": []})
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — Worst-Cell Definition + Classification
# ---------------------------------------------------------------------------
st.sidebar.header("Worst-Cell Definition")
st.sidebar.caption("Decides which cells enter the Worst Cell list.")
d = st.session_state.criteria["definition"]
st.sidebar.markdown("**RF:** (RSRP ≤ threshold **OR** SINR ≤ threshold) AND Sample Rate ≥ threshold")
d["rsrp"] = st.sidebar.number_input(
    "RSRP ≤ (dBm)", value=float(d["rsrp"]), key="d_rsrp"
)
d["sinr"] = st.sidebar.number_input(
    "SINR ≤ (dB)", value=float(d["sinr"]), key="d_sinr"
)
d["rf_service_rate"] = (
    st.sidebar.number_input(
        "RF Cell Sample Rate ≥ (%)",
        value=float(d["rf_service_rate"] * 100),
        key="d_sr",
    )
    / 100.0
)
st.sidebar.markdown("**Throughput:** DL Throughput ≤ threshold")
d["thput_dl_mbps"] = st.sidebar.number_input(
    "Throughput DL ≤ (Mbps)", value=float(d["thput_dl_mbps"]), key="d_thp"
)
st.sidebar.markdown("**MOS:** MOS ≤ threshold")
d["mos"] = st.sidebar.number_input("MOS ≤", value=float(d["mos"]), key="d_mos")
st.sidebar.markdown("**CST:** Call Setup Time ≥ threshold")
d["cst"] = st.sidebar.number_input(
    "Call Setup Time ≥ (s)", value=float(d.get("cst", 5.0)), key="d_cst"
)

c = st.session_state.criteria["classification"]
with st.sidebar.expander("Criteria-Check classification", expanded=False):
    st.caption("KPI tree splits (RF / Throughput / MOS)")
    c["rsrp"] = st.number_input(
        "RSRP split (dBm)", value=float(c["rsrp"]), key="c_kpi_rsrp"
    )
    c["sinr"] = st.number_input(
        "SINR split (dB)", value=float(c["sinr"]), key="c_kpi_sinr"
    )
    c["dist"] = st.number_input(
        "Distance split (m)", value=float(c["dist"]), key="c_dist"
    )
    c["thput_dl_mbps"] = st.number_input(
        "Throughput gate (Mbps)", value=float(c["thput_dl_mbps"]), key="c_thp"
    )
    c["mos"] = st.number_input("MOS gate", value=float(c["mos"]), key="c_cls_mos")
    c["cst"] = st.number_input(
        "CST gate (s)", value=float(c.get("cst", 5.0)), key="c_cls_cst"
    )
    st.caption("RF score tree")
    c["rf_score_const"] = st.number_input(
        "RF Score const", value=float(c["rf_score_const"]), format="%.4f", key="c_rf"
    )
    c["rf_sinr_const"] = st.number_input(
        "RF SINR Score const",
        value=float(c["rf_sinr_const"]),
        format="%.4f",
        key="c_rf_sinr",
    )
    st.caption("CST Status (order)")
    c["cst_dist_over"] = st.number_input(
        "Dist ≥ (m) → Overshooter", value=float(c.get("cst_dist_over", 1500)), key="c_vd"
    )
    c["cst_rsrp_non_dom"] = st.number_input(
        "Non-dom RSRP ≤", value=float(c.get("cst_rsrp_non_dom", -95)), key="c_vnd"
    )
    c["cst_q_lo"] = st.number_input(
        "Non-dom Dist lo (m)", value=float(c.get("cst_q_lo", 1000)), key="c_vlo"
    )
    c["cst_q_hi"] = st.number_input(
        "Non-dom Dist hi (m)", value=float(c.get("cst_q_hi", 1300)), key="c_vhi"
    )
    c["cst_rsrp_poorcov"] = st.number_input(
        "Poor Coverage RSRP ≤", value=float(c.get("cst_rsrp_poorcov", -90)), key="c_vpc"
    )
    c["cst_sinr_poorq"] = st.number_input(
        "Poor Quality SINR <", value=float(c.get("cst_sinr_poorq", 3)), key="c_vpq"
    )

with st.sidebar.expander("Extra definition gates", expanded=False):
    st.caption("AND-ed on the built-in definition. Known KPIs only.")
    _extra_editor("extra_definition", ["RF", "Throughput", "MOS", "CST"])

with st.sidebar.expander("Extra classification rules", expanded=False):
    st.caption("First match overwrites the built-in label.")
    render_classification_rules()

if st.sidebar.button("Reset criteria to defaults"):
    st.session_state.criteria = copy.deepcopy(DEFAULT_CRITERIA)
    st.rerun()


# ---------------------------------------------------------------------------
# 1. Input files
# ---------------------------------------------------------------------------
st.header("1. Input files")
c1, c2, c3 = st.columns(3)
with c1:
    ps_files = st.file_uploader(
        "TEMS PS export(s)", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="ps"
    )
with c2:
    cs_files = st.file_uploader(
        "TEMS CS export(s)", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="cs"
    )
with c3:
    db_file = st.file_uploader("4G Database (.xlsx)", type=["xlsx", "xls"], key="db")

st.markdown("**Worst-Cell categories to generate**")
st.caption("Uncheck a category to skip it. At least one must be selected.")
cb1, cb2, cb3, cb4 = st.columns(4)
with cb1:
    st.session_state.enable_rf = st.checkbox(
        "RF Worst Cell", value=st.session_state.enable_rf, key="chk_rf"
    )
with cb2:
    st.session_state.enable_thput = st.checkbox(
        "Throughput Worst Cell", value=st.session_state.enable_thput, key="chk_thput"
    )
with cb3:
    st.session_state.enable_mos = st.checkbox(
        "MOS Worst Cell", value=st.session_state.enable_mos, key="chk_mos"
    )
with cb4:
    st.session_state.enable_cst = st.checkbox(
        "CST Worst Cell", value=st.session_state.enable_cst, key="chk_cst"
    )

# ---------------------------------------------------------------------------
# 2. Metric mapping
# ---------------------------------------------------------------------------
st.header("2. Metric mapping")
st.info(
    "Only **mandatory** metrics are auto-mapped. Optional columns are unmapped by default "
    "— open the expander to assign. Each Excel header can be used once (1:1)."
)

ps_headers = _union_headers(ps_files)
cs_headers = _union_headers(cs_files)
db_headers = _union_headers([db_file] if db_file else [])

tab_ps, tab_cs, tab_db = st.tabs(
    ["PS mapping", "CS mapping", "4G Database"]
)
with tab_ps:
    map_ps = render_mapping_table("PS", ps_headers)
with tab_cs:
    map_cs = render_mapping_table("CS", cs_headers)
with tab_db:
    map_db = render_mapping_table("DB", db_headers)

miss_ps = validate_mapping(map_ps, "PS") if ps_files else []
miss_cs = validate_mapping(map_cs, "CS") if cs_files else []
miss_db = validate_mapping(map_db, "DB") if db_file else []

if miss_ps or miss_cs or miss_db:
    with st.expander("Unmapped mandatory metrics", expanded=True):
        if miss_ps:
            st.write("**PS:** " + ", ".join(miss_ps))
        if miss_cs:
            st.write("**CS:** " + ", ".join(miss_cs))
        if miss_db:
            st.write("**Database:** " + ", ".join(miss_db))

# ---------------------------------------------------------------------------
# 3. Generate
# ---------------------------------------------------------------------------
st.header("3. Generate")
if not (ps_files or cs_files):
    st.warning("Upload at least one TEMS PS or CS file.")
elif miss_ps or miss_cs or miss_db:
    st.warning("Map all mandatory metrics before running.")

enabled_cats = []
if st.session_state.enable_rf:
    enabled_cats.append("RF")
if st.session_state.enable_thput:
    enabled_cats.append("Throughput")
if st.session_state.enable_mos:
    enabled_cats.append("MOS")
if st.session_state.enable_cst:
    enabled_cats.append("CST")

if not enabled_cats:
    st.warning("Select at least one Worst-Cell category.")

run_disabled = (
    not (ps_files or cs_files)
    or bool(miss_ps)
    or bool(miss_cs)
    or bool(miss_db)
    or not enabled_cats
)

if st.button("Run Worst-Cells analysis", disabled=run_disabled, type="primary"):
    with st.spinner("Processing…"):
        ps_paths = save_uploads(ps_files, "PS")
        cs_paths = save_uploads(cs_files, "CS")
        db_path = None
        if db_file:
            db_path = os.path.join(HERE, "_upl_db.xlsx")
            with open(db_path, "wb") as fh:
                fh.write(db_file.getbuffer())

        results = run_pipeline(
            ps_paths=ps_paths,
            cs_paths=cs_paths,
            db_path=db_path,
            mappings={"PS": map_ps, "CS": map_cs, "DB": map_db},
            criteria=st.session_state.criteria,
            enabled_cats=enabled_cats,
        )
        st.session_state.results = results
        st.success("Done. See tabs below; export via the button at the bottom.")

# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------
if "results" in st.session_state:
    res = st.session_state.results
    sheet_order = [f"{c} Worst Cells" for c in enabled_cats] + [
        f"{c} All Cells" for c in enabled_cats
    ]
    tabs = st.tabs(sheet_order)
    for i, key in enumerate(sheet_order):
        with tabs[i]:
            df = res.get(key, pd.DataFrame())
            if df is None or df.empty:
                st.write("No data (no qualifying cells, or source metrics absent).")
                continue
            if {"eNB_Part", "Cell_Part"}.issubset(df.columns):
                n_unique = df.groupby(["eNB_Part", "Cell_Part"]).ngroups
                extra = ""
                if "Source Type" in df.columns:
                    extra = f"  |  by source: {df.groupby('Source Type').size().to_dict()}"
                st.write(
                    f"**{key}**: {len(df)} rows  ·  **{n_unique} unique (eNB, Cell)**{extra}"
                )
            else:
                st.write(f"**{key}**: {len(df)} rows")
            st.dataframe(df, width="stretch", height=400)

    st.header("4. Export to Excel")
    if st.button("Export results as Excel"):
        out_path = os.path.join(HERE, "WorstCells_Output.xlsx")
        export_excel(res, out_path)
        st.success(f"Exported to {out_path}")
        with open(out_path, "rb") as fh:
            st.download_button(
                "Download WorstCells_Output.xlsx",
                fh.read(),
                file_name="WorstCells_Output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )