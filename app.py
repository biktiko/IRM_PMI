import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tempfile import gettempdir

from src.data_loader import read_excel_all, build_weights_dict, questions_from_weights
from src.scoring import long_from_base, join_weights, aggregate_scores, extract_10pt_rating
import src.scoring as scoring

st.set_page_config(page_title="Վաճառքի կետերի գնահատման դաշբորդ", layout="wide")

@st.cache_data(show_spinner=False)
def load_excel_bytes(b: bytes):
    return b  # placeholder to leverage cache key

@st.cache_data(show_spinner=True)
def parse_workbooks(path: Path):
    return read_excel_all(path) 

# ===== persistence: save uploads to ./imports and autoload latest =====
IMPORTS_DIR = Path(gettempdir()) / "pmi_uploads"
IMPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _list_saved_excels():
    return sorted(IMPORTS_DIR.glob("*.xls*"), key=lambda p: p.stat().st_mtime, reverse=True)

def _save_uploaded(upl):
    dst = IMPORTS_DIR / upl.name
    dst.write_bytes(upl.getbuffer())
    return dst

st.sidebar.title("Տվյալների ներմուծում")
upl = st.sidebar.file_uploader("Բեռնեք Excel (BR թերթեր + LAS/YAP կշիռ)", type=["xlsx","xls"])

state = st.session_state
if "bases" not in state:
    state.bases = {}
    state.weights = pd.DataFrame()
    state.last_file = None

# Выбор источника: либо новый аплоад -> сохраняем, либо последний файл из imports
selected_path = None
if upl is not None:
    content = load_excel_bytes(upl.getvalue())
    tmp_path = IMPORTS_DIR / upl.name
    tmp_path.write_bytes(content)
    bases_raw, weights_raw = parse_workbooks(tmp_path)
    state.bases = bases_raw
    state.weights = build_weights_dict(weights_raw)
    state.last_file = str(tmp_path)
else:
    existing = _list_saved_excels()
    if existing:
        # авто-подхват последнего файла; при желании можно заменить на selectbox
        selected_path = existing[0]
        st.sidebar.info(f"Բեռնված է imports/{selected_path.name}")

# Если выбран файл и он изменился — перечитать базы и веса
if selected_path is not None and state.get("last_file") != str(selected_path):
    bases_raw, weights_raw = read_excel_all(str(selected_path))
    state.bases = bases_raw
    state.weights = build_weights_dict(weights_raw)
    state.last_file = str(selected_path)

# На каждом прогоне пересобираем рейтинги заново (без накопления в session)
state.ratings_list = []

if not state.bases:
    st.info("Բեռնեք Excel կամ տեղադրեք այն imports պանակում։")
    st.stop()
# ===== end persistence block =====

st.sidebar.markdown("---")

prepared = []
state.ratings_list = []

with st.expander("Տվյալների կարգավորումներ և ֆիլտրեր", expanded=False):
    for scen, base in state.bases.items():
        # comment: normalize scenario name
        scen = scen if scen in ["BR1", "BR2", "BR3", "BR4", "BR5"] else "BR1"

        # comment: filter base by scenario
        df_scene = scoring.filter_by_scenario_column(base, scen)

        # comment: small header (no huge H2)
        st.markdown(f"<h4>{scen} սցենարի մշակումը</h4>", unsafe_allow_html=True)

        # comment: get question keys from weights
        qkeys = questions_from_weights(state.weights, scen)
        if not qkeys:
            st.warning("Կշիռների թերթում տվյալ սցենարի հարցեր չեն հայտնաբերվել։ Խնդրում ենք ստուգել LAS/YAP թերթերը։")
            continue

        # comment: choose store column & skip columns
        options_cols = list(df_scene.columns)
        default_store_idx = options_cols.index("store") if "store" in options_cols else (1 if len(options_cols) > 1 else 0)
        store_col = st.selectbox(f"[{scen}] Խանութի սյունակ", options=options_cols, index=default_store_idx)

        default_skip = [c for c in options_cols[:8] if c != store_col]
        skip_cols = st.multiselect(f"[{scen}] Չհաշվարկվող սյունակներ", options=options_cols, default=default_skip)

        # comment: prepare long form for scoring
        drop_list = [c for c in skip_cols if c in df_scene.columns and c != store_col]
        df_for_long = df_scene.drop(columns=drop_list, errors="ignore")

        long_df = long_from_base(
            df_for_long,
            store_col=store_col,
            non_question_cols=skip_cols,
            scenario=scen,
            qkeys_norm=qkeys,
        )

        # comment: collect ratings per scenario
        ratings_scene = extract_10pt_rating(df_scene, scen, store_col)
        state.ratings_list.append(ratings_scene)

        # comment: join weights and accumulate
        merged = join_weights(long_df, state.weights, scen)
        prepared.append(merged)

    # --- 2) Build df_all and ratings, validate ---
    df_all = pd.concat(prepared, ignore_index=True) if prepared else pd.DataFrame()
    ratings = pd.concat(state.ratings_list, ignore_index=True) if state.ratings_list else pd.DataFrame()

    if df_all.empty:
        st.error("Հնարավոր չեղավ պատրաստել տվյալները։ Ստուգեք ներբեռված ֆայլը և կշիռների թերթերը։")
        st.stop()

# --- 3) Filters AFTER df_all exists ---
    st.header("Ֆիլտրեր")
    stores = sorted(df_all["store"].dropna().unique().tolist())
    scenarios = sorted(df_all["scenario"].dropna().unique().tolist())
    sections = sorted(df_all["section"].dropna().unique().tolist())

    sel_scen = st.multiselect("Սցենարներ", options=scenarios, default=scenarios)
    sel_stores = st.multiselect("Խանութներ", options=stores, default=stores)
    sel_sec = st.multiselect("Բաժիններ", options=sections, default=sections)

# --- 4) Apply filters once selections are made ---
flt = df_all[
    df_all["store"].isin(sel_stores)
    & df_all["scenario"].isin(sel_scen)
    & df_all["section"].isin(sel_sec)
]

with st.expander("Ներդրված տվյալներ"):

    total_records = len(flt)
    stores = flt['store'].unique()
    scenarios = flt['scenario'].nunique()
    sections = flt['section'].nunique()

    st.markdown(f"""
    **Գրառումներ:** {total_records:,}  
    **Խանութներ ({len(stores)}):**  
    {', '.join(stores)}  
    **Սցենարներ:** {scenarios}  
    **Բաժիններ:** {sections}
    """, unsafe_allow_html=True)

from src import ui
pq, ps, psc, pstore, _raw = aggregate_scores(flt)

tab_overview, tab_stores, tab_scen, tab_sections, tab_questions, tab_compare, tab_export = st.tabs(
    ["Ընդհանուր", "Խանութներ", "Սցենարներ", "Բաժիններ", "Հարցեր", "Համեմատել", "Արտահանում"]
)

with tab_overview:
    # --- 1) Chart first ---
    st.markdown("#### Խանութների ընդհանուր վարկանիշ")  # smaller than subheader
    ui.rating_table(pstore.rename(columns={"score_store_pct": "score"}), "score", "Ընդհանուր ")

    st.divider()

    # --- 2) Heatmap second ---
    st.markdown("#### Բաժինների ջերմաքարտեզ")
    ui.heatmap_sections(ps)

    st.divider()

    # --- 3) Table last ---
    if not ratings.empty:
        # Compute once, show clean table
        avg = (
            ratings
            .groupby(["store", "scenario"], as_index=False)["rating_10pt"]
            .mean()
            .round(2)
            .rename(columns={"rating_10pt": "avg_rating_10pt"})
            .sort_values(["avg_rating_10pt", "store"], ascending=[False, True])
            .reset_index(drop=True)
        )

        st.markdown("#### Խանութների միջին գնահատականը (1–10)")
        st.dataframe(avg, use_container_width=True)
    else:
        st.info("Տվյալներ չկան ցուցադրելու համար")


with tab_stores:
    st.subheader("Խանութի պրոֆիլ")
    if not pstore.empty:
        store = st.selectbox("Ընտրեք խանութ", options=sorted(pstore["store"].unique()))
        ui.store_profile(ps, pq, store)
    else:
        st.info("Տվյալներ չկան։")

with tab_scen:
    st.subheader("Ռեյտինգ ըստ սցենարների")
    if not psc.empty:
        scen = st.selectbox("Սցենար", options=sorted(psc["scenario"].unique()))
        ui.rating_table(psc[psc["scenario"]==scen].rename(columns={"score_scenario_pct":"score"}),
                        "score", f"{scen} — խանութների ռեյտինգ")
    else:
        st.info("Տվյալներ չկան։")

with tab_sections:
    st.subheader("Համեմատություն ըստ բաժինների")
    if not ps.empty:
        section = st.selectbox("Բաժին", options=sorted(ps["section"].dropna().unique()))
        df_sec = ps[ps["section"]==section].rename(columns={"score_section_pct":"score"})
        ui.rating_table(df_sec, "score", f"Բաժին «{section}» — խանութների ռեյտինգ")

    # Расчёт веса раздела из вопросов текущего сценария (ΣF)
    w_scen = st.session_state.weights[st.session_state.weights["scenario"] == scen].copy()
    calc = (w_scen.groupby("section", as_index=False)
                  .agg(section_weight_calc=("weight_in_scenario", "sum")))

    calc["section_weight_calc"] = (calc["section_weight_calc"] * 100).round(1)

    if "section_total_weight" in w_scen.columns and w_scen["section_total_weight"].notna().any():
        ref = (w_scen.dropna(subset=["section_total_weight"])
                      .drop_duplicates(subset=["section"])[["section","section_total_weight"]])
        ref["section_total_weight"] = (ref["section_total_weight"] * 100).round(1)

        comp = (calc.merge(ref, on="section", how="outer")
                    .assign(diff=lambda d: (d["section_weight_calc"] - d["section_total_weight"]).round(1)))
    else:
        st.markdown("**Բաժնի հանրագումար կշիռը (ΣF по вопросам)**")
        st.dataframe(calc.sort_values("section"), use_container_width=True)


with tab_questions:
    st.subheader("Մանրամասն ըստ հարցերի")
    if not pq.empty:
        st.dataframe(pq.sort_values(["scenario","section","store","question_key"])[
            ["store","scenario","section","question_key","score_question_pct"]
        ].round(1), use_container_width=True)
    else:
        st.info("Տվյալներ չկան։")

with tab_compare:
    st.subheader("Համեմատել երկու խանութ")
    if len(stores) >= 2 and not psc.empty:
        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox("Խանութ A", options=stores, index=0, key="cmp_a")
        with c2:
            b = st.selectbox("Խանութ B", options=stores, index=1, key="cmp_b")
        if a and b and a != b:
            ui.compare_two(psc, ps, a, b)
    else:
        st.info("Տվյալներ չկան համեմատության համար։")

with tab_export:
    st.subheader("Արտահանում")
    ui.download_buttons({
        "per_store": pstore.round(2),
        "per_scenario": psc.round(2),
        "per_section": ps.round(2),
        "per_question": pq.round(2),
    })

with st.expander("Կշիռների հաշվարկի Debug (վերահսկիչ հաշվարկ)", expanded=False):
    if not df_all.empty:
        dbg_store = st.selectbox("Խանութ (Debug)", options=sorted(df_all["store"].unique()))
        dbg_scen  = st.selectbox("Սցենար (Debug)", options=sorted(df_all["scenario"].unique()))
        from src.scoring import diagnostics_breakdown, diagnostics_summary
        dbg_rows = diagnostics_breakdown(df_all, dbg_store, dbg_scen)
        dbg_sum  = diagnostics_summary(df_all, dbg_store, dbg_scen)

        if dbg_rows.empty:
            st.warning("Չկան տվյալներ ընտրված խանութի / սցենարի համար։")
        else:
            st.markdown("**Մանրամասն ըստ հարցերի (վիճակ / կշիռ / ներդրում):**")
            st.dataframe(dbg_rows, use_container_width=True)

            st.markdown("**Սեկցիաների ամփոփում (ստուգում կշիռների համապատասխանությունը):**")
            st.dataframe(dbg_sum, use_container_width=True)

            scen_calc = dbg_rows["scenario_calc_pct"].dropna().unique()
            if scen_calc.size:
                st.info(f"Սցենարի վերահաշվարկված տոկոսը: {scen_calc[0]:.2f}%")

st.caption("© Դաշբորդ — հաշվարկը հիմնված է «Այո» պատասխանների կշռած բաժնին։ LAS/YAP կշիռները կիրառվում են բաժնի (E) և սցենարի (F) մակարդակներում։")
