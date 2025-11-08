import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import gettempdir
import hashlib, os
from src.data_loader import read_excel_all, build_weights_dict, questions_from_weights
from src.scoring import long_from_base, join_weights, aggregate_scores, extract_10pt_rating
import src.scoring as scoring
from src.utils import _parse_visit_date, pick_col

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# --- Credentials loader (supports multiple secret/env names) ---
def _load_creds():
    def _get(key):
        # пробуем secrets, иначе env
        try:
            return st.secrets[key]
        except Exception:
            return os.getenv(key)
    username = (
        _get("APP_USERNAME")
        or _get("USERNAME_ENV")
        or "PMI2025"
    )
    password_plain = (
        _get("APP_PASSWORD")
        or _get("PASS_ENV")
        or _get("PASS_HASH_ENV")
    )
    password_hash = _get("APP_PASS_HASH")

    # если хэш не задан, но password_plain выглядит как 64 hex — считаем хэшем
    if not password_hash and password_plain and len(password_plain) == 64 and all(c in "0123456789abcdef" for c in password_plain.lower()):
        password_hash = password_plain
        password_plain = None
    return username, password_plain, password_hash

APP_USERNAME, APP_PASSWORD_PLAIN, APP_PASSWORD_HASH = _load_creds()

def _check_password(inp: str) -> bool:
    # 1) если есть хэш — проверяем по SHA256
    if APP_PASSWORD_HASH:
        return _sha256(inp) == APP_PASSWORD_HASH
    # 2) fallback plain
    if APP_PASSWORD_PLAIN:
        return inp == APP_PASSWORD_PLAIN
    return False

st.set_page_config(page_title="Վաճառքի կետերի գնահատման դաշբորդ", layout="wide")

# --- AUTH ---
if st.sidebar.button("Logout / Դուրս գալ", disabled=not st.session_state.get("auth_ok")):
    for k in ("auth_ok","auth_user"):
        st.session_state.pop(k, None)
    st.rerun()

if not st.session_state.get("auth_ok"):
    st.title("Մուտք համակարգ")
    with st.form("login_form"):
        u = st.text_input("Login", value="")
        p = st.text_input("Password", type="password", value="")
        ok = st.form_submit_button("Login")
        if ok:
            if u == APP_USERNAME and _check_password(p):
                st.session_state["auth_ok"] = True
                st.session_state["auth_user"] = u
                st.success("Մուտքը հաջող է")
                st.rerun()
            else:
                st.error("Սխալ login կամ password")
    st.stop()
# --- конец авторизации, ниже основной код ---

@st.cache_data(show_spinner=False)
def load_excel_bytes(b: bytes):
    return b

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
upl = st.sidebar.file_uploader("Բեռնեք Excel ֆայլը", type=["xlsx","xls"])

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
    st.info("Բեռնեք Excel ֆայլը")
    st.stop()
# ===== end persistence block =====

st.sidebar.markdown("---")

prepared = []
state.ratings_list = []

# Helper: robust picker по нормализованному имени


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
        auto_store = (
            pick_col(df_scene, keys=[
                "store","Մասնաճյուղի անվանում","Խանութ","Խանութի անվանում","Shop","Store"
            ]) or (options_cols[1] if len(options_cols) > 1 else options_cols[0])
        )
        default_store_idx = options_cols.index(auto_store)
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

    # === Visits & comments extraction (robust) ===
    def _to_minutes(series: pd.Series) -> pd.Series:
        # datetime -> минуты
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.dt.hour * 60 + series.dt.minute + series.dt.second / 60.0
        # timedelta -> минуты
        if pd.api.types.is_timedelta64_dtype(series):
            return series.dt.total_seconds() / 60.0
        # numeric (Excel time as fraction of day)
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float) * 24.0 * 60.0
        # strings: նորմալիզում ենք հայկական երկկետ "։"
        s = series.astype(str).str.strip()
        s = s.replace({"": np.nan, "None": np.nan})
        s = s.str.replace("\u0589", ":", regex=False)  # Armenian :
        s = s.str.replace(".", ":", regex=False)       # երբեմն կետեր
        # извлечь HH:MM
        ext = s.str.extract(r'^\s*(?P<h>\d{1,2})[:](?P<m>\d{2})(?::(?P<s>\d{2}))?\s*$')
        h = pd.to_numeric(ext["h"], errors="coerce")
        m = pd.to_numeric(ext["m"], errors="coerce")
        sec = pd.to_numeric(ext["s"], errors="coerce").fillna(0)
        return h * 60 + m + sec / 60.0

    visits_rows = []
    comments_rows = []

    for scen_name, base_df in state.bases.items():
        dfb = base_df.copy()

        store_col = pick_col(dfb, keys=["store","Մասնաճյուղի անվանում","Խանութ","Խանութի անվանում","Shop","Store"]) or dfb.columns[0]
        start_col = pick_col(dfb, keys=["Այցելության սկիզբ","Visit start"]) or pick_col(dfb, contains=["սկիզ","start"])
        end_col   = pick_col(dfb, keys=["Այցելության ավարտ","Visit end"])   or pick_col(dfb, contains=["ավարտ","end"])
        dur_col   = pick_col(dfb, keys=["Այցի ընդհանուր տևողություն","Duration"]) or pick_col(dfb, contains=["տևող","dur"])
        date_col  = pick_col(dfb, keys=["Այցելության ամսաթիվ","Visit date"]) or pick_col(dfb, contains=["ամսաթ","date"])  # NEW

        has_time = bool(start_col or end_col or dur_col)
        if has_time:
            start_min = _to_minutes(dfb.get(start_col, pd.Series(index=dfb.index, dtype="float")))
            end_min   = _to_minutes(dfb.get(end_col,   pd.Series(index=dfb.index, dtype="float")))
            dur_min   = _to_minutes(dfb.get(dur_col,   pd.Series(index=dfb.index, dtype="float")))

            # вычисляем из start/end
            calc_min = end_min - start_min
            calc_min = calc_min.mask(calc_min < 0, calc_min + 24*60)  # переход через полночь

            # если длительность не дана — берём из расчёта
            duration_min = dur_min.copy()
            duration_min = duration_min.where(duration_min.notna(), calc_min)

            # Базовая дата визита (если нет — фиктивная)
            visit_date_raw = dfb.get(date_col) if date_col else None
            visit_date = _parse_visit_date(visit_date_raw)  # NEW robust parsing
            base_date = visit_date.where(visit_date.notna(), pd.NaT)

            # если нет даты, не добавляем фиктивную 1900-01-01; оставляем NaT
            if base_date.notna().any():
                visit_start = base_date + pd.to_timedelta(start_min, unit="m")
                visit_end   = base_date + pd.to_timedelta(end_min,   unit="m")
                cross_mid = (end_min.notna() & start_min.notna() & (end_min < start_min))
                visit_end = visit_end.where(~cross_mid, visit_end + pd.Timedelta(days=1))

                tmp = pd.DataFrame({
                    "store": dfb[store_col],  # НЕ превращаем NaN в "nan"
                    "scenario": scen_name if scen_name in ["BR1","BR2","BR3","BR4","BR5"] else "BR1",
                    "visit_start": visit_start,
                    "visit_end": visit_end,
                    "visit_duration_min": duration_min
                })
                tmp = tmp.dropna(subset=["store"])  # убрать строки без магазина
                tmp["hour"] = tmp["visit_start"].dt.hour
                def _tod(h):
                    if pd.isna(h): return "Unknown"
                    h = int(h)
                    if 6 <= h < 12:  return "Morning"
                    if 12 <= h < 18: return "Day"
                    if 18 <= h < 22: return "Evening"
                    return "Night"
                tmp["time_of_day"] = tmp["hour"].map(_tod)
                visits_rows.append(tmp)

        # Открытые ответы
        com1_col = pick_col(dfb, keys=["Մեկնաբանություն"]) or pick_col(dfb, contains=["մեկն"])
        # Несколько вариантов написания вопроса про улучшения
        com2_col = (
            pick_col(dfb, keys=["Ի՞նչ կարելի է անել փորձը բարելավելու համար", "Ի՞նչ կարելի է անել փորձը բարելավելու համար։"])
            or pick_col(dfb, contains=["բարելավ","ինչ"])
        )
        for c_name, c_col in [("Comment", com1_col), ("Improvement", com2_col)]:
            if c_col and c_col in dfb.columns:
                cc = dfb[[store_col, c_col]].copy()
                cc.columns = ["store","text"]
                cc["scenario"] = scen_name if scen_name in ["BR1","BR2","BR3","BR4","BR5"] else "BR1"
                cc["type"] = c_name
                cc = cc.dropna(subset=["text"])
                # убрать Այո/Ոչ и пустяки
                cc["text"] = cc["text"].astype(str).str.strip()
                cc = cc[cc["text"].str.len() >= 5]
                cc = cc[~cc["text"].isin(["Այո","Ոչ","Yes","No"])]
                comments_rows.append(cc)

    visits_df = pd.concat(visits_rows, ignore_index=True) if visits_rows else pd.DataFrame()
    comments_df = pd.concat(comments_rows, ignore_index=True) if comments_rows else pd.DataFrame()

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

tab_overview, tab_stores, tab_scen, tab_sections, tab_questions, tab_compare, tab_export, tab_visits = st.tabs(
    ["Ընդհանուր", "Խանութներ", "Սցենարներ", "Բաժիններ", "Հարցեր", "Համեմատել", "Արտահանում", "Այցելություններ"]  # NEW
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
    # сценарий для сравнения весов
    if not state.weights.empty:
        scen_for_weights = st.selectbox(
            "Սցենար (կշիռների համեմատության համար)",
            options=sorted(state.weights["scenario"].dropna().unique()),
            key="sec_weight_scen"
        )
        w_scen = state.weights[state.weights["scenario"] == scen_for_weights].copy()
        # вычислить ΣF (сумма weight_in_scenario по разделу)
        if not w_scen.empty and "weight_in_scenario" in w_scen.columns:
            calc = (w_scen.groupby("section", as_index=False)
                          .agg(section_weight_calc=("weight_in_scenario","sum"))
                          .assign(section_weight_calc=lambda d: (d["section_weight_calc"]*100).round(1)))
        else:
            calc = pd.DataFrame(columns=["section","section_weight_calc"])
    else:
        w_scen = pd.DataFrame()
        calc = pd.DataFrame(columns=["section","section_weight_calc"])

    if not ps.empty:
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

with tab_visits:
    st.subheader("Այցելությունների վերլուծություն")
    if visits_df.empty and comments_df.empty:
        st.info("Այցելությունների կամ մեկնաբանությունների տվյալներ չկան։")
    else:
        if not visits_df.empty:
            # Фильтры вертикально и без NaN
            v_scen = st.multiselect(
                "Սցենար(ներ)",
                options=sorted(visits_df["scenario"].dropna().unique()),
                default=sorted(visits_df["scenario"].dropna().unique()),
                key="v_scen"
            )
            v_store = st.multiselect(
                "Խանութ(ներ)",
                options=sorted(visits_df["store"].dropna().unique()),
                default=sorted(visits_df["store"].dropna().unique()),
                key="v_store"
            )

            # Отфильтрованные визиты + убрать NAN-строки для таблиц
            vflt = visits_df[
                visits_df["scenario"].isin(v_scen) & visits_df["store"].isin(v_store)
            ].dropna(subset=["store","scenario","visit_duration_min"], how="any")

            # Кол-во визитов и средняя длительность по магазину и сценарию
            agg = (vflt.groupby(["store","scenario"], as_index=False)
                        .agg(visits=("visit_duration_min","count"),
                             avg_duration_min=("visit_duration_min","mean"))
                        .assign(avg_duration_min=lambda d: d["avg_duration_min"].round(1)))
            st.markdown("**Միջին տևողություն և այցելությունների թվաքանակ**")
            st.dataframe(agg.sort_values(["scenario","store"]), use_container_width=True)

            # Кол-во визитов по магазину (все сценарии)
            agg_store = (visits_df[visits_df["store"].isin(v_store)]
                            .dropna(subset=["store"])
                            .groupby("store", as_index=False)
                            .agg(visits=("visit_duration_min","count")))
            st.markdown("**Այցելությունները ըստ խանութների (բոլորև սցենարներով)**")
            st.dataframe(agg_store.sort_values("store"), use_container_width=True)

            # Блок распределения по времени суток
            dist = (vflt.groupby(["scenario","time_of_day"], as_index=False)
                          .size()
                          .rename(columns={"size":"visits"}))
            st.markdown("**Այցելումները ըստ օրվա մասերի**")
            st.dataframe(
                dist.pivot(index="time_of_day", columns="scenario", values="visits").fillna(0).astype(int),
                use_container_width=True
            )

            # Список визитов (формат dd.mm.yyyy hh:mm) и без NaN-строк
            lst = (vflt.dropna(subset=["visit_start","visit_end"])
                        .sort_values(["store","scenario","visit_start"])
                        .loc[:, ["store","scenario","visit_start","visit_end","visit_duration_min","time_of_day"]]
                        .assign(
                            visit_start=lambda d: d["visit_start"].dt.strftime("%d.%m.%Y %H:%M"),
                            visit_end=lambda d: d["visit_end"].dt.strftime("%d.%m.%Y %H:%M"),
                            visit_duration_min=lambda d: d["visit_duration_min"].round(1)
                        ))
            st.markdown("**Այցելումների ժամերն ու տևողությունները**")
            st.dataframe(lst, use_container_width=True)

        st.divider()

        if not comments_df.empty:
            c_scen = st.multiselect(
                "Սցենար(ներ) (մեկնաբանություններ)",
                options=sorted(comments_df["scenario"].dropna().unique()),
                default=sorted(comments_df["scenario"].dropna().unique()),
                key="c_scen"
            )
            c_store = st.multiselect(
                "Խանուտ(ներ) (մեկնաբանություններ)",
                options=sorted(comments_df["store"].dropna().unique()),
                default=sorted(comments_df["store"].dropna().unique()),
                key="c_store"
            )
            cflt = comments_df[
                comments_df["scenario"].isin(c_scen) & comments_df["store"].isin(c_store)
            ].dropna(subset=["store","text"])

            st.markdown("**Այցելողների մեկնաբանությունները**")
            st.dataframe(
                cflt.sort_values(["type","store","scenario"])[["store","scenario","type","text"]],
                use_container_width=True
            )
# ...existing code...

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
