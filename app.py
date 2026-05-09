import streamlit as st
import re
import os
from dotenv import load_dotenv

# Load .env from src/.env
load_dotenv("src/.env")

def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)


import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import gettempdir
import hashlib, os
from src.data_loader import read_excel_all, build_weights_dict, questions_from_weights
from src.scoring import long_from_base, join_weights, aggregate_scores, extract_10pt_rating
import src.scoring as scoring
from src.utils import _parse_visit_date, pick_col, brand_theme, brand_bar_chart, _normalize_store_col
import altair as alt
from src import scoring
# NEW — для аудио/Supabase
import uuid, mimetypes
from datetime import datetime
from src.supabase import _sb_client, _sb_public_url, _sb_upload, _sb_list_all, _sb_delete, replace_excel_file
from src.audio_ui import render_audio_tab

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# --- Credentials loader (supports multiple secret/env names) ---
def _load_creds():
    def _get(key):
        # probuyem secrets, inacze env
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

    # ete hesh@ chi sahmanvats, bayc password_plain@ 64 heqs tzeq@ hanelu — hamarvum enq hesh e
    if not password_hash and password_plain and len(password_plain) == 64 and all(c in "0123456789abcdef" for c in password_plain.lower()):
        password_hash = password_plain
        password_plain = None
    return username, password_plain, password_hash

APP_USERNAME, APP_PASSWORD_PLAIN, APP_PASSWORD_HASH = _load_creds()

def _check_password(inp: str) -> bool:
    # 1) ete ka hesh՝ stugum enq SHA256-ov
    if APP_PASSWORD_HASH:
        return _sha256(inp) == APP_PASSWORD_HASH
    # 2) fallback plain
    if APP_PASSWORD_PLAIN:
        return inp == APP_PASSWORD_PLAIN
    return False

st.set_page_config(page_title="Վաճառքի կետերի գնահատման դաշբորդ", layout="wide")

# --- AUTH ---
# Logout moved to bottom of sidebar


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
# upl = st.sidebar.file_uploader("Բեռնեք Excel ֆայլը", type=["xlsx","xls"])

state = st.session_state
if "bases" not in state:
    state.bases = {}
    state.weights = pd.DataFrame()
    state.last_file = None

alt.themes.register("brand", brand_theme)
alt.themes.enable("brand")

# --- NEW: Supabase Auto-Load ---
DATA_BUCKET = get_secret("SUPABASE_DATA_BUCKET", "data")  # bucket name
sb_client, _ = _sb_client()

# Duplicate reload button removed


with st.sidebar.expander("Վերբեռնել նոր Excel (Supabase)"):
    st.warning("Ուշադրություն. Նոր ֆայլը կավելացվի (չի ջնջի հինը):")
    upl_file = st.file_uploader("Ընտրեք .xlsx/.xls", type=["xlsx", "xls"], key="sb_upl_new")
    if upl_file:
        if st.button("Հաստատել և ուղարկել", type="primary"):
            if not sb_client:
                st.error("Supabase client-ը հասանելի չէ:")
            else:
                with st.spinner("Բեռնում ենք Supabase..."):
                    res = replace_excel_file(sb_client, DATA_BUCKET, upl_file)
                    if res:
                        st.success("Հաջողվեց!")
                        st.cache_data.clear()
                        st.rerun()

@st.cache_data(show_spinner="Загрузка данных из Supabase...", ttl=3600)
def load_remote_excels():
    if not sb_client:
        return []
    # NEW: Load ALL excels
    from src.supabase import get_all_excels
    return get_all_excels(sb_client, DATA_BUCKET)

remote_files = load_remote_excels()
selected_paths = []

if remote_files:
    st.sidebar.success(f"Supabase: գտնվել է {len(remote_files)} ֆայլ")
    for content, name in remote_files:
        tmp_path = IMPORTS_DIR / name
        tmp_path.write_bytes(content)
        selected_paths.append(tmp_path)
else:
    # Fallback: check local cache
    existing = _list_saved_excels()
    if existing:
        selected_paths = existing
        st.sidebar.warning(f"Supabase-ում ֆայլ չկա, օգտագործվում են: {[f.name for f in existing]}")
    else:
        st.sidebar.error(f"Supabase: ֆայլ չի գտնվել '{DATA_BUCKET}' բաժնում")

# Parse ALL files and merge
all_bases = {}
all_weights = pd.DataFrame()
loaded_files_sig = str([str(p) for p in selected_paths])

if selected_paths and state.get("last_files_sig") != loaded_files_sig:
    merged_bases = {}
    merged_weights_list = []
    
    stage_to_file = {}
    
    for p in selected_paths:
        try:
            b, w = parse_workbooks(p)
            
            for k, v in b.items():
                s = v.get("_stage_", "Retail")
                if isinstance(s, pd.Series): s = s.iloc[0]
                if k in ["BR1", "BR2", "BR3", "BR4", "BR5"]: s = "Retail"
                elif k in ["LAS_SA", "LAS_SFP", "LAS_SFP_CC", "HS_SA_YAP"]: s = "Stage2"
                elif k == "Stage3": s = "Stage3"
                
                # If we haven't seen this stage yet, record this file as the owner
                if s not in stage_to_file:
                    stage_to_file[s] = p
                
                # Only keep data if it comes from the owning (newest) file
                if stage_to_file[s] == p:
                    if k in merged_bases:
                        merged_bases[k] = pd.concat([merged_bases[k], v], ignore_index=True)
                    else:
                        merged_bases[k] = v
            
            # Merge weights
            w_df = build_weights_dict(w)
            if not w_df.empty:
                merged_weights_list.append(w_df)
                
        except Exception as e:
            st.error(f"Error parsing {p.name}: {e}")

    state.bases = merged_bases
    state.weights = pd.concat(merged_weights_list, ignore_index=True) if merged_weights_list else pd.DataFrame()
    # Deduplicate weights if needed
    if not state.weights.empty:
        # Normalize scenario names in weights too
        state.weights["scenario"] = state.weights["scenario"].astype(str).str.strip()
        state.weights = state.weights.drop_duplicates(subset=["scenario","qkey_norm"], keep="last")
        
    state.last_files_sig = loaded_files_sig
    
    # Отладочная информация (показываем какие этапы найдены в каких файлах)
    with st.sidebar.expander("Отладка загрузки файлов", expanded=False):
        for s, p in stage_to_file.items():
            st.write(f"**{s}**: {p.name}")

# На каждом прогоне пересобираем рейтинги заново (без накопления в session)
state.ratings_list = []

if not state.bases:
    st.info("Բեռնեք Excel ֆայլը")
    st.stop()

# --- NAVIGATION ---
# --- NAVIGATION ---
st.sidebar.markdown("---")

# Reload button moved here for better context
if st.sidebar.button("🔄 Թարմացնել տվյալները (Reload)"):
    st.cache_data.clear()
    for key in list(state.keys()):
        del state[key]
    st.rerun()

nav_mode = st.sidebar.radio("Բաժին", ["Փուլ 1 | Retail", "Փուլ 2 | SA & HS", "Փուլ 3", "Audio"])

# Logout at the bottom
st.sidebar.markdown("---")
if st.sidebar.button("Logout", disabled=not st.session_state.get("auth_ok")):
    for k in ("auth_ok","auth_user"):
        st.session_state.pop(k, None)
    st.rerun()

if nav_mode == "Audio":
    # Pass required arguments: sb_client and bucket name
    # We use a default bucket "audio" if not specified, but let's check what src/supabase.py uses or what secrets have.
    # In src/supabase.py: bucket = st.secrets.get("SUPABASE_BUCKET", "audio")
    # In app.py: DATA_BUCKET = st.secrets.get("SUPABASE_DATA_BUCKET", "data") -> this is for Excel files!
    # Audio usually uses a different bucket.
    # Let's check how it was used before.
    # It seems we need the audio bucket.
    AUDIO_BUCKET = get_secret("SUPABASE_BUCKET", "audio")
    render_audio_tab(sb_client, AUDIO_BUCKET)
    st.stop()

# Filter bases by stage
if nav_mode == "Փուլ 1 | Retail":
    target_stage = "Retail"
elif nav_mode == "Փուլ 2 | SA & HS":
    target_stage = "Stage2"
elif nav_mode == "Փուլ 3":
    target_stage = "Stage3"
else:
    target_stage = "Unknown"
filtered_bases = {}

for scen, df in state.bases.items():
    # Check _stage_ flag (added in data_loader)
    # If missing, assume Retail (legacy)
    # If missing, assume Retail (legacy)
    stage = df.get("_stage_", "Retail")
    # Handle Series/DataFrame ambiguity if _stage_ is a column
    if isinstance(stage, pd.Series):
        stage = stage.iloc[0] if not stage.empty else "Retail"
    
    # ROBUST FILTERING: Override stage based on known scenario names
    # This prevents "Retail" scenarios (BR1-5) from appearing in Stage 2 even if _stage_ tag is wrong
    if scen in ["BR1", "BR2", "BR3", "BR4", "BR5"]:
        actual_stage = "Retail"
    elif scen in ["LAS_SA", "LAS_SFP", "LAS_SFP_CC", "HS_SA_YAP"]:
        actual_stage = "Stage2"
    else:
        actual_stage = stage # Fallback to the tag

    if actual_stage == target_stage:
        filtered_bases[scen] = df

if not filtered_bases:
    st.info(f"Տվյալներ չկան {nav_mode}-ի համար")
    # Debug info
    with st.expander("Debug Info (Data Loading)", expanded=False):
        st.write(f"Target Stage: {target_stage}")
        st.write("All Loaded Bases:")
        for k, v in state.bases.items():
            s = v.get("_stage_", "Missing")
            if isinstance(s, pd.Series): s = s.iloc[0]
            st.write(f"- {k}: {s} (Rows: {len(v)})")
        
        # Specific check for HS_SA_YAP
        if "HS_SA_YAP" in state.bases:
            st.write("HS_SA_YAP found in bases.")
        else:
            st.write("HS_SA_YAP NOT found in bases.")
            
        if not state.weights.empty and "HS_SA_YAP" in state.weights["scenario"].unique():
             st.write("HS_SA_YAP found in weights.")
        else:
             st.write("HS_SA_YAP NOT found in weights.")
    st.stop()

# Use filtered bases for the rest of the app
active_bases = filtered_bases

# Debug: Show what is active
# with st.expander("Debug: Active Data", expanded=False):
#     st.write(f"Active Scenarios: {list(active_bases.keys())}")
#     if not state.weights.empty:
#         st.write(f"Loaded Weight Scenarios: {state.weights['scenario'].unique().tolist()}")
# We need to temporarily swap state.bases to active_bases for the logic below to work without changing everything
# But state.bases is used in 'prepared' loop. 
# Let's just use active_bases in the loop.

# ===== end persistence block =====

st.sidebar.markdown("---")

prepared = []
state.ratings_list = []

with st.container():  # скрытый պատրաստողական բլոկ առանց UI
    for scen, base in active_bases.items():
        # comment: normalize scenario name
        scen = scen if scen in ["BR1", "BR2", "BR3", "BR4", "BR5"] else scen # Allow new scenarios

        # comment: filter base by scenario
        # Only apply scenario column filtering for Retail (BR) scenarios
        if scen in ["BR1", "BR2", "BR3", "BR4", "BR5"]:
             df_scene = scoring.filter_by_scenario_column(base, scen)
        else:
             # For Stage 2, the sheet IS the scenario data
             df_scene = base.copy()

        # NEW: detect employee and stage columns
        emp_col_orig = pick_col(df_scene, keys=["SE Անուն/ազգանուն", "Employee", "Worker", "Աշխատակցի անունը՝", "Աշխատակցի անունը"])
        # Stage column is removed from UI but we might still parse it if present
        stage_col_orig = pick_col(df_scene, keys=["Փուլ", "Stage", "Phase"])
        
        if emp_col_orig:
            df_scene = df_scene.rename(columns={emp_col_orig: "employee"})
        if stage_col_orig:
            df_scene = df_scene.rename(columns={stage_col_orig: "stage"})

        # comment: small header (no huge H2)
        # st.markdown(f"<h4>{scen} սցենարի մշակումը</h4>", unsafe_allow_html=True)

        # comment: get question keys from weights
        qkeys = questions_from_weights(state.weights, scen)
        if not qkeys:
            # For Stage 2, we might not have weights loaded correctly if names don't match
            # But we added logic in data_loader to map them.
            st.warning(f"Կշիռների թերթում {scen} սցենարի հարցեր չեն հայտնաբերվել։")
            continue

        # comment: choose store column & skip columns
        options_cols = list(df_scene.columns)
        # авто-պարզել խանութի սյունակը (без UI)
        store_col = (
            pick_col(df_scene, keys=["store","Մասնաճյուղի անվանում","Խանութ","Խանութի անվանում","Shop","Store", "Մասնաճյուղի՝ խանութի անվանում"])
            or (options_cols[1] if len(options_cols) > 1 else options_cols[0])
        )

        # ՆՈՐՄԱԼԻԶԱՑԻԱ ՆԱԶՎԱՆԻՅ ՄԱԳԱԶԻՆՈՎ (чтобы 'Խանութ  A' == 'Խանութ A')
        if store_col in df_scene.columns:
            df_scene[store_col] = _normalize_store_col(df_scene[store_col])

        # список пропускаемых колонок — по умолчанию (без UI)
        # For Stage 2, questions start at col E (index 4), so we should only skip A-D (indices 0-3)
        if nav_mode == "Փուլ 2 | SA & HS":
             skip_cols = [c for c in options_cols[:4] if c != store_col]
        elif nav_mode == "Փուլ 3":
             skip_cols = [c for c in options_cols[:7] if c != store_col]
        else:
             # For Stage 1, keep original behavior (skip 8)
             skip_cols = [c for c in options_cols[:8] if c != store_col]

        # comment: prepare long form for scoring
        drop_list = [c for c in skip_cols if c in df_scene.columns and c != store_col]
        # Ensure we don't drop the new columns if they were caught in skip_cols
        if "employee" in drop_list: drop_list.remove("employee")
        if "stage" in drop_list: drop_list.remove("stage")

        df_for_long = df_scene.drop(columns=drop_list, errors="ignore")

        extra_ids = []
        if "employee" in df_for_long.columns: extra_ids.append("employee")
        if "stage" in df_for_long.columns: extra_ids.append("stage")

        long_df = long_from_base(
            df_for_long,
            store_col=store_col,
            non_question_cols=skip_cols,
            scenario=scen,
            qkeys_norm=qkeys,
            extra_id_cols=extra_ids
        )
        
        if nav_mode == "Փուլ 3" and scen == "Stage3":
            try:
                from src.weights_stage3 import apply_stage3_rules
                long_df = apply_stage3_rules(long_df)
            except Exception as e:
                st.error(f"Error applying Stage 3 rules: {e}")
        
        # Fallback: If long_df is empty (no questions matched) and it's HS_SA_YAP, try fuzzy matching
        if long_df.empty and scen == "HS_SA_YAP" and qkeys:
            st.toast(f"Attempting fuzzy match for {scen}...", icon="⚠️")
            # Create a map of norm(col) -> col
            col_map = {scoring.norm(c): c for c in df_for_long.columns}
            # Try to find matches for qkeys
            fuzzy_qkeys = []
            for qk in qkeys:
                # 1. Try contains
                for nc, c in col_map.items():
                    if qk in nc or nc in qk:
                        fuzzy_qkeys.append(nc)
            
            if fuzzy_qkeys:
                long_df = long_from_base(
                    df_for_long,
                    store_col=store_col,
                    non_question_cols=skip_cols,
                    scenario=scen,
                    qkeys_norm=fuzzy_qkeys,
                    extra_id_cols=extra_ids
                )
                st.toast(f"Fuzzy match found {len(fuzzy_qkeys)} columns for {scen}", icon="✅")

        # comment: collect ratings per scenario
        ratings_scene = extract_10pt_rating(df_scene, scen, store_col)
        state.ratings_list.append(ratings_scene)

        # comment: join weights and accumulate
        merged = join_weights(long_df, state.weights, scen)
        prepared.append(merged)

    # --- 2) Build df_all and ratings, validate ---
    df_all = pd.concat(prepared, ignore_index=True) if prepared else pd.DataFrame()
    ratings = pd.concat(state.ratings_list, ignore_index=True) if state.ratings_list else pd.DataFrame()

    # финальная нормализация на всякий случай (если что-то проскочило)
    if not df_all.empty and "store" in df_all.columns:
        df_all["store"] = _normalize_store_col(df_all["store"])
        # Fix: Remove nan/empty stores
        df_all = df_all.dropna(subset=["store"])
        df_all = df_all[df_all["store"].astype(str).str.strip() != "nan"]
        df_all = df_all[df_all["store"].astype(str).str.strip() != ""]

    # STRICT STAGE FILTERING (Final Guard)
    if not df_all.empty:
        # Normalize scenario names to ensure exact matching
        df_all["scenario"] = df_all["scenario"].astype(str).str.strip()
        
        if nav_mode == "Փուլ 1 | Retail":
            # Keep only BR1-BR5
            df_all = df_all[df_all["scenario"].isin(["BR1", "BR2", "BR3", "BR4", "BR5"])]
        elif nav_mode == "Փուլ 2 | SA & HS":
            # Keep only Stage 2 scenarios
            # Exclude BR1-BR5 explicitly using regex to be safe
            # Matches ANY scenario starting with BR (case insensitive)
            mask_br = df_all["scenario"].str.contains(r'^BR.*', case=False, regex=True)
            df_all = df_all[(~mask_br) & (df_all["scenario"] != "Stage3")]
        elif nav_mode == "Փուլ 3":
            df_all = df_all[df_all["scenario"] == "Stage3"]

    if df_all.empty:
        st.error("Հնարավոր չեղավ պատրաստել տվյալները (կամ սխալ փուլ)։ Ստուգեք ներբեռված ֆայլը և կշիռների թերթերը։")
        # Debug info for empty result
        with st.expander("Debug Info (Empty Result)", expanded=False):
            st.write(f"Prepared count: {len(prepared)}")
            st.write(f"Active Bases: {list(active_bases.keys())}")
        st.stop()

    # Debug: Check if HS_SA_YAP is in df_all
    if nav_mode == "Փուլ 2 | SA & HS":
        hs_rows = df_all[df_all["scenario"] == "HS_SA_YAP"]
        if hs_rows.empty:
             # It's missing! Check why.
             if "HS_SA_YAP" in active_bases:
                 st.error("HS_SA_YAP data was loaded but dropped during processing!")
                 # Re-run logic to show why
                 base = active_bases["HS_SA_YAP"]
                 qkeys = questions_from_weights(state.weights, "HS_SA_YAP")
                 st.write(f"Weight QKeys count: {len(qkeys)}")
                 
                 # Check intersection
                 norm_cols = [scoring.norm(c) for c in base.columns]
                 intersection = set(norm_cols) & set(qkeys)
                 st.write(f"Matching Columns count: {len(intersection)}")
                 if not intersection:
                     st.write("No columns matched! Check column names.")
                     st.write(f"Data Columns (Norm): {norm_cols[:10]}")
                     st.write(f"Weight QKeys (Norm): {qkeys[:10]}")
             else:
                 st.warning("HS_SA_YAP not found in active_bases.")
    


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

    for scen_name, base_df in active_bases.items():
        dfb = base_df.copy()
        store_col = pick_col(dfb, keys=["store","Մասնաճյուղի անվանում","Խանութ","Խանութի անվանում","Shop","Store", "Մասնաճյուղի՝ խանութի անվանում"]) or dfb.columns[0]
        if store_col in dfb.columns:
            dfb[store_col] = _normalize_store_col(dfb[store_col])
        start_col = pick_col(dfb, keys=["Այցելության սկիզբ","Visit start"]) or pick_col(dfb, contains=["սկիզ","start"])
        end_col   = pick_col(dfb, keys=["Այցելության ավարտ","Visit end"])   or pick_col(dfb, contains=["ավարտ","end"])
        dur_col   = pick_col(dfb, keys=["Այցի ընդհանուր տևողություն","Duration"]) or pick_col(dfb, contains=["տևող","dur"])
        date_col  = pick_col(dfb, keys=["Այցելության ամսաթիվ","Visit date", "Այցելության օր և ժամ"]) or pick_col(dfb, contains=["ամսաթ","date"])

        # Նորմալիզացված ամսաթվի Series (եթե չկա, բոլոր արժեքները NaT)
        if date_col and date_col in dfb.columns:
            base_date = _parse_visit_date(dfb[date_col])
        else:
            base_date = pd.Series(pd.NaT, index=dfb.index)

        has_time = bool(start_col or end_col or dur_col)
        if has_time:
            start_min = _to_minutes(dfb.get(start_col, pd.Series(index=dfb.index, dtype="float")))
            end_min   = _to_minutes(dfb.get(end_col,   pd.Series(index=dfb.index, dtype="float")))
            dur_min   = _to_minutes(dfb.get(dur_col,   pd.Series(index=dfb.index, dtype="float")))

            # Вычисляем длительность из start/end
            calc_min = end_min - start_min
            calc_min = calc_min.mask(calc_min < 0, calc_min + 24*60)  # переход через полночь

            # Итоговая длительность: եթե կա առանձին dur_min — վերցնում ենք այն, այլապես calc_min
            visit_duration_min = dur_min.where(dur_min.notna(), calc_min)

            # Формируем временные метки միայն եթե կա ոչ դատարկ ամսաթիվ
            if base_date.notna().any():
                visit_start = base_date + pd.to_timedelta(start_min, unit="m")
                visit_end   = base_date + pd.to_timedelta(end_min,   unit="m")
                cross_mid = (end_min.notna() & start_min.notna() & (end_min < start_min))
                visit_end = visit_end.where(~cross_mid, visit_end + pd.Timedelta(days=1))
            else:
                visit_start = pd.Series(pd.NaT, index=dfb.index)
                visit_end   = pd.Series(pd.NaT, index=dfb.index)

            tmp = pd.DataFrame({
                "store": dfb[store_col],
                "scenario": scen_name if scen_name in ["BR1","BR2","BR3","BR4","BR5"] else scen_name,
                "visit_start": visit_start,
                "visit_end": visit_end,
                "visit_duration_min": visit_duration_min
            })
            tmp = tmp.dropna(subset=["store"])  # убрать строки առանց խանութի
            if tmp["visit_start"].notna().any():
                tmp["hour"] = tmp["visit_start"].dt.hour
                def _tod(h):
                    if pd.isna(h): return "Unknown"
                    h = int(h)
                    if 6 <= h < 12:  return "Morning"
                    if 12 <= h < 18: return "Day"
                    if 18 <= h < 22: return "Evening"
                    return "Night"
                tmp["time_of_day"] = tmp["hour"].map(_tod)
            else:
                tmp["time_of_day"] = "Unknown"
            visits_rows.append(tmp)

        # Открытые ответы
        com1_col = pick_col(dfb, keys=["Մեկնաբանություն"]) or pick_col(dfb, contains=["մեկն"])
        # Несколько տարբերակների գրություն հարցի բարելավումների մասին
        com2_col = (
            pick_col(dfb, keys=["Ի՞նչ կարելի է անել փորձը բարելավելու համար", "Ի՞նչ կարելի է անել փորձը բարելավելու համար։"])
            or pick_col(dfb, contains=["բարելավ","ինչ"])
        )
        com3_col = pick_col(dfb, keys=["MS-ի ընդհանուր տպավորություններ"]) or pick_col(dfb, contains=["տպավորություն"])
        
        for c_name, c_col in [("Comment", com1_col), ("Improvement", com2_col), ("MS Impression", com3_col)]:
            if c_col and c_col in dfb.columns:
                cc = dfb[[store_col, c_col]].copy()
                cc.columns = ["store","text"]
                cc["scenario"] = scen_name if scen_name in ["BR1","BR2","BR3","BR4","BR5"] else scen_name
                cc["type"] = c_name
                cc = cc.dropna(subset=["text"])
                # убрать Այո/Ոչ и пустяки
                cc["text"] = cc["text"].astype(str).str.strip()
                cc = cc[cc["text"].str.len() >= 5]
                cc = cc[~cc["text"].isin(["Այո","Ոչ","Yes","No"])]
                comments_rows.append(cc)

    visits_df = pd.concat(visits_rows, ignore_index=True) if visits_rows else pd.DataFrame()
    comments_df = pd.concat(comments_rows, ignore_index=True) if comments_rows else pd.DataFrame()

    # Assign Role for Stage 2
    if nav_mode == "Փուլ 2 | SA & HS":
        def _get_role(scen):
            scen = str(scen).upper()
            if "SA" in scen and "HS" not in scen: return "SA" # LAS_SA
            if "SFP" in scen: return "HS" # LAS_SFP, LAS_SFP_CC
            if "HS" in scen: return "HS" # HS_SA_YAP
            return "Other"
        df_all["role"] = df_all["scenario"].apply(_get_role)
        
        # Debug: Check role for HS_SA_YAP
        if "HS_SA_YAP" in df_all["scenario"].unique():
             role_val = df_all[df_all["scenario"]=="HS_SA_YAP"]["role"].iloc[0]
             # st.toast(f"HS_SA_YAP Role: {role_val}", icon="🔍")

# --- 3) Filters AFTER df_all exists ---
with st.expander("Ֆիլտրեր", expanded=False):
    st.header("Ֆիլտրեր")
    
    # Filter options based on CURRENT df_all (which is already filtered by stage)
    stores = sorted(df_all["store"].dropna().unique().tolist())
    scenarios = sorted(df_all["scenario"].dropna().unique().tolist())
    sections = sorted(df_all["section"].dropna().unique().tolist())
    
    # Role filter for Stage 2
    sel_role = []
    if nav_mode == "Փուլ 2 | SA & HS" and "role" in df_all.columns:
        roles = sorted(df_all["role"].dropna().unique().tolist())
        sel_role = st.multiselect("Դեր", options=roles, default=roles)

    # Filter scenarios based on selected role
    if sel_role:
        role_scenarios = df_all[df_all["role"].isin(sel_role)]["scenario"].unique().tolist()
        scenarios = sorted(list(set(scenarios) & set(role_scenarios)))

    sel_scen = st.multiselect("Սցենարներ", options=scenarios, default=scenarios)
    
    # Filter stores based on selected scenarios (optional, but good for context)
    # stores_in_scen = df_all[df_all["scenario"].isin(sel_scen)]["store"].unique().tolist()
    # stores = sorted(list(set(stores) & set(stores_in_scen)))
    
    sel_stores = st.multiselect("Խանութներ", options=stores, default=stores)
    sel_sec = st.multiselect("Բաժիններ", options=sections, default=sections)

    employees = sorted(df_all["employee"].dropna().unique().tolist()) if "employee" in df_all.columns else []
    # Removed Stage filter

    sel_emp = st.multiselect("Աշխատակիցներ", options=employees, default=[]) if employees else []

# --- 4) Apply filters once selections are made ---
mask = (
    df_all["store"].isin(sel_stores)
    & df_all["scenario"].isin(sel_scen)
    & df_all["section"].isin(sel_sec)
)
if employees and sel_emp:
    mask &= df_all["employee"].isin(sel_emp)
if nav_mode == "Stage 2" and sel_role:
    mask &= df_all["role"].isin(sel_role)

flt = df_all[mask]

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

# --- weights per question (merge from raw) ---
if not _raw.empty and not pq.empty:
    qw = (
        _raw.groupby(["store","scenario","section","question_key"], as_index=False)
            .agg(
                weight_in_section=("weight_in_section","first"),
                weight_in_scenario=("weight_in_scenario","first"),
                # NEW: нужен для Այո/Ոչ и чтобы не было KeyError
                answer_bin=("answer_bin","first")
            )
    )
    pq = pq.merge(qw, on=["store","scenario","section","question_key"], how="left")

    # нормализация к долям и проценты для показа
    for c in ["weight_in_section","weight_in_scenario"]:
        if c in pq.columns:
            pq[c] = pd.to_numeric(pq[c], errors="coerce")

    # показываем вес вопроса в сценарии (если его нет — берём вес в секции)
    w_frac = pq["weight_in_scenario"].fillna(pq["weight_in_section"])
    pq["question_weight_pct"] = (w_frac * 100).round(2)

    # вклад вопроса в итог сценария
    pq["weighted_score_pct"] = (pq["score_question_pct"] * w_frac).round(2)

tab_overview, tab_stores, tab_scen, tab_sections, tab_compare, tab_visits = st.tabs(
    ["Ընդհանուր", "Խանութներ", "Սցենարներ", "Բաժիններ", "Համեմատել", "Այցելություններ"]
)

with tab_overview:
    total_scores = scoring.total_score_table(flt)

    if not total_scores.empty:
        # Общая статистика ՍՎԵՐՀՈՒ
        with st.expander("Ընդհանուր վիճակագրություն", expanded=True):
            stores_cnt = int(total_scores["store"].nunique())
            scen_cnt = int(flt["scenario"].nunique())
            avg_pct = float(total_scores["total_score_pct"].mean() * 100)
            med_pct = float(total_scores["total_score_pct"].median() * 100)
            best_row = total_scores.iloc[total_scores["total_score_pct"].idxmax()]
            worst_row = total_scores.iloc[total_scores["total_score_pct"].idxmin()]

            # 1-я строка метрик
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("Խանութներ", stores_cnt)
            r1c2.metric("Միջին %", f"{avg_pct:.1f}%")
            r1c3.metric("Մեդիան %", f"{med_pct:.1f}%")
            r1c4.metric("Սցենարներ", scen_cnt)

            # 2-я строка (длинные названия помещаются лучше)
            r2c1, r2c2 = st.columns(2)
            r2c1.metric("Լավագույն խանութ",
                        f"{best_row['store']} — {best_row['total_score_pct']*100:.1f}%")
            r2c2.metric("Վատագույն խանութ",
                        f"{worst_row['store']} — {worst_row['total_score_pct']*100:.1f}%")

            scen_avgs = scoring.scenario_score_table(flt)
            if not scen_avgs.empty:
                scen_mean = (
                    scen_avgs.groupby("scenario", as_index=False)
                             .agg(avg_pct=("scenario_score_pct", "mean"))
                             .assign(avg_pct=lambda d: d["avg_pct"].round(1))
                             .rename(columns={"scenario": "Սցենար", "avg_pct": "Միջին %"})
                             .sort_values("Սցենար")
                )
                st.markdown("**Միջին արդյունքը ըստ սցենարի (%)**")
                # Вертикальные отдельные столбцы
                ch = alt.Chart(scen_mean).mark_bar().encode(
                    x=alt.X("Սցենար:N", sort=None, title="Սցենար"),
                    y=alt.Y("Միջին %:Q", title="Միջին %", scale=alt.Scale(domain=[0, 100])),
                    tooltip=["Սցենար", alt.Tooltip("Միջին %:Q", format=".1f")]
                ).properties(height=320)
                txt = ch.mark_text(baseline="bottom", dy=-4).encode(
                    text=alt.Text("Միջին %:Q", format=".1f")
                )
                st.altair_chart(ch + txt, use_container_width=True)

        # Далее общий рейтинг
        st.markdown("#### Խանութների ընդհանուր վարկանիշ")
        st.caption(scoring.total_score_caption_hy())
        scores_df = total_scores.rename(columns={"total_score_pct_display": "score"})[["store", "score"]]
        ui.rating_table(scores_df, "score", "Ընդհանուր ")
    else:
        st.info("Տվյալներ չկան ընդհանուր վարկանիշի համար։")

    st.divider()

    # # --- 2) Heatmap second ---
    # st.markdown("#### Բաժինների ջերմաքարտեզ")
    # ui.heatmap_sections(ps)

    # st.divider()

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

        st.markdown("#### Խանութների տպավորությունը (1–10)")
        st.dataframe(avg, use_container_width=True)
    else:
        st.info("Տվյալներ չկան ցուցադրելու համար")


with tab_stores:
    st.subheader("Խանութի պրոֆիլ")
    st.caption(scoring.caption_store_profile_hy())

    # Только выбор магазина сначала
    store_options = sorted(df_all["store"].dropna().unique()) if "store" in df_all.columns else []
    if not store_options:
        st.info("Խանութներ չկան ցուցադրելու համար։")
    else:
        sel_store = st.selectbox("Ընտրեք խանութ", options=store_options, key="store_profile_store")

        # Сценарные проценты для выбранного магазина (все сценарии сразу)
        scen_scores_full = scoring.scenario_score_table(df_all)
        scen_scores_store = scen_scores_full[scen_scores_full["store"] == sel_store].copy()
        scen_scores_store = scen_scores_store.rename(columns={"scenario":"Սցենար", "scenario_score_pct":"Արդյունք %"})

        st.markdown("**Սցենարների արդյունքները (% հնարավոր առավելագույնից)**")
        if scen_scores_store.empty:
            st.warning("Տվյալներ չկան ընտրված խանութի համար։")
        else:
            height = max(160, 26 * len(scen_scores_store))
            ch1 = alt.Chart(scen_scores_store).mark_bar(size=22).encode(
                y=alt.Y("Սցենար:N", sort='-x', title="Սցենար"),
                x=alt.X("Արդյունք %:Q", title="Արդյունք %", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Սցենար", alt.Tooltip("Արդյունք %:Q", format=".0f")]
            ).properties(height=height)
            tx1 = ch1.mark_text(align="left", dx=4).encode(text=alt.Text("Արդյունք %:Q", format=".0f"))
            st.altair_chart(ch1 + tx1, use_container_width=True)
            st.dataframe(scen_scores_store.sort_values("Սցենար"), use_container_width=True)

        st.divider()

        # Теперь выбор одного сценария для детализации
        scen_options_store = sorted(scen_scores_store["Սցենար"].unique()) if not scen_scores_store.empty else []
        if not scen_options_store:
            st.info("Սցենարներ չկան տվյալ խանութի համար։")
        else:
            sel_scen = st.selectbox("Ընտրեք սցենար մանրամասն տեսնելու համար", options=scen_options_store, key="store_profile_scen")

            prof = scoring.store_profile_breakdown(df_all, sel_store, sel_scen)
            if not prof.empty:
                # применить порядок
                prof = scoring.apply_section_order(prof, "section")
                sec_summary = (
                    prof.groupby("section", as_index=False)
                        .agg(section_score_pct=("section_score_pct","first"))
                        .rename(columns={"section":"Բաժին", "section_score_pct":"Բաժնի արդյունք %"})
                )

                # Сортировка по категории
                sec_summary["Բաժին"] = pd.Categorical(
                    sec_summary["Բաժին"],
                    categories=scoring.SECTION_ORDER,
                    ordered=True
                )
                sec_summary = sec_summary.sort_values("Բաժին")

                st.markdown(f"**Բաժինների արդյունքները ({sel_scen})**")
                height2 = max(160, 26 * len(sec_summary))
                ch2 = alt.Chart(sec_summary).mark_bar(size=22).encode(
                    y=alt.Y("Բաժին:N", sort='-x', title="Բաժին"),
                    x=alt.X("Բաժնի արդյունք %:Q", title="Արդյունք %", scale=alt.Scale(domain=[0, 100])),
                    tooltip=["Բաժին", alt.Tooltip("Բաժնի արդյունք %:Q", format=".0f")]
                ).properties(height=height2)
                tx2 = ch2.mark_text(align="left", dx=4).encode(text=alt.Text("Բաժնի արդյունք %:Q", format=".0f"))
                st.altair_chart(ch2 + tx2, use_container_width=True)
                st.dataframe(sec_summary, use_container_width=True)

                st.markdown(f"**Ըստ հարցերի ({sel_scen})**")
                q_cols = prof[["section","question_key","answer","weight_share_pct","earned_pct"]].rename(columns={
                    "section":"Բաժին",
                    "question_key":"Հարց",
                    "answer":"Պատասխան",
                    "weight_share_pct":"Քաշի բաժինը %",
                    "earned_pct":"Ստացված %"
                }).sort_values(["Բաժին","Հարց"])
                st.dataframe(q_cols, use_container_width=True)

                scen_pct = prof["scenario_score_pct"].iloc[0]
                col_m, col_p = st.columns([1,3])
                with col_m:
                    st.metric(label=f"Ընդհանուր արդյունքը ({sel_scen})", value=f"{scen_pct:.2f}%")
                with col_p:
                    st.progress(min(max(float(scen_pct)/100.0, 0.0), 1.0))

with tab_scen:
    st.subheader("Ռեյտինգ ըստ սցենարների")
    st.caption(scoring.caption_scenario_page_hy())

    # Используем отфильтрованные глобально данные flt
    scen_list = sorted(flt["scenario"].dropna().unique()) if "scenario" in flt.columns else []
    if not scen_list:
        st.info("Սցենարներ չկան։")
    else:
        sel_scen = st.selectbox("Սցենար", options=scen_list, key="scen_main")

        # Возможные разделы для выбранного сценария
        scen_df = flt[flt["scenario"] == sel_scen].copy()
        scen_df["w"] = pd.to_numeric(scen_df.get("weight_in_scenario"), errors="coerce").fillna(0.0)
        scen_df = scen_df[scen_df["w"] > 0]
        if "question_key" in scen_df.columns:
            scen_df = scen_df[~scoring._opinion_mask_from_key(scen_df["question_key"])]

        section_opts = sorted(scen_df["section"].dropna().unique()) if "section" in scen_df.columns else []
        # Порядок разделов (опционально)
        section_opts = sorted(
            [s for s in section_opts if s in scoring.SECTION_ORDER],
            key=lambda x: scoring.SECTION_ORDER.index(x)
        ) + [s for s in section_opts if s not in scoring.SECTION_ORDER]
        section_opts_ui = ["Բոլոր բաժինները"] + section_opts
        sel_section = st.selectbox("Բաժին", options=section_opts_ui, key="scen_section")

        # Вопросы доступны только если выбран конкретный раздел
        if sel_section != "Բոլոր բաժինները":
            qdf = scen_df[scen_df["section"] == sel_section]
            question_opts = sorted(qdf["question_key"].dropna().unique()) if "question_key" in qdf.columns else []
            question_opts_ui = ["Բոլոր հարցերը"] + question_opts
            sel_question = st.selectbox("Հարց", options=question_opts_ui, key="scen_question")
        else:
            sel_question = "Բոլոր հարցերը"

        # Расчет
        if sel_question != "Բոլոր հարցերը":
            scores_subset = scoring.scenario_subset_scores(flt, sel_scen, question=sel_question)
            label = f"Հարցի արդյունք % ({sel_question})"
        elif sel_section != "Բոլոր բաժինները":
            scores_subset = scoring.scenario_subset_scores(flt, sel_scen, section=sel_section)
            label = f"Բաժնի արդյունք % ({sel_section})"
        else:
            scores_subset = scoring.scenario_subset_scores(flt, sel_scen)
            label = "Սցենարի ընդհանուր %"

        if scores_subset.empty:
            st.warning("Տվյալներ չկան ընտրված ֆիլտրի համար։")
        else:
            # Таблица рейтинга
            tbl = scores_subset.rename(columns={"store":"store","value_pct":label}).sort_values(label, ascending=False)
            ui.rating_table(tbl.rename(columns={label:"score"}), "score", label)

            # Гистограмма
            ch = alt.Chart(tbl).mark_bar().encode(
                x=alt.X("store:N", sort="-y", title="Խանութ"),
                y=alt.Y(f"{label}:Q", title="%", scale=alt.Scale(domain=[0,100])),
                tooltip=["store", alt.Tooltip(label, format=".1f")]
            ).properties(height=360)
            tx = ch.mark_text(baseline="bottom", dy=-4).encode(text=alt.Text(label, format=".0f"))
            st.altair_chart(ch + tx, use_container_width=True)

            # Детализация (если выбран раздел или вопрос)
            if sel_question != "Բոլոր հարցերը" or sel_section != "Բոլոր բաժինները":
                st.markdown("**Մանրամասներ ըստ խանութի (աստիճանավորված ըստ процենտա)**")
                st.dataframe(tbl, use_container_width=True)

with tab_sections:
    st.subheader("Բաժինների արդյունքներ")
    st.caption(scoring.caption_sections_page_hy())

    # === Фильтры ===
    all_weight_scens = sorted(state.weights["scenario"].dropna().unique()) if not state.weights.empty else []
    
    # Strict filtering for Sections tab based on nav_mode
    if nav_mode == "Փուլ 1 | Retail":
        # Keep only BR1-BR5 (exact match or regex if needed)
        scen_opts = [s for s in all_weight_scens if s in ["BR1", "BR2", "BR3", "BR4", "BR5"]]
    else:
        # Stage 2: Exclude BR1-BR5 using regex
        # Matches ANY scenario starting with BR (case insensitive)
        scen_opts = [s for s in all_weight_scens if not re.match(r'^BR.*', str(s), re.I)]
        
    # Ensure options are present in current data (df_all)
    # This respects the strict filtering done earlier on df_all
    if not df_all.empty:
        available_scens = set(df_all["scenario"].unique())
        scen_opts = [s for s in scen_opts if s in available_scens]
        

    # EXTRA GUARD: Remove BR scenarios again just in case
    if nav_mode == "Փուլ 2 | SA & HS":
        scen_opts = [s for s in scen_opts if "BR" not in str(s).upper()]

    sel_scenarios = st.multiselect(
        "Սցենարներ",
        options=scen_opts,
        default=scen_opts,
        key="sec_scen_multi"
    )

    store_opts = sorted(df_all["store"].dropna().unique()) if not df_all.empty else []
    sel_section_stores = st.multiselect(
        "Խանութներ",
        options=store_opts,
        default=[],  # Default empty
        key="sec_store_multi",
        placeholder="Ընտրեք խանութներ (դատարկ = բոլորը)"
    )
    
    # Logic: if empty, use all
    actual_stores = sel_section_stores if sel_section_stores else store_opts

    # === Результаты по разделам: процент от максимума (по выбранным сценарием/магазинам) ===
    sec_scores = scoring.section_scores(flt, scenarios=sel_scenarios, stores=actual_stores)
    if sec_scores.empty:
        st.info("Չկան տվյալներ ընտրված ֆիլտրերով։")
    else:
        sec_scores = scoring.apply_section_order(sec_scores, "section")
        tbl = (
            sec_scores
            .rename(columns={
                "store":"Խանութ",
                "scenario":"Սցենար",
                "section":"Բաժին",
                "section_pct":"Արդյունք %"
            })
        )
        tbl["Բաժին"] = pd.Categorical(tbl["Բաժին"], categories=scoring.SECTION_ORDER, ordered=True)
        tbl = tbl.sort_values(["Բաժին","Սցենար","Արդյունք %"], ascending=[True, True, False])
        st.dataframe(tbl, use_container_width=True)

    # Таблица вопросов (также привести порядок)
    if not pq.empty and sel_scenarios:
        # Use actual_stores here too
        pq_filtered = pq[pq["scenario"].isin(sel_scenarios) & pq["store"].isin(actual_stores)].copy()
        if "section" in pq_filtered.columns:
            pq_filtered = scoring.apply_section_order(pq_filtered, "section")
            pq_filtered["section"] = pd.Categorical(pq_filtered["section"], categories=scoring.SECTION_ORDER, ordered=True)
        # EXCLUDE: «MS-ի ընդհանուր տպավորություններ» и «Ի՞նչ կարելի է անել փորձը բարելավելու համար» (и их вариации)
        if "question_key" in pq_filtered.columns:
            mask_opinion = scoring._opinion_mask_from_key(pq_filtered["question_key"])
            pq_filtered = pq_filtered[~mask_opinion]

        if not pq_filtered.empty:
            # Այո/Ոչ: сперва пробуем answer_bin, иначе берём score_question_pct (0/1)
            ans = pd.to_numeric(pq_filtered.get("answer_bin"), errors="coerce")
            if "score_question_pct" in pq_filtered.columns:
                ans = ans.fillna(pd.to_numeric(pq_filtered["score_question_pct"], errors="coerce"))
            pq_filtered["ans_label"] = np.where(
                ans >= 1, "Այո",
                np.where(ans == 0, "Ոչ", "")
            )

            st.markdown("**Հարցերի պատասխանները**")
            st.dataframe(
                pq_filtered.sort_values(["scenario","section","store","question_key"])[
                    ["store","scenario","section","question_key","ans_label"]
                ]
                .rename(columns={
                    "store":"Խանութ",
                    "scenario":"Սցենար",
                    "section":"Բաժին",
                    "question_key":"Հարց",
                    "ans_label":"Պատասխան"
                }),
                use_container_width=True
            )
        else:
            st.info("Չկան հարցեր տվյալ ֆիլտրերով։")

with tab_compare:
    st.subheader("Համեմատել երկու խանութ")
    st.caption(scoring.caption_compare_page_hy())

    stores = sorted(flt["store"].dropna().unique()) if not flt.empty else []
    scens  = sorted(flt["scenario"].dropna().unique()) if not flt.empty else []

    if len(stores) < 2:
        st.info("Համեմատելու համար անհրաժեշտ են առնվազն երկու խանութ։")
    else:
        col_a, col_b, col_s = st.columns([1,1,1.2])
        with col_a:
            store_a = st.selectbox("Խանութ A", options=stores, key="cmp_store_a")
        with col_b:
            store_b = st.selectbox("Խանութ B", options=[s for s in stores if s != store_a], key="cmp_store_b")
        with col_s:
            scen = st.selectbox("Սցենար", options=["Բոլոր սցենարները"] + scens, key="cmp_scen")

        sel_section = "Բոլոր բաժինները"
        sel_question = "Բոլոր հարցերը"
        if scen != "Բոլոր սցենարները":
            base = flt[flt["scenario"] == scen].copy()
            base["w"] = pd.to_numeric(base.get("weight_in_scenario"), errors="coerce").fillna(0.0)
            base = base[base["w"] > 0]
            if "question_key" in base.columns:
                base = base[~scoring._opinion_mask_from_key(base["question_key"])]
            sect_opts = ["Բոլոր բաժինները"] + sorted(base["section"].dropna().unique()) if "section" in base.columns else ["Բոլոր բաժինները"]
            sel_section = st.selectbox("Բաժին", options=sect_opts, key="cmp_section")
            if sel_section != "Բոլոր բաժինները":
                qdf = base[base["section"] == sel_section]
                q_opts = ["Բոլոր հարցերը"] + sorted(qdf["question_key"].dropna().unique()) if "question_key" in qdf.columns else ["Բոլոր հարցերը"]
                sel_question = st.selectbox("Հարց", options=q_opts, key="cmp_question")

        if scen == "Բոլոր սցենարները":
            ts = scoring.total_score_table(flt)
            pair = ts[ts["store"].isin([store_a, store_b])].copy()
            pair["value_pct"] = (pair["total_score_pct"] * 100).round(2)
            label = "Ընդհանուր % (բոլոր սցենարներ)"
        else:
            kw = {}
            if sel_question != "Բոլոր հարցերը":
                kw["question"] = sel_question
            elif sel_section != "Բոլոր բաժինները":
                kw["section"] = sel_section
            pair = scoring.scenario_subset_scores(flt, scen, **kw)
            pair = pair[pair["store"].isin([store_a, store_b])].copy()
            label = (
                f"Սցենարի % ({scen})" if not kw else
                (f"Բաժնի % ({sel_section})" if "section" in kw else f"Հարցի % ({sel_question})")
            )

        if pair.empty or pair["store"].nunique() < 2:
            st.warning("Տվյալներ չկան ընտրված ֆիլտրերով։")
        else:
            va = float(pair.loc[pair["store"] == store_a, "value_pct"].iloc[0])
            vb = float(pair.loc[pair["store"] == store_b, "value_pct"].iloc[0])
            mcol1, mcol2 = st.columns(2)
            mcol1.metric(f"{store_a}", f"{va:.2f}%")
            mcol2.metric(f"{store_b}", f"{vb:.2f}%")

            chart_df = pair.rename(columns={"store":"Խանութ","value_pct":label})
            ch = alt.Chart(chart_df).mark_bar(size=60).encode(
                x=alt.X("Խանութ:N", sort=None),
                y=alt.Y(f"{label}:Q", scale=alt.Scale(domain=[0,100]), title="%"),
                color="Խանութ:N",
                tooltip=["Խանութ", alt.Tooltip(label, format=".2f")]
            ).properties(height=320)
            tx = ch.mark_text(baseline="bottom", dy=-4).encode(text=alt.Text(label, format=".0f"))
            st.altair_chart(ch + tx, use_container_width=True)

            if scen != "Բոլոր սցենարները" and sel_question == "Բոլոր հարցերը":
                sec = scoring.section_scores(flt, scenarios=[scen], stores=[store_a, store_b])
                if sel_section != "Բոլոր բաժինները":
                    sec = sec[sec["section"] == sel_section]
                if not sec.empty:
                    sec = scoring.apply_section_order(sec.copy())
                    sec = sec.rename(columns={"store":"Խանութ","section":"Բաժին","section_pct":"Արդյունք %"})
                    ch2 = alt.Chart(sec).mark_bar().encode(
                        x=alt.X("Բաժին:N", title="Բաժին"),
                        y=alt.Y("Արդյունք %:Q", scale=alt.Scale(domain=[0,100]), title="%"),
                        color="Խանութ:N",
                        tooltip=["Խանութ","Բաժին", alt.Tooltip("Արդյունք %:Q", format=".1f")]
                    ).properties(height=360)
                    st.altair_chart(ch2, use_container_width=True)

# --- NEW: Visits tab restored ---
with tab_visits:
    st.subheader("Այցելություններ")
    st.caption("Այստեղ ներկայացված են այցերի քանակը, միջին տևողությունը, ժամանակային բաշխվածությունը "
               "և այցելողների մեկնաբանությունները։ Ֆիլտրերը կիրառվում են միայն այս բաժնում։")

    if visits_df is None or visits_df.empty:
        st.info("Այցելությունների տվյալներ չկան։")
    else:
        # CLEAN: убрать 'nan'/пустые магазины заранее
        vbase = visits_df.copy()
        s_clean = vbase["store"].astype(str).str.strip()
        vbase = vbase[s_clean.notna() & (s_clean != "") & (s_clean.str.lower() != "nan")].copy()

        # Фильтры (только для этой страницы)
        v_stores = sorted(vbase["store"].dropna().unique().tolist())
        v_scens = sorted(vbase["scenario"].dropna().unique().tolist())
        with st.expander("Ֆիլտրեր (Այցելություններ)", expanded=True):
            f_scens = st.multiselect("Սցենարներ", options=v_scens, default=v_scens, key="vis_scen")
            f_stores = st.multiselect("Խանութներ", options=v_stores, default=[], key="vis_store", placeholder="Ընտրեք խանութներ (դատարկ = բոլորը)")
            max_dur = st.slider("Մակս. տևողություն (րոպե) — հեռացնել ծայրահեղ արժեքները",
                                min_value=30, max_value=180, value=60, step=5)

        # Logic: if empty, use all
        actual_v_stores = f_stores if f_stores else v_stores
        
        vflt = vbase[vbase["scenario"].isin(f_scens) & vbase["store"].isin(actual_v_stores)].copy()
        
        # Strict Stage Filtering for Visits
        if nav_mode == "Փուլ 1 | Retail":
             vflt = vflt[vflt["scenario"].isin(["BR1", "BR2", "BR3", "BR4", "BR5"])]
        elif nav_mode == "Փուլ 2 | SA & HS":
             mask_br = vflt["scenario"].astype(str).str.contains(r'^BR.*', case=False, regex=True)
             vflt = vflt[~mask_br]
             
        vflt["visit_duration_min"] = pd.to_numeric(vflt["visit_duration_min"], errors="coerce")

        # FILTER: исключить явные выбросы по длительности
        before_len = len(vflt)
        vflt = vflt[(vflt["visit_duration_min"].isna()) | ((vflt["visit_duration_min"] >= 0) & (vflt["visit_duration_min"] <= max_dur))]
        removed_outliers = before_len - len(vflt)
        if removed_outliers > 0:
            st.caption(f"Ցուցադրման համար հեռացվել է {removed_outliers} այց, որոնց տևողությունը > {max_dur} րոպե (հավանաբար սխալ գրառում).")

        # Tоп-метрики
        total_visits = int(len(vflt))
        avg_dur = float(vflt["visit_duration_min"].mean()) if total_visits else 0.0
        c1, c2 = st.columns(2)
        c1.metric("Այցելությունների քանակ", total_visits)
        c2.metric("Միջին տևողություն (րոպե)", f"{avg_dur:.0f}")

        # 1) Միջին տևողություն և այցելությունների թվաքանակ
        st.markdown("### Միջին տևողություն և այցելությունների թվաքանակ")
        agg = (
            vflt.groupby(["store","scenario"], as_index=False)
                .agg(visits=("store","count"), avg_duration_min=("visit_duration_min","mean"))
        )
        agg["avg_duration_min"] = agg["avg_duration_min"].round(0).astype("Int64")
        st.dataframe(agg.sort_values(["store","scenario"]), use_container_width=True)

        # 2) Այցելությունները ըստ խանութների (բոլոր սցենարներով)
        st.markdown("### Այցելությունները ըստ խանութների (բոլոր սցենարներով)")
        by_store = vflt.groupby("store", as_index=False).size().rename(columns={"size":"visits"})
        st.dataframe(by_store.sort_values("visits", ascending=False), use_container_width=True)

        h = max(240, 24 * len(by_store))
        ch_store = alt.Chart(by_store).mark_bar(size=20).encode(
            y=alt.Y("store:N", sort='-x', title="Խանութ"),
            x=alt.X("visits:Q", title="Այցելությունների քանակ"),
            tooltip=["store","visits"]
        ).properties(height=h)
        st.altair_chart(ch_store, use_container_width=True)

        # 3) Այցելումները ըստ օրվա մասերի
        st.markdown("### Այցելումները ըստ օրվա մասերի")
        tod = (
            vflt.groupby(["time_of_day","scenario"], as_index=False)
                .size().rename(columns={"size":"visits"})
        )
        order_tod = ["Morning","Day","Evening","Night","Unknown"]
        tod["time_of_day"] = pd.Categorical(tod["time_of_day"], categories=order_tod, ordered=True)
        st.dataframe(
            tod.pivot_table(index="time_of_day", columns="scenario", values="visits", fill_value=0),
            use_container_width=True
        )

        ch_tod = alt.Chart(tod).mark_bar().encode(
            x=alt.X("time_of_day:N", title="Օրվա մաս"),
            y=alt.Y("visits:Q", title="Այցելություններ"),
            color=alt.Color("scenario:N", title="Սցենար"),
            tooltip=["time_of_day","scenario","visits"]
        ).properties(height=320)
        st.altair_chart(ch_tod, use_container_width=True)

        # 4) Այցելումների ժամերն ու տևողությունները
        st.markdown("### Այցելումների ժամերն ու տևողությունները")
        cols = ["store","scenario","visit_start","visit_end","visit_duration_min","time_of_day"]
        st.dataframe(vflt[cols].sort_values(["store","scenario","visit_start"]), use_container_width=True)

        if vflt["visit_start"].notna().any():
            ch_scatter = alt.Chart(vflt).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X("visit_start:T", title="Սկիզբ"),
                y=alt.Y("visit_duration_min:Q", title="Տևողություն (րոպե)", scale=alt.Scale(domain=[0, max_dur])),
                color=alt.Color("scenario:N", title="Սցենար"),
                tooltip=["store","scenario","visit_start","visit_end",
                         alt.Tooltip("visit_duration_min:Q", format=".0f")]
            ).properties(height=320)
            st.altair_chart(ch_scatter, use_container_width=True)

    # 5) Այցելողների մեկնաբանությունները
    st.markdown("### Այցելողների մեկնաբանությունները")
    if comments_df is None or comments_df.empty:
        st.info("Մեկնաբանություններ չկան։")
    else:
        cflt = comments_df.copy()
        s_clean_c = cflt["store"].astype(str).str.strip()
        cflt = cflt[s_clean_c.notna() & (s_clean_c != "") & (s_clean_c.str.lower() != "nan")]
        if 'scenario' in cflt and 'store' in cflt:
            cflt = cflt[cflt["scenario"].isin(f_scens) & cflt["store"].isin(f_stores)]
        st.dataframe(cflt.sort_values(["store","scenario"]), use_container_width=True)

with st.expander("Տվյալների բազա", expanded=False):
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
