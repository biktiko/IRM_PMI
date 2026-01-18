from typing import Dict, List
import re

import pandas as pd
import numpy as np

from .common import norm

SCENARIO_NAME_MAP = {1:"BR1", 2:"BR2", 3:"BR3", 4:"BR4", 5:"BR5"}

BASE_PATS = [r'^\s*BR\s*([1-5])\s*$', r'^\s*YAP[-_\s]*BR\s*([1-5])\s*$']
LAS_RE = r'^\s*LAS[_\s-]*Brand\s*Retail\s*\(\s*([1-5])\s*\)\s*$'
YAP_RE = r'^\s*YAP[-_\s]*Brand\s*Retail\s*\(\s*([1-5])\s*\)\s*$'

# Stage 2 Patterns
STAGE2_DATA_MAP = {
    "LAS_SA": "LAS_SA",
    "LAS_SFP": "LAS_SFP",
    "LAS_SFP&CC": "LAS_SFP_CC", # Normalized key
    "HS_SA_YAP": "HS_SA_YAP"
}

# Regex for Stage 2 Data Sheets (flexible matching)
S2_LAS_SA_RE = r'^\s*LAS\s*[_]?\s*SA\s*$'
S2_LAS_SFP_RE = r'^\s*LAS\s*[_]?\s*SFP\s*$'
S2_LAS_SFP_CC_RE = r'^\s*LAS\s*[_]?\s*SFP\s*&\s*CC\s*$'
S2_HS_SA_YAP_RE = r'^\s*HS\s*[_]?\s*SA\s*[_]?\s*YAP\s*$'
S2_HS_YAP_RE = r'^\s*HS\s*\(\s*YAP\s*\)\s*$' # For HS (YAP)

# Map Data Sheet -> Weight Sheet regex/name
# We will try to find weight sheets by name or pattern
S2_WEIGHTS_MAP = {
    "LAS_SA": [r'^\s*LAS\s*[_]?\s*Shop\s*Assistant\s*$'],
    "LAS_SFP": [r'^\s*LAS\s*[_]?\s*SFP\s*Hostess\s*$'],
    "LAS_SFP_CC": [r'^\s*LAS\s*[_]?\s*SFP\s*&\s*CC\s*Hostess\s*$'],
    "HS_SA_YAP": [
        r'^\s*HS\s*\(\s*YAP\s*\)\s*$', 
        # r'^\s*HS\s*[_]?\s*SA\s*[_]?\s*YAP\s*$', # REMOVED: Matches Data Sheet!
        r'^\s*HS\s*YAP\s*$',
        # r'.*HS.*YAP.*' # REMOVED: Too broad, might match data sheet
    ]
}

def _as_base(name: str):
    # Распознаём базовые листы BR1..BR5
    for pat in BASE_PATS:
        m = re.match(pat, name, flags=re.I)
        if m:
            return SCENARIO_NAME_MAP.get(int(m.group(1)))
    m2 = re.match(r'^\s*BR\s*([1-5])', name, flags=re.I)
    if m2:
        return SCENARIO_NAME_MAP.get(int(m2.group(1)))
    return None

def _identify_stage2_sheet(name: str):
    if re.match(S2_LAS_SA_RE, name, re.I): return "LAS_SA"
    if re.match(S2_LAS_SFP_RE, name, re.I): return "LAS_SFP"
    if re.match(S2_LAS_SFP_CC_RE, name, re.I): return "LAS_SFP_CC"
    if re.match(S2_HS_SA_YAP_RE, name, re.I): return "HS_SA_YAP"
    # if re.match(S2_HS_YAP_RE, name, re.I): return "HS_SA_YAP" # REMOVED: HS (YAP) is weights, not data!
    return None

def read_excel_all(xls_file):
    xls = pd.ExcelFile(xls_file)
    sheets = xls.sheet_names

    bases: Dict[str, pd.DataFrame] = {}
    weights: Dict[str, List[pd.DataFrame]] = {}

    # 1. Retail Stage (BR1..BR5)
    for sh in sheets:
        scen = _as_base(sh)
        if scen:
            bases[scen] = xls.parse(sh, dtype=object)
            # Mark as Retail
            bases[scen]["_stage_"] = "Retail"

    # 2. Stage 2 Data Sheets
    for sh in sheets:
        s2_key = _identify_stage2_sheet(sh)
        if s2_key:
            bases[s2_key] = xls.parse(sh, dtype=object)
            bases[s2_key]["_stage_"] = "Stage2"

    # 3. Weights (Retail - General)
    if "Weights" in sheets:
        dfw = xls.parse("Weights", dtype=object)
        wnorm = _normalize_weights_sheet(dfw)
        for scen, wpart in wnorm.groupby("scenario"):
            weights.setdefault(scen, []).append(wpart.drop(columns=["scenario"]))

    # 4. Weights (Retail - LAS/YAP specific sheets)
    for sh in sheets:
        m_las = re.match(LAS_RE, sh, flags=re.I)
        m_yap = re.match(YAP_RE, sh, flags=re.I)
        if m_las:
            idx = int(m_las.group(1))
            scen = SCENARIO_NAME_MAP[idx]
            df = xls.parse(sh, header=0, dtype=object)
            w = _normalize_las_sheet(df)
            w["scenario"] = scen
            w["source"] = "LAS"
            weights.setdefault(scen, []).append(w)
        elif m_yap:
            idx = int(m_yap.group(1))
            scen = SCENARIO_NAME_MAP[idx]
            df = xls.parse(sh, header=0, dtype=object)
            w = _normalize_las_sheet(df)
            w["scenario"] = scen
            w["source"] = "YAP"
            weights.setdefault(scen, []).append(w)

    # 5. Weights (Stage 2)
    # We iterate sheets again to find weights for Stage 2
    for sh in sheets:
        # Check against S2_WEIGHTS_MAP
        found_key = None
        for key, pats in S2_WEIGHTS_MAP.items():
            for pat in pats:
                if re.match(pat, sh, re.I):
                    found_key = key
                    break
            if found_key: break
        
        if found_key:
            df = xls.parse(sh, header=0, dtype=object)
            
            # Special handling for HS (YAP) which has shifted columns
            if found_key == "HS_SA_YAP" and re.match(r'^\s*HS\s*\(\s*YAP\s*\)\s*$', sh, re.I):
                w = _normalize_hs_yap_sheet(df)
            else:
                w = _normalize_las_sheet(df)
                
            w["scenario"] = found_key # Use the key (e.g. LAS_SA) as scenario name
            w["source"] = "Stage2_Weights"
            weights.setdefault(found_key, []).append(w)

    return bases, weights

def _normalize_hs_yap_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Special normalizer for HS (YAP) weights sheet which is shifted left:
    Col A: Section
    Col B: Question
    Col D: Weight in Section
    Col E: Weight in Scenario
    """
    col_A = df.columns[0] if len(df.columns) > 0 else None
    col_B = df.columns[1] if len(df.columns) > 1 else None
    col_D = df.columns[3] if len(df.columns) > 3 else None
    col_E = df.columns[4] if len(df.columns) > 4 else None

    w = df.copy()
    w["section"] = (w[col_A] if col_A in w.columns else None).ffill().fillna("Անվանված չէ")

    # Section header detection: A is not empty, B is empty? 
    # Or based on color/structure.
    # In screenshot: "1. First Contact" is in A.
    # Question rows: "First Contact" in A, Question in B.
    # Actually, looking at screenshot:
    # Row 10: "1. Առաջին..." in A.
    # Row 11: "First Contact" in A, Question in B.
    # So if B is not empty, it's a question?
    
    # Let's assume if B has text, it's a question.
    w["question_key"] = w[col_B] if col_B in w.columns else None
    
    # Filter out metadata rows (top 8 rows)
    # Metadata rows usually have empty D/E or non-numeric
    # Real questions have weights in D/E
    
    def pct(x):
        return _to_pct_number(x)

    w["weight_in_section"]  = (w[col_D].map(pct) if col_D in w.columns else np.nan)
    w["weight_in_scenario"] = (w[col_E].map(pct) if col_E in w.columns else np.nan)
    
    # Fill weights down? Usually weights are per question in this sheet?
    # Screenshot shows weights on every question row.
    # But let's ffill just in case, or maybe not if they are explicit.
    # Screenshot: 100% and 10% are on the question row.
    
    # Filter: Must have a question key AND at least one weight or be a section header
    # But wait, section header "1. ..." has 10% in E?
    # Row 10: "1. ..." in A. E has "10%".
    # So section header has weight in scenario.
    
    is_section_header = w[col_A].astype(str).str.match(r'^\s*\d+\.\s*', na=False)
    
    w["section_total_weight"] = np.where(is_section_header, w["weight_in_scenario"], np.nan)
    w["section_total_weight"] = w["section_total_weight"].ffill()
    
    # If it's a section header, question_key should be None
    w["question_key"] = np.where(is_section_header, None, w["question_key"])
    
    # Drop rows without question key
    w = w.dropna(subset=["question_key"]).copy()
    
    # Drop rows that are likely metadata (e.g. "Mystery Shopper" in A)
    # Metadata rows usually don't have weights?
    # Or check if question_key is not one of the known metadata labels
    # But simpler: check if weights are valid?
    # No, some questions might have 0 weight.
    
    w["qkey_norm"] = w["question_key"].astype(str).map(norm)
    
    return w[["section","question_key","qkey_norm","weight_in_section","weight_in_scenario","section_total_weight"]]

def _to_pct_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", ".")
    if s.endswith("%"):
        try:
            return float(s[:-1].strip())/100.0
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _normalize_weights_sheet(df: pd.DataFrame) -> pd.DataFrame:
    # Ожидаем колонки: scenario, section, question_key, weight_in_section (E), weight_in_scenario (F)
    cols = {c: norm(c) for c in df.columns}
    ren = {}
    for c, k in cols.items():
        if k in ("scenario","scen","սցենար"):
            ren[c] = "scenario"
        elif k in ("section","բաժին"):
            ren[c] = "section"
        elif k in ("question_key","question","հարց","qkey"):
            ren[c] = "question_key"
        elif ("section" in k and "weight" in k) or k in ("e","weight_e"):
            ren[c] = "weight_in_section"
        elif ("scenario" in k and "weight" in k) or k in ("f","weight_f"):
            ren[c] = "weight_in_scenario"
        elif "section_total_weight" in k or k in ("section_total","total_section_weight"):
            ren[c] = "section_total_weight"

    out = df.rename(columns=ren)
    required = {"scenario","section","question_key","weight_in_section","weight_in_scenario"}
    if not required.issubset(out.columns):
        # Возвращаем пустой DataFrame в унифицированном формате
        return pd.DataFrame(columns=["section","question_key","qkey_norm","weight_in_section","weight_in_scenario","section_total_weight","scenario"])

    out["qkey_norm"] = out["question_key"].astype(str).map(norm)
    for c in ("weight_in_section","weight_in_scenario","section_total_weight"):
        if c in out.columns:
            out[c] = out[c].apply(_to_pct_number)

    return out[["section","question_key","qkey_norm","weight_in_section","weight_in_scenario","section_total_weight","scenario"]]

def _normalize_las_sheet(df: pd.DataFrame) -> pd.DataFrame:
    col_B = df.columns[1] if len(df.columns) > 1 else None
    col_C = df.columns[2] if len(df.columns) > 2 else None
    col_D = df.columns[3] if len(df.columns) > 3 else None
    col_E = df.columns[4] if len(df.columns) > 4 else None  # вес секции (обычно)
    col_F = df.columns[5] if len(df.columns) > 5 else None  # вес сценария (обычно)

    w = df.copy()
    w["section"] = (w[col_B] if col_B in w.columns else None).ffill().fillna("Անվանված չէ")

    # строка-заголовок раздела: B пусто, C начинается с "1. "
    # OR: B contains "Բաժին" or similar
    # For HS (YAP), sometimes B is not empty. Let's be more flexible.
    # If C starts with digit dot, it's likely a section header.
    is_section_header = (w[col_C].astype(str).str.match(r'^\s*\d+\.\s*', na=False))

    def pct(x):
        return _to_pct_number(x)

    # Detect if we should use D/E or E/F
    # Check if F is empty/null but D has data
    use_DE = False
    if col_F and col_D:
        # Count non-nulls in F vs D (simple heuristic)
        # Or check if D contains "100%" or "10%" strings
        # User said HS (YAP) uses D and E.
        # Let's check if D has numeric/pct values
        d_vals = w[col_D].astype(str).str.contains(r'\d', na=False).sum()
        f_vals = w[col_F].astype(str).str.contains(r'\d', na=False).sum()
        if d_vals > f_vals:
            use_DE = True

    if use_DE:
        # Weights are in D (Section) and E (Scenario)
        w_sec_col = col_D
        w_scen_col = col_E
    else:
        # Standard: E (Section) and F (Scenario)
        w_sec_col = col_E
        w_scen_col = col_F

    # Веса, с протяжкой вниз (фикс объединённых ячеек)
    w["weight_in_section"]  = (w[w_sec_col].map(pct) if w_sec_col in w.columns else np.nan)
    w["weight_in_scenario"] = (w[w_scen_col].map(pct) if w_scen_col in w.columns else np.nan)
    w["weight_in_section"]  = w["weight_in_section"].ffill()
    w["weight_in_scenario"] = w["weight_in_scenario"].ffill()

    # Суммарный вес раздела из строки-заголовка (E у шапки)
    # Use w_sec_col instead of col_E
    w["section_total_weight"] = np.where(is_section_header, (w[w_sec_col].map(pct) if w_sec_col in w.columns else np.nan), np.nan)
    w["section_total_weight"] = w["section_total_weight"].ffill()

    # Вопросы: НЕ заголовочные строки
    # Ensure we don't pick up section headers as questions
    w["question_key"] = np.where(is_section_header, None, w[col_C] if col_C in w.columns else None)

    # Отфильтруем пустые вопросы
    w = w.dropna(subset=["question_key"]).copy()

    w["qkey_norm"] = w["question_key"].astype(str).map(norm)

    return w[["section","question_key","qkey_norm","weight_in_section","weight_in_scenario","section_total_weight"]]

def build_weights_dict(weights_raw: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    frames = []
    for scen, items in weights_raw.items():
        lst = items if isinstance(items, list) else [items]
        for w in lst:
            w = w.copy()
            if "scenario" not in w.columns:
                w["scenario"] = scen
            cols = ["scenario","section","question_key","qkey_norm","weight_in_section","weight_in_scenario","section_total_weight"]
            if "source" in w.columns:
                cols.append("source")
            w = w[cols]
            for c in ("weight_in_section","weight_in_scenario","section_total_weight"):
                if c in w.columns:
                    w[c] = pd.to_numeric(w[c], errors="coerce")
            frames.append(w)

    if not frames:
        return pd.DataFrame(columns=["scenario","section","question_key","qkey_norm","weight_in_section","weight_in_scenario","section_total_weight"])

    out = pd.concat(frames, ignore_index=True)

    # Если есть LAS и YAP для одного qkey, оставляем LAS (предпочтение LAS)
    if "source" in out.columns:
        # True/False сортировка: LAS будет первым
        out = (out.sort_values(by="source", key=lambda s: s.ne("LAS"))
                   .drop_duplicates(subset=["scenario","qkey_norm"], keep="first"))
        out = out.drop(columns=["source"], errors="ignore")
    else:
        out = out.drop_duplicates(subset=["scenario","qkey_norm"], keep="last")

    return out.reset_index(drop=True)

def questions_from_weights(weights_df: pd.DataFrame, scenario: str) -> List[str]:
    if weights_df is None or weights_df.empty:
        return []
    # Try exact match
    # Normalize weights_df scenario names temporarily for matching
    w_scen = weights_df["scenario"].astype(str).str.strip()
    q = weights_df.loc[w_scen==scenario.strip(), "qkey_norm"].dropna().unique().tolist()
    
    if not q:
        # Try case-insensitive match
        scen_lower = scenario.strip().lower()
        # Create a mapping of lower->original
        scen_map = {s.strip().lower(): s for s in weights_df["scenario"].unique()}
        if scen_lower in scen_map:
            mapped_scen = scen_map[scen_lower]
            # Use the mapped scenario name to find questions
            q = weights_df.loc[weights_df["scenario"]==mapped_scen, "qkey_norm"].dropna().unique().tolist()
            
    return sorted(q)
