from typing import Dict, List
import re

import pandas as pd
import numpy as np

from .common import norm

SCENARIO_NAME_MAP = {1:"BR1", 2:"BR2", 3:"BR3", 4:"BR4", 5:"BR5"}

BASE_PATS = [r'^\s*BR\s*([1-5])\s*$', r'^\s*YAP[-_\s]*BR\s*([1-5])\s*$']
LAS_RE = r'^\s*LAS[_\s-]*Brand\s*Retail\s*\(\s*([1-5])\s*\)\s*$'
YAP_RE = r'^\s*YAP[-_\s]*Brand\s*Retail\s*\(\s*([1-5])\s*\)\s*$'

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

def read_excel_all(xls_file):
    xls = pd.ExcelFile(xls_file)
    sheets = xls.sheet_names

    bases: Dict[str, pd.DataFrame] = {}
    weights: Dict[str, List[pd.DataFrame]] = {}

    # Базы по сценарию
    for sh in sheets:
        scen = _as_base(sh)
        if scen:
            bases[scen] = xls.parse(sh, dtype=object)

    # Общий лист Weights (кастомный)
    if "Weights" in sheets:
        dfw = xls.parse("Weights", dtype=object)
        wnorm = _normalize_weights_sheet(dfw)
        for scen, wpart in wnorm.groupby("scenario"):
            weights.setdefault(scen, []).append(wpart.drop(columns=["scenario"]))

    # LAS / YAP весовые листы
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

    return bases, weights

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
    col_E = df.columns[4] if len(df.columns) > 4 else None  # вес секции
    col_F = df.columns[5] if len(df.columns) > 5 else None  # вес сценария

    w = df.copy()
    w["section"] = (w[col_B] if col_B in w.columns else None).ffill().fillna("Անվանված չէ")

    # строка-заголовок раздела: B пусто, C начинается с "1. "
    is_section_header = (w[col_B].isna()) & (w[col_C].astype(str).str.match(r'^\s*\d+\.\s*', na=False))

    def pct(x):
        return _to_pct_number(x)

    # Веса E/F, с протяжкой вниз (фикс объединённых ячеек)
    w["weight_in_section"]  = (w[col_E].map(pct) if col_E in w.columns else np.nan)
    w["weight_in_scenario"] = (w[col_F].map(pct) if col_F in w.columns else np.nan)
    w["weight_in_section"]  = w["weight_in_section"].ffill()
    w["weight_in_scenario"] = w["weight_in_scenario"].ffill()

    # Суммарный вес раздела из строки-заголовка (E у шапки)
    w["section_total_weight"] = np.where(is_section_header, (w[col_E].map(pct) if col_E in w.columns else np.nan), np.nan)
    w["section_total_weight"] = w["section_total_weight"].ffill()

    # Вопросы: НЕ заголовочные строки
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
    q = weights_df.loc[weights_df["scenario"]==scenario, "qkey_norm"].dropna().unique().tolist()
    return sorted(q)
