from typing import Tuple
import pandas as pd
import numpy as np
import re

from .common import to_binary, norm, excel_col_to_idx
 
def filter_by_scenario_column(df: pd.DataFrame, scen: str) -> pd.DataFrame:
    # найдём колонку, где встречаются BR1..BR5
    candidates = [c for c in df.columns
                  if df[c].astype(str).str.contains(r'\bBR\s*[1-5]\b', case=False, na=False).any()]
    if candidates:
        col = candidates[0]
        return df[df[col].astype(str).str.contains(scen, case=False, na=False)]
    return df

def long_from_base(df: pd.DataFrame, store_col: str, non_question_cols: list, scenario: str, qkeys_norm: list) -> pd.DataFrame:
    norm_map = {c: norm(c) for c in df.columns}
    cols = [c for c,nc in norm_map.items() if nc in qkeys_norm]
    id_cols = [store_col] if store_col in df.columns else []
    long_df = df.melt(id_vars=id_cols, value_vars=cols, var_name="question_key", value_name="answer_raw")
    long_df = long_df.rename(columns={store_col:"store"})
    long_df["scenario"] = scenario
    long_df["qkey_norm"] = long_df["question_key"].astype(str).map(norm)
    long_df["answer_bin"] = long_df["answer_raw"].map(to_binary)
    return long_df

def join_weights(long_df: pd.DataFrame, weights_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    w = weights_df[weights_df["scenario"]==scenario].copy()
    # гарантированно нужные колонки
    base_cols = ["qkey_norm","section","weight_in_section","weight_in_scenario"]
    opt_cols = []
    if "section_total_weight" in w.columns:
        opt_cols.append("section_total_weight")
    w = w[base_cols + opt_cols]
    merged = long_df.merge(w, on="qkey_norm", how="left")
    if "section" not in merged.columns:
        merged["section"] = "Անվանված չէ"
    if "section_total_weight" not in merged.columns:
        merged["section_total_weight"] = np.nan
    return merged

def aggregate_scores(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pq = (df_all.groupby(["store","scenario","section","question_key"], dropna=False)
                .apply(lambda g: _weighted(g, "weight_in_section"))
                .reset_index(name="score_question_pct"))
    ps = (df_all.groupby(["store","scenario","section"], dropna=False)
                .apply(lambda g: _weighted(g, "weight_in_section"))
                .reset_index(name="score_section_pct"))
    psc = (df_all.groupby(["store","scenario"], dropna=False)
                .apply(lambda g: _weighted(g, "weight_in_scenario"))
                .reset_index(name="score_scenario_pct"))
    pstore = (psc.groupby("store", dropna=False)["score_scenario_pct"]
                .mean().reset_index(name="score_store_pct"))
    return pq, ps, psc, pstore, df_all

def _weighted(g: pd.DataFrame, weight_col: str) -> float:
    sub = g.dropna(subset=["answer_bin", weight_col])  # ← без веса не считаем
    if sub.empty:
        return float("nan")
    num = (sub["answer_bin"] * sub[weight_col]).sum()
    den = sub[weight_col].sum()
    return float(num/den*100.0) if den != 0 else float("nan")

SCORE10_COL_BY_SCEN = {"BR1":"AV","BR2":"AW","BR3":"AS","BR4":"AX","BR5":"W"}

def extract_10pt_rating(df: pd.DataFrame, scen: str, store_col: str) -> pd.DataFrame:
    letter = SCORE10_COL_BY_SCEN.get(scen)
    if not letter: 
        return pd.DataFrame(columns=["store","scenario","rating_10pt"])
    idx = excel_col_to_idx(letter)
    if idx < 0 or idx >= len(df.columns):
        return pd.DataFrame(columns=["store","scenario","rating_10pt"])
    col = df.columns[idx]
    out = (df[[store_col, col]]
           .rename(columns={store_col:"store", col:"rating_10pt"})
           .assign(scenario=scen))
    # в число
    out["rating_10pt"] = pd.to_numeric(out["rating_10pt"], errors="coerce")
    return out.dropna(subset=["store"])

def diagnostics_breakdown(df: pd.DataFrame, store: str, scenario: str) -> pd.DataFrame:
    sub = df[(df["store"]==store) & (df["scenario"]==scenario)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["numerator_section"]  = sub["answer_bin"] * sub["weight_in_section"]
    sub["numerator_scenario"] = sub["answer_bin"] * sub["weight_in_scenario"]
    sub["section_calc_pct"] = (
        sub.groupby(["section"], dropna=False)
           .apply(lambda g: (g["numerator_section"].sum()/g["weight_in_section"].sum()*100.0)
                  if g.dropna(subset=["answer_bin","weight_in_section"]).size else np.nan)
           .reindex(sub["section"]).values
    )
    scen_num = sub["numerator_scenario"].sum()
    scen_den = sub["weight_in_scenario"].sum()
    scenario_pct = (scen_num/scen_den*100.0) if scen_den else np.nan
    sub["scenario_calc_pct"] = scenario_pct
    if "section_total_weight" not in sub.columns:
        sub["section_total_weight"] = np.nan
    sub["weight_in_section_pct"] = (sub["weight_in_section"]*100).round(2)
    sub["weight_in_scenario_pct"] = (sub["weight_in_scenario"]*100).round(3)
    return sub[[
        "store","scenario","section","question_key",
        "answer_raw","answer_bin",
        "weight_in_section","weight_in_section_pct","numerator_section",
        "weight_in_scenario","weight_in_scenario_pct","numerator_scenario",
        "section_calc_pct","scenario_calc_pct",
        "section_total_weight"
    ]]

def diagnostics_summary(df: pd.DataFrame, store: str, scenario: str) -> pd.DataFrame:
    """
    Сводка по разделам: сумма весов, сумма использованных весов, процент.
    """
    sub = diagnostics_breakdown(df, store, scenario)
    if sub.empty:
        return sub
    grp = (sub.groupby("section", dropna=False)
             .agg(
                 questions=("question_key","count"),
                 answered=("answer_bin","count"),
                 weight_sum=("weight_in_section","sum"),
                 section_total_weight=("section_total_weight","max"),
                 numerator=("numerator_section","sum"),
                 denominator=("weight_in_section","sum"),
             )
           )
    grp["section_pct_calc"] = (grp["numerator"]/grp["denominator"]*100.0).round(2)
    grp["weight_match"] = (grp["section_total_weight"].round(6) == grp["denominator"].round(6))
    return grp.reset_index()

def derive_section_total(df: pd.DataFrame) -> pd.DataFrame:
    # Предполагаем: заголовок раздела имеет weight_in_section NaN, weight_in_scenario задан (или отдельная колонка raw)
    header_mask = df["weight_in_section"].isna() & df["weight_in_scenario"].notna()
    # Создать карту раздел -> вес
    header_rows = df[header_mask].dropna(subset=["section","weight_in_scenario"])[["section","weight_in_scenario"]]
    header_rows = header_rows.rename(columns={"weight_in_scenario":"section_total_weight"})
    df = df.merge(header_rows, on="section", how="left", suffixes=("","_hdr"))
    # Если уже была колонка section_total_weight — не трогать, иначе использовать _hdr
    if "section_total_weight" in df.columns:
        df["section_total_weight"] = df["section_total_weight"].fillna(df["section_total_weight_hdr"])
    else:
        df["section_total_weight"] = df["section_total_weight_hdr"]
    df = df.drop(columns=[c for c in ["section_total_weight_hdr"] if c in df.columns])
    # Пересчитать weight_in_scenario для вопросов (где weight_in_section не NaN)
    mask_q = df["weight_in_section"].notna() & df["section_total_weight"].notna()
    df.loc[mask_q, "weight_in_scenario"] = df.loc[mask_q, "weight_in_section"] * df.loc[mask_q, "section_total_weight"]
    return df

def _to_frac(x):
    # преобразует '15%', '15,0', 15, 0.15 -> 0.15
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.replace('%','').replace(',','.')
        try:
            x = float(s)
        except:
            return np.nan
    if x > 1:  # было в процентах
        return x/100.0
    return float(x)

def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["weight_in_section","weight_in_scenario","section_total_weight"]:
        if col in df.columns:
            df[col] = df[col].map(_to_frac)
    return df

def recalc_section_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # если есть оба множителя — пересчитать F
    if {"weight_in_section","section_total_weight"}.issubset(df.columns):
        m = df["weight_in_section"].notna() & df["section_total_weight"].notna()
        df.loc[m, "weight_in_scenario"] = df.loc[m, "weight_in_section"] * df.loc[m, "section_total_weight"]

    # если section_total_weight одинаков для всех или пуст — заменить на ΣF
    out = []
    for scen, g in df.groupby("scenario"):
        sums = (g.groupby("section", as_index=False)
                  .agg(section_total_calc=("weight_in_scenario","sum")))
        g = g.merge(sums, on="section", how="left")
        if "section_total_weight" not in g.columns:
            g["section_total_weight"] = g["section_total_calc"]
        else:
            uniq = g["section_total_weight"].dropna().unique()
            if len(uniq) <= 1 or np.allclose(uniq, [0.30], atol=1e-6):
                g["section_total_weight"] = g["section_total_calc"]
        g = g.drop(columns=["section_total_calc"])
        out.append(g)
    return pd.concat(out, ignore_index=True)

def _opinion_mask_from_key(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(False, index=[])
    txt = s.astype(str)
    patt_rating = r"0\s*[-–—]\s*10|0\s*to\s*10|0\s*\/\s*10"
    patt_comment_hy = r"Ի՞նչ\s+կարելի\s+է\s+անել\s+փորձը\s+բարելավելու\s+համար"
    patt_ms_overall_hy = r"MS-ի\s+ընդհանուր\s+տպավորություններ"
    return (
        txt.str.contains(patt_rating, flags=re.IGNORECASE, regex=True, na=False)
        | txt.str.contains(patt_comment_hy, flags=re.IGNORECASE, regex=True, na=False)
        | txt.str.contains(patt_ms_overall_hy, flags=re.IGNORECASE, regex=True, na=False)
    )

# --- Total score (по всем сценариям) ---
def total_score_table(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Общий процент от максимально возможных баллов по всем сценариям для каждого магазина.
    Требует столбцы: store, answer_bin (0/1), weight_in_scenario (0..1).
    """
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["store","total_points","max_points","total_score_pct","total_score_pct_display"])

    d = df_all.copy()
    d["w"] = pd.to_numeric(d.get("weight_in_scenario"), errors="coerce").fillna(0.0)
    if "question_key" in d.columns:
        d = d[~_opinion_mask_from_key(d["question_key"])]
    d["ans"] = pd.to_numeric(d.get("answer_bin"), errors="coerce").fillna(0.0)
    d["num"] = d["ans"] * d["w"]
    g = d.groupby("store", as_index=False).agg(
        total_points=("num","sum"),
        max_points=("w","sum"),
    )
    g["total_score_pct"] = np.where(g["max_points"]>0, g["total_points"]/g["max_points"], np.nan)
    g["total_score_pct_display"] = (g["total_score_pct"]*100).round(2)
    return g[["store","total_points","max_points","total_score_pct","total_score_pct_display"]]

def total_score_caption_hy() -> str:
    return (
        "Այս ցուցիչը ցույց է տալիս խանութի հավաքած միավորների ընդհանուր բաժինը "
        "հնարավոր առավելագույնից՝ բոլոր սցենարների համադրությամբ։ 100% նշանակում է կատարյալ կատարում։"
    )

# --- Процент по каждому сценарию (для всех магазинов) ---
def scenario_score_table(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Процент выполнения по каждому сценарию для каждого магазина (0..100).
    Учитываются только вопросы с положительным весом, 'мнение' вопросы исключены.
    """
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["store","scenario","scenario_score_pct"])
    d = df_all.copy()

    d["w"] = pd.to_numeric(d.get("weight_in_scenario"), errors="coerce").fillna(0.0)
    d["ans"] = pd.to_numeric(d.get("answer_bin"), errors="coerce").fillna(0.0)
    d = d[d["w"] > 0]
    if "question_key" in d.columns:
        d = d[~_opinion_mask_from_key(d["question_key"])]
    d["num"] = d["ans"] * d["w"]

    g = d.groupby(["store","scenario"], as_index=False).agg(
        gained=("num","sum"),
        max_w=("w","sum"),
    )
    g["scenario_score_pct"] = np.where(g["max_w"]>0, g["gained"]/g["max_w"]*100, np.nan).round(2)
    return g[["store","scenario","scenario_score_pct"]]

# --- Профиль магазина по выбранному сценарию (только проценты) ---
def store_profile_breakdown(df_all: pd.DataFrame, store: str, scenario: str) -> pd.DataFrame:
    """
    Профиль магазина по выбранному сценарию: только проценты.
    Исключаются вопросы без веса и 'мнение' (0–10 оценка и «Ի՞նչ կարելի է անել...»).
    """
    cols = ["section","question_key","answer","weight_share_pct","earned_pct","section_score_pct","scenario_score_pct"]
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=cols)
    sub = df_all[(df_all["store"]==store) & (df_all["scenario"]==scenario)].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)
    sub["w"] = pd.to_numeric(sub.get("weight_in_scenario"), errors="coerce").fillna(0.0)
    sub["ans"] = pd.to_numeric(sub.get("answer_bin"), errors="coerce").fillna(0.0)
    sub = sub[sub["w"] > 0]
    if "question_key" in sub.columns:
        sub = sub[~_opinion_mask_from_key(sub["question_key"])]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    sub["weight_share_pct"] = (sub["w"]*100).round(2)
    sub["earned_pct"] = (sub["ans"]*sub["w"]*100).round(2)

    # По разделам
    sec = (sub.groupby("section", dropna=False)
             .agg(num=("earned_pct","sum"), den=("weight_share_pct","sum")))
    sec["section_score_pct"] = np.where(sec["den"]>0, sec["num"]/sec["den"]*100, np.nan).round(2)
    sub = sub.merge(sec["section_score_pct"], on="section", how="left")

    # По сценарию
    scen_num = sub["earned_pct"].sum()
    scen_den = sub["weight_share_pct"].sum()
    scen_pct = round(scen_num/scen_den*100, 2) if scen_den>0 else np.nan
    sub["scenario_score_pct"] = scen_pct

    # Метка ответа
    def _ans_label(a_raw, a_bin):
        try:
            a_bin = float(a_bin)
        except Exception:
            return str(a_raw)
        if np.isnan(a_bin):
            return str(a_raw)
        return "Այո" if a_bin==1 else "Ոչ"
    sub["answer"] = sub.apply(lambda r: _ans_label(r.get("answer_raw"), r.get("ans")), axis=1)

    out = sub[cols].sort_values(["section","question_key"]).reset_index(drop=True)
    return out

def caption_store_profile_hy() -> str:
    return (
        "Այս բաժինը ցույց է տալիս ընտրված խանութի արդյունքները "
        "Հաշվարկվում է՝ որքան տոկոս է հավաքվել հնարավոր առավելագույնից տվյալ սցենարում։ "
    )

def caption_scenario_table_hy() -> str:
    return (
        "Ամփոփ աղյուսակ, որտեղ յուրաքանչյուր տողը խանութ–սցենար զույգի արդյունքի տոկոսն է "
        "հնարավոր առավելագույնից։"
    )

def caption_scenario_page_hy() -> str:
    return (
        "Այս էջը ցույց է տալիս խանութների կատարողականը ընտրված սցենարում՝ տոկոսով հնարավոր առավելագույնից։ "
        "Կարող եք ընտրել մեկ բաժին կամ մեկ հարց. այդ դեպքում հաշվարկը վերաբերում է միայն այդ հատվածին։ "
    )

def caption_sections_page_hy() -> str:
    return (
        "Այս բաժինը ցույց է տալիս խանութների արդյունքները ըստ բաժինների՝ որպես հավաքված միավորների "
        "տոկոս հնարավոր առավելագույնից"
    )

def section_scores(
    df_all: pd.DataFrame,
    scenarios: list[str] | None = None,
    stores: list[str] | None = None
) -> pd.DataFrame:
    """
    Проценты (0..100) по разделам для каждого магазина и сценария.
    Учитывает только вопросы с положительным весом, исключает 'мнение'-вопросы.
    """
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["store","scenario","section","section_pct"])

    d = df_all.copy()
    if scenarios:
        d = d[d["scenario"].isin(scenarios)]
    if stores:
        d = d[d["store"].isin(stores)]
    if d.empty:
        return pd.DataFrame(columns=["store","scenario","section","section_pct"])

    d["w"] = pd.to_numeric(d.get("weight_in_scenario"), errors="coerce").fillna(0.0)
    d["ans"] = pd.to_numeric(d.get("answer_bin"), errors="coerce").fillna(0.0)

    d = d[d["w"] > 0]
    if "question_key" in d.columns:
        d = d[~_opinion_mask_from_key(d["question_key"])]

    if d.empty:
        return pd.DataFrame(columns=["store","scenario","section","section_pct"])

    d["num"] = d["ans"] * d["w"]
    g = d.groupby(["store","scenario","section"], as_index=False).agg(
        gained=("num","sum"),
        max_w=("w","sum")
    )
    g["section_pct"] = np.where(g["max_w"] > 0, g["gained"] / g["max_w"] * 100, np.nan).round(2)
    return g[["store","scenario","section","section_pct"]]

# --- Рейтинг по выбранному сценарию/секции/вопросу ---
def scenario_subset_scores(
    df_all: pd.DataFrame,
    scenario: str,
    section: str | None = None,
    question: str | None = None
) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["store","value_pct"])
    d = df_all[df_all["scenario"] == scenario].copy()
    if d.empty:
        return pd.DataFrame(columns=["store","value_pct"])
    d["w"] = pd.to_numeric(d.get("weight_in_scenario"), errors="coerce").fillna(0.0)
    d["ans"] = pd.to_numeric(d.get("answer_bin"), errors="coerce").fillna(0.0)
    d = d[d["w"] > 0]
    if "question_key" in d.columns:
        d = d[~_opinion_mask_from_key(d["question_key"])]
    # приоритет: вопрос -> раздел
    if question and "question_key" in d.columns:
        d = d[d["question_key"] == question]
    elif section and "section" in d.columns:
        d = d[d["section"] == section]
    if d.empty:
        return pd.DataFrame(columns=["store","value_pct"])
    d["gained"] = d["ans"] * d["w"]
    g = d.groupby("store", as_index=False).agg(
        gained_sum=("gained","sum"),
        max_sum=("w","sum")
    )
    g["value_pct"] = np.where(g["max_sum"]>0, g["gained_sum"]/g["max_sum"]*100, np.nan).round(2)
    return g[["store","value_pct"]]