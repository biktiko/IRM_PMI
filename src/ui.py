import streamlit as st
import pandas as pd
import altair as alt
from src.utils import (
    brand_bar_chart,
    brand_colors,
)

def rating_table(df: pd.DataFrame, score_col: str, title: str):
    if df.empty:
        st.info("Տվյալներ չկան։")
        return
    chart = brand_bar_chart(df, x="store", y=score_col, title=title)
    st.altair_chart(chart, use_container_width=True, theme=None)
    st.dataframe(df.sort_values(score_col, ascending=False).reset_index(drop=True),
                 use_container_width=True)

def heatmap_sections(ps: pd.DataFrame):
    if ps.empty:
        st.info("Տվյալներ չկան։")
        return
    df = ps.rename(columns={"score_section_pct": "score"}).copy()
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("section:N", title="", sort=None),
            y=alt.Y("store:N", title="", sort=None),
            color=alt.Color("score:Q", title="Score", scale=alt.Scale(range=brand_colors)),
            tooltip=["store","section","score"]
        )
        .properties(title="Բաժինների ջերմաքարտեզ", height=420)
    )
    st.altair_chart(chart, use_container_width=True, theme=None)

def store_profile(ps: pd.DataFrame, pq: pd.DataFrame, store: str):
    st.markdown(f"### {store}")
    sect = ps[ps["store"] == store].rename(columns={"score_section_pct": "score"}).copy()
    if not sect.empty:
        c1, c2 = st.columns([2,1])
        with c1:
            bar = brand_bar_chart(sect, x="section", y="score", title="Բաժինների վարկանիշ")
            st.altair_chart(bar, use_container_width=True, theme=None)
        with c2:
            cols = ["section","score"]
            if "weighted_section_score_pct" in sect.columns:
                cols.append("weighted_section_score_pct")
            st.dataframe(sect[cols].sort_values("score", ascending=False),
                         use_container_width=True)

    # Вопросы (без графика): полный текст, бинарный ответ, вес
    qdf = pq[pq["store"] == store].copy()
    if qdf.empty:
        return
    st.markdown("#### Հարցերի արդյունքները")
    label_col = "question_text" if "question_text" in qdf.columns else "question_key"

    # Ответ Այո/Ոչ (если 100% Да, иначе Нет; при частичных процентах считаем Нет)
    ans = qdf["score_question_pct"].apply(lambda v: "Այո" if pd.notna(v) and float(v) == 100 else "Ոչ")
    out = pd.DataFrame({
        "Հարց": qdf[label_col],
        "հարց": ans,  # пользователь просил так назвать колонку с ответом
    })
    if "question_weight_pct" in qdf.columns:
        out["հարցի կշիռը"] = qdf["question_weight_pct"].round(2)

    st.dataframe(out.reset_index(drop=True), use_container_width=True)

def compare_two(psc: pd.DataFrame, ps: pd.DataFrame, store_a: str, store_b: str):
    st.markdown("#### Սցենարների համեմատություն")
    if psc is not None and not psc.empty:
        a = (psc[psc["store"] == store_a][["scenario","score_scenario_pct"]]
             .rename(columns={"score_scenario_pct": f"score_{store_a}"}))
        b = (psc[psc["store"] == store_b][["scenario","score_scenario_pct"]]
             .rename(columns={"score_scenario_pct": f"score_{store_b}"}))
        scen = (a.merge(b, on="scenario", how="outer")
                  .assign(diff=lambda d: d[f"score_{store_a}"] - d[f"score_{store_b}"]))
        st.dataframe(scen.sort_values("scenario").round(2), use_container_width=True)
    else:
        st.info("Տվյալներ սցենարների համար չկան։")

    st.markdown("#### Բաժինների համեմատություն")
    if ps is not None and not ps.empty:
        a2 = (ps[ps["store"] == store_a][["section","score_section_pct"]]
              .rename(columns={"score_section_pct": f"score_{store_a}"}))
        b2 = (ps[ps["store"] == store_b][["section","score_section_pct"]]
              .rename(columns={"score_section_pct": f"score_{store_b}"}))
        sec = (a2.merge(b2, on="section", how="outer")
                 .assign(diff=lambda d: d[f"score_{store_a}"] - d[f"score_{store_b}"]))
        st.dataframe(sec.sort_values("section").round(2), use_container_width=True)
    else:
        st.info("Տվյալներ բաժինների համար չկան։")

def download_buttons(dfs: dict[str, pd.DataFrame]):
    for name, df in dfs.items():
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(f"Download {name}.csv", csv, file_name=f"{name}.csv", mime="text/csv")

