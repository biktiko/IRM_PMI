
import streamlit as st
import plotly.express as px

# def rating_table(df, value_col: str, title: str):
#     st.markdown(f"### {title}")
#     if df.empty:
#         st.info("Տվյալներ չկան ընտրած ֆիլտրերով։")
#         return
#     tbl = df.sort_values(value_col, ascending=False).reset_index(drop=True)
#     st.dataframe(tbl, use_container_width=True)
#     fig = px.bar(tbl, x="store", y=value_col)
#     st.plotly_chart(fig, use_container_width=True)

def rating_table(df, value_col: str, title: str):
    st.markdown(f"### {title}")
    if df.empty:
        st.info("Տվյալներ չկան ընտրած ֆիլտրերով։")
        return

    tbl = df.sort_values(value_col, ascending=False).reset_index(drop=True)

    # 1) сначала график
    fig = px.bar(tbl, x="store", y=value_col)
    st.plotly_chart(fig, use_container_width=True)

    # 2) потом таблица
    st.dataframe(tbl, use_container_width=True)

def heatmap_sections(ps):
    if ps.empty:
        st.info("Տվյալներ չկան։")
        return
    pivot = ps.pivot_table(index="store", columns="section", values="score_section_pct", aggfunc="mean")
    fig = px.imshow(pivot, aspect="auto", origin="lower", labels=dict(color="Score %"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pivot.round(1), use_container_width=True)

def store_profile(ps, pq, store: str):
    st.subheader(f"Խանութի պրոֆիլ — {store}")
    df = ps[ps["store"]==store]
    if df.empty:
        st.info("Տվյալներ չկան։")
        return
    st.markdown("**Բաժիններ (միջին % հարցերի վրա):**")
    st.dataframe(df.sort_values(["scenario","section"])[["scenario","section","score_section_pct"]].round(1),
                 use_container_width=True)
    st.markdown("**Հարցեր (մանրամասն):**")
    df_q = pq[pq["store"]==store]
    st.dataframe(df_q.sort_values(["scenario","section","question_key"])[
        ["scenario","section","question_key","score_question_pct"]
    ].round(1), use_container_width=True)

def compare_two(psc, ps, store_a, store_b):
    st.subheader(f"Համեմատություն — {store_a} vs {store_b}")
    pools = psc[psc["store"].isin([store_a, store_b])]
    if pools.empty:
        st.info("Տվյալներ չկան համեմատության համար։")
        return
    fig = px.bar(pools, x="scenario", y="score_scenario_pct", color="store", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Բաժիններ (բոլոր սցենարներում):**")
    sec = ps[ps["store"].isin([store_a, store_b])]
    st.dataframe(sec.sort_values(["section","scenario","store"])[
        ["store","scenario","section","score_section_pct"]
    ].round(1), use_container_width=True)

def download_buttons(dfs: dict):
    for name, df in dfs.items():
        st.download_button(
            f"{name} — ներբեռնել CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{name}.csv",
            mime="text/csv"
        )
