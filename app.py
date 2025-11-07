import streamlit as st

st.set_page_config(page_title="PMI Test", layout="wide")
st.write("BOOT")

try:
    from src.data_loader import read_excel_all, build_weights_dict, questions_from_weights
    from src.scoring import long_from_base, join_weights, aggregate_scores, extract_10pt_rating
    import src.scoring as scoring
    import src.ui as ui
    st.write("Imports OK")
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

st.info("Загрузите файл слева.")
upl = st.sidebar.file_uploader("Excel", type=["xlsx","xls"])
if upl:
    st.write("Start parse...")
    from pathlib import Path
    from tempfile import gettempdir
    p = Path(gettempdir()) / upl.name
    p.write_bytes(upl.getvalue())
    try:
        bases_raw, weights_raw = read_excel_all(p)
        st.success(f"Bases: {list(bases_raw.keys())}, weights rows: {sum(len(v) for v in weights_raw.values())}")
    except Exception as e:
        st.error(f"Parse error: {e}")