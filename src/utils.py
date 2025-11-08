import pandas as pd
from src.common import norm

def _parse_visit_date(series: pd.Series) -> pd.Series:
    """
    Парсинг столбца даты визита:
    - dd.mm.yy или dd.mm.yyyy (армянские разделители '․' U+2024 и '։' U+0589 тоже)
    - Может содержать лишние символы вокруг
    - Числовые значения (Excel serial) -> преобразование
    Правило двухзначного года: 00–29 -> 2000+, 30–99 -> 1900+.
    """
    if series is None:
        return pd.Series(pd.NaT, index=[0])
    # Если числовой тип (Excel serial)
    if pd.api.types.is_numeric_dtype(series):
        base = pd.Timestamp("1899-12-30")
        return series.apply(
            lambda x: (base + pd.Timedelta(days=float(x))) if pd.notna(x) else pd.NaT
        )

    s = series.astype(str).str.strip()
    # нормализуем нестандартные точки
    s = (s.str.replace("\u2024", ".", regex=False)
           .str.replace("\u0589", ".", regex=False)
           .str.replace("․", ".", regex=False))
    # вырезаем всё кроме цифр и ./-
    s = s.str.replace(r"[^0-9./\-]", "", regex=True)

    ext = s.str.extract(r'(?P<d>\d{1,2})[.\-/](?P<m>\d{1,2})[.\-/](?P<y>\d{2,4})')
    d = pd.to_numeric(ext["d"], errors="coerce")
    m = pd.to_numeric(ext["m"], errors="coerce")
    y_raw = ext["y"]
    y = pd.to_numeric(y_raw, errors="coerce")

    # двухзначный год
    mask2 = y_raw.str.len() == 2
    y_adj = y.copy()
    y_adj[mask2 & (y <= 29)] = 2000 + y_adj[mask2 & (y <= 29)]
    y_adj[mask2 & (y >= 30)] = 1900 + y_adj[mask2 & (y >= 30)]
    dt = pd.to_datetime(
        pd.DataFrame({"year": y_adj, "month": m, "day": d}),
        errors="coerce"
    )
    return dt

def pick_col(df: pd.DataFrame, keys=None, contains=None):
    cols = list(df.columns)
    if keys:
        wanted = {norm(k) for k in keys}
        for c in cols:
            if norm(str(c)) in wanted:
                return c
    if contains:
        for c in cols:
            n = norm(str(c))
            if any(sub in n for sub in contains):
                return c
    return None