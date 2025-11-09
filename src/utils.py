import pandas as pd
import altair as alt
from src.common import norm

# Робастный парсинг столбца даты визита (dd.mm.yy/yyy, Excel-serial, смешанные символы)
def _parse_visit_date(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(pd.NaT, index=[])
    s = series.copy()

    # Excel-serial (числа) -> дата
    if pd.api.types.is_numeric_dtype(s):
        base = pd.Timestamp("1899-12-30")
        return pd.to_datetime(base) + pd.to_timedelta(s.astype(float), unit="D")

    s = s.astype(str).str.strip()
    s = s.replace({"": None, "None": None})
    # унификация разделителей
    s = (s.str.replace("\u2024", ".", regex=False)     # one dot leader
           .str.replace("\u0589", ".", regex=False)    # Armenian :
           .str.replace("․", ".", regex=False)
           .str.replace("/", ".", regex=False)
           .str.replace("-", ".", regex=False))
    # оставим только допустимые символы
    s = s.str.replace(r"[^0-9.]", "", regex=True)

    # пробуем dd.mm.yyyy и dd.mm.yy (yy -> 20xx/19xx)
    ext = s.str.extract(r'(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{2,4})')
    d = pd.to_numeric(ext["d"], errors="coerce")
    m = pd.to_numeric(ext["m"], errors="coerce")
    y_raw = ext["y"].fillna("")
    y = pd.to_numeric(y_raw, errors="coerce")

    # 2-значный год: 00-29 -> 2000+, 30-99 -> 1900+
    y2_mask = y_raw.str.len() == 2
    y_adj = y.copy()
    y_adj[y2_mask & (y <= 29)] = 2000 + y_adj[y2_mask & (y <= 29)]
    y_adj[y2_mask & (y >= 30)] = 1900 + y_adj[y2_mask & (y >= 30)]

    dt = pd.to_datetime(pd.DataFrame({"year": y_adj, "month": m, "day": d}),
                        errors="coerce")
    return dt

def pick_col(df: pd.DataFrame, keys=None, contains=None):
    cols = list(df.columns)
    if keys:
        wanted = {norm(k) for k in keys}
        for c in cols:
            if norm(str(c)) in wanted:
                return c
    if contains:
        low = [norm(str(c)) for c in cols]
        for i, n in enumerate(low):
            if any(sub in n for sub in contains):
                return cols[i]
    return None

# Брендовая палитра
brand_colors = ["#529093", "#8497B0", "#AD8BA0", "#FFC000"]

def brand_theme():
    return {
        "config": {
            "range": {
                "category": brand_colors,
                "ordinal": brand_colors,
                "ramp": brand_colors,
                "diverging": brand_colors,
                "heatmap": brand_colors,
            },
            "background": "transparent",
            "mark": {"color": brand_colors[0]},
            "bar": {"cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
            "line": {"strokeWidth": 3},
            "axis": {
                "labelColor": "#1F2A37",
                "titleColor": "#1F2A37",
                "gridColor": "#E5E7EB",
                "domainColor": "#94A3B8",
                "labelLimit": 1000
            },
            "legend": {"labelColor": "#1F2A37", "titleColor": "#1F2A37"},
            "title": {"color": "#1F2A37", "fontSize": 18, "fontWeight": "bold"},
            "view": {"stroke": None}
        }
    }

# Универсальные фабрики графиков
def apply_brand(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(stroke=None)

def brand_bar_chart(df: pd.DataFrame, x: str, y: str, title: str | None = None,
                    color: str | None = None, tooltip: list[str] | None = None,
                    height: int = 340) -> alt.Chart:
    tooltip = tooltip or [x, y]
    return apply_brand(
        alt.Chart(df)
          .mark_bar(color=color or brand_colors[0])
          .encode(
              x=alt.X(x, sort=None, title=""),
              y=alt.Y(y, title=""),
              tooltip=tooltip
          )
          .properties(title=title, height=height)
    )

def brand_heatmap(df: pd.DataFrame, x: str, y: str, value: str,
                  title: str | None = None, value_title: str | None = None,
                  height: int = 420) -> alt.Chart:
    return apply_brand(
        alt.Chart(df)
          .mark_rect()
          .encode(
              x=alt.X(x, sort=None, title=""),
              y=alt.Y(y, sort=None, title=""),
              color=alt.Color(value, title=value_title or value,
                              scale=alt.Scale(range=brand_colors)),
              tooltip=[x, y, value]
          ).properties(title=title, height=height)
    )

def shorten_labels(series: pd.Series, max_len: int = 28) -> pd.Series:
    return series.apply(lambda s: s if len(str(s)) <= max_len else str(s)[:max_len-1] + "…")

def brand_hbar(df: pd.DataFrame, y: str, x: str, title: str | None = None,
               full_label_col: str | None = None, height: int | None = None) -> alt.Chart:
    rows = len(df)
    base_row_h = 22
    if height is None:
        height = min(900, max(320, rows * base_row_h))
    tooltip = [y, x]
    if full_label_col:
        tooltip.append(full_label_col)
    return apply_brand(
        alt.Chart(df)
          .mark_bar(color=brand_colors[0])
          .encode(
              y=alt.Y(y, sort="-x", title=""),
              x=alt.X(x, title=""),
              tooltip=tooltip
          ).properties(title=title, height=height)
    )

def _normalize_store_col(s: pd.Series) -> pd.Series:
    # NBSP/узкие пробелы -> обычный, схлопываем, убираем вокруг разделителей
    return (
        s.astype(str)
         .str.replace("\u00A0", " ", regex=False)   # NBSP
         .str.replace("\u2009", " ", regex=False)   # thin space
         .str.replace("\u202F", " ", regex=False)   # narrow NBSP
         .str.replace(r"\s+", " ", regex=True)
         # унифицируем разные дефисы
         .str.replace(r"[‐‑‒–—−]", "-", regex=True)
         # убираем пробелы вокруг разделителей: հայկական «՝», двոետочие, запятая, дефիս, слэш, вертикальная черта
         .str.replace(r"\s*([՝,:;|\-/])\s*", r"\1", regex=True)
         .str.strip()
    )