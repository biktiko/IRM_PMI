import json
import mimetypes
import uuid
from datetime import datetime, date
from pathlib import Path
import streamlit as st

from src.supabase import _sb_public_url, _sb_upload, _sb_list_all, _sb_delete

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg"}

def _sidecar(path: str) -> str:
    return f"{path}.meta.json"

def _load_meta(sb, bucket: str, path: str, default_title: str):
    meta = {"title": default_title, "description": "", "saved_at": None}
    try:
        raw = sb.storage.from_(bucket).download(_sidecar(path))
        if raw:
            obj = json.loads(raw.decode("utf-8"))
            meta["title"] = obj.get("title", meta["title"])
            meta["description"] = obj.get("description", meta["description"])
            meta["saved_at"] = obj.get("saved_at")
    except Exception:
        pass
    return meta

def _save_meta(sb, bucket: str, path: str, title: str, description: str):
    payload = json.dumps({
        "title": title,
        "description": description,
        "saved_at": datetime.utcnow().isoformat()  # UTC
    })
    _sb_upload(sb, bucket, _sidecar(path), payload.encode("utf-8"), "application/json")

def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # поддержка "...Z" и без таймзоны
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def _get_stage_and_category(title: str):
    title = title.strip()
    stage = None
    category = None
    
    # Stage 1
    if title.startswith("BR"):
        stage = "1ին փուլ"
        for cat in ["BR1", "BR2", "BR3", "BR4", "BR5"]:
            if title.startswith(cat):
                category = cat
                break
            
    # Stage 2
    else:
        # Stage 2 prefixes (sorted by length desc)
        stage2_cats = ["LAS_SFP&CC", "LAS_SA", "LAS_SFP", "HS_YAP", "SA_YAP"]
        for cat in stage2_cats:
            if title.startswith(cat):
                stage = "2րդ փուլ"
                category = cat
                break
                
    return stage, category

def render_audio_tab(sb, bucket: str):
    st.markdown("""
    <style>
      .audio-card{
        border:2px solid #d1d5db;
        border-radius:16px;
        padding:16px 20px;
        margin:18px 0;
        background:#fff;
        box-shadow:0 2px 4px rgba(0,0,0,0.05);
      }
      .audio-title{font-weight:600;font-size:16px;margin:0 0 4px;}
      .audio-date{font-size:12px;color:#4b5563;margin:0 0 6px;}
      .audio-meta{font-size:12px;color:#6b7280;margin:0 0 10px;}
      .audio-desc{font-size:13px;color:#374151;margin:0 0 12px;}
      .audio-actions{margin-top:8px;}
      .audio-card audio{width:100%;outline:none;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("#### Վերբեռնում")
    col_f, col_r = st.columns([2, 3])
    with col_f:
        upl = st.file_uploader("Ֆայլ", type=["mp3", "wav", "m4a", "ogg"], accept_multiple_files=False, key="audio_upl")
    with col_r:
        title_in = st.text_input("Վերնագիր", value="", key="audio_title")
        desc_in = st.text_area("Նկարագրություն", value="", height=80, key="audio_desc")
        do_upload = st.button("Վերբեռնել", type="primary", key="audio_upload_btn")

    # Загружаем в Supabase только по кнопке
    if do_upload:
        if upl is None:
            st.warning("Ընտրեք ֆայլը, հետո սեղմեք «Վերբեռնել».")
        elif upl.size > 200 * 1024 * 1024:
            st.error("Ֆայլը մեծ է (>200MB)")
        else:
            raw = upl.read()
            mime = mimetypes.guess_type(upl.name)[0] or "audio/mpeg"
            safe = upl.name.replace(" ", "_")
            path = f"{datetime.utcnow():%Y%m%d}/{uuid.uuid4().hex}_{safe}"
            try:
                _sb_upload(sb, bucket, path, raw, mime)  # аудио
                _save_meta(sb, bucket, path, title_in or Path(safe).stem, desc_in or "")  # sidecar JSON с saved_at
                st.success("Բեռնվել է")
                st.rerun()
            except Exception as e:
                st.error(f"Չստացվեց վերբեռնել: {e}")

    st.markdown("#### Ֆայլերի ցուցակ")

    # Листинг только аудио
    raw_files = _sb_list_all(sb, bucket)
    items = []
    for f in raw_files:
        path = f.get("full_path") or ""
        name = Path(path).name
        if not path or name.startswith("."):
            continue
        if Path(name).suffix.lower() not in AUDIO_EXTS:
            continue
        items.append(f)

    if not items:
        st.info("Ֆայլեր դեռ չկան։")
        return

    # Сначала соберём записи с метаданными, чтобы отфильтровать по дате
    records = []
    for f in items:
        path = f.get("full_path")
        filename = Path(path).name
        meta = _load_meta(sb, bucket, path, default_title=filename)
        saved_dt = _parse_iso(meta.get("saved_at"))
        # fallback: updated_at из storage (UTC)
        if saved_dt is None:
            upd = f.get("updated_at")
            saved_dt = _parse_iso(upd) or datetime.utcnow()
        size = (f.get("metadata") or {}).get("size")
        stg, cat = _get_stage_and_category(meta["title"])
        records.append({
            "path": path,
            "filename": filename,
            "title": meta["title"],
            "description": meta["description"],
            "saved_dt": saved_dt,
            "size": size,
            "updated_at": f.get("updated_at"),
            "stage": stg,
            "category": cat
        })

    # Панель поиска + фильтр по дате
    st.markdown("### Ֆիլտրեր")
    q = st.text_input("Փնտրել ըստ վերնագրի կամ անվանման", "", key="audio_search")
    
    col_stg, col_cat, col_d1, col_d2 = st.columns(4)
    with col_stg:
        stage_options = ["Բոլորը", "1ին փուլ", "2րդ փուլ"]
        selected_stage = st.selectbox("Փուլ", stage_options, key="filter_stage")
    with col_cat:
        cat_options = ["Բոլորը"]
        if selected_stage == "1ին փուլ":
            cat_options += ["BR1", "BR2", "BR3", "BR4", "BR5"]
        elif selected_stage == "2րդ փուլ":
            cat_options += ["LAS_SA", "LAS_SFP", "LAS_SFP&CC", "HS_YAP", "SA_YAP"]
        else:
            cat_options += ["BR1", "BR2", "BR3", "BR4", "BR5", "LAS_SA", "LAS_SFP", "LAS_SFP&CC", "HS_YAP", "SA_YAP"]
            
        selected_category = st.selectbox("Կատեգորիա", cat_options, key="filter_category")
    min_dt = min(r["saved_dt"].date() for r in records)
    max_dt = max(r["saved_dt"].date() for r in records)
    with col_d1:
        from_date = st.date_input("Սկիզբ", value=min_dt, min_value=min_dt, max_value=max_dt, key="audio_from")
    with col_d2:
        to_date = st.date_input("Վերջ", value=max_dt, min_value=min_dt, max_value=max_dt, key="audio_to")

    def _pass_filters(r):
        if selected_stage != "Բոլորը" and r["stage"] != selected_stage:
            return False
        if selected_category != "Բոլորը" and r["category"] != selected_category:
            return False
        if q:
            if q.lower() not in r["title"].lower() and q.lower() not in r["filename"].lower():
                return False
        d = r["saved_dt"].date()
        return (from_date <= d <= to_date)

    view = [r for r in records if _pass_filters(r)]
    st.caption(f"Ցուցադրված է {len(view)} / {len(records)} աուդիո")
    if not view:
        st.info("Ոչինչ չի գտնվել ըստ ֆիլտրերի։")
        return

    # Сортировка по дате добавления (новые сверху)
    view.sort(key=lambda r: r["saved_dt"], reverse=True)

    for r in view:
        path = r["path"]
        filename = r["filename"]
        title = r["title"]
        desc = r["description"]
        size_kb = f"{int(r['size'])//1024} KB" if isinstance(r["size"], (int, float)) else ""
        added_human = r["saved_dt"].strftime("%Y-%m-%d %H:%M UTC")
        url = _sb_public_url(sb, bucket, path)

        # Build full card HTML (вся карточка в одном HTML чтобы рамка охватывала и плеер)
        audio_tag = f"<audio controls src='{url}'></audio>" if url else ""
        card_html = f"""
        <div class="audio-card">
          <div class="audio-title">{title}</div>
          <div class="audio-date">Ավելացվել է՝ {added_human}</div>
          <div class="audio-meta">{filename} · {size_kb}</div>
          {f'<div class="audio-desc">{desc}</div>' if desc else ''}
          {audio_tag}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Actions (delete + edit) вне HTML, но визуально внутри благодаря margin
        act_c1, act_c2 = st.columns([1,5])
        with act_c1:
            if st.button("🗑️ Ջնջել", key=f"del_{path}"):
                if _sb_delete(sb, bucket, [path, _sidecar(path)]):
                    st.success("Ջնջված է")
                    st.rerun()
        with act_c2:
            with st.expander("Խմբագրել", expanded=False):
                new_title = st.text_input("Վերնագիր", value=title, key=f"title_{path}")
                new_desc = st.text_area("Նկարագրություն", value=desc, height=80, key=f"desc_{path}")
                if st.button("Պահպանել", key=f"save_{path}"):
                    try:
                        _save_meta(sb, bucket, path, new_title.strip(), new_desc.strip())
                        st.success("Պահպանված է")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Չստացվեց պահել: {e}")