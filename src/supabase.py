# NEW: инициализация Supabase
import streamlit as st
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None


def _sb_client() -> tuple:
    """
    Возвращает (client, bucket).
    Требует supabase>=2.6.0 в requirements.txt.
    """
    url = st.secrets.get("SUPABASE_URL", "https://gfokwqubpzpnvsjhyhex.supabase.co")
    key = st.secrets.get("SUPABASE_ANON_KEY", "")
    bucket = st.secrets.get("SUPABASE_BUCKET", "audio")
    if not url or not key:
        st.error("Нет Supabase URL / KEY в secrets.toml")
        return None, bucket
    if create_client is None:
        st.error("Нет пакета supabase. Добавьте в requirements.txt: supabase>=2.6.0")
        return None, bucket
    return create_client(url, key), bucket

def _sb_public_url(sb, bucket: str, path: str) -> str:
    """
    Возвращает публичный URL. Если SDK не отдает словарь с ключом publicUrl,
    формируем ձեռքով:
    """
    try:
        res = sb.storage.from_(bucket).get_public_url(path)
        # В supabase-py v2 обычно dict с 'publicUrl'
        if isinstance(res, dict):
            return (
                res.get("publicUrl")
                or res.get("publicURL")
                or res.get("data", {}).get("publicUrl")
                or ""
            )
        # Иногда возвращает просто строку
        if isinstance(res, str):
            return res
    except Exception:
        pass
    # Фолбэк: ручная сборка
    base = st.secrets.get("SUPABASE_URL", "").rstrip("/")
    if base:
        return f"{base}/storage/v1/object/public/{bucket}/{path}"
    return ""

def _sb_upload(sb, bucket: str, path: str, data: bytes, mime: str):
    """
    Правильный upload: файл + headers (file_options) с x-upsert='true'.
    """
    return sb.storage.from_(bucket).upload(
        path,
        data,
        {
            "content-type": mime,
            "x-upsert": "true"  # строка, не bool
        }
    )

def _sb_delete(sb, bucket: str, paths: list[str]):
    try:
        sb.storage.from_(bucket).remove(paths)
        return True
    except Exception as e:
        st.error(f"Удалить не удалось: {e}")
        return False

def _sb_list_all(sb, bucket: str) -> list[dict]:
    """
    Возвращает плоский список файлов, проходя по подпапкам (датам).
    """
    out = []
    try:
        root = sb.storage.from_(bucket).list(path="")
    except Exception as e:
        st.error(f"List error: {e}")
        return out
    if not root:
        return out
    for obj in root:
        name = obj.get("name")
        if not name:
            continue
        # Если это каталог (нет точки), обходим его
        if "." not in name:
            try:
                sub = sb.storage.from_(bucket).list(path=name)
                for f in sub:
                    fn = f.get("name")
                    if fn:
                        full = f"{name}/{fn}"
                        f["full_path"] = full
                        out.append(f)
            except Exception:
                continue
        else:
            obj["full_path"] = name
            out.append(obj)
    return out