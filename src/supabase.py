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


def get_latest_excel(sb, bucket: str) -> tuple[bytes, str]:
    """
    Находит и скачивает последний (или единственный) Excel файл из бакета.
    Возвращает (bytes, filename) или (None, None).
    """
    try:
        files = sb.storage.from_(bucket).list()
        # Фильтруем только excel файлы
        excels = [
            f for f in files 
            if f.get("name", "").lower().endswith((".xlsx", ".xls"))
        ]
        
        if not excels:
            return None, None
            
        # Сортируем по дате создания (created_at), берем последний
        # created_at обычно строка ISO
        excels.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        latest = excels[0]
        filename = latest["name"]
        
        # Скачиваем
        res = sb.storage.from_(bucket).download(filename)
        return res, filename
        
    except Exception as e:
        st.error(f"Ошибка при загрузке Excel из Supabase: {e}")
        return None, None

def replace_excel_file(sb, bucket: str, file_obj) -> bool:
    """
    Загружает новый Excel файл и удаляет все старые, чтобы остался ровно один.
    """
    try:
        # 1. Подготовка
        filename = file_obj.name
        # Простая санитарная обработка имени
        filename = filename.replace(" ", "_")
        
        data = file_obj.getvalue()
        # mime type
        mime = file_obj.type
        if not mime:
            if filename.lower().endswith(".xlsx"):
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif filename.lower().endswith(".xls"):
                mime = "application/vnd.ms-excel"
            else:
                mime = "application/octet-stream"

        # 2. Загрузка нового файла (upsert=true)
        _sb_upload(sb, bucket, filename, data, mime)
        
        # 3. Удаление старых файлов (всех Excel, кроме только что загруженного)
        # Используем _sb_list_all чтобы найти все файлы
        all_files = _sb_list_all(sb, bucket)
        to_delete = []
        for f in all_files:
            # _sb_list_all возвращает dict с ключом "name" (или full_path в нашей реализации)
            # В нашей реализации _sb_list_all возвращает плоский список, где name - это имя файла, 
            # а full_path - полный путь.
            # Но если bucket плоский, name достаточно.
            # Важно: Supabase storage list возвращает имена.
            
            fname = f.get("name")
            # Если это тот же файл, что мы только что загрузили — пропускаем
            if fname == filename:
                continue
                
            # Если это Excel — в список на удаление
            if fname and fname.lower().endswith((".xlsx", ".xls")):
                to_delete.append(fname)
        
        if to_delete:
            _sb_delete(sb, bucket, to_delete)
            
        return True
    except Exception as e:
        st.error(f"Ошибка при замене файла: {e}")
        return False