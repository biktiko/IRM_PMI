
import re

YES_ALIASES = {
    "այո","այո՛","да","yes","y","true","1","ուն","ուն.","կա","есть","հա","haa","ayo"
}
NO_ALIASES = {
    "ոչ","no","нет","false","0","չունի","չկա","չուն.","չէ"
}

def norm(s):
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s, flags=re.M).strip().lower()
    return s.replace("՝","'").replace("․",".")

def to_binary(answer):
    s = norm(answer)
    if s in YES_ALIASES or s.startswith(("այո","да","yes")):
        return 1.0
    if s in NO_ALIASES or s.startswith(("ոչ","нет","no")):
        return 0.0
    return None

def excel_col_to_idx(col):
    """Excel column name (e.g., 'AV') -> zero-based index"""
    col = col.strip().upper()
    n = 0
    for ch in col:
        if not ('A' <= ch <= 'Z'):
            continue
        n = n*26 + (ord(ch)-ord('A')+1)
    return n-1
