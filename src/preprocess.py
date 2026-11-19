import html
import re
import urllib.parse


def normalize(s: str) -> str:
    s = str(s)
    for _ in range(3):
        d = _unquote_u(urllib.parse.unquote_plus(s))
        if d == s:
            break
        s = d
    s = html.unescape(s)                       # &quot; &apos; -> " '
    # ký tự Unicode trông giống dấu nháy / gạch
    s = (s.replace("’", "'").replace("‘", "'").replace("′", "'").replace("＇", "'")
           .replace("“", '"').replace("”", '"')
           .replace("–", "--").replace("—", "--"))
    s = s.replace("\x00", "")                 # byte NUL chèn trước dấu nháy
    s = re.sub(r"%(?![0-9a-fA-F]{2})", "", s) # '%' lẻ chèn giữa ký tự (kiểu percentage)
    s = s.lower()
    s = re.sub(r"/\*.*?\*/", " ", s)           # comment inline -> khoảng trắng
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([^\w\s])\s*", r"\1", s)   # bỏ khoảng trắng quanh dấu câu
    return s.strip()
def _unquote_u(s: str) -> str:
    return re.sub(r"%u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)