import random
import string
import pandas as pd

random.seed(7)


def rs(n=4):
    return "".join(random.choice(string.ascii_letters) for _ in range(n))


def rn():
    return str(random.randint(1000, 9999))


# ---------- Tấn công: sinh theo khuôn cú pháp ----------
def attack():
    a, b, w1, w2 = rn(), rn(), rs(), rs()
    pre = random.choice(["", "'", "\"", "')", "%'", "1'", "1) ", "admin'"])
    end = random.choice(["", "--", "-- ", "#"])
    conj = random.choice(["AND", "OR", "and", "or"])
    t = random.choice([
        f"{pre} {conj} {a}={a} {conj} '{w1}'='{w1}",
        f"{pre} {conj} {a}={b} {conj} '{w1}'='{w2}",
        f"{pre} {conj} ('{w1}' LIKE '{w1}",
        f"{pre} {conj} {a}={a} {conj} ({b}={b}",
        f"{pre} {conj} '{w1}'='{w1}' -- ",
        f"{pre} {conj} 1=1 #",
        f"{pre} HAVING {random.randint(0,1)}={random.randint(0,1)}",
        f"{pre} {conj} a='a",
        f"{pre} {conj} (SELECT {a} FROM (SELECT(SLEEP({random.randint(1,9)})))x)",
        f"{pre} {conj} {a}=CAST((CHR(1)||CHR(2))||(SELECT 1)::text AS NUMERIC)",
        f"{pre}; IF({a}={a}) SELECT {a} ELSE DROP FUNCTION {w1}--",
        f"{pre} {conj} {a}=CONVERT(INT,(SELECT CHAR(88)))--",
        f"{pre}+benchmark({random.randint(1000,9999)},SHA1(1))+'",
        f"{pre} {conj} {a}=LIKE('{w1}',UPPER(HEX(RANDOMBLOB({b}))))",
        # ORDER BY kèm hàm trễ (không chỉ ORDER BY trần)
        f"{pre} ORDER BY {random.randint(1,30)},SLEEP({random.randint(1,9)}){end}",
        f"{pre} ORDER BY SLEEP({random.randint(1,9)}){end}",
        f"{pre} {conj} {a}={a} ORDER BY {random.randint(1,30)}{end}",
                f"{pre} {conj} (CASE WHEN {a}={b} THEN {a} ELSE NULL END)",
        f"(CASE WHEN {a}={a} THEN {a} ELSE NULL END){end}",
        f"{pre} {conj} {a}=(CASE WHEN ({a}={a}) THEN {a} ELSE 1/(SELECT 0) END)",
    ])
    return t.strip()


# ---------- Benign: từ khóa SQL lẫn trong văn thường ----------
benign_t = [
    "SELECT một món từ danh sách {x}", "Select {x} từ menu bên dưới", "select {x} to continue",
    "Chọn (SELECT) {x} hoặc {y}", "DROP một tin nhắn cho {x}", "Drop {x} off at the station",
    "UNION {x} và {y} họp lúc 9h", "Update {x}: giao hàng chậm", "DELETE ảnh cũ của {x}",
    "Insert {x} into the box", "Where is {x}? gần {y}", "order by {x} hoặc {y}",
    "{x} AND {y}", "{x} OR {y}", "{x} and {y} -- cảm ơn", "{x}; {y}",
    "Create {x} for {y}", "Grant {x} a wish", "Exec {x} meeting at 3pm",
    # và/hoặc, so sánh số, toán học thường ngày
    "{x} and/or {y}", "{x} và/hoặc {y}", "{n} or {m} guests are coming",
    "{n} hoặc {m} người đi cùng {x}", "{n} or {m} cái cũng được", "if a > b and b > c then a > c",
    "x < {n} and y > {m}", "{n} + {m} = {k}", "{k} = {n} - {m}", "{n} > {m} or {m} > {n}",
    "tuổi từ {n} đến {m} and above", "{x} or {y}? tùy bạn", "yes or no; {x} and {y}",
    "Giá {n}k or {m}k tùy size {x}", "{n}x{m}cm; {x} and {y}", "đặt hàng (order by) ngày {n}",
    "sort by price: {n} to {m}", "{x}: {n}=ok, {m}=no",
]
words = ["An", "Bình", "size M", "đơn hàng", "bàn 4", "cà phê", "trà sữa", "O'Hara", "menu",
         "gói cước", "phòng 305", "áo thun", "giày nam", "Hà Nội", "kế hoạch Q4", "Lan", "Minh",
         "ca sáng", "sách", "vé xe"]


def benign():
    n, m, k = random.randint(1, 99), random.randint(1, 99), random.randint(1, 200)
    return random.choice(benign_t).format(x=random.choice(words), y=random.choice(words), n=n, m=m, k=k)


N_ATT, N_BEN = 450, 600
att = sorted({attack() for _ in range(N_ATT * 4)})[:N_ATT]
ben = sorted({benign() for _ in range(N_BEN * 6)})[:N_BEN]

aug = pd.DataFrame({"Sentence": att + ben,
                    "Label": [1] * len(att) + [0] * len(ben),
                    "source": ["synthetic_attack"] * len(att) + ["synthetic_benign2"] * len(ben)})
aug.to_csv("data/processed/augment.csv", index=False, encoding="utf-8")
print(aug.groupby(["source", "Label"]).size())
print(aug.sample(8, random_state=1).to_string())