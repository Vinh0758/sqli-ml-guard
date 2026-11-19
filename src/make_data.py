import random
import pandas as pd
from preprocess import normalize

random.seed(42)
N_HARD = 1500

vn_ho = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi"]
vn_dem = ["Văn", "Thị", "Minh", "Quốc", "Thu", "Ngọc"]
vn_ten = ["An", "Bình", "Hoa", "Lan", "Dũng", "Hương", "Nam", "Linh"]
en_name = ["O'Brien", "D'Angelo", "O'Connor", "McDonald", "Smith", "Anna-Marie", "John", "Mary"]
streets = ["Nguyễn Trãi", "Quang Trung", "Lê Lợi", "Trần Phú", "Giải Phóng"]
districts = ["Hà Đông", "Cầu Giấy", "Thanh Xuân", "Hải Châu", "Quận 1"]
cities = ["Hà Nội", "Đà Nẵng", "TP.HCM", "Huế"]
prods = ["laptop 15\"", "iPhone 15", "áo thun trắng đen", "giày chạy bộ", "tai nghe bluetooth",
         "C++ programming book", "black & white t-shirt (size M)", "100% cotton shirt"]
mails = ["an", "binh.tran", "lan_pham", "nam99", "o'brien"]


def name():
    return random.choice([
        f"{random.choice(vn_ho)} {random.choice(vn_dem)} {random.choice(vn_ten)}",
        random.choice(en_name) + " " + random.choice(en_name),
    ])


def addr():
    return f"{random.randint(1, 300)} {random.choice(streets)}, {random.choice(districts)}, {random.choice(cities)}"


slots = {
    "name": name,
    "addr": addr,
    "prod": lambda: random.choice(prods),
    "email": lambda: f"{random.choice(mails)}{random.randint(1, 999)}@example.com",
    "phone": lambda: "09" + "".join(random.choice("0123456789") for _ in range(8)),
    "n": lambda: str(random.randint(1, 5000)),
    "code": lambda: "SALE-" + str(random.randint(2020, 2026)) + "-" + str(random.randint(10, 99)),
    "date": lambda: f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(2020, 2026)}",
}

templates = [
    "{name}", "{addr}", "{email}", "{phone}", "{date}", "{prod}",
    "tìm {prod} giá dưới {n}k", "search: {prod}", "where can i buy {prod}",
    "please select an option from the list", "select your plan: basic or premium",
    "union of european countries", "how to drop a class in university",
    "insert coupon code {code}", "update my profile picture {n}", "delete my account please",
    "order by price ascending", "order #{n} - {name}",
    "it's a great product; i love it -- highly recommend", "{name}'s order is delayed",
    "price > {n} and rating >= 4", "C++ / C# tutorial", "100% cotton # bestseller",
    "Tôi muốn chọn (select) size {n} cho {name}", "Giao hàng đến {addr}, gọi {phone}",
    "what's up? it's me, {name}", "1 = 1 is true in math class", "{n} + {n} = {n}",
    "how to write a select statement in sql", "difference between union and union all",
    "my username is {name} and my password hint is 'first pet'",
    "Cảm ơn bạn! Sản phẩm rất tốt - sẽ ủng hộ tiếp (5 sao)", "Liên hệ: {email} hoặc {phone}",
    "{prod} OR {prod}", "{name} AND {name}", "comment: don't forget; call me -- {phone}",
]

hard = set()
tries = 0
while len(hard) < N_HARD and tries < N_HARD * 20:
    tries += 1
    t = random.choice(templates)
    hard.add(t.format(**{k: f() for k, f in slots.items()}))

# --- Dữ liệu Kaggle ---
raw = pd.read_csv("data/raw/sqli_utf8.csv")
raw["Label"] = pd.to_numeric(raw["Label"], errors="coerce")
raw = raw.dropna(subset=["Sentence", "Label"]).astype({"Label": int})
raw["source"] = "kaggle"

extra = pd.DataFrame({"Sentence": sorted(hard), "Label": 0, "source": "synthetic_benign"})
df = pd.concat([raw[["Sentence", "Label", "source"]], extra], ignore_index=True)

# loại trùng SAU khi chuẩn hóa để tránh rò rỉ train/test
df["norm"] = df["Sentence"].map(normalize)
before = len(df)
df = df.drop_duplicates(subset="norm")
df = df[df["norm"].str.len() > 0]
df = df[df["Sentence"].str.len() <= 2000].drop(columns="norm")
print(f"Loại {before - len(df)} mẫu trùng/rỗng/quá dài")

df.to_csv("data/processed/dataset.csv", index=False, encoding="utf-8")
print(df.groupby(["source", "Label"]).size())