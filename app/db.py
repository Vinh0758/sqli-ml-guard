import os
import sqlite3

BACKEND = os.getenv("DB_BACKEND", "sqlite")      # sqlite | postgres
SQLITE_PATH = os.getenv("SQLITE_PATH", "demo.db")
PG_PORT = int(os.getenv("PG_PORT", "5433"))       # cổng PostgreSQL trên máy bạn
PH = "?" if BACKEND == "sqlite" else "%s"         # ký hiệu tham số mỗi DB một khác


def init_sqlite():
    con = sqlite3.connect(SQLITE_PATH)
    con.executescript("""
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS users;
        CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, price REAL);
        CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT, password TEXT);
        INSERT INTO products(name, price) VALUES
          ('Laptop',900),('Phone',500),('Mouse',20),('Keyboard',45),('Monitor',220);
        INSERT INTO users(username, password) VALUES
          ('admin','fake_admin_pw_lab'),('alice','fake_alice_pw_lab'),('bob','fake_bob_pw_lab');
    """)
    con.commit()
    con.close()


def get_conn():
    if BACKEND == "sqlite":
        return sqlite3.connect(SQLITE_PATH)
    if BACKEND == "postgres":
        import psycopg2
        return psycopg2.connect(host="127.0.0.1", port=PG_PORT, dbname="demo",
                                user="lab", password="lab_only_password")
    raise ValueError(f"DB_BACKEND không hợp lệ: {BACKEND}")


def run(sql, params=None):
    """Chạy một câu SQL, trả về danh sách dòng.
    params=None nghĩa là SQL đã bị nối chuỗi sẵn (dùng cho endpoint có lỗ hổng)."""
    con = get_conn()
    try:
        cur = con.cursor()
        if params is None:
            cur.execute(sql)
        else:
            cur.execute(sql, params)
        rows = cur.fetchall()
        return [[float(v) if hasattr(v, "as_tuple") else v for v in r] for r in rows]
    finally:
        con.close()


if BACKEND == "sqlite":
    init_sqlite()