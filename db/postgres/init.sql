CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    price NUMERIC(10,2) NOT NULL
);
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    password TEXT NOT NULL
);
INSERT INTO products (name, price) VALUES
  ('Laptop', 900), ('Phone', 500), ('Mouse', 20), ('Keyboard', 45), ('Monitor', 220);
INSERT INTO users (username, password) VALUES
  ('admin', 'fake_admin_pw_lab'), ('alice', 'fake_alice_pw_lab'), ('bob', 'fake_bob_pw_lab');