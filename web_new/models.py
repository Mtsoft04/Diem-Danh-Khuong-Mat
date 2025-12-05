# web_new/models.py
import sqlite3

DB_PATH = "../database.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_users():
    conn = get_db()
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)')
    conn.execute('INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)', ('admin', '123456'))
    conn.commit()
    conn.close()

init_users()