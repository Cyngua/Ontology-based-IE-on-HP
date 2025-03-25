'''
create tables for the clinical notes data mart
'''
# utils
import sqlite3
conn = sqlite3.connect("database.sqlite")

# Create Tables
def create_tables():
    with open("crud/create_tables.sql", "r") as f:
        conn.executescript(f.read())
    conn.close()

if __name__ == "__main__": 
    create_tables()