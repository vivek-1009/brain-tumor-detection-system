import sqlite3

conn = sqlite3.connect('appointment.db')
cur = conn.cursor()

# Create doctors table
cur.execute('''
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_username TEXT NOT NULL UNIQUE,
        doctor_password TEXT NOT NULL,
        doctor_location TEXT NOT NULL
    )
''')

conn.commit()
conn.close()
print("Table 'doctors' created successfully.")
