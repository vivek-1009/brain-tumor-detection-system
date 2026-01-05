import sqlite3

conn = sqlite3.connect('appointment.db')
cur = conn.cursor()
cur.execute("SELECT doctor_name, doctor_location FROM appointments")
rows = cur.fetchall()
for row in rows:
    print(row)
conn.close()
