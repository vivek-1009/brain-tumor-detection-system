import csv
from io import StringIO
from flask import Flask, flash, make_response, render_template, request, send_from_directory, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np
import os
import datetime
import sqlite3
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models
model_classifier = load_model('models/balanced_mri_model.h5')  # Tumor classifier
model_binary = load_model('models/tumor_classifier_model.h5')  # MRI validator

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
class_label = {0: 'MRI SCAN', 1: 'NON MRI SCAN'}

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database init
def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY,
                        username TEXT,
                        timestamp TEXT,
                        result TEXT,
                        confidence TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
    conn.commit()
    conn.close()
  

def create_appointments_table():
    conn = sqlite3.connect('appointment.db')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_name TEXT NOT NULL,
            doctor_location TEXT NOT NULL,
            user_name TEXT NOT NULL,
            user_location TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Pending'
        )
    ''')
    conn.commit()
    conn.close()
create_appointments_table()

init_db()

# Language
LANG = {
    'en': {
        'title': 'MRI Tumor Detection System',
        'desc': 'Upload an MRI image to detect if there is a tumor and its type.',
        'select_image': 'Select MRI Image:',
        'btn': 'Upload and Detect',
        'confidence': 'Confidence',
        'location': 'Your Location',
        'doctors': 'Recommended Doctors'
    },
    'hi': {
        'title': 'एमआरआई ट्यूमर डिटेक्शन सिस्टम',
        'desc': 'एमआरआई इमेज अपलोड करें और ट्यूमर की जांच करें।',
        'select_image': 'एमआरआई इमेज चुनें:',
        'btn': 'अपलोड और डिटेक्ट करें',
        'confidence': 'विश्वास स्तर',
        'location': 'आपका स्थान',
        'doctors': 'अनुशंसित डॉक्टर'
    }
}
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resizing
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# MRI detection
mri_test_path = "static/MRI-Studies-of-the-Patients-Pituitary-Gland-MRI-of-the-pituitary-gland-showing-a.png"  # ✅ path to known MRI image
non_mri_test_path = "static/pexels-efrem-efre-2786187-31754122.jpg"  
def is_mri(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model_binary.predict(img_array)[0][0]
    print(f"[DEBUG] MRI Confidence: {prediction}")  # <= Add this line

    return prediction > 0.90 # Increase threshold to avoid false positives
# Run tests
print("✅ MRI test result:", is_mri(mri_test_path))         # Expect: True
print("❌ Non-MRI test result:", is_mri(non_mri_test_path)) # Expect: False
def predict_tumor(image_path, location):
    print("[DEBUG] Checking if image is MRI...")

    mri_result = is_mri(image_path)  # ✅ Call once and save
    print("[DEBUG] Is MRI:", mri_result)

    if not mri_result:
        print("[DEBUG] Not an MRI. Aborting prediction.")
        return None, None, "The uploaded image is not an MRI scan. Please upload a valid MRI image.", []

    print("[DEBUG] MRI confirmed. Continuing with tumor prediction.")

    # Continue with prediction if it's an MRI
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model_classifier.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class_index]
    doctors_db = {
            'glioma': {
                'Delhi': [
                    {
                        'name': "Dr. Rajesh Gupta",
                        'clinic': "Apollo Hospital",
                        'location': "Delhi",
                        'rating': 4,
                        'specialty': "Neurosurgeon",
                        'contact': "+91 9876543210"
                    },
                    {
                        'name': "Dr. Pooja Bhalla",
                        'clinic': "Fortis Hospital",
                        'location': "Delhi",
                        'rating': 5,
                        'specialty': "Neurologist",
                        'contact': "+91 9123456780"
                    }
                ],
                'Mumbai': [
                    {
                        'name': "Dr. Aditi Sharma",
                        'clinic': "Lilavati Hospital",
                        'location': "Mumbai",
                        'rating': 4.5,
                        'specialty': "Neurosurgeon",
                        'contact': "+91 9988776655"
                    },
                    {
                        'name': "Dr. Vikram Singh",
                        'clinic': "Kokilaben Hospital",
                        'location': "Mumbai",
                        'rating': 4.8,
                        'specialty': "Neurologist",
                        'contact': "+91 9988776655"
                    }
                ],
                'Chennai': [
                    {
                        'name': "Dr. Aditi Verma",
                        'clinic': "Lilavati Hospital",
                        'location': "Chennai",
                        'rating': 4.5,
                        'specialty': "Neurosurgeon",
                        'contact': "+91 9988776655"
                    },
                    {
                        'name': "Dr. Vikram Sharma",
                        'clinic': "Kokilaben Hospital",
                        'location': "Chennai",
                        'rating': 4.8,
                        'specialty': "Neurologist",
                        'contact': "+91 9988776655"
                    }
                ],
                'Bangalore': [
                    {
                        'name': "Dr. Aditya Vats",
                        'clinic': "Lilavati Hospital",
                        'location': "Bangalore",
                        'rating': 4.5,
                        'specialty': "Neurosurgeon",
                        'contact': "+91 9988776655"
                    },
                    {
                        'name': "Dr. Vikram Ahuja",
                        'clinic': "Kokilaben Hospital",
                        'location': "Bangalore",
                        'rating': 4.8,
                        'specialty': "Neurologist",
                        'contact': "+91 9988776655"
                    }
                ]
            },
            'meningioma': {
                'Delhi': [
                    {
                        'name': "Dr. Anjali Verma",
                        'clinic': "Max Hospital",
                        'location': "Delhi",
                        'rating': 4.2,
                        'specialty': "Neurosurgeon",
                        'contact': "+91 9988776655"
                    },
                    {
                        'name': "Dr. Suresh Kumar",
                        'clinic': "AIIMS",
                        'location': "Delhi",
                        'rating': 4.7,
                        'specialty': "Neurologist",
                        'contact': "+91 9988776655"
                    }
                ],
                'Mumbai': [
                    {
                        'name': "Dr. Neha Desai",
                        'clinic': "Jaslok Hospital",
                        'location': "Mumbai",
                        'rating': 4.6,
                        'specialty': "Neurosurgeon",
                        'contact': "+91 9988776655"
                    },
                    {
                        'name': "Dr. Rahul Mehta",
                        'clinic': "Siddhivinayak Hospital",
                        'location': "Mumbai",
                        'rating': 4.9,
                        'specialty': "Neurologist",
                        'contact': "+91 9988776655"
                    }
                ],
                'Chennai': [
                    {
                        'name': "Dr. Priya Nair",
                        'clinic': "Apollo Hospital",
                        'location': "Chennai",
                        'rating': 4.5,
                        'specialty': "Neurosurgeon",
                        'contact': "+91 9988776655"
                    },
                    {
                        'name': "Dr. Karthik Iyer",
                        'clinic': "Fortis Hospital",
                        'location': "Chennai",
                        'rating': 4.8,
                        'specialty': "Neurologist",
                        'contact': "+91 9988776655"
                    }
                ],      
                'Bangalore': [
                    {
                        'name': "Dr. Ramesh Rao",
                        'clinic': "Manipal Hospital",
                        'location': "Bangalore",
                        'rating': 4.5,
                        'specialty': "Neurosurgeon",
                        'contact': "+91 9988776655"
                    },
                    {
                        'name': "Dr. Kavita Rao",
                        'clinic': "Narayana Health",
                        'location': "Bangalore",
                        'rating': 4.8,
                        'specialty': "Neurologist",
                        'contact': "+91 9988776655"
                    }
                ]    
            },                 
             'pituitary': {
        'Delhi': [
            {
                'name': "Dr. Rakesh Nair",
                'clinic': "BLK Hospital",
                'location': "Delhi",
                'rating': 4.5,
                'specialty': "Endocrinologist",
                'contact': "+91 9876512340"
            },
            {
                'name': "Dr. Meera Saxena",
                'clinic': "AIIMS",
                'location': "Delhi",
                'rating': 4.7,
                'specialty': "Neurosurgeon",
                'contact': "+91 9812345678"
            }
        ],
        'Mumbai': [
            {
                'name': "Dr. Sanjay Kulkarni",
                'clinic': "Nanavati Hospital",
                'location': "Mumbai",
                'rating': 4.6,
                'specialty': "Endocrinologist",
                'contact': "+91 9988771122"
            },
            {
                'name': "Dr. Asha Patil",
                'clinic': "Wockhardt Hospital",
                'location': "Mumbai",
                'rating': 4.8,
                'specialty': "Neurosurgeon",
                'contact': "+91 9090909090"
            }
        ],
        'Chennai': [
            {
                'name': "Dr. Ravi Kumar",
                'clinic': "Apollo Hospital",
                'location': "Chennai",
                'rating': 4.5,
                'specialty': "Endocrinologist",
                'contact': "+91 9988776655"
            },
            {
                'name': "Dr. Priya Iyer",
                'clinic': "Fortis Hospital",
                'location': "Chennai",
                'rating': 4.8,
                'specialty': "Neurosurgeon",
                'contact': "+91 9988776655"
            }
        ],
        'Bangalore': [
            {
                'name': "Dr. Suresh Rao",
                'clinic': "Manipal Hospital",
                'location': "Bangalore",
                'rating': 4.5,
                'specialty': "Endocrinologist",
                'contact': "+91 9988776655"
            },
            {
                'name': "Dr. Kavitha Reddy",
                'clinic': "Narayana Health",
                'location': "Bangalore",
                'rating': 4.8,
                'specialty': "Neurosurgeon",
                'contact': "+91 9988776655"
            }
        ]
    }
}
        # Decision
    if predicted_label == 'notumor':
        message = "You are fully fit and fine. No tumor detected."
        doctor_list = []
    elif predicted_label in doctors_db:
        message = f"Tumor detected: {predicted_label.capitalize()}"
        doctor_list = doctors_db.get(predicted_label, {}).get(location, [])
    else:
        message = "Tumor detected but no doctor recommendations available for this type."
        doctor_list = [] 
    doctor_list = doctors_db.get(predicted_label, {}).get(location, [])

    return predicted_label, confidence_score, message, doctor_list



@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('loginc'))  # Make sure this is correct


    lang = request.args.get('lang', 'en')
    labels = LANG.get(lang, LANG['en'])
    result = confidence = message = None
    doctors = []
    file_path = ""

    if request.method == 'POST':
        file = request.files.get('file')
        location = request.form.get('location', 'Delhi')

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result, confidence, message, doctors = predict_tumor(file_path, location)

            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO history (username, timestamp, result, confidence) VALUES (?, ?, ?, ?)",
                           (session['username'], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            result, f"{confidence*100:.2f}"))
            conn.commit()
            conn.close()

            file_path = f'/uploads/{filename}'

    return render_template('index.html',
                           result=result,
                           confidence=f"{confidence*100:.2f}" if confidence else None,
                           message=message,
                           doctors=doctors,
                           file_path=file_path,
                           lang=lang,
                           labels=labels)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect('/login')
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error="Username already exists.")
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect('/')
        else:
            return render_template('index.html', error="Invalid credentials.")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history WHERE username = ?", (session['username'],))
    rows = cursor.fetchall()
    conn.close()
    history_data = rows
    return render_template('history.html', history=history_data)


@app.route('/download-history/pdf')
def download_history_pdf():
    if 'username' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history WHERE username = ?", (session['username'],))
    rows = cursor.fetchall()
    conn.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction History for {session['username']}", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(20, 10, txt="ID", border=1)
    pdf.cell(50, 10, txt="Timestamp", border=1)
    pdf.cell(50, 10, txt="Result", border=1)
    pdf.cell(30, 10, txt="Confidence", border=1)
    pdf.ln()

    for row in rows:
        pdf.cell(20, 10, txt=str(row[0]), border=1)
        pdf.cell(50, 10, txt=str(row[2]), border=1)
        pdf.cell(50, 10, txt=str(row[3]), border=1)
        pdf.cell(30, 10, txt=str(row[4]), border=1)
        pdf.ln()

    pdf_output = pdf.output(dest='S').encode('latin1')
    response_obj = make_response(pdf_output)
    response_obj.headers['Content-Type'] = 'application/pdf'
    response_obj.headers['Content-Disposition'] = f'attachment; filename=history_{session["username"]}.pdf'
    return response_obj


@app.route('/download-history/csv')
def download_history_csv():
    if 'username' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, result, confidence FROM history WHERE username = ?", (session['username'],))
    data = cursor.fetchall()
    conn.close()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Tumor Type', 'Confidence (%)'])
    writer.writerows(data)
    output.seek(0)

    response_obj = make_response(output.getvalue())
    response_obj.headers['Content-Disposition'] = f'attachment;filename=history_{session["username"]}.csv'
    response_obj.headers['Content-Type'] = 'text/csv'
    return response_obj


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/symptoms')
def symptoms():
    # You can also pass data if needed
    return render_template('symptoms.html')
@app.route('/loginc')
def loginc():
    return render_template('loginc.html')
@app.route('/doctorregister', methods=['GET', 'POST'])
def doctorregister(): 
    if request.method == 'POST':
        doctor_username = request.form['doctor_username']
        doctor_password = request.form['doctor_password']
        doctor_location = request.form['doctor_location']

        conn = sqlite3.connect('appointment.db')
        cur = conn.cursor()

        # Check if doctor already exists
        cur.execute("SELECT * FROM doctors WHERE doctor_username=?", (doctor_username,))
        existing_doctor = cur.fetchone()

        if existing_doctor:
            conn.close()
            return render_template('doctorregister.html', error="Doctor already registered.")

        # Insert new doctor
        cur.execute("INSERT INTO doctors (doctor_username,doctor_password,doctor_location) VALUES (?, ?, ?)",
                    (doctor_username,doctor_password,doctor_location))
        conn.commit()
        conn.close()

        return redirect('/doctorlogin')  # Redirect to doctor login after successful registration

    return render_template('doctorregister.html')

from flask import session, redirect, render_template, request



@app.route('/book_appointment')
def book_appointment():
    name = request.args.get('name')
    location = request.args.get('location')
    contact = request.args.get('contact')
    return render_template('book_appointment.html', name=name, location=location, contact=contact)
@app.route('/confirm-appointment', methods=['POST'])
@app.route('/confirm-appointment', methods=['POST'])
def confirm_appointment():
    doctor_name = request.form['doctor_name']
    doctor_location = request.form['doctor_location']
    # doctor_contact = request.form['doctor_contact']  # ❌ remove this
    user_name = request.form['user_name']
    user_location = request.form['user_location']
    date = request.form['date']
    time = request.form['time']

    # 🔸 Save to database (removed doctor_contact)
    conn = sqlite3.connect('appointment.db')
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO appointments 
        (doctor_name, doctor_location, user_name, user_location, date, time, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (doctor_name, doctor_location, user_name, user_location, date, time, 'Pending'))
    conn.commit()
    conn.close()

    return f"""
    <div style='text-align:center; margin-top:50px; font-family:sans-serif;'>
      <h2>✅ Appointment Confirmed!</h2>
      <p><strong>Doctor:</strong> {doctor_name} ({doctor_location})</p>
      <p><strong>Patient:</strong> {user_name} from {user_location}</p>
      <p><strong>Date & Time:</strong> {date} at {time}</p>
    </div>
    """



@app.route('/doctorlogin', methods=['GET', 'POST'])
def doctorlogin():
    if request.method == 'POST':
        doctor_username = request.form['doctor_username'].strip()
        doctor_password = request.form['doctor_password'].strip()
        doctor_location = request.form['doctor_location']

        conn = sqlite3.connect('appointment.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM doctors WHERE doctor_username=? AND doctor_password=? AND doctor_location=?",
                    (doctor_username, doctor_password, doctor_location))
        doctor = cur.fetchone()
        conn.close()

        if doctor:
            # Store doctor name + location in session
            session['doctor_name'] = doctor_username.strip()
            session['doctor_location'] = doctor_location.strip()
            return redirect('/appointments')
        else:
            return render_template('doctorlogin.html', error="Invalid credentials or location.")
    
    return render_template('doctorlogin.html')
@app.route('/appointments')
def appointments():
    if 'doctor_name' not in session or 'doctor_location' not in session:
        return redirect('/doctorlogin')

    print("SESSION doctor_name:", session['doctor_name'])
    print("SESSION doctor_location:", session['doctor_location'])

    doctor_name = session['doctor_name']
    doctor_location = session['doctor_location']

    conn = sqlite3.connect('appointment.db')
    cur = conn.cursor()
    cur.execute('''
        SELECT id, user_name, user_location, date, time, status 
        FROM appointments 
        WHERE doctor_name=? AND doctor_location=?
    ''', (doctor_name, doctor_location))
    appointments = cur.fetchall()
    conn.close()

    print("Fetched Appointments:", appointments)  # ✅ Check this

    return render_template('appointments.html', appointments=appointments, doctor=doctor_name)




@app.route('/confirm/<int:appointment_id>', methods=['POST'])
def confirm_appointment_status(appointment_id):
    conn = sqlite3.connect('appointment.db')
    cur = conn.cursor()
    cur.execute('UPDATE appointments SET status = ? WHERE id = ?', ('Confirmed', appointment_id))
    conn.commit()
    conn.close()
    return redirect('/appointments')

@app.route('/doctorlogout')
def doctorlogout():
    session.clear()
    return redirect('/doctorlogin')



if __name__ == '__main__':
    

    app.run(debug=True)