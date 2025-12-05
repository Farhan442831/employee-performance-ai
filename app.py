# app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
# Import Keras/TensorFlow
from tensorflow.keras.models import load_model 
# app.py (di bagian import)

# Import Keras/TensorFlow
from tensorflow.keras.models import load_model 
# Tambahkan baris ini untuk mendapatkan fungsi metrics dan loss
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError

# Import komponen untuk Fuzzy Logic
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- 1. Konfigurasi Awal dan Load Assets ---
app = Flask(__name__)

# app.py, GANTI BARIS INI
# Load Model ANN, Scaler, dan Daftar Fitur
# app.py, GANTI DENGAN KODE INI
try:
    # Definisikan objek custom untuk mengatasi konflik nama 'mse' dan 'mae'
    custom_objects = {
        'mse': MeanSquaredError(), 
        'mae': MeanAbsoluteError()
    }
    
    ann_model = load_model('ann_salary_model.h5', custom_objects=custom_objects)
    scaler = joblib.load('scaler_for_ann.pkl')
    ann_features = joblib.load('ann_model_features.pkl')
    print("Model ANN dan aset berhasil dimuat.")
# ...
    scaler = joblib.load('scaler_for_ann.pkl')
    ann_features = joblib.load('ann_model_features.pkl')
    print("Model ANN dan aset berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model ANN atau aset. Pastikan file ada di sini: {e}")
    # Jika gagal load, aplikasi tidak bisa berjalan
    exit()
# --- 2. Fungsi Fuzzy Logic (Fuzzy Inference System) ---
def predict_fuzzy_salary(age_input, rating_input):
    # Mendefinisikan Variabel Linguistik (Input dan Output)
    age = ctrl.Antecedent(np.arange(20, 71, 1), 'age')
    rating = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'rating')
    salary = ctrl.Consequent(np.arange(40, 200, 1), 'salary')

    # Fungsi Keanggotaan (Membership Functions)
    age['young'] = fuzz.trimf(age.universe, [20, 20, 35])
    age['mid'] = fuzz.trimf(age.universe, [30, 45, 60])
    age['old'] = fuzz.trimf(age.universe, [55, 70, 70])

    rating['low'] = fuzz.trimf(rating.universe, [0, 0, 2.5])
    rating['medium'] = fuzz.trimf(rating.universe, [2, 3.5, 4.5])
    rating['high'] = fuzz.trimf(rating.universe, [4, 5, 5])
    
    salary['low'] = fuzz.trimf(salary.universe, [40, 40, 80])
    salary['medium'] = fuzz.trimf(salary.universe, [70, 110, 150])
    salary['high'] = fuzz.trimf(salary.universe, [140, 200, 200])

    # Rules (Aturan IF-THEN)
    rule1 = ctrl.Rule(age['young'] & rating['low'], salary['low'])
    rule2 = ctrl.Rule(age['mid'] & rating['medium'], salary['medium'])
    rule3 = ctrl.Rule(age['old'] & rating['high'], salary['high'])
    rule4 = ctrl.Rule(age['young'] & rating['high'], salary['medium'])
    rule5 = ctrl.Rule(age['old'] & rating['low'], salary['medium'])
    rule6 = ctrl.Rule(age['mid'] & rating['low'], salary['low'])
    rule7 = ctrl.Rule(age['mid'] & rating['high'], salary['high'])

    salary_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
    salary_prediction = ctrl.ControlSystemSimulation(salary_ctrl)

    try:
        salary_prediction.input['age'] = age_input
        salary_prediction.input['rating'] = rating_input
        salary_prediction.compute()
        return salary_prediction.output['salary']
    except ValueError:
        return np.nan 

# --- 3. Routing Flask ---

@app.route('/')
def home():
    # Menampilkan form input
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil Input dari Form (harus sama dengan nama 'name' di index.html)
        age = float(request.form['age'])
        rating = float(request.form['rating'])
        # Asumsi: input 'python_yn' dan 'Location_NY' (contoh kolom dummy) tersedia di form
        python_yn = float(request.form['python_yn']) 
        loc_ny = float(request.form.get('Location_NY', 0)) # Ambil nilai, default 0 jika tidak ada

        # 2. Persiapan Dataframe untuk ANN
        # Buat dataframe 1 baris dengan kolom sesuai ann_features (PENTING: urutan harus sama!)
        input_data = pd.DataFrame(0, index=[0], columns=ann_features)
        
        # Masukkan nilai input yang diambil
        input_data['age'] = age
        input_data['Rating'] = rating
        input_data['python_yn'] = python_yn
        
        # Contoh input untuk kolom dummy hasil One-Hot Encoding
        # Kamu harus menyesuaikan nama kolom dummy ini ('Location_NY') sesuai dengan yang ada di file 'ann_model_features.pkl'
        if 'Location_New York' in input_data.columns: # Ganti 'Location_New York' dengan nama kolom yang sesuai
             input_data['Location_New York'] = loc_ny 
        
        # 3. Scaling Input ANN
        input_scaled = scaler.transform(input_data)
        
        # 4. Prediksi Model ANN
        ann_prediction = ann_model.predict(input_scaled)[0][0]
        
        # 5. Prediksi Model Fuzzy (hanya butuh age dan rating)
        fuzzy_prediction = predict_fuzzy_salary(age, rating)
        
        # 6. Format Output
        ann_output = f"US$ {ann_prediction:.2f}K"
        fuzzy_output = f"US$ {fuzzy_prediction:.2f}K" if not np.isnan(fuzzy_prediction) else "N/A (Input out of range)"

        return render_template('result.html', 
                               ann_salary=ann_output,
                               fuzzy_salary=fuzzy_output)

    except Exception as e:
        return render_template('result.html', error=f"Error saat prediksi: {e}")

if __name__ == '__main__':
    # Mode produksi, hanya berjalan sekali.
    app.run(debug=False) # <-- Ubah menjadi False