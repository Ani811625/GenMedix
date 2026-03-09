import json
import os
import random
import string
import smtplib
import ssl
from datetime import datetime, timedelta
from threading import Thread
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv
import io
from functools import wraps

import stripe
from flask import (
    Flask, request, jsonify, render_template, make_response, 
    Response, redirect, url_for, session, flash, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import inspect

import joblib
import pandas as pd
import shap
import numpy as np
from weasyprint import HTML
from supabase import create_client, Client

# =======================================================
# 1. APP CONFIGURATION & SETUP
# =======================================================

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'enterprise-secure-key-2026')

DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL.replace("postgres://", "postgresql://")
else:
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'project.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET')

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
MAIL_USERNAME = os.environ.get("MAIL_USERNAME") 
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD") 
ADMIN_RECEIVER_EMAIL = os.environ.get("MAIL_USERNAME") 

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try: 
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except: 
        pass

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- MASTER ADMIN SECURITY LOGIC ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('admin_login'))
            
        # Check if the user's email exists in the authorized Admin table
        is_admin = AdminEmail.query.filter_by(email=current_user.email).first()
        if not is_admin:
            flash("Unauthorized Access: Terminal locked.", "danger")
            return redirect(url_for('dashboard'))
            
        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def inject_global_data():
    basedir = os.path.abspath(os.path.dirname(__file__))
    broadcast_data = None
    broadcast_file = os.path.join(basedir, 'broadcast.json')
    
    if os.path.exists(broadcast_file):
        try:
            with open(broadcast_file, 'r') as f:
                broadcast_data = json.load(f)
        except Exception: 
            pass
    
    is_admin = False
    if current_user.is_authenticated:
        try:
            if AdminEmail.query.filter_by(email=current_user.email).first():
                is_admin = True
        except Exception: 
            pass
            
    return dict(current_user=current_user, global_broadcast=broadcast_data, is_admin=is_admin)

@app.before_request
def check_maintenance_and_db():
    # SUPABASE SAFE INIT: Will only add missing tables, won't drop existing ones.
    try:
        inspector = inspect(db.engine)
        if not inspector.has_table("admin_email"):
             with app.app_context():
                db.create_all()
                
        # Auto-seed the Master Admin Account
        with app.app_context():
            if not AdminEmail.query.filter_by(email='aniruddhas387@gmail.com').first():
                db.session.add(AdminEmail(email='aniruddhas387@gmail.com', added_by='SYSTEM_INIT'))
                
                # Check if you already made an account in Supabase
                existing_user = User.query.filter_by(email='aniruddhas387@gmail.com').first()
                if not existing_user:
                    master_user = User(
                        full_name="System Administrator", 
                        email='aniruddhas387@gmail.com', 
                        medical_reg_id="MASTER-01"
                    )
                    master_user.set_password("Admin@1234")
                    db.session.add(master_user)
                
                db.session.commit()
    except Exception: 
        pass

    # Maintenance Lockout Check
    basedir = os.path.abspath(os.path.dirname(__file__))
    maintenance_file = os.path.join(basedir, 'maintenance.json')
    
    allowed_endpoints = ['static', 'login', 'logout', 'admin_login']
    
    if request.endpoint and request.endpoint not in allowed_endpoints:
        if os.path.exists(maintenance_file):
            try:
                is_admin = current_user.is_authenticated and AdminEmail.query.filter_by(email=current_user.email).first()
            except Exception: 
                is_admin = False
            
            if is_admin:
                pass # Admins bypass maintenance
            else:
                end_time = None
                try:
                    with open(maintenance_file, 'r') as f:
                        m_data = json.load(f)
                        end_time = m_data.get("end_time")
                except Exception: 
                    pass
                return render_template('maintenance.html', end_time=end_time), 503

# =======================================================
# 2. DATABASE MODELS
# =======================================================

class AdminEmail(db.Model):
    email = db.Column(db.String(150), primary_key=True)
    added_by = db.Column(db.String(150))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Subscription(db.Model):
    email = db.Column(db.String(150), primary_key=True)
    stripe_customer_id = db.Column(db.String(100))
    plan_type = db.Column(db.String(50))
    max_seats = db.Column(db.Integer)
    current_users = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    medical_reg_id = db.Column(db.String(100), unique=True)
    subscription_email = db.Column(db.String(150), db.ForeignKey('subscription.email'))
    
    patients = db.relationship('Patient', backref='doctor', lazy=True, cascade="all, delete-orphan")
    notes = db.relationship('Note', backref='doctor', lazy=True, cascade="all, delete-orphan")
    
    def set_password(self, password): 
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password): 
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    dob = db.Column(db.String(10)) 
    gender = db.Column(db.String(10))
    aadhar = db.Column(db.String(14), index=True)
    country = db.Column(db.String(50))
    address = db.Column(db.String(200))
    phone = db.Column(db.String(20))
    emergency_contact = db.Column(db.String(100))
    blood_group = db.Column(db.String(5))
    allergies = db.Column(db.Text)
    medical_history = db.Column(db.Text)
    current_medications = db.Column(db.Text)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    reports = db.relationship('Report', backref='patient', lazy=True, cascade="all, delete-orphan")
    notes = db.relationship('Note', backref='patient', lazy=True, cascade="all, delete-orphan")

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    drug_name = db.Column(db.String(50), nullable=False)
    predicted_dose = db.Column(db.String(50))
    model_used = db.Column(db.String(50))
    confidence = db.Column(db.String(20))
    doctor_name = db.Column(db.String(150))
    report_data_json = db.Column(db.Text)
    pdf_storage_path = db.Column(db.String(200))
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    note_text = db.Column(db.Text, nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# =======================================================
# 3. AI MODELS & HELPERS
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

try:
    base_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_base_v1.pkl'))
    base_model_columns = joblib.load(os.path.join(MODEL_DIR, 'base_model_columns.pkl'))
    
    enhanced_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_enhanced_v1.pkl'))
    enhanced_model_columns = joblib.load(os.path.join(MODEL_DIR, 'model_columns.pkl'))
    
    enhanced_explainer = shap.TreeExplainer(enhanced_model)
    base_explainer = shap.TreeExplainer(base_model)
    
    diabetes_model = joblib.load(os.path.join(MODEL_DIR, 'diabetes_model_v1.pkl'))
    diabetes_model_columns = joblib.load(os.path.join(MODEL_DIR, 'diabetes_model_columns.pkl'))
    diabetes_explainer = shap.TreeExplainer(diabetes_model)
except Exception as e: 
    print(f"Error loading models: {e}")

def get_interaction_warnings(checked_drugs_list):
    warnings = []
    if not checked_drugs_list: 
        return warnings
    if "Amiodarone" in checked_drugs_list: 
        warnings.append("<strong>Severe Interaction: Amiodarone.</strong> Potentiates Warfarin effect. Reduce dose by 30-50%.")
    if "Fluconazole" in checked_drugs_list: 
        warnings.append("<strong>Severe Interaction: Fluconazole.</strong> Strong CYP2C9 inhibitor. Significant dose reduction likely required.")
    if "Bactrim" in checked_drugs_list: 
        warnings.append("<strong>Severe Interaction: TMP/SMX (Bactrim).</strong> Potentiates Warfarin effect. Monitor INR closely.")
    if "Rifampin" in checked_drugs_list: 
        warnings.append("<strong>Severe Interaction: Rifampin.</strong> Reduces Warfarin effect. Dose increase may be required.")
    if "Carbamazepine" in checked_drugs_list: 
        warnings.append("<strong>Interaction: Carbamazepine.</strong> Reduces Warfarin effect.")
    return warnings

def run_model_prediction(patient_data_dict):
    is_enhanced = any(key.startswith('CYP2C9') or key.startswith('VKORC1') for key in patient_data_dict.keys())
    
    if is_enhanced:
        model_to_use = enhanced_model
        columns_to_use = enhanced_model_columns
        explainer_to_use = enhanced_explainer
        model_name = "Enhanced (Clinical + Genome)"
    else:
        model_to_use = base_model
        columns_to_use = base_model_columns
        explainer_to_use = base_explainer
        model_name = "Base (Clinical-Only)"
        
    patient_df = pd.DataFrame([patient_data_dict]).reindex(columns=columns_to_use, fill_value=0)
    prediction_array = model_to_use.predict(patient_df)
    predicted_dose = round(prediction_array[0], 2)
    
    std_dev = np.std([tree.predict(patient_df) for tree in model_to_use.estimators_])
    shap_values = explainer_to_use.shap_values(patient_df)[0]
    top_indices = np.argsort(np.abs(shap_values))[-5:] 
    
    shap_explanation = {
        patient_df.columns[i]: round(shap_values[i], 2) 
        for i in reversed(top_indices) if np.abs(shap_values[i]) > 0
    }
    
    return {
        "prediction": predicted_dose, 
        "model_name": model_name, 
        "shap_explanation": shap_explanation, 
        "std_dev": std_dev
    }

def get_confidence_score(std_dev):
    if std_dev < 0.5: 
        return "High", "Model estimators are in strong agreement."
    elif std_dev < 1.0: 
        return "Medium", "Model estimators show variance. Use with caution."
    else: 
        return "Low", "Significant disagreement in estimators. Proceed with caution."

def get_human_explanation(shap_dict):
    explanations = []
    for feature, value in shap_dict.items():
        if feature == "Weight__kg_":
            display_name = "Weight"
        elif feature == "Height__cm_":
            display_name = "Height"
        elif feature.startswith("CYP2C9"):
            display_name = "CYP2C9 Genotype"
        elif feature.startswith("VKORC1"):
            display_name = "VKORC1 Genotype"
        else:
            display_name = feature.replace("Race_", "")
            
        direction = "increased" if value > 0 else "decreased"
        explanations.append(f"<strong>{display_name}</strong> {direction} the dose recommendation.")
    return explanations

def get_clinical_suggestions(shap_dict, confidence):
    suggestions = []
    for feature in shap_dict.keys():
        if "VKORC1" in feature and shap_dict[feature] < -0.5: 
            suggestions.append("<strong>High Sensitivity:</strong> VKORC1 genotype suggests lower dose requirement.")
        if "CYP2C9" in feature and shap_dict[feature] < -0.5: 
            suggestions.append("<strong>Slow Metabolizer:</strong> CYP2C9 genotype suggests slower clearance.")
            
    if "Weight__kg_" in shap_dict and shap_dict["Weight__kg_"] > 1.0: 
        suggestions.append("High body weight is a major factor increasing the dose.")
        
    if confidence == "Low": 
        suggestions.append("<strong>Low Confidence:</strong> Review all data carefully.")
        
    if not suggestions: 
        suggestions.append("Standard dosing protocol advised. Monitor INR as per guidelines.")
        
    return suggestions

def process_prediction_data(form_data):
    patient_info_dict = {
        "patient_name": form_data.get('patient_name'), 
        "patient_dob": form_data.get('patient_dob'), 
        "patient_gender": form_data.get('patient_gender'), 
        "patient_country": form_data.get('patient_country'), 
        "patient_address": form_data.get('patient_address')
    }
    
    safety_data_dict = {
        "is_pregnant": "Yes" if form_data.get('is_pregnant') else "No", 
        "active_bleeding": "Yes" if form_data.get('active_bleeding') else "No", 
        "platelet_count": form_data.get('platelet_count') or "Not Provided", 
        "baseline_inr": form_data.get('baseline_inr') or "Not Provided"
    }
    
    clinical_data_dict = {
        "Age": float(form_data.get('Age')), 
        "Height__cm_": float(form_data.get('Height__cm_')), 
        "Weight__kg_": float(form_data.get('Weight__kg_'))
    }
    
    race = form_data.get('Race')
    cyp2c9 = form_data.get('CYP2C9_genotypes')
    vkorc1 = form_data.get('VKORC1_genotype')
    
    if race: 
        clinical_data_dict[race] = 1.0
    if cyp2c9: 
        clinical_data_dict[cyp2c9] = 1.0
    if vkorc1: 
        clinical_data_dict[vkorc1] = 1.0
        
    clinical_info_display = {
        "Age": form_data.get('Age'), 
        "Height__cm_": form_data.get('Height__cm_'), 
        "Weight__kg_": form_data.get('Weight__kg_'), 
        "Race_Display": race.split('_')[-1] if race else "N/A", 
        "CYP2C9_Display": cyp2c9.split('__')[-1].replace('_', '/*') if cyp2c9 else "N/A", 
        "VKORC1_Display": vkorc1.split('_')[-1] if vkorc1 else "N/A"
    }
    
    pred_data = run_model_prediction(clinical_data_dict) 
    confidence, conf_expl = get_confidence_score(pred_data['std_dev'])
    suggestions = get_clinical_suggestions(pred_data['shap_explanation'], confidence)
    
    if form_data.get('is_pregnant'): 
        suggestions.append("<strong>CONTRAINDICATION:</strong> Patient marked Pregnant.")
    if form_data.get('active_bleeding'): 
        suggestions.append("<strong>CONTRAINDICATION:</strong> Active bleeding detected.")
        
    results_dict = {
        "predicted_dose_mg_per_week": pred_data['prediction'], 
        "model_used": pred_data['model_name'], 
        "confidence_score": confidence, 
        "confidence_explanation": conf_expl, 
        "human_explanation": get_human_explanation(pred_data['shap_explanation']), 
        "clinical_suggestions": suggestions, 
        "shap_explanation": pred_data.get('shap_explanation', {}), 
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "report_id": f"GM-{datetime.now().strftime('%Y%m%d')}-{abs(hash(patient_info_dict['patient_name'])) % 10000}"
    }
    
    return patient_info_dict, clinical_info_display, safety_data_dict, results_dict

def run_diabetes_prediction(patient_data_dict):
    patient_df = pd.DataFrame([patient_data_dict]).reindex(columns=diabetes_model_columns, fill_value=0)
    probabilities = diabetes_model.predict_proba(patient_df)[0]
    prob_positive = round(probabilities[1] * 100, 1)
    
    if prob_positive >= 60.0: 
        risk_level = "High Risk"
        confidence_score = "High" if prob_positive > 80 else "Medium"
    elif prob_positive >= 40.0: 
        risk_level = "Pre-Diabetic"
        confidence_score = "Medium"
    else: 
        risk_level = "Low Risk"
        confidence_score = "High" if prob_positive < 20 else "Medium"
        
    shap_values = diabetes_explainer.shap_values(patient_df)
    
    if hasattr(shap_values, 'values'): 
        shap_values = shap_values.values
        
    if isinstance(shap_values, list): 
        shap_values_for_instance = shap_values[1][0] 
    else:
        shap_array = np.array(shap_values)
        if len(shap_array.shape) == 3:
            shap_values_for_instance = shap_array[0, :, 1]
        else:
            shap_values_for_instance = shap_array[0]
            
    top_indices = np.argsort(np.abs(shap_values_for_instance))[-5:]
    
    shap_explanation = {
        patient_df.columns[i]: round(float(shap_values_for_instance[i]), 3) 
        for i in reversed(top_indices) if abs(float(shap_values_for_instance[i])) > 0.01
    }
    
    return {
        "prediction": risk_level, 
        "probability": prob_positive, 
        "model_name": "Diabetes Genomic Classifier", 
        "shap_explanation": shap_explanation, 
        "confidence_score": confidence_score
    }

def get_diabetes_clinical_suggestions(shap_dict, risk_level):
    suggestions = []
    
    if risk_level == "High Risk": 
        suggestions.append("<strong>Action Required:</strong> Immediate HbA1c testing and endocrinology consult recommended.")
    elif risk_level == "Pre-Diabetic": 
        suggestions.append("<strong>Preventative Care:</strong> Recommend lifestyle intervention, dietary changes, and weight management.")
        
    if shap_dict.get('Glucose', 0) > 0.05: 
        suggestions.append("Elevated fasting glucose is a primary driver of this risk profile.")
    if shap_dict.get('BMI', 0) > 0.05: 
        suggestions.append("High BMI is significantly elevating diabetes risk. Target 5-10% body weight reduction.")
    if shap_dict.get('TCF7L2_Risk_Variant', 0) > 0.05: 
        suggestions.append("<strong>Genomic Risk:</strong> Patient carries the TCF7L2 high-risk allele, indicating a strong genetic predisposition.")
        
    if not suggestions: 
        suggestions.append("Continue routine annual screening and maintain a healthy lifestyle.")
        
    return suggestions

def process_diabetes_data(form_data):
    patient_info_dict = {
        "patient_name": form_data.get('patient_name'), 
        "patient_dob": form_data.get('patient_dob'), 
        "patient_gender": form_data.get('patient_gender')
    }
    
    safety_data_dict = { 
        "is_pregnant": "Not Applicable", 
        "active_bleeding": "Not Applicable" 
    }
    
    clinical_data_dict = {
        "Pregnancies": float(form_data.get('Pregnancies', 0)), 
        "Glucose": float(form_data.get('Glucose', 90)), 
        "BloodPressure": float(form_data.get('BloodPressure', 80)), 
        "SkinThickness": float(form_data.get('SkinThickness', 20)), 
        "Insulin": float(form_data.get('Insulin', 80)), 
        "BMI": float(form_data.get('BMI', 25.0)), 
        "DiabetesPedigree": float(form_data.get('DiabetesPedigree', 0.5)), 
        "Age": float(form_data.get('Age', 30)), 
        "TCF7L2_Risk_Variant": float(form_data.get('TCF7L2_Risk_Variant', 0))
    }
    
    clinical_info_display = {
        "Age": form_data.get('Age'), 
        "BMI": form_data.get('BMI'), 
        "Fasting Glucose": f"{form_data.get('Glucose')} mg/dL", 
        "Blood Pressure": f"{form_data.get('BloodPressure')} mmHg", 
        "Family History Pedigree": form_data.get('DiabetesPedigree'), 
        "TCF7L2 Variant": "Detected (High Risk)" if clinical_data_dict['TCF7L2_Risk_Variant'] == 1 else "Not Detected"
    }
    
    pred_data = run_diabetes_prediction(clinical_data_dict)
    
    results_dict = {
        "predicted_dose_mg_per_week": pred_data['prediction'], 
        "probability_score": pred_data['probability'], 
        "model_used": pred_data['model_name'], 
        "confidence_score": pred_data['confidence_score'], 
        "confidence_explanation": f"The AI model predicts a {pred_data['probability']}% probability of pathology.", 
        "clinical_suggestions": get_diabetes_clinical_suggestions(pred_data['shap_explanation'], pred_data['prediction']), 
        "shap_explanation": pred_data['shap_explanation'], 
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "report_id": f"DIA-{datetime.now().strftime('%Y%m%d')}-{abs(hash(patient_info_dict['patient_name'])) % 10000}"
    }
    
    return patient_info_dict, clinical_info_display, safety_data_dict, results_dict

# =======================================================
# 4. APP ROUTES (Public & User)
# =======================================================

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    try:
        df = pd.read_csv('data/warfarin.csv')
        headers = df.columns.tolist()
        rows = df.head(200).to_dict('records')
    except Exception: 
        headers = []
        rows = []
    return render_template('dataset.html', headers=headers, rows=rows, row_count=len(rows), showing_count=len(rows))

@app.route('/pricing')
def pricing(): 
    return render_template('pricing.html', key=STRIPE_PUBLISHABLE_KEY)

@app.route('/about')
def about(): 
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: 
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user, remember=True)
            flash('Login Successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid Email or Password.', 'danger')
        
    return render_template('login.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        is_admin = AdminEmail.query.filter_by(email=current_user.email).first()
        if is_admin:
            return redirect(url_for('admin_dashboard'))
            
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Verify they are an authorized Admin in the database
        admin_record = AdminEmail.query.filter_by(email=email).first()
        if not admin_record:
            flash("Unauthorized: Your email is not registered as an Administrator.", "danger")
            return redirect(url_for('admin_login'))
            
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid credentials.", "danger")
            
    return render_template('admin_login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('home'))

# --- PATIENT HUB ---
@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    if request.method == 'POST':
        aadhar = request.form.get('aadhar').replace(' ', '')
        dob = request.form.get('dob')
        patient = Patient.query.filter_by(aadhar=aadhar, dob=dob).first()
        if patient:
            session['patient_aadhar'] = aadhar
            session['patient_name'] = patient.full_name
            return redirect(url_for('patient_dashboard'))
        flash("Invalid Credentials.", "danger")
    return render_template('patient_login.html')

@app.route('/patient/dashboard')
def patient_dashboard():
    if 'patient_aadhar' not in session: 
        return redirect(url_for('patient_login'))
        
    aadhar = session['patient_aadhar']
    patient_records = Patient.query.filter_by(aadhar=aadhar).all()
    patient_ids = [p.id for p in patient_records]
    
    reports = Report.query.filter(Report.patient_id.in_(patient_ids)).order_by(Report.generated_at.desc()).all()
    aadhar_display = f"{aadhar[:4]} {aadhar[4:8]} {aadhar[8:]}"
    
    return render_template('patient_dashboard.html', reports=reports, patient_name=session['patient_name'], aadhar_display=aadhar_display)

@app.route('/patient/logout')
def patient_logout():
    session.pop('patient_aadhar', None)
    session.pop('patient_name', None)
    return redirect(url_for('home'))

# --- DOCTOR HUB ---
@app.route('/dashboard')
@login_required
def dashboard():
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.full_name).all()
    return render_template('dashboard.html', patients=patients, total_reports=0, total_patients=len(patients), gender_data={}, blood_data={}, confidence_data={})

@app.route('/add_patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    if request.method == 'POST':
        new_patient = Patient(
            full_name=request.form.get('full_name'), 
            aadhar=request.form.get('aadhar').replace(' ', ''), 
            dob=request.form.get('dob'), 
            gender=request.form.get('gender'), 
            doctor_id=current_user.id
        )
        db.session.add(new_patient)
        db.session.commit()
        flash('Patient added.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('add_patient.html')

@app.route('/view_patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: 
        return redirect(url_for('dashboard'))
        
    reports = Report.query.filter_by(patient_id=patient.id).order_by(Report.generated_at.desc()).all()
    notes = Note.query.filter_by(patient_id=patient.id).order_by(Note.created_at.desc()).all()
    return render_template('view_patient.html', patient=patient, reports=reports, notes=notes)

@app.route('/patient/<int:patient_id>/new_assessment', methods=['GET'])
@login_required
def new_assessment(patient_id): 
    patient = Patient.query.get_or_404(patient_id)
    return render_template('assessment_type.html', patient=patient)

@app.route('/patient/<int:patient_id>/select_disease', methods=['GET'])
@login_required
def select_disease(patient_id): 
    patient = Patient.query.get_or_404(patient_id)
    return render_template('select_disease.html', patient=patient)

@app.route('/patient/<int:patient_id>/select_drug', methods=['GET'])
@login_required
def select_drug(patient_id): 
    patient = Patient.query.get_or_404(patient_id)
    return render_template('select_drug.html', patient=patient)

@app.route('/patient/<int:patient_id>/redirect_form', methods=['POST'])
@login_required
def redirect_to_drug_form(patient_id):
    if request.form.get('drug_name') == 'warfarin': 
        return redirect(url_for('warfarin_form', patient_id=patient_id))
    return redirect(url_for('dashboard'))

@app.route('/patient/<int:patient_id>/warfarin_form', methods=['GET'])
@login_required
def warfarin_form(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    try: 
        calculated_age = datetime.today().year - datetime.strptime(patient.dob, '%Y-%m-%d').year
    except Exception: 
        calculated_age = 0
    return render_template('warfarin_form.html', patient=patient, calculated_age=calculated_age)

@app.route('/patient/<int:patient_id>/generate_warfarin_report', methods=['POST'])
@login_required
def generate_warfarin_report(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient_info, clinical_info, safety_info, results = process_prediction_data(request.form)
    
    full_report_data = {
        "patient_info": patient_info, 
        "clinical_info": clinical_info, 
        "safety_info": safety_info, 
        "results": results, 
        "doctor_name": request.form.get('doctor_name')
    }
    
    new_report = Report(
        drug_name="Warfarin", 
        predicted_dose=f"{results['predicted_dose_mg_per_week']} mg/week", 
        model_used=results['model_used'], 
        confidence=results['confidence_score'], 
        doctor_name=request.form.get('doctor_name'), 
        report_data_json=json.dumps(full_report_data), 
        patient_id=patient.id
    )
    db.session.add(new_report)
    db.session.commit()
    
    return make_response(render_template('display_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=request.form.get('doctor_name'), report_obj=new_report))

@app.route('/patient/<int:patient_id>/diabetes_form', methods=['GET'])
@login_required
def diabetes_form(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    try: 
        calculated_age = datetime.today().year - datetime.strptime(patient.dob, '%Y-%m-%d').year
    except Exception: 
        calculated_age = 0
    return render_template('diabetes_form.html', patient=patient, calculated_age=calculated_age)

@app.route('/patient/<int:patient_id>/generate_diabetes_report', methods=['POST'])
@login_required
def generate_diabetes_report(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient_info, clinical_info, safety_info, results = process_diabetes_data(request.form)
    
    full_report_data = {
        "patient_info": patient_info, 
        "clinical_info": clinical_info, 
        "safety_info": safety_info, 
        "results": results, 
        "doctor_name": request.form.get('doctor_name')
    }
    
    new_report = Report(
        drug_name="Type 2 Diabetes Assessment", 
        predicted_dose=results['predicted_dose_mg_per_week'], 
        model_used=results['model_used'], 
        confidence=results['confidence_score'], 
        doctor_name=request.form.get('doctor_name'), 
        report_data_json=json.dumps(full_report_data), 
        patient_id=patient.id
    )
    db.session.add(new_report)
    db.session.commit()
    
    return make_response(render_template('display_disease_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=request.form.get('doctor_name'), report_obj=new_report))

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    report = Report.query.get_or_404(report_id)
    report_data = json.loads(report.report_data_json)
    
    if report.drug_name == "Type 2 Diabetes Assessment":
        template_name = 'display_disease_report.html'
    else:
        template_name = 'display_report.html'
        
    return make_response(render_template(
        template_name, 
        patient_info=report_data.get('patient_info'), 
        clinical_info=report_data.get('clinical_info'), 
        safety_info=report_data.get('safety_info', {}), 
        results=report_data.get('results'), 
        doctor_name=report_data.get('doctor_name'), 
        report_obj=report
    ))


# =======================================================
# 5. ENTERPRISE ADMIN COMMAND CENTER
# =======================================================

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    total_doctors = User.query.count()
    total_patients = Patient.query.count()
    total_reports = Report.query.count()
    total_subscriptions = Subscription.query.count()
    
    all_doctors = User.query.order_by(User.id.desc()).all()
    all_subs = Subscription.query.order_by(Subscription.email).all()
    all_admins = AdminEmail.query.all()
    
    basedir = os.path.abspath(os.path.dirname(__file__))
    maintenance_active = os.path.exists(os.path.join(basedir, 'maintenance.json'))
    
    current_broadcast = {}
    broadcast_file = os.path.join(basedir, 'broadcast.json')
    if os.path.exists(broadcast_file):
        try:
            with open(broadcast_file, 'r') as f:
                current_broadcast = json.load(f)
        except Exception:
            pass
            
    return render_template(
        'admin_dashboard.html', 
        total_doctors=total_doctors, 
        total_patients=total_patients,
        total_reports=total_reports, 
        total_subs=total_subscriptions,
        all_doctors=all_doctors, 
        all_subs=all_subs, 
        all_admins=all_admins,
        maintenance_active=maintenance_active, 
        current_broadcast=current_broadcast
    )

@app.route('/admin/system/add_admin', methods=['POST'])
@login_required
@admin_required
def admin_add_admin():
    new_email = request.form.get('email')
    existing_admin = AdminEmail.query.filter_by(email=new_email).first()
    
    if existing_admin:
        flash("Email already has admin privileges.", "warning")
    else:
        new_admin = AdminEmail(email=new_email, added_by=current_user.email)
        db.session.add(new_admin)
        db.session.commit()
        flash(f"Granted admin privileges to {new_email}.", "success")
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/system/toggle_maintenance', methods=['POST'])
@login_required
@admin_required
def admin_toggle_maintenance():
    basedir = os.path.abspath(os.path.dirname(__file__))
    maintenance_file = os.path.join(basedir, 'maintenance.json')
    
    if os.path.exists(maintenance_file):
        os.remove(maintenance_file)
        flash("System is now ONLINE.", "success")
    else:
        end_time = request.form.get('end_time')
        with open(maintenance_file, 'w') as f:
            json.dump({"status": "OFFLINE", "end_time": end_time}, f)
        flash(f"KILL SWITCH ENGAGED. Lockout active until {end_time}.", "danger")
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/system/broadcast', methods=['POST'])
@login_required
@admin_required
def admin_broadcast():
    basedir = os.path.abspath(os.path.dirname(__file__))
    broadcast_file = os.path.join(basedir, 'broadcast.json')
    message = request.form.get('message')
    
    if message and message.strip():
        data = {
            "message": message.strip(),
            "bg_color": request.form.get('bg_color', '#000000'),
            "text_color": request.form.get('text_color', '#ffffff'),
            "is_promo": request.form.get('is_promo') == 'on',
            "promo_code": request.form.get('promo_code', ''),
            "promo_link": request.form.get('promo_link', '#')
        }
        with open(broadcast_file, 'w') as f: 
            json.dump(data, f)
        flash("Rich Broadcast transmitted.", "success")
    else:
        if os.path.exists(broadcast_file): 
            os.remove(broadcast_file)
        flash("Broadcast cleared.", "info")
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/doctor/<int:doc_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_doctor(doc_id):
    if current_user.id == doc_id: 
        flash("Action Blocked: You cannot delete your own account.", "danger")
        return redirect(url_for('admin_dashboard'))
        
    doctor = User.query.get_or_404(doc_id)
    doc_name = doctor.full_name
    
    if doctor.subscription_email:
        sub = Subscription.query.filter_by(email=doctor.subscription_email).first()
        if sub and sub.current_users > 0: 
            sub.current_users -= 1
            
    db.session.delete(doctor)
    db.session.commit()
    flash(f"Physician {doc_name} and all their clinical data have been permanently wiped.", "success")
    
    return redirect(url_for('admin_dashboard'))

# --- LICENSE MANAGER ROUTES ---

@app.route('/admin/license/add', methods=['POST'])
@login_required
@admin_required
def admin_add_license():
    email = request.form.get('email')
    plan_type = request.form.get('plan_type')
    max_seats = request.form.get('max_seats')
    
    sub = Subscription.query.filter_by(email=email).first()
    if sub:
        sub.plan_type = plan_type
        sub.max_seats = int(max_seats)
        sub.is_active = True
        flash(f"License successfully updated for {email}.", "success")
    else:
        new_sub = Subscription(
            email=email,
            stripe_customer_id="MANUAL_OVERRIDE",
            plan_type=plan_type,
            max_seats=int(max_seats),
            is_active=True
        )
        db.session.add(new_sub)
        flash(f"New manual license provisioned for {email}.", "success")
        
    db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/license/revoke', methods=['POST'])
@login_required
@admin_required
def admin_revoke_license():
    email = request.form.get('email')
    sub = Subscription.query.filter_by(email=email).first()
    
    if sub:
        sub.is_active = False
        sub.max_seats = 0
        db.session.commit()
        flash(f"License for {email} has been immediately revoked.", "warning")
        
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/doctor/create', methods=['POST'])
@login_required
@admin_required
def admin_create_doctor():
    email = request.form.get('email')
    full_name = request.form.get('full_name')
    reg_id = request.form.get('medical_reg_id')
    password = request.form.get('password')
    license_email = request.form.get('subscription_email')

    if User.query.filter_by(email=email).first():
        flash("Email already registered to an existing physician.", "danger")
        return redirect(url_for('admin_dashboard'))

    sub = Subscription.query.filter_by(email=license_email).first()
    if not sub or not sub.is_active:
        flash("Invalid or inactive License Email provided.", "danger")
        return redirect(url_for('admin_dashboard'))

    if sub.current_users >= sub.max_seats:
        flash("License seat limit reached. Upgrade the license first.", "danger")
        return redirect(url_for('admin_dashboard'))

    new_doc = User(
        full_name=full_name, 
        email=email, 
        medical_reg_id=reg_id, 
        subscription_email=license_email
    )
    new_doc.set_password(password)
    
    sub.current_users += 1
    db.session.add(new_doc)
    db.session.commit()
    
    flash(f"Account for Dr. {full_name} successfully created under {license_email}.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/doctor/<int:doc_id>/edit', methods=['POST'])
@login_required
@admin_required
def admin_edit_doctor(doc_id):
    doctor = User.query.get_or_404(doc_id)
    
    # Check if the user is trying to edit a Master Admin
    if AdminEmail.query.filter_by(email=doctor.email).first():
        flash("Cannot edit a Master Admin account through this form.", "danger")
        return redirect(url_for('admin_dashboard'))

    doctor.full_name = request.form.get('full_name')
    doctor.email = request.form.get('email')
    doctor.medical_reg_id = request.form.get('medical_reg_id')
    
    db.session.commit()
    flash(f"Physician profile for Dr. {doctor.full_name} updated successfully.", "success")
    return redirect(url_for('admin_dashboard'))


if __name__ == '__main__':
    app.run(debug=True)