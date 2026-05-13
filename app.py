import json
import os
import random
import string
import smtplib
import ssl
import dill
import lime.lime_tabular
from datetime import datetime, timedelta
from threading import Thread
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv
import io
from functools import wraps

# --- THIRD PARTY IMPORTS ---
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
from sqlalchemy import inspect, text

# --- ML & PDF IMPORTS ---
import joblib
import pandas as pd
import shap
import numpy as np
from weasyprint import HTML

# --- SUPABASE IMPORT ---
from supabase import create_client, Client

# =======================================================
# 1. APP CONFIGURATION & SETUP
# =======================================================

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key')

# --- DATABASE CONFIGURATION ---
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL.replace("postgres://", "postgresql://")
else:
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'project.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- STRIPE CONFIGURATION ---
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET')

# --- MAIL CONFIGURATION ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
MAIL_USERNAME = os.environ.get("MAIL_USERNAME") 
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD") 
ADMIN_RECEIVER_EMAIL = os.environ.get("MAIL_USERNAME") 

# --- SUPABASE CLIENT SETUP (Auth & Storage) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try: supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except: pass

# --- LOGIN MANAGER ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Please log in to access the clinical dashboard."
login_manager.login_message_category = "info"

# --- MASTER ADMIN SECURITY LOGIC ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('admin_login'))
        is_admin = AdminEmail.query.filter_by(email=current_user.email).first()
        if not is_admin:
            flash("🛡️ Access Denied: Administrator privileges required. This incident has been logged.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.context_processor
def inject_global_data():
    basedir = os.path.abspath(os.path.dirname(__file__))
    broadcast_data = None
    broadcast_file = os.path.join(basedir, 'broadcast.json')
    if os.path.exists(broadcast_file):
        try:
            with open(broadcast_file, 'r') as f:
                broadcast_data = json.load(f)
        except Exception: pass
    
    is_admin = False
    if current_user.is_authenticated:
        try:
            if AdminEmail.query.filter_by(email=current_user.email).first():
                is_admin = True
        except: pass
            
    return dict(current_user=current_user, global_broadcast=broadcast_data, is_admin=is_admin)

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
    
    is_beta_tester = db.Column(db.Boolean, default=False)
    
    subscription_email = db.Column(db.String(150), db.ForeignKey('subscription.email'))
    patients = db.relationship('Patient', backref='doctor', lazy=True, cascade="all, delete-orphan")
    notes = db.relationship('Note', backref='doctor', lazy=True, cascade="all, delete-orphan")
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)

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
# 3. ISOLATED MODEL LOADING (Prevents Cascading Failures)
# =======================================================

@app.before_request
def check_maintenance_and_db():
    try:
        inspector = inspect(db.engine)
        if not inspector.has_table("user") or not inspector.has_table("admin_email"):
             with app.app_context(): db.create_all()
             
        with app.app_context():
            try:
                db.session.execute(text('ALTER TABLE "user" ADD COLUMN is_beta_tester BOOLEAN DEFAULT FALSE'))
                db.session.commit()
            except Exception:
                db.session.rollback() 
                
        with app.app_context():
            if not AdminEmail.query.filter_by(email='aniruddhas387@gmail.com').first():
                db.session.add(AdminEmail(email='aniruddhas387@gmail.com', added_by='SYSTEM_INIT'))
                if not User.query.filter_by(email='aniruddhas387@gmail.com').first():
                    master_user = User(full_name="System Administrator", email='aniruddhas387@gmail.com', medical_reg_id="MASTER-01")
                    master_user.set_password("Admin@1234")
                    db.session.add(master_user)
                db.session.commit()
    except: pass

    basedir = os.path.abspath(os.path.dirname(__file__))
    maintenance_file = os.path.join(basedir, 'maintenance.json')
    if request.endpoint and request.endpoint not in ['static', 'login', 'logout', 'admin_login']:
        if os.path.exists(maintenance_file):
            try: is_admin = current_user.is_authenticated and AdminEmail.query.filter_by(email=current_user.email).first()
            except: is_admin = False
            
            if not is_admin:
                end_time, m_title, m_message = None, "System Upgrades in Progress", "GenMedix is currently locked down for updates."
                try:
                    with open(maintenance_file, 'r') as f:
                        m_data = json.load(f)
                        end_time = m_data.get("end_time")
                        if m_data.get("title"): m_title = m_data.get("title")
                        if m_data.get("message"): m_message = m_data.get("message")
                except: pass
                return render_template('maintenance.html', end_time=end_time, m_title=m_title, m_message=m_message), 503

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 1. Base Model Load
try:
    base_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_base_v1.pkl'))
    base_model_columns = joblib.load(os.path.join(MODEL_DIR, 'base_model_columns.pkl'))
    base_explainer = shap.TreeExplainer(base_model)
except Exception as e: print(f"Base Model Load Error: {e}")

# 2. Enhanced Model Load
try:
    enhanced_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_enhanced_v1.pkl'))
    enhanced_model_columns = joblib.load(os.path.join(MODEL_DIR, 'model_columns.pkl'))
    enhanced_explainer = shap.TreeExplainer(enhanced_model)
except Exception as e: print(f"Enhanced Model Load Error: {e}")

# 3. Diabetes Model Load
try:
    diabetes_model = joblib.load(os.path.join(MODEL_DIR, 'diabetes_model_v1.pkl'))
    diabetes_model_columns = joblib.load(os.path.join(MODEL_DIR, 'diabetes_model_columns.pkl'))
    diabetes_explainer = shap.TreeExplainer(diabetes_model)
except Exception as e: print(f"Diabetes Model Load Error: {e}")

# 4. Vancomycin Model Load
try:
    vancomycin_model = joblib.load(os.path.join(MODEL_DIR, 'vancomycin', 'vancomycin_xgb_model_v1.pkl'))
    vancomycin_model_columns = joblib.load(os.path.join(MODEL_DIR, 'vancomycin', 'vancomycin_model_columns.pkl'))
    vancomycin_explainer = shap.TreeExplainer(vancomycin_model)
except Exception as e: print(f"Vancomycin Model Load Error: {e}")

# 5. Warfarin LIME Explainer Load (With fallback paths)
base_lime_explainer = None
try:
    wf_path = os.path.join(MODEL_DIR, 'warfarin', 'warfarin_lime_explainer.pkl')
    if not os.path.exists(wf_path): wf_path = os.path.join(MODEL_DIR, 'warfarin_lime_explainer.pkl')
    with open(wf_path, 'rb') as f:
        base_lime_explainer = dill.load(f)
except Exception as e: print(f"Warfarin LIME Load Error: {e}")

# 6. Vancomycin LIME Explainer Load (With fallback paths)
vancomycin_lime_explainer = None
try:
    v_path = os.path.join(MODEL_DIR, 'vancomycin', 'vancomycin_lime_explainer.pkl')
    if not os.path.exists(v_path): v_path = os.path.join(MODEL_DIR, 'vancomycin_lime_explainer.pkl')
    with open(v_path, 'rb') as f:
        vancomycin_lime_explainer = dill.load(f)
except Exception as e: print(f"Vancomycin LIME Load Error: {e}")


# =======================================================
# 4. ML LOGIC (Warfarin, Diabetes, Vancomycin)
# =======================================================
def get_interaction_warnings(checked_drugs_list):
    warnings = []
    if not checked_drugs_list: return warnings
    if "Amiodarone" in checked_drugs_list: warnings.append("<strong>Severe Interaction: Amiodarone.</strong> Potentiates Warfarin effect. Reduce dose by 30-50%.")
    if "Fluconazole" in checked_drugs_list: warnings.append("<strong>Severe Interaction: Fluconazole.</strong> Strong CYP2C9 inhibitor. Significant dose reduction likely required.")
    if "Bactrim" in checked_drugs_list: warnings.append("<strong>Severe Interaction: TMP/SMX (Bactrim).</strong> Potentiates Warfarin effect. Monitor INR closely.")
    if "Rifampin" in checked_drugs_list: warnings.append("<strong>Severe Interaction: Rifampin.</strong> Reduces Warfarin effect. Dose increase may be required.")
    if "Carbamazepine" in checked_drugs_list: warnings.append("<strong>Interaction: Carbamazepine.</strong> Reduces Warfarin effect.")
    return warnings

def run_model_prediction(patient_data_dict):
    is_enhanced = any(key.startswith('CYP2C9') or key.startswith('VKORC1') for key in patient_data_dict.keys())
    model_to_use, columns_to_use, explainer_to_use, model_name = (enhanced_model, enhanced_model_columns, enhanced_explainer, "Enhanced (Clinical + Genome)") if is_enhanced else (base_model, base_model_columns, base_explainer, "Base (Clinical-Only)")
    
    patient_df = pd.DataFrame([patient_data_dict]).reindex(columns=columns_to_use, fill_value=0)
    prediction_array = model_to_use.predict(patient_df)
    predicted_dose = round(prediction_array[0], 2)
    std_dev = np.std([tree.predict(patient_df) for tree in model_to_use.estimators_])
    
    # SHAP Generation
    shap_values = explainer_to_use.shap_values(patient_df)[0]
    top_indices = np.argsort(np.abs(shap_values))[-5:] 
    shap_explanation = {patient_df.columns[i]: round(shap_values[i], 2) for i in reversed(top_indices) if np.abs(shap_values[i]) > 0}
    
    # LIME Generation with Wrapper
    lime_explanation = {}
    if base_lime_explainer is not None:
        try:
            numeric_row = patient_df.iloc[0].apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            def lime_predict_wrapper(numpy_data):
                temp_df = pd.DataFrame(numpy_data, columns=columns_to_use)
                return model_to_use.predict(temp_df)
                
            lime_exp = base_lime_explainer.explain_instance(
                data_row=numeric_row,
                predict_fn=lime_predict_wrapper
            )
            lime_explanation = {feat: round(weight, 2) for feat, weight in lime_exp.as_list()[:4]}
        except Exception as e:
            print(f"LIME Execution Error (Warfarin): {e}")

    return {
        "prediction": predicted_dose, 
        "model_name": model_name, 
        "shap_explanation": shap_explanation, 
        "lime_explanation": lime_explanation,
        "std_dev": std_dev
    }

def get_confidence_score(std_dev):
    if std_dev < 0.5: return "High", "Model estimators are in strong agreement."
    elif std_dev < 1.0: return "Medium", "Model estimators show variance. Use with caution."
    else: return "Low", "Significant disagreement in estimators. Proceed with caution."

def get_human_explanation(shap_dict):
    explanations = []
    for feature, value in shap_dict.items():
        if feature in ["Weight__kg_", "Weight_kg"]: display_name = "Weight"
        elif feature in ["Height__cm_", "Height_cm"]: display_name = "Height"
        elif feature == "Serum_Creatinine": display_name = "Serum Creatinine"
        elif feature == "Calculated_CrCl": display_name = "Est. Creatinine Clearance"
        elif feature == "HLA_A_32_01_Risk": display_name = "HLA Toxicity Risk"
        elif feature == "agr_Group_II_Mutation": display_name = "agr Resistance Mutation"
        elif feature.startswith("CYP2C9"): display_name = "CYP2C9 Genotype"
        elif feature.startswith("VKORC1"): display_name = "VKORC1 Genotype"
        else: display_name = feature.replace("Race_", "").replace("_", " ")
        explanations.append(f"<strong>{display_name}</strong> {'increased' if value > 0 else 'decreased'} the dose recommendation.")
    return explanations

def get_clinical_suggestions(shap_dict, confidence):
    suggestions = []
    for feature in shap_dict.keys():
        if "VKORC1" in feature and shap_dict[feature] < -0.5: suggestions.append("<strong>High Sensitivity:</strong> VKORC1 genotype suggests lower dose requirement.")
        if "CYP2C9" in feature and shap_dict[feature] < -0.5: suggestions.append("<strong>Slow Metabolizer:</strong> CYP2C9 genotype suggests slower clearance.")
    if "Weight__kg_" in shap_dict and shap_dict["Weight__kg_"] > 1.0: suggestions.append("High body weight is a major factor increasing the dose.")
    if confidence == "Low": suggestions.append("<strong>Low Confidence:</strong> Review all data carefully.")
    if not suggestions: suggestions.append("Standard dosing protocol advised. Monitor INR as per guidelines.")
    return suggestions

def process_prediction_data(form_data):
    patient_info_dict = {"patient_name": form_data.get('patient_name'), "patient_dob": form_data.get('patient_dob'), "patient_gender": form_data.get('patient_gender'), "patient_country": form_data.get('patient_country'), "patient_address": form_data.get('patient_address')}
    safety_data_dict = {"is_pregnant": "Yes" if form_data.get('is_pregnant') else "No", "active_bleeding": "Yes" if form_data.get('active_bleeding') else "No", "platelet_count": form_data.get('platelet_count') or "Not Provided", "baseline_inr": form_data.get('baseline_inr') or "Not Provided"}
    clinical_data_dict = {"Age": float(form_data.get('Age')), "Height__cm_": float(form_data.get('Height__cm_')), "Weight__kg_": float(form_data.get('Weight__kg_'))}
    race, cyp2c9, vkorc1 = form_data.get('Race'), form_data.get('CYP2C9_genotypes'), form_data.get('VKORC1_genotype')
    if race: clinical_data_dict[race] = 1.0
    if cyp2c9: clinical_data_dict[cyp2c9] = 1.0
    if vkorc1: clinical_data_dict[vkorc1] = 1.0
    clinical_info_display = {"Age": form_data.get('Age'), "Height__cm_": form_data.get('Height__cm_'), "Weight__kg_": form_data.get('Weight__kg_'), "Race_Display": race.split('_')[-1] if race else "N/A", "CYP2C9_Display": cyp2c9.split('__')[-1].replace('_', '/*') if cyp2c9 else "N/A", "VKORC1_Display": vkorc1.split('_')[-1] if vkorc1 else "N/A"}
    pred_data = run_model_prediction(clinical_data_dict) 
    confidence, conf_expl = get_confidence_score(pred_data['std_dev'])
    suggestions = get_clinical_suggestions(pred_data['shap_explanation'], confidence)
    if form_data.get('is_pregnant'): suggestions.append("<strong>CONTRAINDICATION:</strong> Patient marked Pregnant.")
    if form_data.get('active_bleeding'): suggestions.append("<strong>CONTRAINDICATION:</strong> Active bleeding detected.")
    
    results_dict = {
        "predicted_dose_mg_per_week": pred_data['prediction'], 
        "model_used": pred_data['model_name'], 
        "confidence_score": confidence, 
        "confidence_explanation": conf_expl, 
        "human_explanation": get_human_explanation(pred_data['shap_explanation']), 
        "clinical_suggestions": suggestions, 
        "shap_explanation": pred_data.get('shap_explanation', {}), 
        "lime_explanation": pred_data.get('lime_explanation', {}),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "report_id": f"GM-{datetime.now().strftime('%Y%m%d')}-{abs(hash(patient_info_dict['patient_name'])) % 10000}"
    }
    return patient_info_dict, clinical_info_display, safety_data_dict, results_dict

def run_diabetes_prediction(patient_data_dict):
    patient_df = pd.DataFrame([patient_data_dict]).reindex(columns=diabetes_model_columns, fill_value=0)
    probabilities = diabetes_model.predict_proba(patient_df)[0]
    prob_positive = round(probabilities[1] * 100, 1)
    if prob_positive >= 60.0: risk_level, confidence_score = "High Risk", "High" if prob_positive > 80 else "Medium"
    elif prob_positive >= 40.0: risk_level, confidence_score = "Pre-Diabetic", "Medium"
    else: risk_level, confidence_score = "Low Risk", "High" if prob_positive < 20 else "Medium"
    shap_values = diabetes_explainer.shap_values(patient_df)
    if hasattr(shap_values, 'values'): shap_values = shap_values.values
    if isinstance(shap_values, list): shap_values_for_instance = shap_values[1][0] 
    else:
        shap_array = np.array(shap_values)
        shap_values_for_instance = shap_array[0, :, 1] if len(shap_array.shape) == 3 else shap_array[0]
    top_indices = np.argsort(np.abs(shap_values_for_instance))[-5:]
    shap_explanation = {patient_df.columns[i]: round(float(shap_values_for_instance[i]), 3) for i in reversed(top_indices) if abs(float(shap_values_for_instance[i])) > 0.01}
    return {"prediction": risk_level, "probability": prob_positive, "model_name": "Diabetes Genomic Classifier (RF)", "shap_explanation": shap_explanation, "confidence_score": confidence_score}

def get_diabetes_clinical_suggestions(shap_dict, risk_level):
    suggestions = []
    if risk_level == "High Risk": suggestions.append("<strong>Action Required:</strong> Immediate HbA1c testing and endocrinology consult recommended.")
    elif risk_level == "Pre-Diabetic": suggestions.append("<strong>Preventative Care:</strong> Recommend lifestyle intervention, dietary changes, and weight management.")
    if shap_dict.get('Glucose', 0) > 0.05: suggestions.append("Elevated fasting glucose is a primary driver of this risk profile.")
    if shap_dict.get('BMI', 0) > 0.05: suggestions.append("High BMI is significantly elevating diabetes risk. Target 5-10% body weight reduction.")
    if shap_dict.get('TCF7L2_Risk_Variant', 0) > 0.05: suggestions.append("<strong>Genomic Risk:</strong> Patient carries the TCF7L2 high-risk allele, indicating a strong genetic predisposition.")
    if not suggestions: suggestions.append("Continue routine annual screening and maintain a healthy lifestyle.")
    return suggestions

def process_diabetes_data(form_data):
    patient_info_dict = {"patient_name": form_data.get('patient_name'), "patient_dob": form_data.get('patient_dob'), "patient_gender": form_data.get('patient_gender')}
    safety_data_dict = { "is_pregnant": "Not Applicable", "active_bleeding": "Not Applicable" }
    clinical_data_dict = {"Pregnancies": float(form_data.get('Pregnancies', 0)), "Glucose": float(form_data.get('Glucose', 90)), "BloodPressure": float(form_data.get('BloodPressure', 80)), "SkinThickness": float(form_data.get('SkinThickness', 20)), "Insulin": float(form_data.get('Insulin', 80)), "BMI": float(form_data.get('BMI', 25.0)), "DiabetesPedigree": float(form_data.get('DiabetesPedigree', 0.5)), "Age": float(form_data.get('Age', 30)), "TCF7L2_Risk_Variant": float(form_data.get('TCF7L2_Risk_Variant', 0))}
    clinical_info_display = {"Age": form_data.get('Age'), "BMI": form_data.get('BMI'), "Fasting Glucose": f"{form_data.get('Glucose')} mg/dL", "Blood Pressure": f"{form_data.get('BloodPressure')} mmHg", "Family History Pedigree": form_data.get('DiabetesPedigree'), "TCF7L2 Variant": "Detected (High Risk)" if clinical_data_dict['TCF7L2_Risk_Variant'] == 1 else "Not Detected"}
    pred_data = run_diabetes_prediction(clinical_data_dict)
    results_dict = {"predicted_dose_mg_per_week": pred_data['prediction'], "probability_score": pred_data['probability'], "model_used": pred_data['model_name'], "confidence_score": pred_data['confidence_score'], "confidence_explanation": f"The AI model predicts a {pred_data['probability']}% probability of pathology.", "clinical_suggestions": get_diabetes_clinical_suggestions(pred_data['shap_explanation'], pred_data['prediction']), "shap_explanation": pred_data['shap_explanation'], "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "report_id": f"DIA-{datetime.now().strftime('%Y%m%d')}-{abs(hash(patient_info_dict['patient_name'])) % 10000}"}
    return patient_info_dict, clinical_info_display, safety_data_dict, results_dict

def run_vancomycin_prediction(patient_data_dict):
    patient_df = pd.DataFrame([patient_data_dict]).reindex(columns=vancomycin_model_columns, fill_value=0)
    
    prediction_array = vancomycin_model.predict(patient_df)
    raw_dose = float(prediction_array[0])
    
    rounded_dose = round(raw_dose / 250.0) * 250.0
    final_clinical_dose = max(500.0, min(4000.0, rounded_dose))
    
    # SHAP Generation
    shap_values = vancomycin_explainer.shap_values(patient_df)
    if isinstance(shap_values, list): shap_values_for_instance = shap_values[0][0]
    elif len(shap_values.shape) == 3: shap_values_for_instance = shap_values[0, :, 0]
    else: shap_values_for_instance = shap_values[0]
        
    top_indices = np.argsort(np.abs(shap_values_for_instance))[-5:]
    shap_explanation = {patient_df.columns[i]: round(float(shap_values_for_instance[i]), 2) for i in reversed(top_indices) if abs(float(shap_values_for_instance[i])) > 0.01}
    
    # LIME Generation with Wrapper
    lime_explanation = {}
    if vancomycin_lime_explainer is not None:
        try:
            numeric_row = patient_df.iloc[0].apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            def vanc_lime_wrapper(numpy_data):
                temp_df = pd.DataFrame(numpy_data, columns=vancomycin_model_columns)
                return vancomycin_model.predict(temp_df)
                
            lime_exp = vancomycin_lime_explainer.explain_instance(
                data_row=numeric_row,
                predict_fn=vanc_lime_wrapper
            )
            lime_explanation = {feat: round(weight, 2) for feat, weight in lime_exp.as_list()[:4]}
        except Exception as e:
            print(f"LIME Execution Error (Vancomycin): {e}")

    crcl = patient_data_dict.get('Calculated_CrCl', 100)
    confidence_score = "High" if 30 <= crcl <= 120 else "Medium"
    
    return {
        "prediction": int(final_clinical_dose), 
        "model_name": "Vancomycin XGBoost (Clinical Guardrails)", 
        "shap_explanation": shap_explanation, 
        "lime_explanation": lime_explanation,
        "confidence_score": confidence_score
    }

def process_vancomycin_data(form_data):
    patient_info_dict = {"patient_name": form_data.get('patient_name'), "patient_dob": form_data.get('patient_dob'), "patient_gender": form_data.get('Gender'), "patient_country": "N/A", "patient_address": "N/A"}
    safety_data_dict = {"is_pregnant": "Not Applicable", "active_bleeding": "Not Applicable", "platelet_count": "Not Applicable", "baseline_inr": "Not Applicable"}
    
    age = float(form_data.get('Age', 30))
    weight = float(form_data.get('Weight_kg', 70))
    height = float(form_data.get('Height_cm', 170))
    scr = float(form_data.get('Serum_Creatinine', 1.0))
    gender = form_data.get('Gender', 'Male')
    hla = 1.0 if form_data.get('hla_risk') == 'Yes' else 0.0
    agr = 1.0 if form_data.get('agr_mutation') == 'Yes' else 0.0
    gender_male = 1.0 if gender == 'Male' else 0.0
    
    crcl = ((140 - age) * weight) / (72 * scr)
    if gender == 'Female': crcl *= 0.85
    crcl = round(crcl, 1)

    clinical_data_dict = {
        'Age': age, 'Weight_kg': weight, 'Height_cm': height,
        'Serum_Creatinine': scr, 'Calculated_CrCl': crcl,
        'HLA_A_32_01_Risk': hla, 'agr_Group_II_Mutation': agr,
        'Gender_Male': gender_male
    }
    
    clinical_info_display = {
        "Age": age, "Weight (kg)": weight, "Serum Creatinine": f"{scr} mg/dL",
        "Est. CrCl": f"{crcl} mL/min",
        "HLA-A*32:01": "Positive (High Risk)" if hla == 1.0 else "Negative",
        "agr Group II": "Detected" if agr == 1.0 else "Not Detected"
    }
    
    pred_data = run_vancomycin_prediction(clinical_data_dict)
    
    suggestions = []
    if hla == 1.0: suggestions.append("<strong>CRITICAL WARNING:</strong> HLA-A*32:01 risk allele detected. High risk of DRESS syndrome. AI reduced recommended dose. Proceed with caution.")
    if agr == 1.0: suggestions.append("<strong>EFFICACY WARNING:</strong> agr Group II mutation detected in bacteria. Potential vancomycin tolerance. AI increased recommended dose.")
    if crcl < 30: suggestions.append("<strong>RENAL IMPAIRMENT:</strong> Est. CrCl < 30 mL/min. Pulse dosing by levels is recommended over continuous standard dosing.")
    if not suggestions: suggestions.append("Parameters indicate standard clearance. Monitor trough levels before the 4th dose.")
    
    results_dict = {
        "predicted_dose_mg_per_week": pred_data['prediction'], 
        "model_used": pred_data['model_name'], 
        "confidence_score": pred_data['confidence_score'], 
        "confidence_explanation": "Model evaluated pharmacokinetic CrCl and genome markers via log-normal distribution.", 
        "human_explanation": get_human_explanation(pred_data['shap_explanation']), 
        "clinical_suggestions": suggestions, 
        "shap_explanation": pred_data['shap_explanation'],
        "lime_explanation": pred_data.get('lime_explanation', {}),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "report_id": f"VANC-{datetime.now().strftime('%Y%m%d')}-{abs(hash(patient_info_dict['patient_name'])) % 10000}"
    }
    
    return patient_info_dict, clinical_info_display, safety_data_dict, results_dict

# =======================================================
# 6. APP ROUTES
# =======================================================

@app.route('/')
def home(): return render_template('index.html')

@app.route('/dataset')
def dataset():
    try:
        df = pd.read_csv('data/warfarin.csv')
        headers = df.columns.tolist()
        rows = df.head(200).to_dict('records')
        row_count = len(df)
    except: headers, rows, row_count = [], [], 0
    return render_template('dataset.html', headers=headers, rows=rows, row_count=row_count, showing_count=len(rows))

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message_body = request.form.get('message')
    if MAIL_USERNAME and MAIL_PASSWORD:
        try:
            msg = MIMEMultipart()
            msg['From'] = MAIL_USERNAME
            msg['To'] = ADMIN_RECEIVER_EMAIL
            msg['Subject'] = f"GenMedix Support: {subject}"
            body = f"<h3>New Contact Request</h3><p><strong>Name:</strong> {name}</p><p><strong>Email:</strong> {email}</p><hr><p><strong>Message:</strong></p><p>{message_body}</p>"
            msg.attach(MIMEText(body, 'html'))
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
                server.login(MAIL_USERNAME, MAIL_PASSWORD)
                server.sendmail(MAIL_USERNAME, ADMIN_RECEIVER_EMAIL, msg.as_string())
            flash('Message sent! We will contact you shortly.', 'success')
        except Exception as e: flash('Error sending message. Please try again later.', 'danger')
    else: flash('Message received (Demo Mode).', 'success')
    return redirect(url_for('about'))

@app.route('/pricing')
def pricing(): return render_template('pricing.html', key=STRIPE_PUBLISHABLE_KEY)

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    data = request.json
    plan_type, email = data.get('plan_type'), data.get('email')
    if plan_type == 'Individual': amount, product_name, max_seats = 1500, 'Individual Plan (1 Doctor)', 1
    else: amount, product_name, max_seats = 25000, 'Enterprise Plan (50 Doctors)', 50
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'], customer_email=email,
            line_items=[{'price_data': {'currency': 'usd', 'product_data': {'name': product_name}, 'unit_amount': amount}, 'quantity': 1}],
            mode='payment', success_url=url_for('register', _external=True), cancel_url=url_for('pricing', _external=True),
            metadata={'plan_type': plan_type, 'max_seats': max_seats, 'customer_email': email}
        )
        return jsonify({'id': session.id})
    except Exception as e: return jsonify(error=str(e)), 403

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    try: event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except: return 'Invalid payload', 400
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        new_sub = Subscription(email=session['metadata']['customer_email'], stripe_customer_id=session['customer'], plan_type=session['metadata']['plan_type'], max_seats=int(session['metadata']['max_seats']), is_active=True)
        db.session.merge(new_sub)
        db.session.commit()
    return jsonify(success=True)

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        if AdminEmail.query.filter_by(email=current_user.email).first(): return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        email, password = request.form.get('email'), request.form.get('password')
        if not AdminEmail.query.filter_by(email=email).first():
            flash("Unauthorized: Your email is not registered as an Administrator.", "danger")
            return redirect(url_for('admin_login'))
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            return redirect(url_for('admin_dashboard'))
        else: flash("Invalid credentials.", "danger")
    return render_template('admin_login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user, remember=True)
            flash('Login Successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid Email or Password.', 'danger')
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            if supabase:
                try:
                    supabase.auth.sign_in_with_otp({"email": email})
                    session['reset_email'] = email
                    flash('An OTP has been sent to your email address.', 'success')
                    return redirect(url_for('verify_reset_otp'))
                except Exception as e: flash('Error communicating with Supabase.', 'danger')
            else: flash('Supabase is not configured properly.', 'danger')
        else:
            flash('If that email is registered, an OTP has been sent.', 'info')
            session['reset_email'] = email 
            return redirect(url_for('verify_reset_otp'))
    return render_template('forgot_password.html')

@app.route('/verify_reset_otp', methods=['GET', 'POST'])
def verify_reset_otp():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    email = session.get('reset_email')
    if not email: return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        try:
            supabase.auth.verify_otp({"email": email, "token": request.form.get('otp'), "type": "email"})
            session['can_reset_password'] = True
            flash("OTP Verified! Please enter your new password.", "success")
            return redirect(url_for('set_new_password'))
        except: flash("Invalid or expired OTP.", "danger")
    return render_template('verify_reset_otp.html', email=email)

@app.route('/set_new_password', methods=['GET', 'POST'])
def set_new_password():
    if not session.get('can_reset_password') or not session.get('reset_email'): return redirect(url_for('login'))
    if request.method == 'POST':
        new_password, confirm_password = request.form.get('password'), request.form.get('confirm_password')
        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('set_new_password'))
        user = User.query.filter_by(email=session.get('reset_email')).first()
        if user:
            user.set_password(new_password)
            db.session.commit()
            session.pop('reset_email', None)
            session.pop('can_reset_password', None)
            flash('Password updated! You can now log in.', 'success')
            return redirect(url_for('login'))
        flash('Error updating password.', 'danger')
        return redirect(url_for('login'))
    return render_template('reset_password.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email, license_email = request.form.get('email'), request.form.get('license_email')
        password, password_confirm = request.form.get('password'), request.form.get('password_confirm')
        if password != password_confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        sub = Subscription.query.filter_by(email=license_email).first()
        if not sub or not sub.is_active:
            flash('No active subscription found.', 'danger')
            return redirect(url_for('register'))
        if sub.current_users >= sub.max_seats:
            flash(f'License limit reached.', 'danger')
            return redirect(url_for('register'))
        if sub.plan_type == 'Individual' and license_email != email:
            flash('Individual Plan violation.', 'danger')
            return redirect(url_for('register'))
        new_doctor = User(full_name=request.form.get('full_name'), email=email, medical_reg_id=request.form.get('medical_reg_id'), subscription_email=license_email)
        new_doctor.set_password(password)
        try:
            sub.current_users += 1
            db.session.add(new_doctor)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {e}', 'danger')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('reset_email', None)
    session.pop('can_reset_password', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    action = request.form.get('action')
    if request.method == 'POST':
        if action == 'update_details':
            current_user.full_name = request.form.get('full_name')
            current_user.email = request.form.get('email')
            db.session.commit()
            flash('Details updated.', 'success')
        elif action == 'delete_account':
            if not current_user.check_password(request.form.get('password')):
                flash("Incorrect password.", "danger")
                return redirect(url_for('account'))
            if current_user.subscription_email:
                sub = Subscription.query.filter_by(email=current_user.subscription_email).first()
                if sub and sub.current_users > 0: sub.current_users -= 1
            db.session.delete(current_user)
            db.session.commit()
            logout_user()
            flash("Account deleted.", "success")
            return redirect(url_for('home'))
    return render_template('account.html')

@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    if request.method == 'POST':
        aadhar = request.form.get('aadhar').replace(' ', '')
        patient = Patient.query.filter_by(aadhar=aadhar, dob=request.form.get('dob')).first()
        if patient:
            session['patient_aadhar'] = aadhar
            session['patient_name'] = patient.full_name
            return redirect(url_for('patient_dashboard'))
        flash("Invalid Credentials.", "danger")
    return render_template('patient_login.html')

@app.route('/patient/dashboard')
def patient_dashboard():
    if 'patient_aadhar' not in session: return redirect(url_for('patient_login'))
    aadhar = session['patient_aadhar']
    patient_ids = [p.id for p in Patient.query.filter_by(aadhar=aadhar).all()]
    reports = Report.query.filter(Report.patient_id.in_(patient_ids)).order_by(Report.generated_at.desc()).all()
    return render_template('patient_dashboard.html', reports=reports, patient_name=session['patient_name'], aadhar_display=f"{aadhar[:4]} {aadhar[4:8]} {aadhar[8:]}")

@app.route('/patient/logout')
def patient_logout():
    session.pop('patient_aadhar', None)
    session.pop('patient_name', None)
    flash("Patient logged out.", "success")
    return redirect(url_for('home'))

@app.route('/patient/download/<int:report_id>')
def download_archived_report_patient(report_id):
    if 'patient_aadhar' not in session: return redirect(url_for('patient_login'))
    report = Report.query.get_or_404(report_id)
    if report.patient.aadhar != session['patient_aadhar']: abort(403)
    if not report.pdf_storage_path or not supabase: return redirect(url_for('patient_dashboard'))
    res = supabase.storage.from_("medical_reports").create_signed_url(report.pdf_storage_path, 60)
    return redirect(res['signedURL'])

@app.route('/dashboard')
@login_required
def dashboard():
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.full_name).all()
    total_reports = db.session.query(Report).join(Patient).filter(Patient.doctor_id == current_user.id).count()
    gender_data = {'Male': 0, 'Female': 0, 'Other': 0}
    blood_data = {}
    confidence_data = {'High': 0, 'Medium': 0, 'Low': 0}
    for p in patients:
        gender_data[p.gender if p.gender in ['Male', 'Female'] else 'Other'] += 1
        bg = p.blood_group if p.blood_group else 'Unknown'
        blood_data[bg] = blood_data.get(bg, 0) + 1
    for r in Report.query.filter(Report.patient_id.in_([p.id for p in patients])).all():
        if r.confidence in confidence_data: confidence_data[r.confidence] += 1
    return render_template('dashboard.html', patients=patients, total_reports=total_reports, total_patients=len(patients), gender_data=gender_data, blood_data=blood_data, confidence_data=confidence_data)

@app.route('/export/patients')
@login_required
def export_patients_csv():
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.full_name).all()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Patient Name', 'Aadhar ID', 'Date of Birth', 'Gender', 'Blood Group', 'Phone Number', 'Emergency Contact', 'Allergies', 'Country'])
    for p in patients: cw.writerow([p.full_name, p.aadhar, p.dob, p.gender, p.blood_group or 'N/A', p.phone or 'N/A', p.emergency_contact or 'N/A', p.allergies or 'None', p.country or 'N/A'])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=GenMedix_Patient_Export.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/add_patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    if request.method == 'POST':
        clean_aadhar = request.form.get('aadhar').replace(' ', '')
        if len(clean_aadhar) != 12 or not clean_aadhar.isdigit():
            flash('Invalid ID format.', 'danger')
            return render_template('add_patient.html')
        if Patient.query.filter_by(aadhar=clean_aadhar, doctor_id=current_user.id).first():
            flash('You have already registered this patient.', 'warning')
            return render_template('add_patient.html')
        new_patient = Patient(
            full_name=request.form.get('full_name'), aadhar=clean_aadhar, dob=request.form.get('dob'),
            gender=request.form.get('gender'), blood_group=request.form.get('blood_group'),
            country=request.form.get('country'), address=request.form.get('address'),
            phone=request.form.get('phone'), emergency_contact=request.form.get('emergency_contact'),
            allergies=request.form.get('allergies'), medical_history=request.form.get('medical_history'),
            current_medications=request.form.get('current_medications'), doctor_id=current_user.id
        )
        db.session.add(new_patient)
        db.session.commit()
        flash('Patient added successfully.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('add_patient.html')

@app.route('/view_patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    reports = Report.query.filter_by(patient_id=patient.id).order_by(Report.generated_at.desc()).all()
    notes = Note.query.filter_by(patient_id=patient.id).order_by(Note.created_at.desc()).all()
    return render_template('view_patient.html', patient=patient, reports=reports, notes=notes)

@app.route('/patient/<int:patient_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        patient.full_name = request.form.get('full_name')
        patient.dob = request.form.get('dob')
        patient.gender = request.form.get('gender')
        patient.country = request.form.get('country')
        patient.address = request.form.get('address')
        db.session.commit()
        flash('Details updated.', 'success')
        return redirect(url_for('view_patient', patient_id=patient.id))
    return render_template('edit_patient.html', patient=patient)

@app.route('/patient/<int:patient_id>/delete', methods=['POST'])
@login_required
def delete_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    db.session.delete(patient)
    db.session.commit()
    flash("Patient deleted.", "success")
    return redirect(url_for('dashboard'))

@app.route('/patient/<int:patient_id>/new_assessment', methods=['GET'])
@login_required
def new_assessment(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    return render_template('assessment_type.html', patient=patient)

@app.route('/patient/<int:patient_id>/select_disease', methods=['GET'])
@login_required
def select_disease(patient_id): return render_template('select_disease.html', patient=Patient.query.get_or_404(patient_id))

@app.route('/patient/<int:patient_id>/select_drug', methods=['GET'])
@login_required
def select_drug(patient_id): return render_template('select_drug.html', patient=Patient.query.get_or_404(patient_id))

@app.route('/patient/<int:patient_id>/redirect_form', methods=['POST'])
@login_required
def redirect_to_drug_form(patient_id):
    drug = request.form.get('drug_name')
    if drug == 'warfarin': return redirect(url_for('warfarin_form', patient_id=patient_id))
    if drug == 'vancomycin': return redirect(url_for('vancomycin_form', patient_id=patient_id))
    return redirect(url_for('dashboard'))

@app.route('/patient/<int:patient_id>/warfarin_form', methods=['GET'])
@login_required
def warfarin_form(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    calculated_age = 0
    try:
        dob = datetime.strptime(patient.dob, '%Y-%m-%d')
        today = datetime.today()
        calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except: pass
    return render_template('warfarin_form.html', patient=patient, calculated_age=calculated_age)

@app.route('/patient/<int:patient_id>/generate_warfarin_report', methods=['POST'])
@login_required
def generate_warfarin_report(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient_info, clinical_info, safety_info, results = process_prediction_data(request.form)
    doctor_name = request.form.get('doctor_name')
    interacting_drugs = request.form.getlist('interacting_drugs')
    interaction_warnings = get_interaction_warnings(interacting_drugs)

    full_report_data = {
        "patient_info": patient_info, "clinical_info": clinical_info, "safety_info": safety_info, 
        "results": results, "doctor_name": doctor_name, "interacting_drugs": interacting_drugs
    }

    html_string = render_template('display_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=None, interaction_warnings=interaction_warnings)
    pdf_bytes = HTML(string=html_string).write_pdf()

    pdf_path = None
    if supabase:
        try:
            filename = f"report_{patient.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            supabase.storage.from_("medical_reports").upload(path=filename, file=pdf_bytes, file_options={"content-type": "application/pdf"})
            pdf_path = filename 
        except Exception as e: pass

    new_report = Report(
        drug_name="Warfarin", predicted_dose=f"{results['predicted_dose_mg_per_week']} mg/week",
        model_used=results['model_used'], confidence=results['confidence_score'],
        doctor_name=doctor_name, report_data_json=json.dumps(full_report_data),
        patient_id=patient.id, pdf_storage_path=pdf_path 
    )
    db.session.add(new_report)
    db.session.commit()
    return make_response(render_template('display_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=request, interaction_warnings=interaction_warnings, report_obj=new_report))

@app.route('/patient/<int:patient_id>/diabetes_form', methods=['GET'])
@login_required
def diabetes_form(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    calculated_age = 0
    try:
        dob = datetime.strptime(patient.dob, '%Y-%m-%d')
        today = datetime.today()
        calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except: pass
    return render_template('diabetes_form.html', patient=patient, calculated_age=calculated_age)

@app.route('/patient/<int:patient_id>/generate_diabetes_report', methods=['POST'])
@login_required
def generate_diabetes_report(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient_info, clinical_info, safety_info, results = process_diabetes_data(request.form)
    doctor_name = request.form.get('doctor_name')

    full_report_data = {
        "patient_info": patient_info, "clinical_info": clinical_info,
        "safety_info": safety_info, "results": results, "doctor_name": doctor_name
    }

    html_string = render_template('display_disease_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=None)
    pdf_bytes = HTML(string=html_string).write_pdf()

    new_report = Report(
        drug_name="Type 2 Diabetes Assessment", predicted_dose=results['predicted_dose_mg_per_week'], 
        model_used=results['model_used'], confidence=results['confidence_score'],
        doctor_name=doctor_name, report_data_json=json.dumps(full_report_data),
        patient_id=patient.id, pdf_storage_path=None
    )
    db.session.add(new_report)
    db.session.commit()
    return make_response(render_template('display_disease_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=request, report_obj=new_report))

@app.route('/patient/<int:patient_id>/vancomycin_form', methods=['GET'])
@login_required
def vancomycin_form(patient_id):
    if not current_user.is_beta_tester:
        flash("Access Denied: The Vancomycin module is currently restricted to Beta Testers.", "danger")
        return redirect(url_for('dashboard'))
    patient = Patient.query.get_or_404(patient_id)
    calculated_age = 0
    try:
        dob = datetime.strptime(patient.dob, '%Y-%m-%d')
        today = datetime.today()
        calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except: pass
    return render_template('vancomycin_form.html', patient=patient, calculated_age=calculated_age)

@app.route('/patient/<int:patient_id>/generate_vancomycin_report', methods=['POST'])
@login_required
def generate_vancomycin_report(patient_id):
    if not current_user.is_beta_tester: return redirect(url_for('dashboard'))
    patient = Patient.query.get_or_404(patient_id)
    patient_info, clinical_info, safety_info, results = process_vancomycin_data(request.form)
    doctor_name = request.form.get('doctor_name')
    
    interacting_drugs = request.form.getlist('interacting_drugs')
    interaction_warnings = []
    if "NSAIDs" in interacting_drugs: interaction_warnings.append("<strong>Interaction: NSAIDs</strong>. Concomitant use increases risk of acute kidney injury.")
    if "Aminoglycosides" in interacting_drugs: interaction_warnings.append("<strong>Severe Interaction: Aminoglycosides</strong>. Highly synergistic nephrotoxicity. Monitor renal function closely.")
    if "Piperacillin" in interacting_drugs: interaction_warnings.append("<strong>Interaction: Piperacillin-Tazobactam</strong>. Increased risk of nephrotoxicity compared to vancomycin alone.")

    full_report_data = {
        "patient_info": patient_info, "clinical_info": clinical_info, "safety_info": safety_info, 
        "results": results, "doctor_name": doctor_name, "interacting_drugs": interacting_drugs
    }

    html_string = render_template('display_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=None, interaction_warnings=interaction_warnings)
    pdf_bytes = HTML(string=html_string).write_pdf()

    pdf_path = None
    if supabase:
        try:
            filename = f"report_vanc_{patient.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            supabase.storage.from_("medical_reports").upload(path=filename, file=pdf_bytes, file_options={"content-type": "application/pdf"})
            pdf_path = filename 
        except Exception as e: pass

    new_report = Report(
        drug_name="Vancomycin", predicted_dose=f"{results['predicted_dose_mg_per_week']} mg Daily Target",
        model_used=results['model_used'], confidence=results['confidence_score'],
        doctor_name=doctor_name, report_data_json=json.dumps(full_report_data),
        patient_id=patient.id, pdf_storage_path=pdf_path 
    )
    db.session.add(new_report)
    db.session.commit()
    return make_response(render_template('display_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=request, interaction_warnings=interaction_warnings, report_obj=new_report))

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    report = Report.query.get_or_404(report_id)
    if report.patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    try: report_data = json.loads(report.report_data_json)
    except: return redirect(url_for('view_patient', patient_id=report.patient_id))
    
    if report.drug_name == "Type 2 Diabetes Assessment":
        template_name = 'display_disease_report.html'
        interaction_warnings = []
    else:
        template_name = 'display_report.html'
        interaction_warnings = get_interaction_warnings(report_data.get('interacting_drugs', []))
    
    return make_response(render_template(
        template_name, patient_info=report_data.get('patient_info'), clinical_info=report_data.get('clinical_info'), 
        safety_info=report_data.get('safety_info', {}), results=report_data.get('results'), 
        doctor_name=report_data.get('doctor_name'), request=None, interaction_warnings=interaction_warnings, report_obj=report
    ))

@app.route('/download_archived_report/<int:report_id>')
@login_required
def download_archived_report(report_id):
    report = Report.query.get_or_404(report_id)
    if report.patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    if not report.pdf_storage_path or not supabase: return redirect(url_for('view_report', report_id=report_id))
    try:
        file_bytes = supabase.storage.from_("medical_reports").download(report.pdf_storage_path)
        return Response(file_bytes, mimetype='application/pdf', headers={'Content-Disposition': f'attachment;filename=Report_{report.patient.full_name}.pdf'})
    except Exception as e: return redirect(url_for('view_report', report_id=report_id))

@app.route('/download_report', methods=['POST'])
@login_required
def download_report():
    form_data = request.form
    drug_name = form_data.get('drug_name', '')
    
    if drug_name == 'Vancomycin':
        patient_info, clinical_info, safety_info, results = process_vancomycin_data(form_data)
    else:
        patient_info, clinical_info, safety_info, results = process_prediction_data(form_data)
        
    doctor_name = form_data.get('doctor_name', current_user.full_name)
    html_string = render_template('display_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=None)
    pdf_file = HTML(string=html_string).write_pdf()
    return Response(pdf_file, mimetype='application/pdf', headers={'Content-Disposition': 'attachment;filename=DosageReport.pdf'})

@app.route('/report/<int:report_id>/delete', methods=['POST'])
@login_required
def delete_report(report_id):
    report = Report.query.get_or_404(report_id)
    pid = report.patient.id
    if report.patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    if report.pdf_storage_path and supabase:
        try: supabase.storage.from_("medical_reports").remove([report.pdf_storage_path])
        except: pass
    db.session.delete(report)
    db.session.commit()
    flash("Report deleted.", "success")
    return redirect(url_for('view_patient', patient_id=pid))

@app.route('/patient/<int:patient_id>/add_note', methods=['POST'])
@login_required
def add_note(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    new_note = Note(note_text=request.form.get('note_text'), patient_id=patient.id, doctor_id=current_user.id)
    db.session.add(new_note)
    db.session.commit()
    flash("Note added.", "success")
    return redirect(url_for('view_patient', patient_id=patient_id, _anchor='notes-tab'))

# =======================================================
# 8. SUPER ADMIN COMMAND CENTER
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
    
    warfarin_count = Report.query.filter_by(drug_name='Warfarin').count()
    diabetes_count = Report.query.filter_by(drug_name='Type 2 Diabetes Assessment').count()
    vancomycin_count = Report.query.filter_by(drug_name='Vancomycin').count()
    
    basedir = os.path.abspath(os.path.dirname(__file__))
    maintenance_active = os.path.exists(os.path.join(basedir, 'maintenance.json'))
    
    current_broadcast = {}
    broadcast_file = os.path.join(basedir, 'broadcast.json')
    if os.path.exists(broadcast_file):
        try:
            with open(broadcast_file, 'r') as f:
                current_broadcast = json.load(f)
        except Exception: pass
            
    current_maintenance = {}
    if maintenance_active:
        try:
            with open(os.path.join(basedir, 'maintenance.json'), 'r') as f:
                current_maintenance = json.load(f)
        except: pass
            
    return render_template(
        'admin_dashboard.html', 
        total_doctors=total_doctors, total_patients=total_patients,
        total_reports=total_reports, total_subs=total_subscriptions,
        all_doctors=all_doctors, all_subs=all_subs, all_admins=all_admins,
        maintenance_active=maintenance_active, 
        current_broadcast=current_broadcast,
        current_maintenance=current_maintenance,
        warfarin_count=warfarin_count, diabetes_count=diabetes_count,
        vancomycin_count=vancomycin_count
    )

@app.route('/admin/doctor/<int:doc_id>/toggle_beta', methods=['POST'])
@login_required
@admin_required
def admin_toggle_beta(doc_id):
    doctor = User.query.get_or_404(doc_id)
    doctor.is_beta_tester = not doctor.is_beta_tester
    db.session.commit()
    
    status = "GRANTED" if doctor.is_beta_tester else "REVOKED"
    flash(f"Beta Testing Access {status} for Dr. {doctor.full_name}.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/system/add_admin', methods=['POST'])
@login_required
@admin_required
def admin_add_admin():
    new_email = request.form.get('email')
    if AdminEmail.query.filter_by(email=new_email).first(): flash("Email already has admin privileges.", "warning")
    else:
        db.session.add(AdminEmail(email=new_email, added_by=current_user.email))
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
        m_title = request.form.get('m_title', 'System Upgrades in Progress')
        m_message = request.form.get('m_message', 'GenMedix is currently locked down for critical infrastructure updates. No patient data is at risk. Services will resume shortly.')
        with open(maintenance_file, 'w') as f:
            json.dump({"status": "OFFLINE", "end_time": end_time, "title": m_title, "message": m_message}, f)
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
            "id": str(datetime.now().timestamp()), 
            "type": request.form.get('broadcast_type', 'info'),
            "message": message.strip(),
            "is_dismissible": request.form.get('is_dismissible') == 'on',
            "cta_text": request.form.get('cta_text', ''),
            "cta_link": request.form.get('cta_link', '#')
        }
        with open(broadcast_file, 'w') as f: 
            json.dump(data, f)
        flash("Enterprise Broadcast transmitted.", "success")
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
        if sub and sub.current_users > 0: sub.current_users -= 1
    db.session.delete(doctor)
    db.session.commit()
    flash(f"Physician {doc_name} and all their clinical data have been permanently wiped.", "success")
    return redirect(url_for('admin_dashboard'))

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
        new_sub = Subscription(email=email, stripe_customer_id="MANUAL_OVERRIDE", plan_type=plan_type, max_seats=int(max_seats), is_active=True)
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
        flash("Email already registered.", "danger")
        return redirect(url_for('admin_dashboard'))

    sub = Subscription.query.filter_by(email=license_email).first()
    if not sub or not sub.is_active:
        flash("Invalid License Email.", "danger")
        return redirect(url_for('admin_dashboard'))

    if sub.current_users >= sub.max_seats:
        flash("License limit reached.", "danger")
        return redirect(url_for('admin_dashboard'))

    new_doc = User(full_name=full_name, email=email, medical_reg_id=reg_id, subscription_email=license_email)
    new_doc.set_password(password)
    sub.current_users += 1
    db.session.add(new_doc)
    db.session.commit()
    flash(f"Account for Dr. {full_name} created.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/doctor/<int:doc_id>/edit', methods=['POST'])
@login_required
@admin_required
def admin_edit_doctor(doc_id):
    doctor = User.query.get_or_404(doc_id)
    if AdminEmail.query.filter_by(email=doctor.email).first():
        flash("Cannot edit a Master Admin.", "danger")
        return redirect(url_for('admin_dashboard'))
    doctor.full_name = request.form.get('full_name')
    doctor.email = request.form.get('email')
    doctor.medical_reg_id = request.form.get('medical_reg_id')
    db.session.commit()
    flash(f"Profile updated successfully.", "success")
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)