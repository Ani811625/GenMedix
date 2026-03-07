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
from sqlalchemy import inspect


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
    print(f"--- ✅ SUCCESS: FOUND EXTERNAL DATABASE URL ---")
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL.replace("postgres://", "postgresql://")
else:
    print("--- ⚠️ WARNING: NO DATABASE_URL FOUND. USING LOCAL SQLITE ---")
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'project.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- CRITICAL: INITIALIZE DB BEFORE MODELS ---
db = SQLAlchemy(app)

# --- STRIPE CONFIGURATION ---
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET')

# --- MAIL CONFIGURATION ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

MAIL_USERNAME = os.environ.get("MAIL_USERNAME") # Your Gmail
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD") # Your App Password
ADMIN_RECEIVER_EMAIL = os.environ.get("MAIL_USERNAME") # Send alerts to yourself

# --- SUPABASE CLIENT SETUP (Auth & Storage) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("--- ✅ SUCCESS: CONNECTED TO SUPABASE AUTH & STORAGE ---")
    except Exception as e:
        print(f"--- ❌ ERROR CONNECTING TO SUPABASE: {e} ---")

# --- LOGIN MANAGER ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Please log in to access the clinical dashboard."
login_manager.login_message_category = "info"

# --- MASTER ADMIN CONFIGURATION ---
# The only email address allowed to access the Super Admin Panel
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'aniruddhas387@gmail.com')

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.email != ADMIN_EMAIL:
            flash("🛡️ Access Denied: Administrator privileges required. This incident has been logged.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.context_processor
def inject_user():
    return dict(current_user=current_user)

# =======================================================
# 2. DATABASE MODELS
# =======================================================

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
    
    # Subscription Link
    subscription_email = db.Column(db.String(150), db.ForeignKey('subscription.email'))
    
    patients = db.relationship('Patient', backref='doctor', lazy=True, cascade="all, delete-orphan")
    notes = db.relationship('Note', backref='doctor', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    
    # --- Basic Identity ---
    full_name = db.Column(db.String(150), nullable=False)
    dob = db.Column(db.String(10)) 
    gender = db.Column(db.String(10))
    aadhar = db.Column(db.String(14), index=True)
    
    # --- Contact ---
    country = db.Column(db.String(50))
    address = db.Column(db.String(200))
    phone = db.Column(db.String(20))
    emergency_contact = db.Column(db.String(100))
    
    # --- Clinical Profile ---
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
# 3. LOAD AI MODELS (Warfarin & Diabetes)
# =======================================================

@app.before_request
def check_maintenance_and_db():
    if os.environ.get('MAINTENANCE_MODE') == 'true':
        if request.endpoint and request.endpoint != 'static':
            return render_template('maintenance.html'), 503

    try:
        inspector = inspect(db.engine)
        if not inspector.has_table("user"):
             with app.app_context():
                db.create_all()
    except Exception as e:
        print(f"--- ❌ DB Check Error: {e} ---")

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print("--- Loading models... ---")
try:
    # 1. Warfarin Regression Models
    base_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_base_v1.pkl'))
    base_model_columns = joblib.load(os.path.join(MODEL_DIR, 'base_model_columns.pkl'))
    enhanced_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_enhanced_v1.pkl'))
    enhanced_model_columns = joblib.load(os.path.join(MODEL_DIR, 'model_columns.pkl'))
    
    enhanced_explainer = shap.TreeExplainer(enhanced_model)
    base_explainer = shap.TreeExplainer(base_model)
    
    # 2. NEW: Diabetes Classification Model
    diabetes_model = joblib.load(os.path.join(MODEL_DIR, 'diabetes_model_v1.pkl'))
    diabetes_model_columns = joblib.load(os.path.join(MODEL_DIR, 'diabetes_model_columns.pkl'))
    diabetes_explainer = shap.TreeExplainer(diabetes_model)
    print("--- All models loaded. ---")
except Exception as e:
    print(f"--- FATAL ERROR loading models: {e} ---")
    pass 

# =======================================================
# 4. WARFARIN ML LOGIC
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
    is_enhanced = False
    for key in patient_data_dict.keys():
        if key.startswith('CYP2C9_genotypes') or key.startswith('VKORC1_genotype'):
            is_enhanced = True
            break
            
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

    patient_df = pd.DataFrame([patient_data_dict])
    patient_df = patient_df.reindex(columns=columns_to_use, fill_value=0)
    
    prediction_array = model_to_use.predict(patient_df)
    predicted_dose = round(prediction_array[0], 2)
    
    tree_predictions = [tree.predict(patient_df) for tree in model_to_use.estimators_]
    std_dev = np.std(tree_predictions)
    
    shap_values = explainer_to_use.shap_values(patient_df)
    feature_names = patient_df.columns
    shap_values_for_instance = shap_values[0]
    abs_shap_values = np.abs(shap_values_for_instance)
    top_indices = np.argsort(abs_shap_values)[-5:] 
    
    shap_explanation = {}
    for i in reversed(top_indices): 
        if abs_shap_values[i] > 0:
            feature_name = feature_names[i]
            shap_explanation[feature_name] = round(shap_values_for_instance[i], 2)

    return {"prediction": predicted_dose, "model_name": model_name, "shap_explanation": shap_explanation, "std_dev": std_dev}

def get_confidence_score(std_dev):
    if std_dev < 0.5: return "High", "Model estimators are in strong agreement."
    elif std_dev < 1.0: return "Medium", "Model estimators show variance. Use with caution."
    else: return "Low", "Significant disagreement in estimators. Proceed with caution."

def get_human_explanation(shap_dict):
    explanations = []
    for feature, value in shap_dict.items():
        if feature == "Weight__kg_": display_name = "Weight"
        elif feature == "Height__cm_": display_name = "Height"
        elif feature.startswith("CYP2C9"): display_name = "CYP2C9 Genotype"
        elif feature.startswith("VKORC1"): display_name = "VKORC1 Genotype"
        else: display_name = feature.replace("Race_", "")
        direction = "<strong>increased</strong>" if value > 0 else "<strong>decreased</strong>"
        explanations.append(f"<strong>{display_name}</strong> {direction} the dose recommendation.")
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
        "Weight__kg_": float(form_data.get('Weight__kg_')),
    }
    
    race = form_data.get('Race')
    cyp2c9 = form_data.get('CYP2C9_genotypes')
    vkorc1 = form_data.get('VKORC1_genotype')

    if race: clinical_data_dict[race] = 1.0
    if cyp2c9: clinical_data_dict[cyp2c9] = 1.0
    if vkorc1: clinical_data_dict[vkorc1] = 1.0

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
    human_expl = get_human_explanation(pred_data['shap_explanation'])
    suggestions = get_clinical_suggestions(pred_data['shap_explanation'], confidence)
    
    if form_data.get('is_pregnant'): suggestions.append("<strong>CONTRAINDICATION:</strong> Patient marked Pregnant. Warfarin contraindicated.")
    if form_data.get('active_bleeding'): suggestions.append("<strong>CONTRAINDICATION:</strong> Active bleeding detected.")
    if form_data.get('platelet_count'):
        try:
            if int(form_data.get('platelet_count')) < 50000: suggestions.append("<strong>SAFETY ALERT:</strong> Severe Thrombocytopenia (<50k).")
        except: pass

    results_dict = {
        "predicted_dose_mg_per_week": pred_data['prediction'],
        "model_used": pred_data['model_name'],
        "confidence_score": confidence,
        "confidence_explanation": conf_expl,
        "human_explanation": human_expl,
        "clinical_suggestions": suggestions,
        "shap_explanation": pred_data.get('shap_explanation', {}),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_id": f"GM-{datetime.now().strftime('%Y%m%d')}-{abs(hash(patient_info_dict['patient_name'])) % 10000}"
    }
    return patient_info_dict, clinical_info_display, safety_data_dict, results_dict

# =======================================================
# 5. NEW: DIABETES ML CLASSIFICATION LOGIC
# =======================================================

def run_diabetes_prediction(patient_data_dict):
    patient_df = pd.DataFrame([patient_data_dict]).reindex(columns=diabetes_model_columns, fill_value=0)
    
    # 1. Predict Probability
    probabilities = diabetes_model.predict_proba(patient_df)[0]
    prob_positive = round(probabilities[1] * 100, 1) # Probability of being Diabetic

    # 2. Assign Risk Level based on thresholds
    if prob_positive >= 60.0:
        risk_level = "High Risk"
        confidence_score = "High" if prob_positive > 80 else "Medium"
    elif prob_positive >= 40.0:
        risk_level = "Pre-Diabetic"
        confidence_score = "Medium"
    else:
        risk_level = "Low Risk"
        confidence_score = "High" if prob_positive < 20 else "Medium"

    # 3. SHAP Explainability for Classifiers (Version-Proof Fix)
    shap_values = diabetes_explainer.shap_values(patient_df)
    
    # Extract the raw array if SHAP returned an Explanation object
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values

    # Handle the different ways SHAP returns classification arrays
    if isinstance(shap_values, list):
        # Older SHAP: returns a list where index 1 is the positive class
        shap_values_for_instance = shap_values[1][0] 
    else:
        # Newer SHAP: returns a 3D array (samples, features, classes)
        shap_array = np.array(shap_values)
        if len(shap_array.shape) == 3:
            shap_values_for_instance = shap_array[0, :, 1] # Extract Class 1 (Diabetes) for Sample 0
        else:
            shap_values_for_instance = shap_array[0]

    feature_names = patient_df.columns
    top_indices = np.argsort(np.abs(shap_values_for_instance))[-5:]
    
    # Use float() to guarantee JSON compatibility and prevent Numpy dimension errors
    shap_explanation = {}
    for i in reversed(top_indices):
        val = float(shap_values_for_instance[i])
        if abs(val) > 0.01:
            shap_explanation[feature_names[i]] = round(val, 3)

    return {
        "prediction": risk_level, 
        "probability": prob_positive,
        "model_name": "Diabetes Genomic Classifier (RF)",
        "shap_explanation": shap_explanation,
        "confidence_score": confidence_score
    }

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
    patient_info_dict = {
        "patient_name": form_data.get('patient_name'), 
        "patient_dob": form_data.get('patient_dob'), 
        "patient_gender": form_data.get('patient_gender')
    }
    safety_data_dict = { "is_pregnant": "Not Applicable", "active_bleeding": "Not Applicable" }

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
        "Age": form_data.get('Age'), "BMI": form_data.get('BMI'),
        "Fasting Glucose": f"{form_data.get('Glucose')} mg/dL",
        "Blood Pressure": f"{form_data.get('BloodPressure')} mmHg",
        "Family History Pedigree": form_data.get('DiabetesPedigree'),
        "TCF7L2 Variant": "Detected (High Risk)" if clinical_data_dict['TCF7L2_Risk_Variant'] == 1 else "Not Detected"
    }

    pred_data = run_diabetes_prediction(clinical_data_dict)
    suggestions = get_diabetes_clinical_suggestions(pred_data['shap_explanation'], pred_data['prediction'])

    results_dict = {
        "predicted_dose_mg_per_week": pred_data['prediction'], # Repurposed as Risk Level for the DB structure
        "probability_score": pred_data['probability'],
        "model_used": pred_data['model_name'],
        "confidence_score": pred_data['confidence_score'],
        "confidence_explanation": f"The AI model predicts a {pred_data['probability']}% probability of pathology.",
        "clinical_suggestions": suggestions,
        "shap_explanation": pred_data['shap_explanation'],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_id": f"DIA-{datetime.now().strftime('%Y%m%d')}-{abs(hash(patient_info_dict['patient_name'])) % 10000}"
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
    except:
        headers, rows, row_count = [], [], 0
    return render_template('dataset.html', headers=headers, rows=rows, row_count=row_count, showing_count=len(rows))

@app.route('/about')
def about():
    return render_template('about.html')

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

            body = f"""
            <h3>New Contact Request</h3>
            <p><strong>Name:</strong> {name}</p>
            <p><strong>Email:</strong> {email}</p>
            <hr>
            <p><strong>Message:</strong></p>
            <p>{message_body}</p>
            """
            msg.attach(MIMEText(body, 'html'))

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
                server.login(MAIL_USERNAME, MAIL_PASSWORD)
                server.sendmail(MAIL_USERNAME, ADMIN_RECEIVER_EMAIL, msg.as_string())
            
            flash('Message sent! We will contact you shortly.', 'success')
        except Exception as e:
            print(f"Email Error: {e}")
            flash('Error sending message. Please try again later.', 'danger')
    else:
        print(f"--- FAKE EMAIL SENT ---\nFrom: {email}\nMsg: {message_body}")
        flash('Message received (Demo Mode).', 'success')

    return redirect(url_for('about'))

# --- STRIPE ROUTES ---

@app.route('/pricing')
def pricing():
    return render_template('pricing.html', key=STRIPE_PUBLISHABLE_KEY)

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    data = request.json
    plan_type = data.get('plan_type')
    email = data.get('email')

    if plan_type == 'Individual':
        amount = 1500 
        product_name = 'Individual Plan (1 Doctor)'
        max_seats = 1
    else:
        amount = 25000 
        product_name = 'Enterprise Plan (50 Doctors)'
        max_seats = 50

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=email,
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {'name': product_name},
                    'unit_amount': amount,
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('register', _external=True),
            cancel_url=url_for('pricing', _external=True),
            metadata={
                'plan_type': plan_type, 
                'max_seats': max_seats,
                'customer_email': email
            }
        )
        return jsonify({'id': session.id})
    except Exception as e:
        return jsonify(error=str(e)), 403

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except ValueError as e: return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError as e: return 'Invalid signature', 400

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        customer_email = session['metadata']['customer_email']
        plan_type = session['metadata']['plan_type']
        max_seats = session['metadata']['max_seats']
        customer_id = session['customer']

        new_sub = Subscription(
            email=customer_email,
            stripe_customer_id=customer_id,
            plan_type=plan_type,
            max_seats=int(max_seats),
            is_active=True
        )
        db.session.merge(new_sub)
        db.session.commit()
        print(f"✅ Subscription Activated: {customer_email}")

    return jsonify(success=True)

# --- AUTH ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            flash('Login Successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid Email or Password. Please try again.', 'danger')
            
    return render_template('login.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

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
                except Exception as e:
                    print(f"Supabase OTP Error: {e}")
                    flash('Error communicating with Supabase. Please try again.', 'danger')
            else:
                flash('Supabase is not configured properly.', 'danger')
        else:
            flash('If that email is registered, an OTP has been sent.', 'info')
            session['reset_email'] = email 
            return redirect(url_for('verify_reset_otp'))

    return render_template('forgot_password.html')


@app.route('/verify_reset_otp', methods=['GET', 'POST'])
def verify_reset_otp():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    email = session.get('reset_email')
    if not email:
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        otp = request.form.get('otp')
        try:
            supabase.auth.verify_otp({"email": email, "token": otp, "type": "email"})
            session['can_reset_password'] = True
            flash("OTP Verified! Please enter your new password.", "success")
            return redirect(url_for('set_new_password'))
        except Exception as e:
            print(f"OTP Verification Error: {e}")
            flash("Invalid or expired OTP. Please try again.", "danger")
            
    return render_template('verify_reset_otp.html', email=email)


@app.route('/set_new_password', methods=['GET', 'POST'])
def set_new_password():
    if not session.get('can_reset_password') or not session.get('reset_email'):
        flash("Unauthorized access.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('set_new_password'))

        email = session.get('reset_email')
        user = User.query.filter_by(email=email).first()
        
        if user:
            user.set_password(new_password)
            db.session.commit()
            
            session.pop('reset_email', None)
            session.pop('can_reset_password', None)
            
            flash('Your password has been successfully updated! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Error updating password. User not found.', 'danger')
            return redirect(url_for('login'))

    return render_template('reset_password.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        license_email = request.form.get('license_email')
        name = request.form.get('full_name')
        reg_id = request.form.get('medical_reg_id')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')

        if password != password_confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))

        sub = Subscription.query.filter_by(email=license_email).first()
        if not sub or not sub.is_active:
            flash('No active subscription found for this License Email.', 'danger')
            return redirect(url_for('register'))
            
        if sub.current_users >= sub.max_seats:
            flash(f'License limit reached for this plan.', 'danger')
            return redirect(url_for('register'))

        if sub.plan_type == 'Individual' and license_email != email:
            flash('Individual Plan violation: Register with the subscribed email.', 'danger')
            return redirect(url_for('register'))

        new_doctor = User(full_name=name, email=email, medical_reg_id=reg_id, subscription_email=license_email)
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
    session.clear()
    flash('You have been logged out.', 'success')
    response = make_response(redirect(url_for('home')))
    response.set_cookie('access_token', '', expires=0)
    response.set_cookie('remember_token', '', expires=0)
    return response

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
            return redirect(url_for('account'))
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

# --- PATIENT PORTAL ---

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
        else:
            flash("Invalid Aadhar ID or Date of Birth.", "danger")
    return render_template('patient_login.html')

@app.route('/patient/dashboard')
def patient_dashboard():
    if 'patient_aadhar' not in session: return redirect(url_for('patient_login'))
    aadhar = session['patient_aadhar']
    
    patient_records = Patient.query.filter_by(aadhar=aadhar).all()
    patient_ids = [p.id for p in patient_records]
    all_reports = Report.query.filter(Report.patient_id.in_(patient_ids)).order_by(Report.generated_at.desc()).all()
    
    return render_template('patient_dashboard.html', reports=all_reports, patient_name=session['patient_name'], aadhar_display=f"{aadhar[:4]} {aadhar[4:8]} {aadhar[8:]}")

@app.route('/patient/logout')
def patient_logout():
    session.pop('patient_aadhar', None)
    session.pop('patient_name', None)
    flash("Patient logged out.", "success")
    response = make_response(redirect(url_for('home')))
    return response

@app.route('/patient/download/<int:report_id>')
def download_archived_report_patient(report_id):
    if 'patient_aadhar' not in session: return redirect(url_for('patient_login'))
    report = Report.query.get_or_404(report_id)
    
    if report.patient.aadhar != session['patient_aadhar']: abort(403)
    if not report.pdf_storage_path or not supabase:
        flash("File unavailable.", "warning")
        return redirect(url_for('patient_dashboard'))

    res = supabase.storage.from_("medical_reports").create_signed_url(report.pdf_storage_path, 60)
    return redirect(res['signedURL'])

# --- DOCTOR DASHBOARD ---

@app.route('/dashboard')
@login_required
def dashboard():
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.full_name).all()
    total_reports = db.session.query(Report).join(Patient).filter(Patient.doctor_id == current_user.id).count()
    
    gender_data = {'Male': 0, 'Female': 0, 'Other': 0}
    for p in patients:
        if p.gender == 'Male': gender_data['Male'] += 1
        elif p.gender == 'Female': gender_data['Female'] += 1
        else: gender_data['Other'] += 1

    blood_data = {}
    for p in patients:
        bg = p.blood_group if p.blood_group else 'Unknown'
        blood_data[bg] = blood_data.get(bg, 0) + 1

    patient_ids = [p.id for p in patients]
    reports = Report.query.filter(Report.patient_id.in_(patient_ids)).all()
    confidence_data = {'High': 0, 'Medium': 0, 'Low': 0}
    for r in reports:
        if r.confidence in confidence_data:
            confidence_data[r.confidence] += 1

    return render_template(
        'dashboard.html', 
        patients=patients, 
        total_reports=total_reports, 
        total_patients=len(patients),
        gender_data=gender_data,
        blood_data=blood_data,
        confidence_data=confidence_data
    )

@app.route('/export/patients')
@login_required
def export_patients_csv():
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.full_name).all()
    si = io.StringIO()
    cw = csv.writer(si)
    
    cw.writerow([
        'Patient Name', 'Aadhar ID', 'Date of Birth', 'Gender', 
        'Blood Group', 'Phone Number', 'Emergency Contact', 
        'Allergies', 'Country'
    ])
    
    for p in patients:
        cw.writerow([
            p.full_name,
            p.aadhar,
            p.dob,
            p.gender,
            p.blood_group if p.blood_group else 'N/A',
            p.phone if p.phone else 'N/A',
            p.emergency_contact if p.emergency_contact else 'N/A',
            p.allergies if p.allergies else 'None',
            p.country if p.country else 'N/A'
        ])
        
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
            flash('Invalid Aadhar: Must be 12 digits.', 'danger')
            return render_template('add_patient.html')

        existing_patient = Patient.query.filter_by(aadhar=clean_aadhar, doctor_id=current_user.id).first()
        if existing_patient:
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
        flash(f'Patient {new_patient.full_name} added successfully.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('add_patient.html')

@app.route('/view_patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for('dashboard'))
    reports = Report.query.filter_by(patient_id=patient.id).order_by(Report.generated_at.desc()).all()
    notes = Note.query.filter_by(patient_id=patient.id).order_by(Note.created_at.desc()).all()
    return render_template('view_patient.html', patient=patient, reports=reports, notes=notes)

@app.route('/patient/<int:patient_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for('dashboard'))
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
    flash(f"Patient deleted.", "success")
    return redirect(url_for('dashboard'))

# =======================================================
# 7. CLINICAL HUB & ML ROUTES (Warfarin & Diabetes)
# =======================================================

@app.route('/patient/<int:patient_id>/new_assessment', methods=['GET'])
@login_required
def new_assessment(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    return render_template('assessment_type.html', patient=patient)

@app.route('/patient/<int:patient_id>/select_disease', methods=['GET'])
@login_required
def select_disease(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    return render_template('select_disease.html', patient=patient)

@app.route('/patient/<int:patient_id>/select_drug', methods=['GET'])
@login_required
def select_drug(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    return render_template('select_drug.html', patient=patient)

@app.route('/patient/<int:patient_id>/redirect_form', methods=['POST'])
@login_required
def redirect_to_drug_form(patient_id):
    if request.form.get('drug_name') == 'warfarin': 
        return redirect(url_for('warfarin_form', patient_id=patient_id))
    flash("Invalid drug selected.", "danger")
    return redirect(url_for('dashboard'))

@app.route('/patient/<int:patient_id>/warfarin_form', methods=['GET'])
@login_required
def warfarin_form(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
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
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))

    patient_info, clinical_info, safety_info, results = process_prediction_data(request.form)
    doctor_name = request.form.get('doctor_name')
    interacting_drugs = request.form.getlist('interacting_drugs')
    interaction_warnings = get_interaction_warnings(interacting_drugs)

    full_report_data = {
        "patient_info": patient_info, "clinical_info": clinical_info,
        "safety_info": safety_info, "results": results,
        "doctor_name": doctor_name, "interacting_drugs": interacting_drugs
    }

    html_string = render_template('display_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=None, interaction_warnings=interaction_warnings)
    pdf_bytes = HTML(string=html_string).write_pdf()

    pdf_path = None
    if supabase:
        try:
            filename = f"report_{patient.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            supabase.storage.from_("medical_reports").upload(path=filename, file=pdf_bytes, file_options={"content-type": "application/pdf"})
            pdf_path = filename 
        except Exception as e: print(f"--- ❌ Supabase Upload Failed: {e} ---")

    new_report = Report(
        drug_name="Warfarin", predicted_dose=f"{results['predicted_dose_mg_per_week']} mg/week",
        model_used=results['model_used'], confidence=results['confidence_score'],
        doctor_name=doctor_name, report_data_json=json.dumps(full_report_data),
        patient_id=patient.id, pdf_storage_path=pdf_path 
    )
    db.session.add(new_report)
    db.session.commit()

    return make_response(render_template('display_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=request, interaction_warnings=interaction_warnings, report_obj=new_report))

# --- NEW: DIABETES FORM & REPORT ROUTING ---
@app.route('/patient/<int:patient_id>/diabetes_form', methods=['GET'])
@login_required
def diabetes_form(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    
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
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))

    patient_info, clinical_info, safety_info, results = process_diabetes_data(request.form)
    doctor_name = request.form.get('doctor_name')

    full_report_data = {
        "patient_info": patient_info, "clinical_info": clinical_info,
        "safety_info": safety_info, "results": results, "doctor_name": doctor_name
    }

    # We will build 'display_disease_report.html' in the next step
    html_string = render_template('display_disease_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=None)
    pdf_bytes = HTML(string=html_string).write_pdf()

    # Save to database
    new_report = Report(
        drug_name="Type 2 Diabetes Assessment", # Saving the Disease name
        predicted_dose=results['predicted_dose_mg_per_week'], # Saving the "Risk Level"
        model_used=results['model_used'], confidence=results['confidence_score'],
        doctor_name=doctor_name, report_data_json=json.dumps(full_report_data),
        patient_id=patient.id, pdf_storage_path=None
    )
    db.session.add(new_report)
    db.session.commit()

    return make_response(render_template('display_disease_report.html', patient_info=patient_info, clinical_info=clinical_info, safety_info=safety_info, results=results, doctor_name=doctor_name, request=request, report_obj=new_report))

# --- DYNAMIC REPORT VIEWER (Loads Drug vs Disease HTML based on DB) ---
@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    report = Report.query.get_or_404(report_id)
    if report.patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    try: report_data = json.loads(report.report_data_json)
    except: return redirect(url_for('view_patient', patient_id=report.patient_id))
    
    # Check if it's a Disease Report or a Drug Report to load the correct HTML template
    if report.drug_name == "Type 2 Diabetes Assessment":
        template_name = 'display_disease_report.html'
        interaction_warnings = []
    else:
        template_name = 'display_report.html'
        interaction_warnings = get_interaction_warnings(report_data.get('interacting_drugs', []))
    
    return make_response(render_template(
        template_name, 
        patient_info=report_data.get('patient_info'), 
        clinical_info=report_data.get('clinical_info'), 
        safety_info=report_data.get('safety_info', {}), 
        results=report_data.get('results'), 
        doctor_name=report_data.get('doctor_name'), 
        request=None, 
        interaction_warnings=interaction_warnings, 
        report_obj=report
    ))

@app.route('/download_archived_report/<int:report_id>')
@login_required
def download_archived_report(report_id):
    report = Report.query.get_or_404(report_id)
    if report.patient.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for('dashboard'))
    
    if not report.pdf_storage_path or not supabase:
        flash("No archived PDF found.", "warning")
        return redirect(url_for('view_report', report_id=report_id))

    try:
        file_bytes = supabase.storage.from_("medical_reports").download(report.pdf_storage_path)
        return Response(file_bytes, mimetype='application/pdf', headers={'Content-Disposition': f'attachment;filename=Report_{report.patient.full_name}.pdf'})
    except Exception as e:
        flash("Error retrieving file.", "danger")
        return redirect(url_for('view_report', report_id=report_id))

@app.route('/download_report', methods=['POST'])
@login_required
def download_report():
    form_data = request.form
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
    # 1. Fetch Global Platform Metrics
    total_doctors = User.query.count()
    total_patients = Patient.query.count()
    total_reports = Report.query.count()
    total_subscriptions = Subscription.query.count()
    
    # 2. Fetch Doctors (Recent for dashboard, All for auditing)
    recent_doctors = User.query.order_by(User.id.desc()).limit(5).all()
    all_doctors = User.query.order_by(User.id.desc()).all()
    
    # 3. Calculate Platform Load
    warfarin_count = Report.query.filter_by(drug_name='Warfarin').count()
    diabetes_count = Report.query.filter_by(drug_name='Type 2 Diabetes Assessment').count()
    
    return render_template('admin_dashboard.html', 
                           total_doctors=total_doctors,
                           total_patients=total_patients,
                           total_reports=total_reports,
                           total_subs=total_subscriptions,
                           recent_doctors=recent_doctors,
                           all_doctors=all_doctors,
                           warfarin_count=warfarin_count,
                           diabetes_count=diabetes_count)

@app.route('/admin/doctor/<int:doc_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_doctor(doc_id):
    if current_user.id == doc_id:
        flash("Action Blocked: You cannot delete the Master Admin account.", "danger")
        return redirect(url_for('admin_dashboard'))
        
    doctor = User.query.get_or_404(doc_id)
    doc_name = doctor.full_name
    
    # Free up the license seat before deleting
    if doctor.subscription_email:
        sub = Subscription.query.filter_by(email=doctor.subscription_email).first()
        if sub and sub.current_users > 0: 
            sub.current_users -= 1
            
    db.session.delete(doctor)
    db.session.commit()
    
    flash(f"Physician {doc_name} and all their clinical data have been permanently wiped.", "success")
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)