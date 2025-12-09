import json
import os
import random
import string
from datetime import datetime, timedelta
from threading import Thread

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

# --- APP CONFIGURATION ---
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'

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

db = SQLAlchemy(app)

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

# --- CONTEXT PROCESSOR ---
@app.context_processor
def inject_user():
    return dict(current_user=current_user)

# --- ERROR HANDLERS ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# --- AUTO-CREATE TABLES ---
@app.before_request
def check_maintenance_and_db():
    if os.environ.get('MAINTENANCE_MODE') == 'true':
        if request.endpoint and request.endpoint != 'static':
            return render_template('maintenance.html'), 503

    try:
        inspector = inspect(db.engine)
        existing_tables = inspector.get_table_names()
        if "user" not in existing_tables:
            with app.app_context():
                db.create_all()
    except Exception as e:
        print(f"--- ❌ DB Check Error: {e} ---")

# --- LOGIN MANAGER ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Please log in to access the clinical dashboard."
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- DATABASE MODELS ---

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    medical_reg_id = db.Column(db.String(100), unique=True)
    
    # NOTE: OTP fields removed because Supabase handles verification state now
    
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
    aadhar = db.Column(db.String(12), unique=True)
    country = db.Column(db.String(50))
    address = db.Column(db.String(200))
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

# --- LOAD AI MODELS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print("--- Loading models... ---")
try:
    base_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_base_v1.pkl'))
    base_model_columns = joblib.load(os.path.join(MODEL_DIR, 'base_model_columns.pkl'))
    enhanced_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_enhanced_v1.pkl'))
    enhanced_model_columns = joblib.load(os.path.join(MODEL_DIR, 'model_columns.pkl'))
    
    enhanced_explainer = shap.TreeExplainer(enhanced_model)
    base_explainer = shap.TreeExplainer(base_model)
    print("--- All models loaded. ---")
except Exception as e:
    print(f"--- FATAL ERROR loading models: {e} ---")
    pass 

# --- HELPER FUNCTIONS ---

def get_interaction_warnings(checked_drugs_list):
    warnings = []
    if not checked_drugs_list: return warnings
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
            shap_value = round(shap_values_for_instance[i], 2)
            shap_explanation[feature_name] = shap_value

    return {
        "prediction": predicted_dose,
        "model_name": model_name,
        "shap_explanation": shap_explanation,
        "std_dev": std_dev
    }

def get_confidence_score(std_dev):
    if std_dev < 0.5:
        score = "High"
        explanation = "Model estimators are in strong agreement."
    elif std_dev < 1.0:
        score = "Medium"
        explanation = "Model estimators show variance. Use with caution."
    else:
        score = "Low"
        explanation = "Significant disagreement in estimators. Proceed with caution."
    return score, explanation

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
    
    # Append Safety Warnings
    if form_data.get('is_pregnant'):
        suggestions.append("<strong>CONTRAINDICATION:</strong> Patient marked Pregnant. Warfarin contraindicated.")
    if form_data.get('active_bleeding'):
        suggestions.append("<strong>CONTRAINDICATION:</strong> Active bleeding detected.")
    if form_data.get('platelet_count'):
        try:
            if int(form_data.get('platelet_count')) < 50000:
                suggestions.append("<strong>SAFETY ALERT:</strong> Severe Thrombocytopenia (<50k).")
        except: pass

    results_dict = {
        "predicted_dose_mg_per_week": pred_data['prediction'],
        "model_used": pred_data['model_name'],
        "confidence_score": confidence,
        "confidence_explanation": conf_expl,
        "human_explanation": human_expl,
        "clinical_suggestions": suggestions,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_id": f"GM-{datetime.now().strftime('%Y%m%d')}-{abs(hash(patient_info_dict['patient_name'])) % 10000}"
    }
    
    return patient_info_dict, clinical_info_display, safety_data_dict, results_dict

# --- ROUTES ---

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    DATA_FILE_PATH = 'data/warfarin.csv'
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        headers = df.columns.tolist()
        rows = df.head(200).to_dict('records')
        row_count = len(df)
    except:
        headers, rows, row_count = [], [], 0
    return render_template('dataset.html', headers=headers, rows=rows, row_count=row_count, showing_count=len(rows))

# --- SUPABASE OTP LOGIN FLOW ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # 1. Check Local DB first (Does this user exist?)
        local_user = User.query.filter_by(email=email).first()
        if not local_user:
            flash('No account found with this email.', 'danger')
            return redirect(url_for('login'))
            
        # 2. Check Password (The first Factor)
        if not local_user.check_password(password):
            flash('Incorrect password.', 'danger')
            return redirect(url_for('login'))

        # 3. Trigger Supabase OTP (The Second Factor)
        try:
            if not supabase:
                raise Exception("Supabase Client not configured.")
                
            # Request OTP from Supabase
            res = supabase.auth.sign_in_with_otp({"email": email})
            
            # Store email in session to verify next step
            session['auth_email'] = email
            flash('Two-Factor Code sent to your email by Supabase.', 'info')
            return redirect(url_for('verify_otp'))
            
        except Exception as e:
            print(f"--- ❌ Supabase Auth Error: {e} ---")
            flash('Error sending OTP. Please try again.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    # Security Check: Must have started login flow
    if 'auth_email' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        otp = request.form.get('otp')
        email = session.get('auth_email')
        
        try:
            # 1. Verify Token with Supabase API
            res = supabase.auth.verify_otp({
                "email": email,
                "token": otp,
                "type": "email"
            })
            
            # 2. If no error thrown, OTP is valid. Log in via Flask-Login
            local_user = User.query.filter_by(email=email).first()
            if local_user:
                login_user(local_user)
                session.pop('auth_email', None) # Clear session
                return redirect(url_for('dashboard'))
            else:
                flash("Login verified, but user record missing in DB.", "danger")
                
        except Exception as e:
            print(f"--- ❌ OTP Verification Failed: {e} ---")
            flash("Invalid or expired code. Please try again.", "danger")

    return render_template('verify_otp.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
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
        
        if User.query.filter_by(medical_reg_id=reg_id).first():
            flash('Medical ID already registered.', 'danger')
            return redirect(url_for('register'))

        new_doctor = User(full_name=name, email=email, medical_reg_id=reg_id)
        new_doctor.set_password(password)
        
        try:
            db.session.add(new_doctor)
            db.session.commit()
            
            # NOTE: We do NOT need to create the user in Supabase manually.
            # Supabase auto-creates "Auth Users" the first time we request an OTP for them.
            
            flash('Account created successfully! Please log in.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating account: {e}', 'danger')
            return redirect(url_for('register'))

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear() # Clear any leftover auth data
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.full_name).all()
    total_reports = db.session.query(Report).join(Patient).filter(Patient.doctor_id == current_user.id).count()
    return render_template('dashboard.html', patients=patients, total_reports=total_reports, total_patients=len(patients))

@app.route('/add_patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    if request.method == 'POST':
        aadhar = request.form.get('aadhar')
        if Patient.query.filter_by(aadhar=aadhar).first():
            flash('Aadhar number already registered.', 'danger')
            return render_template('add_patient.html')
        
        new_patient = Patient(
            full_name=request.form.get('full_name'),
            aadhar=aadhar,
            dob=request.form.get('dob'),
            gender=request.form.get('gender'),
            country=request.form.get('country'),
            address=request.form.get('address'),
            doctor_id=current_user.id
        )
        db.session.add(new_patient)
        db.session.commit()
        flash(f'Patient {new_patient.full_name} added.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('add_patient.html')

@app.route('/view_patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))
    reports = Report.query.filter_by(patient_id=patient.id).order_by(Report.generated_at.desc()).all()
    notes = Note.query.filter_by(patient_id=patient.id).order_by(Note.created_at.desc()).all()
    return render_template('view_patient.html', patient=patient, reports=reports, notes=notes)

@app.route('/patient/<int:patient_id>/select_drug', methods=['GET'])
@login_required
def select_drug(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))
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
    if patient.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for('dashboard'))
    
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
    if patient.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for('dashboard'))

    patient_info, clinical_info, safety_info, results = process_prediction_data(request.form)
    
    doctor_name = request.form.get('doctor_name')
    interacting_drugs = request.form.getlist('interacting_drugs')
    interaction_warnings = get_interaction_warnings(interacting_drugs)

    timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
    report_id_display = f"GM-{datetime.now().strftime('%Y%m%d')}-{patient.id}"
    results['report_id'] = report_id_display
    
    full_report_data = {
        "patient_info": patient_info,
        "clinical_info": clinical_info,
        "safety_info": safety_info,
        "results": results,
        "doctor_name": doctor_name,
        "interacting_drugs": interacting_drugs
    }

    # Generate PDF
    html_string = render_template(
        'display_report.html',
        patient_info=patient_info,
        clinical_info=clinical_info,
        safety_info=safety_info,
        results=results,
        doctor_name=doctor_name,
        request=None,
        interaction_warnings=interaction_warnings
    )
    pdf_bytes = HTML(string=html_string).write_pdf()

    # Upload to Supabase Storage
    pdf_path = None
    if supabase:
        try:
            filename = f"report_{patient.id}_{timestamp_str}.pdf"
            supabase.storage.from_("medical_reports").upload(
                path=filename,
                file=pdf_bytes,
                file_options={"content-type": "application/pdf"}
            )
            print(f"--- ✅ PDF Uploaded: {filename} ---")
            pdf_path = filename 
        except Exception as e:
            print(f"--- ❌ Supabase Upload Failed: {e} ---")

    # Save to Database
    new_report = Report(
        drug_name="Warfarin",
        predicted_dose=f"{results['predicted_dose_mg_per_week']} mg/week",
        model_used=results['model_used'],
        confidence=results['confidence_score'],
        doctor_name=doctor_name, 
        report_data_json=json.dumps(full_report_data),
        patient_id=patient.id,
        pdf_storage_path=pdf_path 
    )
    db.session.add(new_report)
    db.session.commit()

    # Render Response
    response = make_response(render_template(
        'display_report.html',
        patient_info=patient_info,
        clinical_info=clinical_info,
        safety_info=safety_info,
        results=results,
        doctor_name=doctor_name,
        request=request, 
        interaction_warnings=interaction_warnings
    ))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

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
        res = supabase.storage.from_("medical_reports").create_signed_url(
            report.pdf_storage_path, 60
        )
        if res and 'signedURL' in res:
            return redirect(res['signedURL'])
        else:
            flash("Could not generate link.", "danger")
            return redirect(url_for('view_report', report_id=report_id))
    except Exception:
        flash("Error retrieving file.", "danger")
        return redirect(url_for('view_report', report_id=report_id))

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    report = Report.query.get_or_404(report_id)
    if report.patient.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for('dashboard'))

    try:
        report_data = json.loads(report.report_data_json)
    except:
        flash("Error: Report corrupted.", "danger")
        return redirect(url_for('view_patient', patient_id=report.patient_id))
    
    saved_interacting_drugs = report_data.get('interacting_drugs', [])
    interaction_warnings = get_interaction_warnings(saved_interacting_drugs)
    
    response = make_response(render_template(
        'display_report.html',
        patient_info=report_data.get('patient_info'),
        clinical_info=report_data.get('clinical_info'),
        safety_info=report_data.get('safety_info', {}),
        results=report_data.get('results'),
        doctor_name=report_data.get('doctor_name'),
        request=None,
        interaction_warnings=interaction_warnings,
        report_obj=report 
    ))
    return response

@app.route('/download_report', methods=['POST'])
@login_required
def download_report():
    form_data = request.form
    patient_info, clinical_info, safety_info, results = process_prediction_data(form_data)
    doctor_name = form_data.get('doctor_name', current_user.full_name)

    html_string = render_template(
        'display_report.html', 
        patient_info=patient_info, 
        clinical_info=clinical_info, 
        safety_info=safety_info,
        results=results,
        doctor_name=doctor_name,
        request=None
    )
    pdf_file = HTML(string=html_string).write_pdf()
    return Response(pdf_file, mimetype='application/pdf', headers={'Content-Disposition': 'attachment;filename=DosageReport.pdf'})

@app.route('/report/<int:report_id>/delete', methods=['POST'])
@login_required
def delete_report(report_id):
    report = Report.query.get_or_404(report_id)
    patient_id = report.patient.id 
    if report.patient.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for('dashboard'))
        
    if report.pdf_storage_path and supabase:
        try:
            supabase.storage.from_("medical_reports").remove([report.pdf_storage_path])
        except Exception: pass
            
    db.session.delete(report)
    db.session.commit()
    flash("Report deleted.", "success")
    return redirect(url_for('view_patient', patient_id=patient_id))

@app.route('/patient/<int:patient_id>/delete', methods=['POST'])
@login_required
def delete_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for('dashboard'))
    
    db.session.delete(patient)
    db.session.commit()
    flash(f"Patient deleted.", "success")
    return redirect(url_for('dashboard'))

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

@app.route('/patient/<int:patient_id>/add_note', methods=['POST'])
@login_required
def add_note(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash("Unauthorized.", "danger")
        return redirect(url_for('dashboard'))
        
    note_text = request.form.get('note_text')
    if not note_text:
        flash("Note cannot be empty.", "danger")
        return redirect(url_for('view_patient', patient_id=patient_id))
        
    new_note = Note(note_text=note_text, patient_id=patient.id, doctor_id=current_user.id)
    db.session.add(new_note)
    db.session.commit()
    flash("Note added.", "success")
    return redirect(url_for('view_patient', patient_id=patient_id, _anchor='notes-tab'))

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
                
            db.session.delete(current_user)
            db.session.commit()
            logout_user()
            flash("Account deleted.", "success")
            return redirect(url_for('home'))

    return render_template('account.html')

if __name__ == '__main__':
    app.run(debug=True)