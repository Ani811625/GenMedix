import json
import os
from datetime import datetime
from flask import (
    Flask, request, jsonify, render_template, make_response, 
    Response, redirect, url_for, session, flash, abort
)
import joblib
import pandas as pd
import shap
import numpy as np
from weasyprint import HTML
from sqlalchemy import inspect

# --- IMPORTS FOR DATABASE, LOGIN & EMAIL ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message

# --- APP CONFIGURATION ---
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'

# --- EMAIL CONFIGURATION ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME') 
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)

# --- DATABASE CONFIGURATION (FIXED) ---
# We correctly fetch the variable by its NAME 'DATABASE_URL'
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    print(f"--- ✅ SUCCESS: FOUND EXTERNAL DATABASE URL ---")
    # Fix for SQLAlchemy: Postgres connection string must start with postgresql://
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL.replace("postgres://", "postgresql://")
else:
    print("--- ⚠️ WARNING: NO DATABASE_URL FOUND. USING LOCAL SQLITE ---")
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'project.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- CONTEXT PROCESSOR (THE FIX FOR "current_user is undefined") ---
@app.context_processor
def inject_user():
    return dict(current_user=current_user)

# --- AUTO-CREATE TABLES ON STARTUP ---
@app.before_request
def check_maintenance_and_db():
    # 1. Check Maintenance Mode
    if os.environ.get('MAINTENANCE_MODE') == 'true':
        if request.endpoint and request.endpoint != 'static':
            return render_template('maintenance.html'), 503

    # 2. Create Tables if they don't exist
    try:
        inspector = inspect(db.engine)
        if not inspector.has_table("user"):
            print("--- No 'user' table found. Creating all tables... ---")
            with app.app_context():
                db.create_all()
            print("--- Database tables created. ---")
    except Exception as e:
        print(f"--- ERROR checking/creating tables: {e} ---")

# --- LOGIN MANAGER SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "You must be logged in to access this page."
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
    
    print("--- All models, column lists, and SHAP explainers loaded successfully. ---")
except Exception as e:
    print(f"--- FATAL ERROR loading models: {e} ---")
    # We continue so the app doesn't crash immediately, but predictions will fail
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
        explanation = "The model's internal estimators are in strong agreement."
    elif std_dev < 1.0:
        score = "Medium"
        explanation = "The model's internal estimators show some variance. Use with caution."
    else:
        score = "Low"
        explanation = "The model's internal estimators have significant disagreement. Verify input and proceed with caution."
    return score, explanation

def get_human_explanation(shap_dict):
    explanations = []
    for feature, value in shap_dict.items():
        if feature == "Weight__kg_": display_name = "Weight"
        elif feature == "Height__cm_": display_name = "Height"
        elif feature.startswith("CYP2C9"): display_name = "CYP2C9 Genotype"
        elif feature.startswith("VKORC1"): display_name = "VKORC1 Genotype"
        else: display_name = feature.replace("Race_", "")

        if value > 0: direction = "<strong>increased</strong>"
        else: direction = "<strong>decreased</strong>"
            
        explanations.append(f"<strong>{display_name}</strong> {direction} the dose recommendation.")
    return explanations

def get_clinical_suggestions(shap_dict, confidence):
    suggestions = []
    for feature in shap_dict.keys():
        if "VKORC1" in feature and shap_dict[feature] < -0.5:
            suggestions.append("<strong>High Sensitivity Detected:</strong> VKORC1 genotype suggests a lower dose requirement.")
        if "CYP2C9" in feature and shap_dict[feature] < -0.5:
            suggestions.append("<strong>Slow Metabolizer Detected:</strong> CYP2C9 genotype suggests slower drug clearance.")

    if "Weight__kg_" in shap_dict and shap_dict["Weight__kg_"] > 1.0:
         suggestions.append("Patient's high body weight is a major factor increasing the dose.")
    
    if confidence == "Low":
        suggestions.append("<strong>Low Model Confidence:</strong> Please review all patient data and proceed with extra caution.")

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
    return patient_info_dict, clinical_info_display, results_dict

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
    except FileNotFoundError:
        flash(f"Error: The dataset file could not be found at '{DATA_FILE_PATH}'.", "danger")
        headers, rows, row_count = [], [], 0
    except Exception as e:
        flash(f"An error occurred while reading the data: {e}", "danger")
        headers, rows, row_count = [], [], 0

    return render_template('dataset.html', headers=headers, rows=rows, row_count=row_count, showing_count=len(rows))

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

            # Send Welcome Email
            try:
                msg = Message("Welcome to GenMedix!", recipients=[email])
                msg.body = f"""
                Dear Dr. {name},

                Welcome to GenMedix! Your clinician account has been successfully created.
                You can now log in to your dashboard to manage patients and generate AI-powered dosage reports.

                Login here: https://genmedix-app.onrender.com/login

                Best regards,
                The GenMedix Team
                """
                mail.send(msg)
                flash('Account created! A welcome email has been sent to your inbox.', 'success')
            except Exception as e:
                print(f"Error sending email: {e}")
                flash('Account created, but we could not send the welcome email. You can still login.', 'warning')

        except Exception as e:
            db.session.rollback()
            flash(f'Error creating account: {e}', 'danger')
            return redirect(url_for('register'))

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user is None or not user.check_password(password):
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

        login_user(user)
        return redirect(url_for('dashboard'))

    return render_template('login.html') 

@app.route('/logout')
@login_required
def logout():
    logout_user()
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
            flash('A patient with this Aadhar number is already registered.', 'danger')
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
        flash(f'Patient {new_patient.full_name} added successfully!', 'success')
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
        flash("Unauthorized access.", "danger")
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
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))

    patient_info_dict = {
        "patient_name": patient.full_name,
        "patient_dob": patient.dob,
        "patient_gender": patient.gender,
        "patient_country": patient.country,
        "patient_address": patient.address 
    }
    clinical_data_dict = {
        "Age": float(request.form.get('Age')),
        "Height__cm_": float(request.form.get('Height__cm_')),
        "Weight__kg_": float(request.form.get('Weight__kg_')),
    }
    doctor_name = request.form.get('doctor_name')
    interacting_drugs = request.form.getlist('interacting_drugs')
    race = request.form.get('Race')
    cyp2c9 = request.form.get('CYP2C9_genotypes')
    vkorc1 = request.form.get('VKORC1_genotype')

    if race: clinical_data_dict[race] = 1.0
    if cyp2c9: clinical_data_dict[cyp2c9] = 1.0
    if vkorc1: clinical_data_dict[vkorc1] = 1.0

    clinical_info_display = {
        "Age": request.form.get('Age'),
        "Height__cm_": request.form.get('Height__cm_'),
        "Weight__kg_": request.form.get('Weight__kg_'),
        "Race_Display": race.split('_')[-1] if race else "N/A",
        "CYP2C9_Display": cyp2c9.split('__')[-1].replace('_', '/*') if cyp2c9 else "N/A",
        "VKORC1_Display": vkorc1.split('_')[-1] if vkorc1 else "N/A"
    }

    pred_data = run_model_prediction(clinical_data_dict) 
    confidence, conf_expl = get_confidence_score(pred_data['std_dev'])
    human_expl = get_human_explanation(pred_data['shap_explanation'])
    suggestions = get_clinical_suggestions(pred_data['shap_explanation'], confidence)
    interaction_warnings = get_interaction_warnings(interacting_drugs)

    results_dict = {
        "predicted_dose_mg_per_week": pred_data['prediction'],
        "model_used": pred_data['model_name'],
        "confidence_score": confidence,
        "confidence_explanation": conf_expl,
        "human_explanation": human_expl,
        "clinical_suggestions": suggestions,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_id": f"GM-{datetime.now().strftime('%Y%m%d')}-{patient.id}"
    }
    
    full_report_data = {
        "patient_info": patient_info_dict,
        "clinical_info": clinical_info_display,
        "results": results_dict,
        "doctor_name": doctor_name,
        "interacting_drugs": interacting_drugs
    }

    new_report = Report(
        drug_name="Warfarin",
        predicted_dose=f"{pred_data['prediction']} mg/week",
        model_used=pred_data['model_name'],
        confidence=confidence,
        doctor_name=doctor_name, 
        report_data_json=json.dumps(full_report_data),
        patient_id=patient.id
    )
    db.session.add(new_report)
    db.session.commit()

    response = make_response(render_template(
        'display_report.html',
        patient_info=patient_info_dict,
        clinical_info=clinical_info_display,
        results=results_dict,
        doctor_name=doctor_name,
        request=request, 
        interaction_warnings=interaction_warnings
    ))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    report = Report.query.get_or_404(report_id)
    if report.patient.doctor_id != current_user.id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))

    try:
        report_data = json.loads(report.report_data_json)
    except:
        flash("Error: Report data is corrupted.", "danger")
        return redirect(url_for('view_patient', patient_id=report.patient_id))
    
    saved_interacting_drugs = report_data.get('interacting_drugs', [])
    interaction_warnings = get_interaction_warnings(saved_interacting_drugs)
    
    response = make_response(render_template(
        'display_report.html',
        patient_info=report_data.get('patient_info'),
        clinical_info=report_data.get('clinical_info'),
        results=report_data.get('results'),
        doctor_name=report_data.get('doctor_name'),
        request=None,
        interaction_warnings=interaction_warnings
    ))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/download_report', methods=['POST'])
@login_required
def download_report():
    form_data = request.form.to_dict()
    patient_info, clinical_info, results = process_prediction_data(form_data)
    doctor_name = form_data.get('doctor_name', current_user.full_name)

    html_string = render_template(
        'display_report.html', 
        patient_info=patient_info, 
        clinical_info=clinical_info, 
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
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))
    db.session.delete(report)
    db.session.commit()
    flash("Report deleted successfully.", "success")
    return redirect(url_for('view_patient', patient_id=patient_id))

@app.route('/patient/<int:patient_id>/delete', methods=['POST'])
@login_required
def delete_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))
    db.session.delete(patient)
    db.session.commit()
    flash(f"Patient '{patient.full_name}' deleted.", "success")
    return redirect(url_for('dashboard'))

@app.route('/patient/<int:patient_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        new_aadhar = request.form.get('aadhar')
        if new_aadhar != patient.aadhar:
            if Patient.query.filter_by(aadhar=new_aadhar).first():
                flash('That Aadhar number is already assigned to another patient.', 'danger')
                return render_template('edit_patient.html', patient=patient)
        
        patient.full_name = request.form.get('full_name')
        patient.aadhar = new_aadhar
        patient.dob = request.form.get('dob')
        patient.gender = request.form.get('gender')
        patient.country = request.form.get('country')
        patient.address = request.form.get('address')
        db.session.commit()
        flash('Patient details updated.', 'success')
        return redirect(url_for('view_patient', patient_id=patient.id))
    
    return render_template('edit_patient.html', patient=patient)

@app.route('/patient/<int:patient_id>/add_note', methods=['POST'])
@login_required
def add_note(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))
        
    note_text = request.form.get('note_text')
    if not note_text:
        flash("Note cannot be empty.", "danger")
        return redirect(url_for('view_patient', patient_id=patient_id))
        
    new_note = Note(note_text=note_text, patient_id=patient.id, doctor_id=current_user.id)
    db.session.add(new_note)
    db.session.commit()
    flash("Note added successfully.", "success")
    return redirect(url_for('view_patient', patient_id=patient_id, _anchor='notes-tab'))

@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    action = request.form.get('action')
    if request.method == 'POST':
        if action == 'update_details':
            new_email = request.form.get('email')
            new_reg_id = request.form.get('medical_reg_id')

            if new_email != current_user.email and User.query.filter_by(email=new_email).first():
                flash('Email already in use.', 'danger')
                return redirect(url_for('account'))
            
            if new_reg_id != current_user.medical_reg_id and User.query.filter_by(medical_reg_id=new_reg_id).first():
                flash('Medical ID already in use.', 'danger')
                return redirect(url_for('account'))

            current_user.full_name = request.form.get('full_name')
            current_user.email = new_email
            current_user.medical_reg_id = new_reg_id
            db.session.commit()
            flash('Details updated successfully.', 'success')
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