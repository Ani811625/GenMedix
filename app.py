import json
from flask import (
    Flask, request, jsonify, render_template, make_response, 
    Response, redirect, url_for, session, flash, abort
)
from datetime import datetime
import joblib
import pandas as pd
import shap
import numpy as np
from weasyprint import HTML
import io
import os
import click
from flask.cli import with_appcontext # <-- YOU ARE MISSING THIS LINE

# --- NEW IMPORTS FOR DATABASE & LOGIN ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# --- APP, CACHE, & DB CONFIGURATION ---
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# basedir = os.path.abspath(os.path.dirname(__file__))
# app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'project.db')
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'

# --- NEW DATABASE CONFIGURATION ---
# Use the Render DB URL if available (on deployment), 
# otherwise, fall back to the local SQLite DB for testing.
DATABASE_URL = os.environ.get('postgresql://genmedix_db_user:gxBDPM2SpxXONZ9rXhVfx8v60VlRzIEb@dpg-d44ah63uibrs73eltqh0-a/genmedix_db')
if DATABASE_URL:
    # On Render, DATABASE_URL will be set automatically.
    # We may need to replace 'postgres://' with 'postgresql://' for SQLAlchemy
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL.replace("postgres://", "postgresql://")
else:
    # For local development
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'project.db')
# --- END NEW CONFIG ---
# --- ADD THIS DEBUG LINE ---
print(f"--- CONNECTING TO DATABASE: {app.config['SQLALCHEMY_DATABASE_URI']} ---")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- LOGIN MANAGER SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Name of the login function
login_manager.login_message = "You must be logged in to access this page."
login_manager.login_message_category = "info" # Bootstrap class

# --- DATABASE MODELS ---

class User(db.Model, UserMixin):
    """
    Represents a Clinician (Doctor) user account.
    """
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False) # Doctors will log in with email
    password_hash = db.Column(db.String(256))
    medical_reg_id = db.Column(db.String(100), unique=True) # Their professional ID
    
    # This creates a "one-to-many" link: One Doctor -> Many Patients
    patients = db.relationship('Patient', backref='doctor', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    """
    Represents a Patient, who is "owned" by a doctor.
    """
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    dob = db.Column(db.String(10)) # Store as YYYY-MM-DD
    gender = db.Column(db.String(10))
    aadhar = db.Column(db.String(12), unique=True) # Patient's Aadhar
    country = db.Column(db.String(50))
    address = db.Column(db.String(200))
    
    # Foreign Key to link this Patient to a User (Doctor)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # This creates a "one-to-many" link: One Patient -> Many Reports
    reports = db.relationship('Report', backref='patient', lazy=True, cascade="all, delete-orphan")
    notes = db.relationship('Note', backref='patient', lazy=True, cascade="all, delete-orphan")

class Report(db.Model):
    """
    Represents a single dosage report, linked to one Patient.
    """
    id = db.Column(db.Integer, primary_key=True)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    drug_name = db.Column(db.String(50), nullable=False)
    predicted_dose = db.Column(db.String(50))
    model_used = db.Column(db.String(50))
    confidence = db.Column(db.String(20))
    doctor_name = db.Column(db.String(150))
    report_data_json = db.Column(db.Text) 
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

# --- ADD THIS NEW MODEL ---
class Note(db.Model):
    """
    Represents a clinical note for a patient.
    """
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    note_text = db.Column(db.Text, nullable=False)

    # Foreign Key to link this Note to a Patient
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

    # Foreign Key to link this Note to the Doctor who wrote it
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
# --- END OF NEW MODEL ---

# --- User Loader Function for Flask-Login ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- DEFINE FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- LOAD EVERYTHING ON STARTUP ---
print("--- Loading models... ---")
try:
    # Build absolute paths to each file
    base_model_path = os.path.join(MODEL_DIR, 'random_forest_base_v1.pkl')
    base_cols_path = os.path.join(MODEL_DIR, 'base_model_columns.pkl')
    enhanced_model_path = os.path.join(MODEL_DIR, 'random_forest_enhanced_v1.pkl')
    enhanced_cols_path = os.path.join(MODEL_DIR, 'model_columns.pkl')

    # Load models and columns
    base_model = joblib.load(base_model_path)
    base_model_columns = joblib.load(base_cols_path)
    enhanced_model = joblib.load(enhanced_model_path)
    enhanced_model_columns = joblib.load(enhanced_cols_path)
    
    # Initialize SHAP Explainers
    enhanced_explainer = shap.TreeExplainer(enhanced_model)
    base_explainer = shap.TreeExplainer(base_model)
    
    print("--- All models, column lists, and SHAP explainers loaded successfully. ---")

except FileNotFoundError as e:
    print(f"--- FATAL ERROR: MODEL FILE NOT FOUND ---")
    print(f"Could not find file: {e.filename}")
    print(f"Please ensure your 'models' folder is in the same directory as app.py and contains all .pkl files.")
    raise 
except Exception as e:
    print(f"--- FATAL ERROR loading models: {e} ---")
    raise 

# --- END OF MODEL LOADING ---


# --- HELPER FUNCTIONS ---

# --- ADD THIS NEW HELPER FUNCTION ---
def get_interaction_warnings(checked_drugs_list):
    """
    Checks a list of drug names and returns specific warning strings
    for any known severe interactions with Warfarin.
    """
    warnings = []
    if not checked_drugs_list:
        return warnings # Return an empty list if no drugs were checked

    if "Amiodarone" in checked_drugs_list:
        warnings.append(
            "<strong>Severe Interaction: Amiodarone.</strong> Potentiates Warfarin effect by inhibiting CYP2C9. "
            "May require a 30-50% reduction in Warfarin dose. Monitor INR extremely closely."
        )
    if "Fluconazole" in checked_drugs_list:
        warnings.append(
            "<strong>Severe Interaction: Fluconazole.</strong> Potentiates Warfarin effect by strongly inhibiting CYP2C9. "
            "Significant dose reduction is likely required. Monitor INR frequently."
        )
    if "Bactrim" in checked_drugs_list:
        warnings.append(
            "<strong>Severe Interaction: TMP/SMX (Bactrim).</strong> Potentiates Warfarin effect. "
            "A dose reduction is often necessary. Monitor INR closely, especially upon starting or stopping."
        )
    if "Rifampin" in checked_drugs_list:
        warnings.append(
            "<strong>Severe Interaction: Rifampin.</strong> *Reduces* Warfarin effect by strongly inducing CYP2C9. "
            "A significant *increase* in Warfarin dose may be required. Monitor INR closely."
        )
    if "Carbamazepine" in checked_drugs_list:
        warnings.append(
            "<strong>Interaction: Carbamazepine.</strong> *Reduces* Warfarin effect by inducing metabolism. "
            "A higher Warfarin dose may be required."
        )
    
    return warnings

def run_model_prediction(patient_data_dict):
    """
    Runs the prediction using the correct model (Base or Enhanced)
    and returns the prediction, model name, and SHAP explanation.
    """
    is_enhanced = False
    for key in patient_data_dict.keys():
        if key.startswith('CYP2C9_genotypes') or key.startswith('VKORC1_genotype'):
            is_enhanced = True
            break
            
    # Select the correct model, columns, and explainer
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

    # Create a DataFrame and reindex to match model's training columns
    patient_df = pd.DataFrame([patient_data_dict])
    patient_df = patient_df.reindex(columns=columns_to_use, fill_value=0)
    
    # --- Run Prediction ---
    prediction_array = model_to_use.predict(patient_df)
    predicted_dose = round(prediction_array[0], 2)
    
    # --- Get Model Confidence (Standard Deviation) ---
    tree_predictions = [tree.predict(patient_df) for tree in model_to_use.estimators_]
    std_dev = np.std(tree_predictions)
    
    # --- Get SHAP Explanation ---
    shap_values = explainer_to_use.shap_values(patient_df)
    feature_names = patient_df.columns
    shap_values_for_instance = shap_values[0]
    
    abs_shap_values = np.abs(shap_values_for_instance)
    top_indices = np.argsort(abs_shap_values)[-5:] # Get indices of top 5
    
    shap_explanation = {}
    for i in reversed(top_indices): # Reverse to show most important first
        if abs_shap_values[i] > 0: # Only show features that had an impact
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
    """
    Converts the model's standard deviation into a human-readable
    confidence score.
    """
    if std_dev < 0.5:
        score = "High"
        explanation = "The model's internal estimators are in strong agreement."
    elif std_dev < 1.0:
        score = "Medium"
        explanation = "The model's internal estimators show some variance. Use with caution."
    else:
        score = "Low"
        explanation = "The model's internal estimators have significant disagreement. This case may be unusual or have conflicting data. Verify input and proceed with extreme caution."
    return score, explanation

def get_human_explanation(shap_dict):
    """
    Converts the SHAP dictionary into a list of human-readable strings.
    """
    explanations = []
    for feature, value in shap_dict.items():
        # Clean up feature names for display
        if feature == "Weight__kg_": display_name = "Weight"
        elif feature == "Height__cm_": display_name = "Height"
        elif feature.startswith("CYP2C9"): display_name = "CYP2C9 Genotype"
        elif feature.startswith("VKORC1"): display_name = "VKORC1 Genotype"
        else: display_name = feature.replace("Race_", "")

        # Create the explanation string
        if value > 0: direction = "<strong>increased</strong>"
        else: direction = "<strong>decreased</strong>"
            
        explanations.append(f"<strong>{display_name}</strong> {direction} the dose recommendation.")
    
    return explanations

def get_clinical_suggestions(shap_dict, confidence):
    """
    Generates actionable clinical suggestions based on the SHAP values.
    """
    suggestions = []
    
    for feature in shap_dict.keys():
        if "VKORC1" in feature and shap_dict[feature] < -0.5:
            suggestions.append("<strong>High Sensitivity Detected:</strong> Patient's VKORC1 genotype strongly suggests a lower dose requirement. Titrate slowly and monitor INR closely.")
        if "CYP2C9" in feature and shap_dict[feature] < -0.5:
            suggestions.append("<strong>Slow Metabolizer Detected:</strong> Patient's CYP2C9 genotype suggests slower drug clearance. A lower dose is likely required to avoid over-anticoagulation.")

    if "Weight__kg_" in shap_dict and shap_dict["Weight__kg_"] > 1.0:
         suggestions.append("Patient's high body weight is a major factor increasing the dose. Monitor for efficacy.")
    
    if confidence == "Low":
        suggestions.append("<strong>Low Model Confidence:</strong> The model found this case to be unusual. Please review all patient data and proceed with extra caution.")

    if not suggestions:
        suggestions.append("Standard dosing protocol advised. Monitor INR as per guidelines.")
        
    return suggestions

# --- END OF HELPER FUNCTIONS ---


# --- PDF DOWNLOAD HELPER FUNCTION ---
def process_prediction_data(form_data):
    """
    Processes form data from the download button, runs prediction,
    and returns all dictionaries needed for the report template.
    This is ONLY for the download button.
    """
    
    # === STEP 1: GET PATIENT INFO ===
    patient_info_dict = {
        "patient_name": form_data.get('patient_name'),
        "patient_dob": form_data.get('patient_dob'),
        "patient_gender": form_data.get('patient_gender'),
        "patient_country": form_data.get('patient_country'),
        "patient_address": form_data.get('patient_address')
    }

    # === STEP 2: GET CLINICAL INFO ===
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

    # === STEP 3: CREATE DISPLAY-FRIENDLY DICTIONARY ===
    clinical_info_display = {
        "Age": form_data.get('Age'),
        "Height__cm_": form_data.get('Height__cm_'),
        "Weight__kg_": form_data.get('Weight__kg_'),
        "Race_Display": race.split('_')[-1] if race else "N/A",
        "CYP2C9_Display": cyp2c9.split('__')[-1].replace('_', '/*') if cyp2c9 else "N/A",
        "VKORC1_Display": vkorc1.split('_')[-1] if vkorc1 else "N/A"
    }

    # === STEP 4: RUN PREDICTION & ANALYSIS (THE NEW, CORRECT WAY) ===
    pred_data = run_model_prediction(clinical_data_dict) 
    confidence, conf_expl = get_confidence_score(pred_data['std_dev'])
    human_expl = get_human_explanation(pred_data['shap_explanation'])
    suggestions = get_clinical_suggestions(pred_data['shap_explanation'], confidence)

    # === STEP 5: ASSEMBLE FINAL RESULTS DICTIONARY ===
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

    # === STEP 6: RETURN ALL DICTIONARIES ===
    return patient_info_dict, clinical_info_display, results_dict


# --- PAGE ROUTES ---
@app.route('/')
def home(): 
    return render_template('index.html')

# --- USER AUTHENTICATION ROUTES ---

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
            flash('Passwords do not match. Please try again.', 'danger')
            return redirect(url_for('register'))

        user_by_email = User.query.filter_by(email=email).first()
        user_by_reg_id = User.query.filter_by(medical_reg_id=reg_id).first()

        if user_by_email:
            flash('An account with this email address already exists.', 'danger')
            return redirect(url_for('register'))
        
        if user_by_reg_id:
            flash('An account with this Medical ID already exists.', 'danger')
            return redirect(url_for('register'))

        new_doctor = User(
            full_name=name,
            email=email,
            medical_reg_id=reg_id
        )
        new_doctor.set_password(password)
        
        db.session.add(new_doctor)
        db.session.commit()

        flash('Account created successfully. Please log in.', 'success')
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
            flash('Invalid email or password. Please try again.', 'danger')
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

# --- DOCTOR & PATIENT MANAGEMENT ROUTES ---

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch all patients associated with the currently logged-in doctor
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.full_name).all()

    # --- NEW CODE ---
    # Count all reports by joining through the patients table
    total_reports = db.session.query(Report).join(Patient).filter(Patient.doctor_id == current_user.id).count()
    # --- END NEW CODE ---

    # Render the new dashboard template, passing the new stats
    return render_template('dashboard.html', patients=patients, total_reports=total_reports, total_patients=len(patients))

@app.route('/add_patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    if request.method == 'POST':
        aadhar = request.form.get('aadhar')
        
        existing_patient = Patient.query.filter_by(aadhar=aadhar).first()
        if existing_patient:
            flash('A patient with this Aadhar number is already registered in the system.', 'danger')
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
        flash("You are not authorized to view this patient.", "danger")
        return redirect(url_for('dashboard'))
        
    reports = Report.query.filter_by(patient_id=patient.id).order_by(Report.generated_at.desc()).all()
    notes = Note.query.filter_by(patient_id=patient.id).order_by(Note.created_at.desc()).all()
    
    return render_template('view_patient.html', patient=patient, reports=reports, notes=notes)

# --- PREDICTION WORKFLOW ROUTES ---

@app.route('/patient/<int:patient_id>/select_drug', methods=['GET'])
@login_required
def select_drug(patient_id):
    patient = Patient.query.get_or_404(patient_id)

    if patient.doctor_id != current_user.id:
        flash("You are not authorized to access this patient.", "danger")
        return redirect(url_for('dashboard'))

    return render_template('select_drug.html', patient=patient)

@app.route('/patient/<int:patient_id>/redirect_form', methods=['POST'])
@login_required
def redirect_to_drug_form(patient_id):
    drug_name = request.form.get('drug_name')
    
    if drug_name == 'warfarin':
        return redirect(url_for('warfarin_form', patient_id=patient_id))
    
    flash("Invalid drug selected.", "danger")
    return redirect(url_for('dashboard'))

@app.route('/patient/<int:patient_id>/warfarin_form', methods=['GET'])
@login_required
def warfarin_form(patient_id):
    patient = Patient.query.get_or_404(patient_id)

    # Security check
    if patient.doctor_id != current_user.id:
        flash("You are not authorized to access this patient.", "danger")
        return redirect(url_for('dashboard'))

    # --- NEW: Calculate Age ---
    calculated_age = 0 # Default
    try:
        dob = datetime.strptime(patient.dob, '%Y-%m-%d')
        today = datetime.today()
        calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        pass # Keep age 0 if DOB is invalid

    return render_template('warfarin_form.html', patient=patient, calculated_age=calculated_age)

@app.route('/patient/<int:patient_id>/generate_warfarin_report', methods=['POST'])
@login_required
def generate_warfarin_report(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    if patient.doctor_id != current_user.id:
        flash("You are not authorized to access this patient.", "danger")
        return redirect(url_for('dashboard'))

    # === STEP 1: GET PATIENT INFO (from database) ===
    patient_info_dict = {
        "patient_name": patient.full_name,
        "patient_dob": patient.dob,
        "patient_gender": patient.gender,
        "patient_country": patient.country,
        "patient_address": patient.address 
    }

    # === STEP 2: GET CLINICAL INFO (from form) ===
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

    # === STEP 3: CREATE DISPLAY-FRIENDLY DICTIONARY ===
    clinical_info_display = {
        "Age": request.form.get('Age'),
        "Height__cm_": request.form.get('Height__cm_'),
        "Weight__kg_": request.form.get('Weight__kg_'),
        "Race_Display": race.split('_')[-1] if race else "N/A",
        "CYP2C9_Display": cyp2c9.split('__')[-1].replace('_', '/*') if cyp2c9 else "N/A",
        "VKORC1_Display": vkorc1.split('_')[-1] if vkorc1 else "N/A"
    }

    # === STEP 4: RUN PREDICTION & ANALYSIS ===
    pred_data = run_model_prediction(clinical_data_dict) 
    confidence, conf_expl = get_confidence_score(pred_data['std_dev'])
    human_expl = get_human_explanation(pred_data['shap_explanation'])
    suggestions = get_clinical_suggestions(pred_data['shap_explanation'], confidence)
    interaction_warnings = get_interaction_warnings(interacting_drugs)
    # === STEP 5: ASSEMBLE FULL REPORT DATA ===
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

    # === STEP 6: SAVE REPORT TO DATABASE ===
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

    # === STEP 7: CREATE RESPONSE WITH NO-CACHE HEADERS ===
    response = make_response(render_template(
        'display_report.html',
        patient_info=patient_info_dict,
        clinical_info=clinical_info_display,
        results=results_dict,
        doctor_name=doctor_name,
        request=request, # Pass the request object so the download button can be shown
        interaction_warnings=interaction_warnings
    ))

    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    report = Report.query.get_or_404(report_id)

    if report.patient.doctor_id != current_user.id:
        flash("You are not authorized to view this report.", "danger")
        return redirect(url_for('dashboard'))

    try:
        report_data = json.loads(report.report_data_json)
    except:
        flash("Error: Report data is corrupted.", "danger")
        return redirect(url_for('view_patient', patient_id=report.patient_id))
    # --- NEW: Re-generate warnings from saved data ---
    saved_interacting_drugs = report_data.get('interacting_drugs', [])
    interaction_warnings = get_interaction_warnings(saved_interacting_drugs)
    # --- END NEW ---
    response = make_response(render_template(
        'display_report.html',
        patient_info=report_data.get('patient_info'),
        clinical_info=report_data.get('clinical_info'),
        results=report_data.get('results'),
        doctor_name=report_data.get('doctor_name'),
        request=None,  # Set request to None to hide the download button
        interaction_warnings=interaction_warnings
    ))
    
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# --- PDF DOWNLOAD ROUTE ---

@app.route('/download_report', methods=['POST'])
@login_required
def download_report():
    form_data = request.form.to_dict()
    
    # Use our new, correct processing function
    patient_info, clinical_info, results = process_prediction_data(form_data)
    
    # Get the doctor name from the form
    doctor_name = form_data.get('doctor_name', current_user.full_name)

    # --- RENDER THE CORRECT TEMPLATE ---
    html_string = render_template(
        'display_report.html', 
        patient_info=patient_info, 
        clinical_info=clinical_info, 
        results=results,
        doctor_name=doctor_name,
        request=None  # Set to None to hide download buttons in the PDF
    )
    
    pdf_file = HTML(string=html_string).write_pdf()
    
    return Response(
        pdf_file, 
        mimetype='application/pdf', 
        headers={'Content-Disposition': 'attachment;filename=DosageReport.pdf'}
    )

@app.route('/report/<int:report_id>/delete', methods=['POST'])
@login_required
def delete_report(report_id):
    report = Report.query.get_or_404(report_id)
    patient_id = report.patient.id # Save the patient ID for redirecting back
    
    # --- SECURITY CHECK ---
    if report.patient.doctor_id != current_user.id:
        flash("You are not authorized to delete this report.", "danger")
        return redirect(url_for('dashboard'))
        
    db.session.delete(report)
    db.session.commit()
    
    flash("Report deleted successfully.", "success")
    # Redirect back to the patient's profile page
    return redirect(url_for('view_patient', patient_id=patient_id))

@app.route('/patient/<int:patient_id>/delete', methods=['POST'])
@login_required
def delete_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    # --- SECURITY CHECK ---
    if patient.doctor_id != current_user.id:
        flash("You are not authorized to delete this patient.", "danger")
        return redirect(url_for('dashboard'))
        
    # Thanks to our 'cascade' update, deleting the patient
    # will automatically delete all of their reports.
    db.session.delete(patient)
    db.session.commit()
    
    flash(f"Patient '{patient.full_name}' and all associated reports have been deleted.", "success")
    # Redirect back to the main dashboard
    return redirect(url_for('dashboard'))

@app.route('/patient/<int:patient_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    # --- SECURITY CHECK ---
    if patient.doctor_id != current_user.id:
        flash("You are not authorized to edit this patient.", "danger")
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        # --- Handle the form submission ---
        new_aadhar = request.form.get('aadhar')
        
        # Check if new Aadhar is already taken by ANOTHER patient
        if new_aadhar != patient.aadhar:
            existing_patient = Patient.query.filter_by(aadhar=new_aadhar).first()
            if existing_patient:
                flash('That Aadhar number is already assigned to another patient.', 'danger')
                return render_template('edit_patient.html', patient=patient)
        
        # Update patient object with new data from the form
        patient.full_name = request.form.get('full_name')
        patient.aadhar = new_aadhar
        patient.dob = request.form.get('dob')
        patient.gender = request.form.get('gender')
        patient.country = request.form.get('country')
        patient.address = request.form.get('address')
        
        db.session.commit()
        
        flash('Patient details updated successfully.', 'success')
        return redirect(url_for('view_patient', patient_id=patient.id))
    
    # --- For a GET request, just show the form pre-filled with patient data ---
    return render_template('edit_patient.html', patient=patient)

@app.route('/patient/<int:patient_id>/add_note', methods=['POST'])
@login_required
def add_note(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    # --- SECURITY CHECK ---
    if patient.doctor_id != current_user.id:
        flash("You are not authorized to access this patient.", "danger")
        return redirect(url_for('dashboard'))
        
    note_text = request.form.get('note_text')
    if not note_text:
        flash("Note cannot be empty.", "danger")
        return redirect(url_for('view_patient', patient_id=patient_id))
        
    # Create the new note and link it to the patient AND the doctor
    new_note = Note(
        note_text=note_text,
        patient_id=patient.id,
        doctor_id=current_user.id
    )
    
    db.session.add(new_note)
    db.session.commit()
    
    flash("Note added successfully.", "success")
    # Redirect back to the patient's profile
    return redirect(url_for('view_patient', patient_id=patient_id, _anchor='notes-tab'))

@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    # 'action' will be either 'update_details' or 'delete_account'
    # This lets us have two separate forms on one page.
    action = request.form.get('action')

    if request.method == 'POST':
        
        if action == 'update_details':
            # --- LOGIC FOR UPDATING DETAILS ---
            new_email = request.form.get('email')
            new_name = request.form.get('full_name')
            new_reg_id = request.form.get('medical_reg_id')

            # Check if new email is already taken by ANOTHER user
            if new_email != current_user.email:
                existing_user = User.query.filter_by(email=new_email).first()
                if existing_user:
                    flash('That email address is already in use.', 'danger')
                    return redirect(url_for('account'))
            
            # Check if new medical ID is already taken by ANOTHER user
            if new_reg_id != current_user.medical_reg_id:
                existing_user = User.query.filter_by(medical_reg_id=new_reg_id).first()
                if existing_user:
                    flash('That medical registration ID is already in use.', 'danger')
                    return redirect(url_for('account'))

            # Update the current user's details
            current_user.full_name = new_name
            current_user.email = new_email
            current_user.medical_reg_id = new_reg_id
            
            db.session.commit()
            flash('Your account details have been updated successfully.', 'success')
            return redirect(url_for('account'))

        elif action == 'delete_account':
            # --- LOGIC FOR DELETING ACCOUNT ---
            # Check if the entered password is correct
            if not current_user.check_password(request.form.get('password')):
                flash("Incorrect password. Account was not deleted.", "danger")
                return redirect(url_for('account'))
                
            # If password is correct, delete the user.
            # The 'cascade' will automatically delete all their patients and reports.
            db.session.delete(current_user)
            db.session.commit()
            
            logout_user() # Log them out
            flash("Your account and all associated data have been permanently deleted.", "success")
            return redirect(url_for('home')) # Redirect to the public homepage

    # For a GET request, just show the account page
    return render_template('account.html')

# --- MAIN RUN ---
if __name__ == '__main__':
    app.run(debug=True)

# --- NEW: DATABASE CREATION COMMAND ---
@app.cli.command("create-db")
@with_appcontext
def create_db():
    """Creates the database tables."""
    db.create_all()
    print("Database tables created successfully.")