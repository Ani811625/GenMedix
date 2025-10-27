
from flask import Flask, request, jsonify, render_template, make_response, Response
from datetime import datetime
import joblib
import pandas as pd
import shap
import numpy as np
from weasyprint import HTML
import io
import os

app = Flask(__name__)
# --- DEFINE FILE PATHS ---
# Gets the absolute path to the directory where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Joins that path with the 'models' folder
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
    raise # This will stop the app and show you the error
except Exception as e:
    print(f"--- FATAL ERROR loading models: {e} ---")
    raise # This will stop the app and show you the error

# --- END OF MODEL LOADING ---

# # --- HELPER FUNCTIONS ---
# def get_confidence_score(model, data):
#     all_tree_predictions = [tree.predict(data) for tree in model.estimators_]
#     std_dev = np.std(all_tree_predictions)
#     if std_dev < 1.0: return "High", "The individual decision trees in the model reached a strong consensus."
#     elif std_dev < 2.5: return "Medium", "There was some variance in the individual tree predictions, but a general consensus was reached."
#     else: return "Low", "There was significant variance in the individual tree predictions. Proceed with extra clinical caution."

# def get_human_explanation(shap_dict):
#     feature_map = {
#         'Age': 'The patient\'s age', 'Height__cm_': 'The patient\'s height', 'Weight__kg_': 'The patient\'s weight',
#         'Race_White': 'Race: White', 'Race_Asian': 'Race: Asian',
#         'Race_Black_or_African_American': 'Race: Black/African American',
#         'CYP2C9_genotypes__1__1': 'CYP2C9 Genotype: *1/*1', 'CYP2C9_genotypes__1__2': 'CYP2C9 Genotype: *1/*2', 'CYP2C9_genotypes__1__3': 'CYP2C9 Genotype: *1/*3',
#         'VKORC1_genotype___1639_G_A__3673___chr16_31015190__rs9923231_A_A': 'VKORC1 Genotype: A/A',
#         'VKORC1_genotype___1639_G_A__3673___chr16_31015190__rs9923231_A_G': 'VKORC1 Genotype: A/G',
#         'VKORC1_genotype___1639_G_A__3673___chr16_31015190__rs9923231_G_G': 'VKORC1 Genotype: G/G'
#     }
#     explanations = []
#     for feature, impact in shap_dict.items():
#         feature_name = feature_map.get(feature, feature.replace('_', ' '))
#         direction = "increasing" if impact > 0 else "decreasing"
#         magnitude = "significantly" if abs(impact) > 2.0 else "moderately"
#         if abs(impact) > 0.01:
#              explanations.append(f"<strong>{feature_name}</strong> was a factor, {magnitude} <strong>{direction}</strong> the required dose.")
#     return explanations if explanations else ["The prediction was based on a complex interaction of all provided features."]

# def get_clinical_suggestions(form_data, model_name, confidence_score):
#     suggestions = []
#     if "Base (Clinical-Only)" in model_name:
#         suggestions.append("<strong>Genomic Data Recommended:</strong> This prediction is based on clinical data only. Accuracy can be significantly improved by providing the patient's CYP2C9 and VKORC1 genotypes.")
#     if confidence_score == "Low":
#         suggestions.append("<strong>Proceed with Caution:</strong> The model's confidence is low, possibly due to an unusual patient profile. Closer monitoring of the patient's INR is strongly advised.")
#     if form_data.get('CYP2C9_genotypes') in ['CYP2C9_genotypes__1__3', 'CYP2C9_genotypes__1__2']:
#         suggestions.append("<strong>Metabolism Alert:</strong> The patient has a reduced-function CYP2C9 genotype, which can decrease warfarin metabolism. A lower starting dose is often recommended.")
#     if form_data.get('VKORC1_genotype') == 'VKORC1_genotype___1639_G_A__3673___chr16_31015190__rs9923231_A_A':
#         suggestions.append("<strong>Sensitivity Alert:</strong> The patient has a VKORC1 A/A genotype, associated with increased sensitivity to warfarin. A lower dose is generally required.")
#     if not suggestions:
#         suggestions.append("Standard dosing protocols and INR monitoring are recommended.")
#     return suggestions

# --- HELPER FUNCTIONS ---
# DELETE your old functions and PASTE this entire block

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
    # Get predictions from all individual trees in the Random Forest
    tree_predictions = [tree.predict(patient_df) for tree in model_to_use.estimators_]
    std_dev = np.std(tree_predictions)
    
    # --- Get SHAP Explanation ---
    shap_values = explainer_to_use.shap_values(patient_df)
    
    # Get feature names and their corresponding shap values
    feature_names = patient_df.columns
    shap_values_for_instance = shap_values[0]
    
    # Get the top 5 most impactful features
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
    # These thresholds are examples; they can be tuned.
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

        # Create the explanation string
        if value > 0:
            direction = "<strong>increased</strong>"
        else:
            direction = "<strong>decreased</strong>"
            
        explanations.append(f"<strong>{display_name}</strong> {direction} the dose recommendation.")
    
    return explanations

def get_clinical_suggestions(shap_dict, confidence):
    """
    Generates actionable clinical suggestions based on the SHAP values.
    """
    suggestions = []
    
    # Check for specific genetic factors
    for feature in shap_dict.keys():
        if "VKORC1" in feature and shap_dict[feature] < -0.5:
            suggestions.append("<strong>High Sensitivity Detected:</strong> Patient's VKORC1 genotype strongly suggests a lower dose requirement. Titrate slowly and monitor INR closely.")
        if "CYP2C9" in feature and shap_dict[feature] < -0.5:
            suggestions.append("<strong>Slow Metabolizer Detected:</strong> Patient's CYP2C9 genotype suggests slower drug clearance. A lower dose is likely required to avoid over-anticoagulation.")

    # Check for weight impact
    if "Weight__kg_" in shap_dict and shap_dict["Weight__kg_"] > 1.0:
         suggestions.append("Patient's high body weight is a major factor increasing the dose. Monitor for efficacy.")
    
    # Check for confidence
    if confidence == "Low":
        suggestions.append("<strong>Low Model Confidence:</strong> The model found this case to be unusual. Please review all patient data and proceed with extra caution.")

    if not suggestions:
        suggestions.append("Standard dosing protocol advised. Monitor INR as per guidelines.")
        
    return suggestions

# --- END OF HELPER FUNCTIONS ---

# --- CENTRAL PROCESSING FUNCTION ---
# def process_prediction_data(form_data):
#     drug_name = form_data.get('drug_name', 'warfarin')
#     model_path = os.path.join('models', drug_name)
    
#     enhanced_model = joblib.load(os.path.join(model_path, 'random_forest_enhanced_v1.pkl'))
#     enhanced_model_columns = joblib.load(os.path.join(model_path, 'model_columns.pkl'))
#     base_model = joblib.load(os.path.join(model_path, 'random_forest_base_v1.pkl'))
#     base_model_columns = joblib.load(os.path.join(model_path, 'base_model_columns.pkl'))
    
#     model_input = {}
#     if form_data.get('Age'): model_input['Age'] = float(form_data.get('Age'))
#     if form_data.get('Height__cm_'): model_input['Height__cm_'] = float(form_data.get('Height__cm_'))
#     if form_data.get('Weight__kg_'): model_input['Weight__kg_'] = float(form_data.get('Weight__kg_'))
#     if form_data.get('Race'): model_input[form_data.get('Race')] = 1.0
#     if form_data.get('CYP2C9_genotypes'): model_input[form_data.get('CYP2C9_genotypes')] = 1.0
#     if form_data.get('VKORC1_genotype'): model_input[form_data.get('VKORC1_genotype')] = 1.0

#     is_enhanced = any(key.startswith('CYP2C9_') or key.startswith('VKORC1_') for key in model_input.keys())
    
#     if is_enhanced: model, columns, model_name = enhanced_model, enhanced_model_columns, "Enhanced (Clinical + Genome)"
#     else: model, columns, model_name = base_model, base_model_columns, "Base (Clinical-Only)"

#     explainer = shap.TreeExplainer(model)
#     patient_df = pd.DataFrame([model_input]).reindex(columns=columns, fill_value=0)
    
#     prediction = model.predict(patient_df)[0]
#     confidence_score, confidence_explanation = get_confidence_score(model, patient_df)
    
#     shap_values = explainer.shap_values(patient_df)[0]
#     top_indices = np.argsort(np.abs(shap_values))[-5:]
#     explanation_dict = {columns[i]: round(shap_values[i], 2) for i in reversed(top_indices) if abs(shap_values[i]) > 0.01}
    
#     patient_info = {k: v for k, v in form_data.items() if k.startswith('patient_')}
#     clinical_info_display = {
#         'Age': form_data.get('Age'), 'Height__cm_': form_data.get('Height__cm_'), 'Weight__kg_': form_data.get('Weight__kg_'),
#         'Race_Display': form_data.get('Race', 'N/A').replace('Race_', ''),
#         'CYP2C9_Display': form_data.get('CYP2C9_genotypes', 'N/A').split('__')[-1].replace('_', '/') if form_data.get('CYP2C9_genotypes') else 'N/A',
#         'VKORC1_Display': form_data.get('VKORC1_genotype', 'N/A').split('_')[-1] if form_data.get('VKORC1_genotype') else 'N/A'
#     }
#     prediction_results = {
#         'predicted_dose_mg_per_week': round(prediction, 2), 'model_used': model_name,
#         'confidence_score': confidence_score, 'confidence_explanation': confidence_explanation,
#         'human_explanation': get_human_explanation(explanation_dict)
#     }
    
#     return patient_info, clinical_info_display, prediction_results


#
# DELETE your old 'def process_prediction_data(form_data):' function
#
# REPLACE it with this entire new function:
#
def process_prediction_data(form_data):
    """
    Processes form data, runs prediction, and returns all dictionaries
    needed for the report template.
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
    # This dictionary holds the raw data for the model
    clinical_data_dict = {
        "Age": float(form_data.get('Age')),
        "Height__cm_": float(form_data.get('Height__cm_')),
        "Weight__kg_": float(form_data.get('Weight__kg_')),
    }
    
    # Add one-hot encoded features from dropdowns
    race = form_data.get('Race')
    cyp2c9 = form_data.get('CYP2C9_genotypes')
    vkorc1 = form_data.get('VKORC1_genotype')

    if race: clinical_data_dict[race] = 1.0
    if cyp2c9: clinical_data_dict[cyp2c9] = 1.0
    if vkorc1: clinical_data_dict[vkorc1] = 1.0

    # === STEP 3: CREATE DISPLAY-FRIENDLY DICTIONARY ===
    # This dictionary is just for showing the values on the report
    clinical_info_display = {
        "Age": form_data.get('Age'),
        "Height__cm_": form_data.get('Height__cm_'),
        "Weight__kg_": form_data.get('Weight__kg_'),
        "Race_Display": race.split('_')[-1] if race else "N/A",
        "CYP2C9_Display": cyp2c9.split('__')[-1].replace('_', '/*') if cyp2c9 else "N/A",
        "VKORC1_Display": vkorc1.split('_')[-1] if vkorc1 else "N/A"
    }

    # === STEP 4: RUN PREDICTION & ANALYSIS (THE NEW, CORRECT WAY) ===
    # This calls the helper function we added earlier
    pred_data = run_model_prediction(clinical_data_dict) 
    
    # THIS IS THE LINE THAT FIXES THE BUG:
    # It correctly calls get_confidence_score with only one argument
    confidence, conf_expl = get_confidence_score(pred_data['std_dev'])
    
    # Get the other helper results
    human_expl = get_human_explanation(pred_data['shap_explanation'])
    suggestions = get_clinical_suggestions(pred_data['shap_explanation'], confidence)

    # === STEP 5: ASSEMBLE FINAL RESULTS DICTIONARY ===
    # This dictionary is passed to the report template
    results_dict = {
        "predicted_dose_mg_per_week": pred_data['prediction'],
        "model_name": pred_data['model_name'],
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
def home(): return render_template('index.html')
@app.route('/select_drug')
def select_drug(): return render_template('select_drug.html')
@app.route('/patient_form', methods=['POST'])
def patient_form():
    drug_name = request.form.get('drug_name')
    return render_template('patient_form.html', drug_name=drug_name)
@app.route('/predict_page', methods=['POST'])
def predict_page():
    patient_info = request.form.to_dict()
    return render_template('predict.html', patient_info=patient_info)
@app.route('/about')
def about(): return render_template('about.html')
@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/background')
def background():
    return render_template('background.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# --- REPORT GENERATION & DOWNLOAD ROUTES ---
# @app.route('/generate_report', methods=['POST'])
# def generate_report():
#     form_data = request.form.to_dict()
#     patient_info, clinical_info, results = process_prediction_data(form_data)
#     # return render_template('report.html', patient_info=patient_info, clinical_info=clinical_info, results=results)
#     # --- REPLACE your old return render_template(...) with this: ---
    
# # 1. Create the response object
#     response = make_response(render_template(
#        'report.html',
#         patient_info=patient_info_dict,
#         clinical_info=clinical_info_display,
#         results=results_dict,
#         request=request 
#     ))

# # 2. Add headers to prevent caching
#     response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '0'

# # 3. Return the modified response
#     return response

@app.route('/generate_report', methods=['POST'])
def generate_report():
    
    # === STEP 1: GET PATIENT INFO (FROM HIDDEN FORM FIELDS) ===
    patient_info_dict = {
        "patient_name": request.form.get('patient_name'),
        "patient_dob": request.form.get('patient_dob'),
        "patient_gender": request.form.get('patient_gender'),
        "patient_country": request.form.get('patient_country'),
        "patient_address": request.form.get('patient_address') # Added this from your form
    }

    # === STEP 2: GET CLINICAL INFO (FROM FORM FIELDS) ===
    # This dictionary holds the raw data for the model
    clinical_data_dict = {
        "Age": float(request.form.get('Age')),
        "Height__cm_": float(request.form.get('Height__cm_')),
        "Weight__kg_": float(request.form.get('Weight__kg_')),
    }
    
    # Add one-hot encoded features from dropdowns
    race = request.form.get('Race')
    cyp2c9 = request.form.get('CYP2C9_genotypes')
    vkorc1 = request.form.get('VKORC1_genotype')

    if race:
        clinical_data_dict[race] = 1.0
    if cyp2c9:
        clinical_data_dict[cyp2c9] = 1.0
    if vkorc1:
        clinical_data_dict[vkorc1] = 1.0

    # === STEP 3: CREATE DISPLAY-FRIENDLY DICTIONARY ===
    # This dictionary is just for showing the values on the report
    clinical_info_display = {
        "Age": request.form.get('Age'),
        "Height__cm_": request.form.get('Height__cm_'),
        "Weight__kg_": request.form.get('Weight__kg_'),
        "Race_Display": race.split('_')[-1] if race else "N/A",
        "CYP2C9_Display": cyp2c9.split('__')[-1].replace('_', '/*') if cyp2c9 else "N/A",
        "VKORC1_Display": vkorc1.split('_')[-1] if vkorc1 else "N/A"
    }

    # === STEP 4: RUN PREDICTION & ANALYSIS ===
    # (This assumes your helper functions from our previous work are in app.py)
    # 1. Run the core model prediction
    pred_data = run_model_prediction(clinical_data_dict) 
    
    # 2. Get confidence score
    confidence, conf_expl = get_confidence_score(pred_data['std_dev'])
    
    # 3. Get SHAP explanation
    human_expl = get_human_explanation(pred_data['shap_explanation'])
    
    # 4. Get clinical suggestions
    suggestions = get_clinical_suggestions(pred_data['shap_explanation'], confidence)

    # === STEP 5: ASSEMBLE FINAL RESULTS DICTIONARY ===
    # This dictionary is passed to the report template
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

    # === STEP 6: CREATE RESPONSE WITH NO-CACHE HEADERS ===
    # (This is the new block you added, now in the correct place)
    response = make_response(render_template(
        'report.html',
        patient_info=patient_info_dict,
        clinical_info=clinical_info_display,
        results=results_dict,
        request=request 
    ))

    # Add headers to prevent caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    # Return the modified response
    return response

@app.route('/download_report', methods=['POST'])
def download_report():
    form_data = request.form.to_dict()
    patient_info, clinical_info, results = process_prediction_data(form_data)
    html_string = render_template('report.html', patient_info=patient_info, clinical_info=clinical_info, results=results)
    pdf_file = HTML(string=html_string).write_pdf()
    return Response(pdf_file, mimetype='application/pdf', headers={'Content-Disposition': 'attachment;filename=DosageReport.pdf'})

if __name__ == '__main__':
    app.run(debug=True)