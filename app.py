
from flask import Flask, request, jsonify, render_template, Response
import joblib
import pandas as pd
import shap
import numpy as np
from weasyprint import HTML
import io
import os

app = Flask(__name__)

# --- HELPER FUNCTIONS ---
def get_confidence_score(model, data):
    all_tree_predictions = [tree.predict(data) for tree in model.estimators_]
    std_dev = np.std(all_tree_predictions)
    if std_dev < 1.0: return "High", "The individual decision trees in the model reached a strong consensus."
    elif std_dev < 2.5: return "Medium", "There was some variance in the individual tree predictions, but a general consensus was reached."
    else: return "Low", "There was significant variance in the individual tree predictions. Proceed with extra clinical caution."

def get_human_explanation(shap_dict):
    feature_map = {
        'Age': 'The patient\'s age', 'Height__cm_': 'The patient\'s height', 'Weight__kg_': 'The patient\'s weight',
        'Race_White': 'Race: White', 'Race_Asian': 'Race: Asian',
        'Race_Black_or_African_American': 'Race: Black/African American',
        'CYP2C9_genotypes__1__1': 'CYP2C9 Genotype: *1/*1', 'CYP2C9_genotypes__1__2': 'CYP2C9 Genotype: *1/*2', 'CYP2C9_genotypes__1__3': 'CYP2C9 Genotype: *1/*3',
        'VKORC1_genotype___1639_G_A__3673___chr16_31015190__rs9923231_A_A': 'VKORC1 Genotype: A/A',
        'VKORC1_genotype___1639_G_A__3673___chr16_31015190__rs9923231_A_G': 'VKORC1 Genotype: A/G',
        'VKORC1_genotype___1639_G_A__3673___chr16_31015190__rs9923231_G_G': 'VKORC1 Genotype: G/G'
    }
    explanations = []
    for feature, impact in shap_dict.items():
        feature_name = feature_map.get(feature, feature.replace('_', ' '))
        direction = "increasing" if impact > 0 else "decreasing"
        magnitude = "significantly" if abs(impact) > 2.0 else "moderately"
        if abs(impact) > 0.01:
             explanations.append(f"<strong>{feature_name}</strong> was a factor, {magnitude} <strong>{direction}</strong> the required dose.")
    return explanations if explanations else ["The prediction was based on a complex interaction of all provided features."]

def get_clinical_suggestions(form_data, model_name, confidence_score):
    suggestions = []
    if "Base (Clinical-Only)" in model_name:
        suggestions.append("<strong>Genomic Data Recommended:</strong> This prediction is based on clinical data only. Accuracy can be significantly improved by providing the patient's CYP2C9 and VKORC1 genotypes.")
    if confidence_score == "Low":
        suggestions.append("<strong>Proceed with Caution:</strong> The model's confidence is low, possibly due to an unusual patient profile. Closer monitoring of the patient's INR is strongly advised.")
    if form_data.get('CYP2C9_genotypes') in ['CYP2C9_genotypes__1__3', 'CYP2C9_genotypes__1__2']:
        suggestions.append("<strong>Metabolism Alert:</strong> The patient has a reduced-function CYP2C9 genotype, which can decrease warfarin metabolism. A lower starting dose is often recommended.")
    if form_data.get('VKORC1_genotype') == 'VKORC1_genotype___1639_G_A__3673___chr16_31015190__rs9923231_A_A':
        suggestions.append("<strong>Sensitivity Alert:</strong> The patient has a VKORC1 A/A genotype, associated with increased sensitivity to warfarin. A lower dose is generally required.")
    if not suggestions:
        suggestions.append("Standard dosing protocols and INR monitoring are recommended.")
    return suggestions

# --- CENTRAL PROCESSING FUNCTION ---
def process_prediction_data(form_data):
    drug_name = form_data.get('drug_name', 'warfarin')
    model_path = os.path.join('models', drug_name)
    
    enhanced_model = joblib.load(os.path.join(model_path, 'random_forest_enhanced_v1.pkl'))
    enhanced_model_columns = joblib.load(os.path.join(model_path, 'model_columns.pkl'))
    base_model = joblib.load(os.path.join(model_path, 'random_forest_base_v1.pkl'))
    base_model_columns = joblib.load(os.path.join(model_path, 'base_model_columns.pkl'))
    
    model_input = {}
    if form_data.get('Age'): model_input['Age'] = float(form_data.get('Age'))
    if form_data.get('Height__cm_'): model_input['Height__cm_'] = float(form_data.get('Height__cm_'))
    if form_data.get('Weight__kg_'): model_input['Weight__kg_'] = float(form_data.get('Weight__kg_'))
    if form_data.get('Race'): model_input[form_data.get('Race')] = 1.0
    if form_data.get('CYP2C9_genotypes'): model_input[form_data.get('CYP2C9_genotypes')] = 1.0
    if form_data.get('VKORC1_genotype'): model_input[form_data.get('VKORC1_genotype')] = 1.0

    is_enhanced = any(key.startswith('CYP2C9_') or key.startswith('VKORC1_') for key in model_input.keys())
    
    if is_enhanced: model, columns, model_name = enhanced_model, enhanced_model_columns, "Enhanced (Clinical + Genome)"
    else: model, columns, model_name = base_model, base_model_columns, "Base (Clinical-Only)"

    explainer = shap.TreeExplainer(model)
    patient_df = pd.DataFrame([model_input]).reindex(columns=columns, fill_value=0)
    
    prediction = model.predict(patient_df)[0]
    confidence_score, confidence_explanation = get_confidence_score(model, patient_df)
    
    shap_values = explainer.shap_values(patient_df)[0]
    top_indices = np.argsort(np.abs(shap_values))[-5:]
    explanation_dict = {columns[i]: round(shap_values[i], 2) for i in reversed(top_indices) if abs(shap_values[i]) > 0.01}
    
    patient_info = {k: v for k, v in form_data.items() if k.startswith('patient_')}
    clinical_info_display = {
        'Age': form_data.get('Age'), 'Height__cm_': form_data.get('Height__cm_'), 'Weight__kg_': form_data.get('Weight__kg_'),
        'Race_Display': form_data.get('Race', 'N/A').replace('Race_', ''),
        'CYP2C9_Display': form_data.get('CYP2C9_genotypes', 'N/A').split('__')[-1].replace('_', '/') if form_data.get('CYP2C9_genotypes') else 'N/A',
        'VKORC1_Display': form_data.get('VKORC1_genotype', 'N/A').split('_')[-1] if form_data.get('VKORC1_genotype') else 'N/A'
    }
    prediction_results = {
        'predicted_dose_mg_per_week': round(prediction, 2), 'model_used': model_name,
        'confidence_score': confidence_score, 'confidence_explanation': confidence_explanation,
        'human_explanation': get_human_explanation(explanation_dict)
    }
    
    return patient_info, clinical_info_display, prediction_results


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
@app.route('/generate_report', methods=['POST'])
def generate_report():
    form_data = request.form.to_dict()
    patient_info, clinical_info, results = process_prediction_data(form_data)
    return render_template('report.html', patient_info=patient_info, clinical_info=clinical_info, results=results)

@app.route('/download_report', methods=['POST'])
def download_report():
    form_data = request.form.to_dict()
    patient_info, clinical_info, results = process_prediction_data(form_data)
    html_string = render_template('report.html', patient_info=patient_info, clinical_info=clinical_info, results=results)
    pdf_file = HTML(string=html_string).write_pdf()
    return Response(pdf_file, mimetype='application/pdf', headers={'Content-Disposition': 'attachment;filename=DosageReport.pdf'})

if __name__ == '__main__':
    app.run(debug=True)