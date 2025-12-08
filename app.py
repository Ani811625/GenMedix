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

# --- NEW IMPORTS FOR MAIL ---
from flask_mail import Mail, Message

from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'

# --- EMAIL CONFIGURATION ---
# We use environment variables so your password isn't exposed in the code
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME') 
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)

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

# ... [Keep your @app.before_request and LoginManager setup exactly as they were] ...
# (I am skipping repeating them to save space, but DO NOT DELETE THEM from your file)

# ... [Keep all your DB Models (User, Patient, Report, Note) exactly the same] ...

# ... [Keep your Model Loading logic exactly the same] ...

# ... [Keep your Helper Functions (get_interaction_warnings, etc.) exactly the same] ...


# --- ROUTES ---

@app.route('/')
def home(): 
    return render_template('index.html')

# --- UPDATED REGISTER ROUTE WITH EMAIL ---
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

        user_by_email = User.query.filter_by(email=email).first()
        if user_by_email:
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        
        user_by_reg_id = User.query.filter_by(medical_reg_id=reg_id).first()
        if user_by_reg_id:
            flash('Medical ID already registered.', 'danger')
            return redirect(url_for('register'))

        # Create User
        new_doctor = User(full_name=name, email=email, medical_reg_id=reg_id)
        new_doctor.set_password(password)
        
        try:
            db.session.add(new_doctor)
            db.session.commit()

            # --- SEND WELCOME EMAIL ---
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

# ... [Keep all other routes (login, dashboard, add_patient, predict, etc.) exactly the same] ...

if __name__ == '__main__':
    app.run(debug=True)