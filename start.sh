#!/usr/bin/env bash

# This script runs the database setup *before* starting the app.

echo "--- Running database setup... ---"
# This command will find your DATABASE_URL and create tables in your PostgreSQL DB
flask create-db

echo "--- Database setup complete. Starting web server... ---"
# This starts your app
gunicorn app:app