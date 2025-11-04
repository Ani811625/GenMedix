#!/usr/bin/env bash

# This script runs the database setup *before* starting the app.
echo "--- Running database setup... ---"

# This command will find your DATABASE_URL and create tables in your PostgreSQL DB
flask create-db

echo "--- Database setup complete. Starting web server... ---"

# This starts your app using Gunicorn
# It binds to the port Render provides
gunicorn app:app --bind 0.0.0.0:$PORT