from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from werkzeug.utils import secure_filename
import os
from pymongo import MongoClient

app = Flask(__name__)

# Load the trained MLM model
# model = joblib.load("model.py")

# Connect to MongoDB
uri = "mongodb+srv://williamjmirabella:<password>@cluster0.hlezfnk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Define API endpoint for recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    # Parse request data
    data = request.get_json()

    # Validate input data
    if 'user_data' not in data:
        return jsonify({'error': 'Invalid input data. Missing "user_data" field.'}), 400

    # Extract user data from request
    user_data = data['user_data']

    # Perform inference with the MLM model
    try:
        # Convert user data to numpy array (adjust as needed based on your model requirements)
        user_data_np = np.array(user_data).reshape(1, -1)

        # Make predictions using the MLM model
        # recommendations = model.predict(user_data_np)

        # Format recommendations (adjust as needed based on your model output)
        recommendations = recommendations.tolist()

        # Return recommendations
        return jsonify({'recommendations': recommendations}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@app.route('/')
def signup_form():
    return render_template('signup.html')


@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    # Hash password and perform other necessary validations

    # Insert user data into MongoDB collection
    user_data = {
        'username': username,
        'email': email,
        'password': password,
        'model_file': None,
        'csv_file': None
    }

    # Handle model file upload
    if 'model_file' in request.files:
        model_file = request.files['model_file']
        if model_file.filename != '':
            # Save uploaded model file to a temporary directory
            model_filename = secure_filename(model_file.filename)
            model_filepath = os.path.join('uploads', model_filename)
            model_file.save(model_filepath)
            # Store model file path in user data
            user_data['model_file'] = model_filepath

    # Handle CSV file upload
    if 'csv_file' in request.files:
        csv_file = request.files['csv_file']
        if csv_file.filename != '':
            # Save uploaded CSV file to a temporary directory
            csv_filename = secure_filename(csv_file.filename)
            csv_filepath = os.path.join('uploads', csv_filename)
            csv_file.save(csv_filepath)
            # Store CSV file path in user data
            user_data['csv_file'] = csv_filepath

    # Insert user data into MongoDB collection
    user_id = user_collection.insert_one(user_data).inserted_id

    return 'User signed up successfully!'


if __name__ == '__main__':
    app.run(debug=True)
