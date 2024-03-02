from flask import Flask, request, jsonify
from flask_mongoengine import MongoEngine
from flask_pymongo import PyMongo
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
import os
import pandas as pd
import joblib
from gridfs import GridFS
from io import BytesIO

app = Flask(__name__)
with app.app_context():
    app.config['MONGODB_SETTINGS'] = {
        'db': 'users',
        'host': 'mongodb://localhost/your_database_name'
    }
    db = MongoEngine(app)
    fs = GridFS(db.get_db())


    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv'}

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Model(db.Document):
    name = db.StringField(required=True)
    model_file = db.StringField(required=True)

def train_model(csv_data, uploaded_file):
    with app.app_context():
        # Save the uploaded CSV file to MongoDB using GridFS
        file_id = fs.put(BytesIO(uploaded_file.read()), filename=uploaded_file.filename)
        
        # Select features for clustering
        selected_features = ['Age', 'Gender', 'Category']
        
        # One-hot encode selected features
        encoded_features = pd.get_dummies(csv_data[selected_features], drop_first=True)
        
        # Combine encoded features with numerical features
        combined_features = pd.concat([encoded_features], axis=1)
        
        # Perform standardization
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined_features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=10, random_state=42)
        csv_data['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Save the trained K-means model
        model_file = 'kmeans_model.pkl'
        joblib.dump(kmeans, os.path.join(app.config['UPLOAD_FOLDER'], model_file))
        
        # Save the model entry in MongoDB
        model_entry = Model(name='KMeans Model', model_file=model_file, csv_file_id=file_id)
        model_entry.save()
        
        return model_file

def predict_cluster(new_data, kmeans_model, encoded_features):
    with app.app_context():
        # Preprocess new data
        selected_features = ['Age', 'Gender', 'Category']
        new_data_selected = new_data[selected_features]
        new_data_encoded = pd.get_dummies(new_data_selected, drop_first=True)
        
        # Add missing columns to new_data_encoded
        model_columns = pd.DataFrame(columns=encoded_features.columns)
        new_data_encoded = new_data_encoded.reindex(columns=model_columns.columns, fill_value=0)
        
        # Normalize the data using mean and std from training data
        new_data_normalized = (new_data_encoded - encoded_features.mean()) / encoded_features.std()
        
        # Predict cluster
        cluster = kmeans_model.predict(new_data_normalized)[0]
        return cluster

def recommend_items(cluster, customer_data):
    with app.app_context():
        cluster_data = customer_data[customer_data['cluster'] == cluster]
        common_items = cluster_data['Item Purchased'].value_counts().index[:3]  # Get top 3 common items
        return common_items

@app.route('/upload', methods=['POST'])
def upload_file():
    with app.app_context():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Load the uploaded CSV file
            csv_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Train the KMeans model
            model_file, encoded_features = train_model(csv_data, file)
            
            # Save the model entry in MongoDB
            model_entry = Model(name='KMeans Model', model_file=model_file)
            model_entry.save()
            
            return jsonify({'message': 'File uploaded and model created successfully'})

@app.route('/recommend', methods=['POST'])
def make_recommendation():
    with app.app_context():
        file_id = request.json['file_id']
        file_data = fs.get(file_id).read()
        customer_data = pd.read_csv(file_data)
        selected_features = ['Age', 'Gender', 'Category']
        new_data = request.json

        encoded_features = pd.get_dummies(customer_data[selected_features], drop_first=True)
        
        # Load the KMeans model
        model_entry = Model.objects().first()
        if not model_entry:
            return jsonify({'error': 'No model found in the database'})
        
        model_file = model_entry.model_file
        kmeans_model = joblib.load(os.path.join(app.config['UPLOAD_FOLDER'], model_file))
        
        # Predict cluster
        predicted_cluster = predict_cluster(new_data, kmeans_model, encoded_features)
        
        # Recommendation
        recommended_items = recommend_items(predicted_cluster, new_data)
        return jsonify({'recommended_items': recommended_items})

if __name__ == '__main__':
    app.run(debug=True)
