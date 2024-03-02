import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
customer_data = pd.read_csv('app/src/data.csv')

# Select features for clustering
selected_features = ['Age', 'Gender', 'Category']

# One-hot encode selected features
encoded_features = pd.get_dummies(customer_data[selected_features], drop_first=True)

# Combine encoded features with numerical features
combined_features = pd.concat([encoded_features], axis=1)

# Perform standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)
print(combined_features)

# Perform K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(scaled_features)

# Save the trained K-means model
joblib.dump(kmeans, 'kmeans_model.pkl')

kmeans_model = joblib.load('kmeans_model.pkl')

# Predict a cluster for new data
def predict_cluster(new_data):
    # Preprocess new data
    new_data_selected = new_data[selected_features]
    new_data_encoded = pd.get_dummies(new_data_selected, drop_first=True)
    
    # Add missing columns to new_data_encoded
    missing_columns = set(encoded_features.columns) - set(new_data_encoded.columns)
    for col in missing_columns:
        new_data_encoded[col] = 0

    # Reorder columns to match training data
    new_data_encoded = new_data_encoded[encoded_features.columns]

    # Normalize the data using mean and std from training data
    new_data_normalized = (new_data_encoded - encoded_features.mean()) / encoded_features.std()
    
    # Predict cluster
    cluster = kmeans_model.predict(new_data_normalized)[0]
    return cluster

def recommend_items(cluster):
    cluster_data = customer_data[customer_data['cluster'] == cluster]
    common_items = cluster_data['Item Purchased'].value_counts().index[:3]  # Get top 3 common items
    return common_items

# Example of using the predict_cluster function
new_data = pd.DataFrame([{'Age': 30, 'Gender': 'Female', 'Category': 'Outerwear'}])
predicted_cluster = predict_cluster(new_data)
print("Predicted Cluster:", predicted_cluster)

recommended_items = recommend_items(predicted_cluster)
print("Recommended Items:", recommended_items)

