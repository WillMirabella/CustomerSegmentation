# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
df = pd.read_csv('data.csv')

# Preprocessing and Feature Engineering
# Encode categorical variables
categorical_cols = ['Gender', 'Category', 'Location', 'Size', 'Color', 'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method']
df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)

# Handling 'Frequency of Purchases' with custom mapping to numeric scale
frequency_mapping = {'Annually': 1, 'Bi-Weekly': 26, 'Quarterly': 4, 'Monthly': 12, 'Weekly': 52, 'Fortnightly': 26, 'Every 3 Months': 4}
df['Frequency of Purchases'] = df['Frequency of Purchases'].map(frequency_mapping).fillna(0)

# Normalize numerical variables
numerical_cols = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Frequency of Purchases']
df[numerical_cols] = MinMaxScaler().fit_transform(df[numerical_cols])

# Generate User and Item Profiles
# For simplicity, we're directly using encoded categorical values; further enhancements could involve embedding layers or more sophisticated encoding.
user_profile_cols = ['Age', 'Gender', 'Location', 'Subscription Status', 'Frequency of Purchases']
item_profile_cols = ['Category', 'Size', 'Color', 'Season']

# Creating a user-item matrix for interaction data
df['Interaction'] = 1  # Assuming existence of a record implies interaction
user_item_interaction_matrix = df.pivot_table(index='Customer ID', columns='Item Purchased', values='Interaction', fill_value=0)

# Calculate user similarity based on user profiles (not just interactions)
user_profiles = df.groupby('Customer ID')[user_profile_cols].mean()  # Simplistic aggregation
user_similarity = cosine_similarity(user_profiles)
user_similarity_df = pd.DataFrame(user_similarity, index=user_profiles.index, columns=user_profiles.index)

# Recommend Items for a given user
def recommend_items(user_id, num_recommendations=5):
    if user_id not in user_similarity_df.index:
        return []

    # Identify similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11].index
    
    # Aggregate items interacted with by similar users
    similar_users_interactions = user_item_interaction_matrix.loc[similar_users].sum().sort_values(ascending=False)
    
    # Filter out items already interacted with by the target user
    target_user_interacted_items = user_item_interaction_matrix.loc[user_id]
    recommendations = similar_users_interactions[~similar_users_interactions.index.isin(target_user_interacted_items[target_user_interacted_items > 0].index)]
    
    return recommendations.head(num_recommendations).index.tolist()

# Example usage
user_id = 440  # Assuming Customer ID is a string; adjust as per your dataset specifics
recommendations = recommend_items(user_id, 2)
print(f'Recommended items for user {user_id}: {recommendations}')
