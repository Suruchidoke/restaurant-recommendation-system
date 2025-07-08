import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed data
df = pd.read_csv('processed_restaurant_data.csv')

# Combine features
def combine_features(row):
    return f"{row['Cuisines']} {row['Price range']} {row['Rating text']}"

df['combined_features'] = df.apply(combine_features, axis=1)

# TF-IDF and cosine similarity
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(df['combined_features'])
similarity_matrix = cosine_similarity(feature_vectors)

# Recommendation function
def recommend_restaurants(restaurant_name, top_n=5):
    if restaurant_name not in df['Restaurant Name'].values:
        print("❌ Restaurant not found. Please try again.")
        return

    idx = df[df['Restaurant Name'] == restaurant_name].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    print(f"\n🍽️ Top {top_n} restaurants similar to '{restaurant_name}':\n")
    for i, score in sorted_scores:
        match = df.iloc[i]
        print(f"🔹 {match['Restaurant Name']} ({match['Cuisines']} | Price: {match['Price range']} | Rating: {match['Rating text']})")

# --- User input ---
user_input = input("🔍 Enter a restaurant name: ")
recommend_restaurants(user_input, top_n=5)
