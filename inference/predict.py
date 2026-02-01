import sys
import json
import pickle
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# ============== Paths ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model files
knn = pickle.load(open(os.path.join(BASE_DIR, "../model/knn_model.pkl"), "rb"))
movies_df = pickle.load(open(os.path.join(BASE_DIR, "../model/movies_dataframe.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "../model/scaler.pkl"), "rb"))

feature_cols = knn["feature_cols"]   # your saved feature columns


# ============== Prepare Input Vector ==============
def prepare_input(input_json):
    genres_input = input_json.get("genre", "")
    if isinstance(genres_input, str):
        genres_input = genres_input.split("|") if genres_input else []

    # Handle explicit None from JSON (e.g. {"year": null})
    year = input_json.get("year")
    if year is None:
        year = movies_df["year"].median()

    avg = input_json.get("average_rating")
    if avg is None:
        avg = movies_df["average_rating"].mean()

    vec = []

    # fill vector based on feature columns
    for col in feature_cols:
        if col in ["year", "average_rating"]:
            vec.append(year if col == "year" else avg)
        else:
            vec.append(1.0 if col in genres_input else 0.0)

    vec = np.array(vec, dtype=float).reshape(1, -1)

    # scale last 2 cols
    vec[:, -2:] = scaler.transform(vec[:, -2:])

    return vec


# ============== Get Recommendations (now 50) ==============
def recommend(input_json, top_n=50):
    vec = prepare_input(input_json)

    # Fetch a larger pool of candidates to filter from
    # We fetch 1000 (or as many as available if less)
    n_candidates = 2000
    distances, indices = knn["nn"].kneighbors(vec, n_neighbors=min(n_candidates, len(movies_df)))

    idxs = indices[0]
    dists = distances[0]

    results = []

    # Get filter criteria
    # Year: +/- 2 years
    target_year = input_json.get("year", None)
    
    # Rating: >= target - 0.5
    target_rating = input_json.get("average_rating", None)

    # Genre: At least one matching genre (if input genres specified)
    input_genres = input_json.get("genre", "")
    if isinstance(input_genres, str) and input_genres:
        target_genres_set = set(input_genres.split("|"))
    else:
        target_genres_set = set()

    for i, idx in enumerate(idxs):
        # Skip the movie itself if it's the exact same query (distance ~ 0 might indicate same movie in training set)
        # But here we are comparing vectors. 
        # Usually idx=0 is the input vector if it was in the training set. 
        # Since we construct a new vector, we just iterate all.
        
        movie = movies_df.iloc[idx]
        
        # --- Apply Filters ---
        
        # 1. Year Filter
        if target_year is not None:
            # allow +/- 2 years
            if not (target_year - 2 <= movie["year"] <= target_year + 2):
                continue

        # 2. Rating Filter
        if target_rating is not None:
             # allow strictly >= target - 0.5
            if not (movie["average_rating"] >= target_rating - 0.5):
                continue
        
        # 3. Genre Filter
        if target_genres_set:
            movie_genres = str(movie["genres"]).split("|")
            # check intersection
            if not target_genres_set.intersection(movie_genres):
                continue

        # If passed all filters, add to results
        results.append({
            "movieId": int(movie["movieId"]),
            "title": movie["title"],
            "genres": movie["genres"],
            "year": int(movie["year"]),
            "rating": float(movie["average_rating"]),
            "distance": float(dists[i])
        })

        if len(results) >= top_n:
            break

    return results


# ============== Entry ==============
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_json = json.loads(sys.argv[1])
    else:
        input_json = json.load(sys.stdin)

    recs = recommend(input_json)
    print(json.dumps({"recommendations": recs}, ensure_ascii=False))
