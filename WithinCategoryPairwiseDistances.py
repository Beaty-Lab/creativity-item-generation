import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
from scipy import stats
from scipy.spatial import ConvexHull, Delaunay
import numpy as np

# Load the sentence transformer model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Function to compute sentence embeddings
def get_embeddings(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to compute pairwise semantic distances (1 - cosine similarity)
def compute_pairwise_distances(embeddings):
    cosine_sim = cosine_similarity(embeddings)
    semantic_distances = 1 - cosine_sim
    upper_triangle_indices = np.triu_indices(semantic_distances.shape[0], k=1)
    upper_triangle_distances = semantic_distances[upper_triangle_indices]
    return upper_triangle_distances

# Function to reduce dimensionality using PCA
def reduce_dimensionality(embeddings, n_components=3):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

# Function to calculate convex hull volume in reduced dimensional space
def compute_convex_hull_volume(embeddings):
    if len(embeddings) < embeddings.shape[1] + 1:
        print(f"Not enough points ({len(embeddings)}) for dimensionality ({embeddings.shape[1]})")
        return np.nan
    try:
        hull = ConvexHull(embeddings)
        return hull.volume
    except Exception as e:
        print(f"Convex hull error: {e}")
        return np.nan

def compute_overlap_volume(hull1, hull2_points):
    try:
        delaunay = Delaunay(hull1.points[hull1.vertices])
        inside = delaunay.find_simplex(hull2_points) >= 0
        overlap_points = hull2_points[inside]
        if len(overlap_points) < hull1.points.shape[1] + 1:
            return 0.0
        overlap_hull = ConvexHull(overlap_points)
        return overlap_hull.volume
    except Exception as e:
        print(f"Error computing overlap volume: {e}")
        return 0.0

# Function to calculate the bounding sphere radius
def compute_bounding_sphere_radius(embeddings):
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return distances.max()

def compute_sphere_overlap(radius1, center1, embeddings2):
    distances = np.linalg.norm(embeddings2 - center1, axis=1)
    overlap_count = np.sum(distances <= radius1)
    return overlap_count

# Load the CSV file into a DataFrame
file_path = 'Half gravity - responses categorized and rated.csv'  # Update with the correct file path
df = pd.read_csv(file_path)

print(df.head(5))

# Initialize a list to store results for the summary DataFrame
summary_data = []
categories_embeddings = {}

# Group by "Category Number" and compute pairwise distances for "Response"
for category, group in df.groupby("Category Number"):
    responses = group["Response"].tolist()
    embeddings = get_embeddings(responses).numpy()
    
    if embeddings.shape[0] < 2:
        print(f"Not enough responses for category {category}")
        continue

    reduced_embeddings = reduce_dimensionality(embeddings, n_components=3)
    distances = compute_pairwise_distances(embeddings)
    
    mean_distance = distances.mean()
    median_distance = np.median(distances)
    mode_distance = stats.mode(distances, keepdims=True)[0][0] if len(distances) > 0 else float('nan')
    min_distance = distances.min()
    max_distance = distances.max()

    convex_hull_volume = compute_convex_hull_volume(reduced_embeddings)
    
    if not np.isnan(convex_hull_volume):
        categories_embeddings[category] = (embeddings, reduced_embeddings)
        overlap_convex_hull_volume = 0.0
        unique_convex_hull_volume = convex_hull_volume
    else:
        overlap_convex_hull_volume = np.nan
        unique_convex_hull_volume = np.nan

    sphere_radius = compute_bounding_sphere_radius(embeddings)
    sphere_center = np.mean(embeddings, axis=0)

    summary_data.append({
        "Category Number": category,
        "Mean": mean_distance,
        "Median": median_distance,
        "Mode": mode_distance,
        "Minimum": min_distance,
        "Maximum": max_distance,
        "Convex Hull Volume": convex_hull_volume,
        "Overlap Convex Hull Volume": overlap_convex_hull_volume,
        "Unique Convex Hull Volume": unique_convex_hull_volume,
        "Sphere Radius": sphere_radius,
        "Overlap Sphere Radius": 0.0,
        "Unique Sphere Radius": sphere_radius
    })

# Calculate overlap areas for categories where convex hull calculation was successful
for i, (category1, (embeddings1, reduced_embeddings1)) in enumerate(categories_embeddings.items()):
    if len(reduced_embeddings1) < reduced_embeddings1.shape[1] + 1:
        continue
    
    hull1 = ConvexHull(reduced_embeddings1)
    overlap_convex_hull_volume = 0.0
    overlap_sphere_radius = 0.0
    sphere_center1 = np.mean(embeddings1, axis=0)
    sphere_radius1 = summary_data[i]["Sphere Radius"]

    for category2, (embeddings2, reduced_embeddings2) in categories_embeddings.items():
        if category1 != category2:
            if len(reduced_embeddings2) >= reduced_embeddings2.shape[1] + 1:
                overlap_convex_hull_volume += compute_overlap_volume(hull1, reduced_embeddings2)
            overlap_sphere_radius += compute_sphere_overlap(sphere_radius1, sphere_center1, embeddings2)

    summary_data[i]["Overlap Convex Hull Volume"] = overlap_convex_hull_volume
    summary_data[i]["Unique Convex Hull Volume"] = max(0, summary_data[i]["Convex Hull Volume"] - overlap_convex_hull_volume)
    summary_data[i]["Overlap Sphere Radius"] = overlap_sphere_radius
    summary_data[i]["Unique Sphere Radius"] = max(0, summary_data[i]["Sphere Radius"] - overlap_sphere_radius)

# Create a summary DataFrame from the results
summary_df = pd.DataFrame(summary_data)

# Display the summary DataFrame
print(summary_df)

# Save the summary DataFrame as a CSV file
summary_df.to_csv('summary_statistics.csv', index=False)
