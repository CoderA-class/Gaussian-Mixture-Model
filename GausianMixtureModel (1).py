import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D

# Load data
df = pd.read_csv('Mall_Customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit GMM
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X_scaled)
clusters = gmm.predict(X_scaled)
df['Cluster'] = clusters
centers = scaler.inverse_transform(gmm.means_)

# Print model parameters
print("GMM Equation: P(x) = Σ π_k × N(x | μ_k, Σ_k)")
print("\nParameters:")
for i in range(5):
    print(f"Cluster {i}: π={gmm.weights_[i]:.3f}, μ=[{centers[i][0]:.0f}, {centers[i][1]:.0f}]")

# Print silhouette score
print(f"\nSilhouette Score: {silhouette_score(X_scaled, clusters):.3f}")

# Plot training set results
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(5):
    mask = df['Cluster'] == i
    plt.scatter(df.loc[mask, 'Annual Income (k$)'],
                df.loc[mask, 'Spending Score (1-100)'],
                c=colors[i], label=f'Cluster {i}', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('GMM Customer Segments')
plt.legend()
plt.show()

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_clusters = []
for train_idx, val_idx in kf.split(X_scaled):
    gmm_fold = GaussianMixture(n_components=5, random_state=42)
    gmm_fold.fit(X_scaled[train_idx])
    fold_clusters.append(gmm_fold.predict(X_scaled[val_idx]))

ari_scores = [adjusted_rand_score(fold_clusters[i], fold_clusters[i+1]) for i in range(4)]
print(f"Cross-validation ARI: {np.mean(ari_scores):.3f}")

# Extra: 3D visualization with Age
X_3d = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].values
scaler_3d = StandardScaler()
X_3d_scaled = scaler_3d.fit_transform(X_3d)
gmm_3d = GaussianMixture(n_components=5, random_state=42)
gmm_3d.fit(X_3d_scaled)
clusters_3d = gmm_3d.predict(X_3d_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(5):
    mask = clusters_3d == i
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
               c=colors[i], alpha=0.6, s=40)
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score')
ax.set_zlabel('Age (years)')
ax.set_title('3D GMM with Age')
plt.show()