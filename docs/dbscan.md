# DBSCAN with Austin Traffic Dataset

## 1. Import & Prepare Data

We'll reuse the dataset from the K-Means section.

```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load dataset (replace with actual filename)
df = pd.read_csv("Austin_traffic_incidents.csv")

# Select location + time features
data = df[['latitude', 'longitude', 'published_date']].dropna()

# Convert date → hour of day
data['published_date'] = pd.to_datetime(data['published_date'])
data['hour'] = data['published_date'].dt.hour

# Features for clustering
X = data[['latitude', 'longitude', 'hour']]

# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

## 2. Run DBSCAN

```
# eps = neighborhood size, min_samples = points to form a dense cluster
dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters = dbscan.fit_predict(X_scaled)

# Add results to dataframe
data['cluster'] = clusters
```

**cluster = -1** means the point was labeled as **noise/outlier**.

---

## 3. Visualize Clusters

```
plt.figure(figsize=(10,6))
plt.scatter(data['longitude'], data['latitude'], c=data['cluster'], cmap='tab10', s=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("DBSCAN Clustering of Traffic Incidents in Austin")
plt.show()
```

This map will show **dense hotspots** as clusters, while scattered incidents are marked as noise (`-1`).

---

## 4. Compare K-Means vs DBSCAN

* **K-Means**: Always forces `k` clusters, even if data doesn't naturally group.
* **DBSCAN**: Finds clusters of arbitrary shapes, filters out noise.
* **Traffic Use Case:**

  * K-Means: Good for broad “region-based” grouping.
  * DBSCAN: Good for detecting **true hotspots** of high density (e.g., one dangerous intersection).


## 5. Mini-Challenges for Students

1. **Parameter Tuning:** Change `eps` (radius of neighborhood) and `min_samples`. How do clusters change?

   * Hint: Smaller `eps` → more noise, more small clusters.
   * Larger `eps` → fewer, bigger clusters.

2. **Compare with K-Means:** Cluster only on location (`lat`, `lon`). Which method gives more meaningful groups?

3. **Noise Analysis:** Look at incidents labeled as `-1`. Are these **rare events**? Where do they happen?

4. **Feature Engineering:** Add `is_weekend` or `rush_hour` as features and rerun DBSCAN. Does it detect weekend vs weekday hotspots?





