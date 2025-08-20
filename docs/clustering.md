# Clustering



## Introduction to Clustering

**What is Clustering?**

**Clustering** is an **unsupervised learning technique** used to group data points that are **similar** to each other.

Unlike **classification or regression** (supervised learning), clustering doesn't require **labels**. Instead, the algorithm looks at the data’s **structure** and tries to discover hidden groupings.

In practice, clustering answers the question:   *"Which data points look alike, and how can we group them?"*


## Supervised vs. Unsupervised Learning

* **Supervised Learning (Classification, Regression):**

  * You give the algorithm **input features** *and* **labels/outputs**.
  * The goal is to **predict labels** for new data.
  * Example (Traffic Dataset): Predict whether an incident will cause a **lane closure** (yes/no).

* **Unsupervised Learning (Clustering, Dimensionality Reduction):**

  * You only give the algorithm **input features** (no labels).
  * The goal is to **discover hidden structure** in the data.
  * Example (Traffic Dataset): Find **hotspots** of incidents in Austin without knowing them in advance.


## Why Clustering is Useful

* Helps make sense of **unlabeled data**.
* Reveals **patterns** that aren't obvious.
* Can serve as a **preprocessing step** for other ML tasks (e.g., group incidents before running classification).

* Real-world applications:

  * **Traffic Analysis:** Identify accident-prone intersections or rush-hour clusters.
  * **Retail:** Group customers by shopping behavior (customer segmentation).
  * **Healthcare:** Discover subgroups of patients with similar symptoms.
  * **Astronomy:** Group galaxies or stars based on physical properties.


## How Clustering Works (Conceptually)

Every clustering algorithm has two main ideas:

  1. **Measure similarity** → Often done with distance (Euclidean, Manhattan, cosine similarity).
  2. **Group points together** → Close points → same cluster; distant points → different clusters.

Example:  If we plot traffic incidents by **latitude & longitude**, nearby points form **geographic clusters** (like downtown Austin vs. I-35 corridor).

---

## Common Clustering Algorithms

1. **K-Means**

   * Partitions data into `k` groups.
   * Each cluster has a "center" (centroid).
   * Simple, fast, but assumes round clusters.

2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

   * Groups dense areas of data.
   * Automatically detects outliers (noise).
   * Works well when clusters are irregular shapes.

3. **Hierarchical Clustering**

   * Builds a **tree of clusters** (dendrogram).
   * Good for understanding relationships, but slower for large datasets.


## Today's Focus

* **K-Means:** A simple and widely used method to introduce clustering concepts.
* **DBSCAN:** A density-based method that's great for finding hotspots and handling outliers.


## Traffic Example:

* We don't know the "true labels" for traffic hotspots.
* But we can group incidents by **location + time of day** to discover clusters like *downtown congestion*, *highway accidents*, etc.


```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset (replace path with actual file)
df = pd.read_csv("Austin_traffic_incidents.csv")

# Preview
print(df.head())

# Select useful features for clustering
data = df[['latitude', 'longitude', 'published_date']].dropna()

# Convert published_date to hour of day (feature engineering)
data['published_date'] = pd.to_datetime(data['published_date'])
data['hour'] = data['published_date'].dt.hour

# Keep only features for clustering
X = data[['latitude', 'longitude', 'hour']]
```

### Sidequest:  

**Why we only used `['latitude', 'longitude', 'hour']` in K-Means**

Clustering algorithms (like K-Means) look for **similarities across the features you feed in**. So the choice of features is crucial.

In the example:

`X = data[['latitude', 'longitude', 'hour']]`

we used only **latitude, longitude, and time of day** because:

* **Latitude & Longitude** give us **spatial patterns** (geographic hotspots).
* **Hour** adds a **temporal dimension** (rush-hour vs late-night clusters).

We dropped the other columns (like incident ID, description, weather, etc.) because:

* Many of them are **categorical or text-based** (e.g., `issue_reported` = "Crash", "Hazard", "Stalled Vehicle"). K-Means doesn't handle categorical/text data directly.

* Including irrelevant or noisy features can **distort the clustering**. For example, if we added a random ID number, the algorithm might cluster based on that meaningless difference.

* For a first pass, it's easier to keep clustering focused on a **clear interpretation** (location + time).


## **When Would We Add More Features?**

We absolutely *can* include more features, but we usually need some **preprocessing/feature engineering**:

* **Categorical Features** (e.g., `issue_reported`):

  * Use **one-hot encoding** (`pd.get_dummies`) so categories become numeric.
  * Example: Crash → [1,0,0], Hazard → [0,1,0], etc.

* **Text Columns** (e.g., descriptions):

  * Use **TF-IDF vectorization** or embeddings.

* **Numeric Features** (e.g., duration, severity):

  * Can be included directly, but remember to **scale** them (StandardScaler).


## Example: Adding "Issue Reported"

Suppose we want to cluster not just by **where/when**, but also by **type of incident**.

```
# One-hot encode categorical feature
incident_type = pd.get_dummies(df['issue_reported'], prefix='issue')

# Combine with lat/lon/hour
X = pd.concat([data[['latitude', 'longitude', 'hour']], incident_type], axis=1)

# Scale everything
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Now the clustering will group incidents not only by *where/when* they happened, but also by *what kind of incident* they were.


Key Point: Clustering is only as good as the **features you choose**.

* Too few features → oversimplified clusters.
* Too many irrelevant features → noisy, meaningless clusters.
* Smart **feature engineering** → insightful, actionable clusters.


### 3. Feature Scaling

Clustering is sensitive to **feature scales** (lat/lon vs hours), so we standardize.

`scaler = StandardScaler()`

`X_scaled = scaler.fit_transform(X)`


### 4. Run K-Means Clustering

```
# Choose number of clusters (k)
k = 5`
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to data
data['cluster'] = clusters
```

---

### 5. Visualize Clusters

We'll plot clusters on a **map-like scatter** (latitutde vs. longitude), colored by cluster.

```
plt.figure(figsize=(10,6))
plt.scatter(data['longitude'], data['latitude'], c=data['cluster'], cmap='tab10', s=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Traffic Incident Clusters in Austin")
plt.show()
```

---

### 6. Elbow Method (Optional, but Great Teaching Point)

To pick the right `k`, we test different values.

```
inertia = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-cluster Sum of Squares)")
plt.title("Elbow Method for Optimal k")
plt.show()
```

---

### 7. Mini-Challenges for Students

1. **Hotspot Clusters**: Cluster only by `latitude` and `longitude`. Where are the biggest clusters of accidents?
2. **Time-based Clusters**: Cluster only by `hour`. What patterns do you see (rush-hour vs late night)?
3. **Feature Engineering Experiment**: Add a new feature: `is_weekend`. Does clustering look different?
4. **Compare k=3 vs k=8**: How does changing cluster size affect results?

---

### 8. Wrap-Up

* **Clustering is exploratory**: There's no “accuracy” metric like classification, instead we look for *insightful groupings*.
* **Austin Traffic Example**: Helps identify geographic \+ temporal hotspots, useful for **urban planning** or **resource allocation**.


## DBSCAN with Austin Traffic Dataset

### 1. Import & Prepare Data

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

### 2. Run DBSCAN

```
# eps = neighborhood size, min_samples = points to form a dense cluster
dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters = dbscan.fit_predict(X_scaled)

# Add results to dataframe
data['cluster'] = clusters
```

**cluster = -1** means the point was labeled as **noise/outlier**.

---

### 3. Visualize Clusters

```
plt.figure(figsize=(10,6))
plt.scatter(data['longitude'], data['latitude'], c=data['cluster'], cmap='tab10', s=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("DBSCAN Clustering of Traffic Incidents in Austin")
plt.show()
```

This map will show **dense hotspots** as clusters, while scattered incidents are marked as noise (`-1`).


### 4. Compare K-Means vs DBSCAN

* **K-Means**: Always forces `k` clusters, even if data doesn't naturally group.
* **DBSCAN**: Finds clusters of arbitrary shapes, filters out noise.
* **Traffic Use Case:**

  * K-Means: Good for broad “region-based” grouping.
  * DBSCAN: Good for detecting **true hotspots** of high density (e.g., one dangerous intersection).


### 5. Mini-Challenges for Students

1. **Parameter Tuning:** Change `eps` (radius of neighborhood) and `min_samples`. How do clusters change?

   * Hint: Smaller `eps` → more noise, more small clusters.
   * Larger `eps` → fewer, bigger clusters.

2. **Compare with K-Means:** Cluster only on location (`lat`, `lon`). Which method gives more meaningful groups?
3. **Noise Analysis:** Look at incidents labeled as `-1`. Are these **rare events**? Where do they happen?
4. **Feature Engineering:** Add `is_weekend` or `rush_hour` as features and rerun DBSCAN. Does it detect weekend vs weekday hotspots?





