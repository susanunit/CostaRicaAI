# Feature Engineering

## Definition 

Feature engineering is the process of **transforming raw data into meaningful inputs (features)** that machine learning models can understand.  

**Why it matters**

* The quality of your features often matters **more than the choice of algorithm**. 

* A well-chosen feature can reveal hidden patterns, making the model smarter and more accurate. 

**Analogy** 

* Imagine you’re predicting whether a basketball player will score. 

  * Raw data might be: *player’s height, team name, jersey number, last game date*. 

  * After feature engineering, useful features might be: *average points per game, shooting percentage in the last 5 games, fatigue level, opponent’s defense rating*. 

* Same raw data, but better structured information. 


### Key Types of Feature Engineering

1. **Cleaning** – Fixing missing values, removing duplicates, converting data types. 

2. **Transformation** – Scaling numbers, normalizing values, encoding categorical data. 

3. **Creation** – Making new features from old ones. 

   * **Example**: from “Published Date” you can create “Hour of Day,” “Day of Week,” “Is Weekend.” 

4. **Selection** – Choosing the most important features to avoid noise.

"Feature engineering is how we turn messy real-world data into the smart signals that help AI make better predictions. It’s less about magic algorithms and more about asking the right questions of your data."

## Feature Engineering Tasks

### 1. Datetime Features (from Published Date)

| Derived Feature | Purpose |
| ----- | ----- |
| Incident Hour (0-23) | Identify time-of-day patterns |
| Day of Week (0-6) | Capture weekday/weekend behavior |
| Weekend Flag | Binary flag (1 \= Weekend, 0 \= Weekday) |

### 2. Categorical Encoding

| Feature | Encoding Method | Notes |
| ----- | ----- | ----- |
| Issue Reported | One-Hot Encoding or Label Encoding | High cardinality may require frequency encoding |
| Agency | One-Hot Encoding | Depends on how many unique agencies there are |

### 3. Spatial Features (Latitude/Longitude)

| Transformation | Purpose |
| ----- | ----- |
| Distance from Downtown (30.2672, \-97.7431) | Proximity to city center |
| Latitude & Longitude Scaling | Normalize for distance-based models |
| Location Clusters (Optional) | KMeans or DBSCAN clustering on coordinates |

### 4. Address Text Feature Engineering (Optional but Valuable)

| Transformation | Purpose |
| ----- | ----- |
| Extract Street Names | e.g., “E 6th St” |
| Road Type Flag | e.g., Highway, Service Road, Blvd, etc. |
| Text Length of Address | Indirect signal for address granularity |

### 5. Feature Scaling

| Feature | Scaling Method |
| ----- | ----- |
| Latitude, Longitude, Distance | MinMaxScaler (scale to 0-1) |
| Time-based Features (if numerical) | StandardScaler (mean 0, std 1\) |

---

## Target & ML Goals

| Task | Target Feature | ML Type |
| ----- | ----- | ----- |
| **Classification of Incident Type** | Issue Reported | Multiclass Classification |
| **Cluster Incident Hotspots** | Latitude/Longitude \+ Time | Clustering (KMeans/DBSCAN) |
| **Bias Detection by Agency** | Agency vs. Incident Types | Clustering/Exploratory Analysis |

---

## Summary of Needed Feature Engineering for Day 2:

| Task | Required? | Complexity Level |
| ----- | ----- | ----- |
| Datetime Features from Published Date | ✅ | Easy |
| Encode Issue Reported | ✅ | Moderate |
| Encode Agency | ✅ | Easy |
| Lat/Lon Scaling | ✅ | Easy |
| Distance from Downtown | ✅ | Moderate |
| Location Clustering | Optional | Moderate |
| Extract Road Type from Address | Optional | Moderate |
| Text Features from Address (length) | Optional | Easy |

## Why Engineer Datetime Features from `Published Date`?

### Raw Column:

| Column Name | Example Value | Problem with Raw Form |
| ----- | ----- | ----- |
| Published Date | `2025-08-03 14:30:00` | A raw timestamp is meaningless to ML models |

---

## Transformations & Their Purpose:

### 1. Incident Hour (0-23)

| Purpose | Why it Matters |
| ----- | ----- |
| Capture time-of-day patterns | Traffic incidents often peak during rush hours (7-9am, 4-6pm) |
| Useful for both classification and clustering | Helps models detect temporal patterns linked to specific incident types (e.g., collisions in the morning, hazards at night) |

### 2. Day of Week (0-6)

| Purpose | Why it Matters |
| ----- | ----- |
| Identify weekday vs weekend patterns | Traffic behavior changes on weekends; incidents like stalled vehicles or hazards may be more common |
| Essential for clustering patterns | Groups incidents based on weekly cycles (e.g., Friday rush hour hotspots) |

### 3. Weekend Flag (Binary 0/1)

| Purpose | Why it Matters |
| ----- | ----- |
| Simplifies weekday/weekend distinction | For simple models, binary features are often more impactful than categorical day-of-week |
| Useful in classification | Helps classify incident types likely to occur on weekends (e.g., events, road closures) |

### 4. Time-of-Day as Cyclical Feature (Optional, Advanced)

| Purpose | Why it Matters |
| ----- | ----- |
| Encode hour using sine/cosine | Prevents misleading distances between 23:00 and 00:00 in clustering models |
| Makes models aware of circular time | Important for KMeans/DBSCAN where distance metrics would otherwise treat 23 and 0 as far apart |

---

## High-Level Why:

* **Traffic incidents are inherently temporal.**  
   Patterns in collisions, hazards, and stalled vehicles **follow time-of-day and day-of-week rhythms**.

* **Machine Learning models don't understand timestamps.**  
   They need **explicit numerical or categorical features** representing patterns (e.g., rush hours, weekends).

* **For Clustering**, time-of-day and day-of-week help reveal "incident patterns" that are **spatial-temporal**:

  * Where and when do collisions spike?

  * Are stalled vehicles more common on weekends?

* **For Classification**, datetime-derived features add valuable predictive signals:

  * If it’s Friday 5 PM, there’s a higher chance it’s a collision.

  * If it’s Sunday afternoon, it might be a hazard or road closure.

---

** Without Datetime Features:**

| Model Task | Without Datetime Features |
| ----- | ----- |
| Classification | Model treats all incidents as temporally equal, losing out on key predictive patterns |
| Clustering | Incidents occurring at different times but same locations may get grouped incorrectly |

## Why Raw Timestamps are “Meaningless” to ML Models:

### 1. Timestamps Are Not Linear or Numeric in a Useful Way

| Problem | Example |
| ----- | ----- |
| Timestamps are large numbers | `2025-08-03 14:30:00` → 1,755,690,600 (UNIX time) |
| ML models (especially tree-based, linear, distance-based) can't extract useful patterns from such large continuous values |  |
| The difference between two timestamps isn’t always meaningful | The numeric difference between `2025-08-03 14:00:00` and `14:30:00` is 1800 seconds, but the **semantic difference is "same hour"** |

---

### 2. Timestamps Encode Multiple Dimensions (Time & Date)

| Aspect | Why It's Problematic |
| ----- | ----- |
| Day of Week | Not directly encoded—models can’t infer it |
| Hour of Day | Hidden inside a long number |
| Weekend vs Weekday | Hidden pattern—models won’t know weekends differ |
| Recurring Cycles | Timestamps don’t indicate cyclical nature of time |

Models need **explicit signals** like:

* "This happened on a Friday"

* "This occurred at 7 AM"

* "This is during the weekend"

---

### 3. Distance-Based Models Get Confused

| Model Type | Why Raw Timestamps Fail |
| ----- | ----- |
| **KNN, KMeans, DBSCAN** | These rely on distances between feature values. A timestamp like `14:30` is just a big number that misrepresents proximity. |
| Example: | 23:00 (11 PM) and 01:00 (1 AM) are 2 hours apart, but numerically seem “far apart” if we use raw numbers. |
| Solution | Transform time into **cyclical features** (sin/cos encoding) or separate Hour & Day features. |

---

### 4. Tree-Based Models Waste Splits

| Model Type | Problem |
| ----- | ----- |
| Decision Trees, Random Forests | Trees will waste splits trying to make sense of a massive continuous timestamp field |
| Example: | It might try to split on "timestamps greater than 1700000000" — which is arbitrary and meaningless for incident patterns |

---

## Why Engineers Derive Features (Hour, Day, Weekend, Cyclical)

| Derived Feature | Makes This Explicit to Model |
| ----- | ----- |
| **Hour of Day (0-23)** | Helps model see morning/evening patterns |
| **Day of Week (0-6)** | Models weekly traffic trends |
| **Weekend Flag (0/1)** | Helps model generalize weekend-specific behaviors |
| **Cyclical Encoding (sin/cos of Hour)** | Helps distance-based models understand time loops from 23:00 to 00:00 |

## Why We Need to Encode `Issue Reported`

### 1. Raw Text is Not Machine-Understandable

| Problem | Example |
| ----- | ----- |
| Raw text strings (e.g., “Crash”, “Hazard”, “Stalled Vehicle”) are not numeric | ML models require numerical representations of features |
| Models can’t calculate distance, similarity, or make splits on strings | A model can’t “compare” the text “Hazard” with “Collision” directly |

---

### 2. Type of Encoding Depends on Use-Case

| Goal | Suggested Encoding | Why? |
| ----- | ----- | ----- |
| **Classification (as Target)** | Leave as raw labels (string) | Scikit-learn classifiers handle string labels as target values |
| **Classification (as Feature)** | One-Hot Encoding (small cardinality) | Converts each category into a binary column (e.g., “is\_hazard”) |
|  | Frequency Encoding (large cardinality) | Replaces category with its frequency (good for high-cardinality issues) |
| **Clustering (as Feature)** | One-Hot Encoding (preferred) | Distance-based clustering needs numerical vectors |

---

### 3. Why One-Hot Encoding is Usually the First Step

| Pros | Cons |
| ----- | ----- |
| Simple, explicit binary representation | Increases dimensionality (one column per unique value) |
| Works well for distance-based models (KMeans, DBSCAN) | Sparse matrix for high cardinality |
| Ensures no ordinal relationship is assumed |  |

Example:

| Issue Reported | One-Hot Columns |
| ----- | ----- |
| Crash | \[1, 0, 0\] |
| Hazard | \[0, 1, 0\] |
| Stalled Vehicle | \[0, 0, 1\] |

---

### 4. Alternative: Frequency Encoding (When Issue List is Long)

| Why Consider This? | When to Use |
| ----- | ----- |
| Reduces dimensionality (single column) | When “Issue Reported” has high cardinality |
| Embeds frequency information | E.g., “Hazard” might occur in 40% of data, “Collision” in 30%, etc. |

Example:

| Issue Reported | Frequency Encoded Value |
| ----- | ----- |
| Hazard | 0.40 |
| Collision | 0.30 |

---

### 5. Why Encoding “Issue Reported” is Crucial for Clustering

| Problem | Impact if Unencoded |
| ----- | ----- |
| Distance-based algorithms (KMeans, DBSCAN) need numerical features | Without encoding, models can’t differentiate categories |
| Raw strings make clusters meaningless | Incidents with the same “Issue Reported” value won’t be treated as “close” unless numerically encoded |

---

## High-Level Why:

* ML models **don’t understand text labels as categorical concepts** unless we explicitly transform them.

* **Encoding ‘Issue Reported’** injects semantic meaning into a format ML models can process—allowing them to group similar incidents or predict categories effectively.

* Choosing **One-Hot vs Frequency Encoding** depends on **cardinality** and **model sensitivity to dimensionality**.

## Why We Need to Engineer Spatial Features (Latitude, Longitude)

### 1. Raw Lat/Lon Coordinates Are Just Numbers

| Problem | Example |
| ----- | ----- |
| Raw Lat/Lon (e.g., 30.2672, \-97.7431) are treated as independent numerical values | ML models don’t inherently understand geographical proximity |
| Distance between two Lat/Lon points is **not linear** in (Lat, Lon) space | Small differences in coordinates could represent meters or miles depending on zoom level |

---

### 2. For Clustering: Lat/Lon Must Reflect Real-World Proximity

| Issue | Why It’s a Problem |
| ----- | ----- |
| KMeans & DBSCAN rely on **distance metrics (Euclidean, Manhattan, etc.)** | Raw Lat/Lon coordinates do not accurately reflect real-world distances |
| Latitude/Longitude are on a spherical surface (Earth) | Euclidean distances in (Lat, Lon) space are distorted |
| Downtown incidents (dense area) will get mixed with outliers if raw coordinates are used | Models fail to group spatial clusters accurately |

---

### 3. Distance Features Provide Better Spatial Context

| Feature | Why It’s Useful |
| ----- | ----- |
| **Distance from Downtown (Austin City Center)** | Allows model to understand how far an incident is from a central reference point (e.g., 6th & Congress) |
| **Distance to nearest known hotspot (Optional)** | Enhances clustering by anchoring around known traffic hubs |

---

### 4. Spatial Clustering Requires Scaling or Transformation

| Method | Why? |
| ----- | ----- |
| **Min-Max Scaling Lat/Lon** | Normalizes spatial ranges for clustering algorithms that are sensitive to feature scales |
| **Haversine Distance (Optional)** | Calculates great-circle distance between two Lat/Lon points—useful for geospatial clustering |

---

### 5. Alternative Approach: Pre-cluster Lat/Lon → Location Group Feature

| What This Does | Why It Helps |
| ----- | ----- |
| Use KMeans/DBSCAN to pre-cluster Lat/Lon into spatial groups (e.g., Downtown, East Austin, Suburbs) | Converts continuous Lat/Lon into a **categorical “Location Cluster” feature** |
| Reduces dimensionality (turns two continuous variables into one categorical group) | Models can learn from location context without dealing with coordinate math |

---

## High-Level Why:

* **Raw Latitude/Longitude values lack context.**  
   The model doesn’t understand that (30.27, \-97.74) is “Downtown” and (30.30, \-97.70) is “East Austin.”

* **Distance metrics on Lat/Lon are misleading unless scaled or converted to real-world distances.**

* For **Clustering**, spatial features often need:

  * Scaling (Min-Max, Standard)

  * Distance from central reference points (Downtown)

  * Optional: Pre-clustered into categorical “zones”

