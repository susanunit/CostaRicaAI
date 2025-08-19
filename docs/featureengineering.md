# Feature Engineering

## 1. Hour of Day (Cyclic Encoding)

**Problem**

* Time of day (0–23) is **cyclical**.
* 23:00 and 00:00 are *close* in time, but numerically they are far apart.
* Linear models think 0 and 23 are extremes unless we fix it.

**Solution: Sin/Cos Transform (Cyclic Encoding)**

* Imagine mapping the hour onto a **circle (like a clock face)**.
* Encode as:

	* `sin(2π * hour / 24)`
	* `cos(2π * hour / 24)`

* This transforms **Hour of Day** into coordinates on a circle — making midnight close to 11 PM and 1 AM.

**Example Code**

```
import numpy as np
data['hour'] = data['published_date'].dt.hour
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
```

---

## 2. Day of Week Encoding

You can represent **Day of the Week** in two ways:

* **One-Hot Encoding** — Create a binary column for each day (Mon, Tue, ..., Sun)
* **Cyclic Encoding (sin/cos)** — Similar to hour, because Sunday and Monday are adjacent.

**Example Code (One-Hot Encoding):**

```
data['day_of_week'] = data['published_date'].dt.dayofweek  # 0=Mon, 6=Sun
data = pd.get_dummies(data, columns=['day_of_week'])
```

**Cyclic Encoding Alternative:**

```
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)  
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
```

---

## 3. Location Grid (Spatial Binning)

**Problem:**  Lat/Lon coordinates are precise, but regression models need broader **zones** (Downtown, North, etc.).

**Solution:**

* Divide the city map into **grid cells (zones)**.
* Example: Every 0.01 degrees of lat/lon is a new grid.
* Count incidents per grid cell.

**Example Code:**

```
data['lat_bin'] = (data['latitude'] * 100).astype(int)
data['lon_bin'] = (data['longitude'] * 100).astype(int)
data['grid_id'] = data['lat_bin'].astype(str) + "_" + data['lon_bin'].astype(str)
```

This gives you zones like:

* Grid 3025\_9771 → Downtown
* Grid 3028\_9773 → North Austin

---

## 4. Incident Type Encoding

* **One-Hot Encoding:** Create binary columns for each type of incident (Crash, Hazard, etc.).
* **Count Encoding (for Time Buckets):** Count how many of each type occur in a given hour or zone.

**One-Hot Example:**

`data = pd.get_dummies(data, columns=['issue_reported'])`

---

## 5. Lagged Incident Counts

**Concept:**

* The number of incidents in the previous hour might influence the current hour.
* For example: Ongoing hazards might spill over into future hours.

**How to Do It:**

* Aggregate incidents per hour.
* Shift the incident count by 1 hour to create a “lagged” feature.

**Example Code:**

```
data['hour_bucket'] = data['published_date'].dt.floor('H')
hourly_counts = data.groupby('hour_bucket').size().reset_index(name='incident_count')
hourly_counts['incident_count_lag_1'] = hourly_counts['incident_count'].shift(1)
```

---

## 6. Rolling Averages (Smoothed Trends)

**Concept:**

* Smooth out noise by taking a rolling average of incidents.
* Example: Average number of incidents in the last 3 hours.

**Example Code**

```
hourly_counts['incident_count_roll3'] = hourly_counts['incident_count'].rolling(window=3).mean()
```

