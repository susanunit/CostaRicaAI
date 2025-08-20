## **Steps to Build the Classifier**

### 1. Define the target

`y = df["issue_reported"]`

This is categorical (many classes).


### 2. Choose features

To predict `issue_reported`, you need features that might correlate with the type of incident:

* **Time features**: hour of day, day of week (e.g., crashes more during rush hour).
* **Location features**: latitude/longitude (e.g., certain areas have recurring hazards).
* **Priority**: severe incidents may be specific categories.

Let’s start simple with **time \+ priority**:

```
df["hour"] = df["published_date"].dt.hour
df["dayofweek"] = df["published_date"].dt.dayofweek
```

Features:

`X = df[["hour", "dayofweek", "priority"]]`

---

### 3. Encode categorical features

If `priority` is categorical, one-hot encode it:

`X = pd.get_dummies(X, columns=["priority"])`

Encode the target (`issue_reported`):

```
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

---

### 4. Train/test split

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
```

---

### 5. Multi-class Logistic Regression

```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=encoder.classes_))
```

---

### 6. Interpretation

* Now you’ll get precision/recall/F1 **for each incident type** (e.g., “Crash Service,” “Traffic Hazard,” etc.).
* Some categories will be easier to predict (clear time/priority signals).
* Others will be harder (overlap in features).

---

Teaching Angle

* Binary vs Multi-class classification → same model, bigger problem.
* Feature importance → time and priority may predict certain issues better than others.
* Misclassifications → e.g., “Crash Service” vs “Traffic Hazard” might overlap.
* Data imbalance → some categories dominate (students see how that hurts accuracy).

