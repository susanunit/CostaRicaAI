Day 2: Classification & Clustering
==================================

## 🎯Learning Objectives

By the end of Day 2, students will:

* Understand the difference between supervised and unsupervised learning
* Implement and explain basic classification (e.g., decision trees) and clustering (e.g., K-means) algorithms
* Use real-world traffic incident data to solve classification and clustering problems
* Interpret model outputs and performance metrics


## 1. Classification

!!! info
	🔍 What is Classification?

	Classification is the task of predicting a label or category for input data. In our case, we might classify an incident based on its description as a "Crash", "Traffic Hazard", or "Road Closure".

### 💡 Real-Life Analogies

* **Spam filter**: Given the contents of an email, predict whether it's "Spam" or "Not Spam".
* **Doctor's diagnosis**: Based on symptoms, classify a disease.
* **Campus security**: Given an incident report, predict if it's high priority.



### 🧰 Algorithm Focus: Decision Trees

#### 🛠️ How It Works  

A decision tree splits the data based on the features to reduce uncertainty.
Each split is a "yes/no" question (e.g., Does the incident description contain 'crash'?)
It continues splitting until it can confidently assign a label.


!!! info
	🧠 **Intuitive Explanation**: Think of playing 20 Questions, where you narrow down the object by asking binary questions. A decision tree is doing exactly that—asking a series of smart questions to guess the right label.


#### 📐 Math Behind It

!!! warning
	SDL These math formulas are unintelligible and appear to be pasted in from LaTex.  

Entropy (information gain): measures uncertainty. Lower entropy = more confidence.

Entropy(S)=

	−∑i=1npilog⁡2(pi)\text{Entropy}(S) = -\sum_{i=1}^{n} p_i \log_2(p_i)Entropy(S)=−i=1∑n​pi​log2​(pi​)

where pip_ipi​ is the proportion of class iii in the current subset.


#### Information Gain:

!!! warning
	SDL Same - this is unintelligible

	IG=Entropy(parent)−∑(samples in splittotal samples⋅Entropy(child))IG = \text{Entropy(parent)} - \sum \left( \frac{\text{samples in split}}{\text{total samples}} \cdot \text{Entropy(child)} \right)IG=Entropy(parent)−∑(total samplessamples in split​⋅Entropy(child))

The algorithm chooses the feature/split that maximizes information gain.


```
Demo Code
(From earlier, modify slightly to connect to concepts)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Features: Bag of words from 'Description'
X = df['Description']
y = df['IssueReport']
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Evaluation
print(classification_report(y_test, model.predict(X_test)))

# Optional: Visualize the decision tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=model.classes_)
plt.show()
```


## 2: Clustering

!!! info
	🔍 What is Clustering?
	Clustering is the task of grouping data points that are similar without knowing the correct labels in advance.

### 💡 Real-Life Analogies

* **Food delivery zones**: Grouping orders by geography to optimize delivery.
* **Netflix genres**: Automatically grouping movies that "feel" similar based on user behavior.
* **Traffic incidents**: Finding geographic or temporal "hotspots" of activity.



### 🧰 Algorithm Focus: K-Means

#### 🛠️ How It Works

1. Choose K cluster centers randomly.
1. Assign each data point to the nearest cluster center.
1. Move each cluster center to the average of its assigned points.
1. Repeat until assignments don't change (converge).


#### 🧠 Intuitive Explanation
Imagine you're running a food truck business. You want to set up 3 food trucks in Austin to reach as many people as possible. K-Means helps you place them where people naturally cluster—by learning where the traffic is!

###	Math Behind It

**Distance metric**: Typically uses Euclidean distance.

 	d(x,μ)=(x1−μ1)2+(x2−μ2)2+⋯+(xn−μn)2d(x, \mu) = \sqrt{(x_1 - \mu_1)^2 + (x_2 - \mu_2)^2 + \dots + (x_n - \mu_n)^2}d(x,μ)=(x1​−μ1​)2+(x2​−μ2​)2+⋯+(xn​−μn​)2​


**Objective**: Minimize the total within-cluster variance:

	∑i=1K∑x∈Ci∥x−μi∥2\sum_{i=1}^{K} \sum_{x \in C_i} \| x - \mu_i \|^2i=1∑K​x∈Ci​∑​∥x−μi​∥2
 where μi\mu_iμi​ is the centroid of cluster CiC_iCi​.



### 💻 Demo Code

```
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

coords = df[['Latitude', 'Longitude']].dropna()

# Fit KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
coords['Cluster'] = kmeans.fit_predict(coords)

# Visualize
plt.scatter(coords['Longitude'], coords['Latitude'], c=coords['Cluster'], cmap='tab10', alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Traffic Incident Clusters in Austin')
plt.show()
```


## 🔁 Wrap-Up Exercise Ideas

* **Classification Challenge**: Classify whether a traffic report is a "Crash" or not based on its description.

* **Clustering Challenge**: Try different values of k and see how cluster shapes change; add day of week to see if time-based clusters emerge.

* **Discussion Prompt**: When would classification be inappropriate? When would clustering fail?


