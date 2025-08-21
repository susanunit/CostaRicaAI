# Table: Recommended Dataset Sizes by Model Type

| Model Type | Typical Minimum Dataset Size | Notes / Considerations 
| ----- | ----- | ----- |
| Linear Regression / Logistic Regression | ~100–500 rows per feature | Works on small datasets; too many features can cause overfitting 
| Decision Trees / Random Forests | ~1,000–5,000 rows | Random Forests need more data to avoid overfitting; tune tree depth 
| Support Vector Machines (SVMs) | ~1,000+ rows | Small datasets are okay if feature space is low-dimensional 
| k-Nearest Neighbors (k-NN) | ~500–1,000 rows | Very sensitive to noise and feature scaling; larger datasets improve stability 
| Gradient Boosting (XGBoost, LightGBM, CatBoost) | ~1,000–10,000+ rows | Can overfit small datasets; early stopping helps 
| Neural Networks / Deep Learning | ~10,000+ rows | Require large datasets; pretraining/transfer learning recommended for smaller datasets 
| Clustering (k-Means, DBSCAN, etc.) | ~500–1,000 rows | More data improves cluster stability; normalization important 
| Text / NLP models (Bag-of-Words / TF-IDF) | ~1,000+ documents | Sparse features; large vocabularies require more data 
| Image / Vision models | ~5,000+ images | Data augmentation helps; deep CNNs require tens of thousands 

