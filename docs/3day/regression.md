## **Day 3: Regression & Forecasting Deep Dive — Detailed Breakdown**

---

### 1. Recap of Regression Concepts 

#### Key Points

* **Linear vs. Nonlinear Relationships:**

	* Show scatterplots: e.g., **Delay vs. Time of Day** (linear trend) and **Delay vs. Traffic Volume** (possible nonlinearity).
	* **Visual Demo**: Fit a Linear Regression line to nonlinear data—highlight residual patterns.

* **When Linear Models Fail:**

	* Visual: Underfitting example with linear model on a curved trend.
	* Discussion: Why “forcing a straight line” doesn’t always work.

* **Feature Engineering’s Role:**

	* Add **Time of Day squared** feature → improves fit for certain periods (rush hour peaks).
	* Discuss **lag features** as a way to capture hidden temporal dependencies.

---

### 2. Linear, Polynomial, Ridge Regression Hands-on

#### Coding Steps

1. **Load Dataset:**

	```
	import pandas as pd  
	data = pd.read_csv('austin_traffic_data.csv')
	```

1.  **Linear Regression (Baseline Model):**

	```
	from sklearn.linear_model import LinearRegression
	X = data[['traffic_volume']]  # Simple feature
	y = data['delay_minutes']
	model = LinearRegression().fit(X, y)
	```

2. Plot predictions vs. actuals.  Discuss high residuals for peak traffic hours.

	**Polynomial Features:**

	```
	from sklearn.preprocessing import PolynomialFeatures
	poly = PolynomialFeatures(degree=2)
	X_poly = poly.fit_transform(X)
	model_poly = LinearRegression().fit(X_poly, y)
	```

3. Show improvement (if any) visually.

**Ridge Regression for Multicollinearity:**

```
from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=1.0).fit(X_poly, y)
```

4. Discuss how Ridge controls coefficient magnitudes.

---

### 3. Decision Trees & Random Forest Regression 

#### Key Points & Code

* **Visual Explanation of Splits:**

	* Interactive demo showing splits on traffic\_volume.

**DecisionTreeRegressor:**

```
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(max_depth=4).fit(X, y)
```

*  Plot predictions — show piecewise constant fits.

**Random Forest Regressor & Feature Importance:**

```
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100).fit(X, y)
importances = rf_model.feature_importances_
print(importances)
```


* Discuss how ensemble of trees captures more complex patterns.
* Visualize feature importances when using multiple features later.

---

### 4. Regression Diagnostics & Residual Analysis 

#### Key Points

**Plot Residuals:**

```
import matplotlib.pyplot as plt
residuals = y - model.predict(X)
plt.scatter(y, residuals)
plt.axhline(0, color='red')
```

* Highlight patterns like funnel shape (heteroscedasticity) or curvature (nonlinearity).

**Compute Metrics:**

```
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print("R²:", r2_score(y, model.predict(X)))
print("MSE:", mean_squared_error(y, model.predict(X)))
print("MAE:", mean_absolute_error(y, model.predict(X)))
```


* **Diagnostic Challenge:**

  * Compare Linear vs Polynomial vs Random Forest residuals.

  * Discussion Prompt: “Which model is underfitting? Overfitting? Why?”

---

### 5. Forecasting: Time Series Regression 

#### **Key Points & Code:**

* **Difference between Regression & Forecasting:**

  * Regression: Predicts “what is.”

  * Forecasting: Predicts “what will be.”

**Creating Lag Features:**

```
data['delay_lag_1'] = data['delay_minutes'].shift(1)
data['delay_lag_2'] = data['delay_minutes'].shift(2)
```

* 

**Rolling Statistics:**

```
data['rolling_avg_3'] = data['delay_minutes'].rolling(window=3).mean()
```

* **Hands-on Demo:**

	* Build a Linear Regression Model using lag features to predict current delay.
	* Plot forecasted vs actuals.

---

### 6. Forecasting Hands-on: Building Lag Features 

#### Lab Steps

* Teams engineer their own lag features.
* Build forecasting models using Linear Regression or Random Forest.
* Plot and interpret **forecast horizon vs accuracy tradeoff**.
* Prompt: “How far into the future can your model forecast before accuracy drops?”

---

### 7. ARIMA & Exponential Smoothing Basics 

#### Key Points

* **ARIMA Components:**

  * p \= autoregressive lags

  * d \= differencing order (trend removal)

  * q \= moving average lags

* **Exponential Smoothing:**

  * SES (Simple Exponential Smoothing)

  * Holt’s Linear Trend Method

  * Holt-Winters (seasonality)

**Code Walkthrough (statsmodels):**

```
from statsmodels.tsa.arima.model import ARIMA
model_arima = ARIMA(data['delay_minutes'], order=(1,1,1))
result = model_arima.fit()
print(result.summary())
```


```
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model_es = ExponentialSmoothing(data['delay_minutes'], trend='add').fit()
```


---

### 8. Forecasting Challenge 

#### Structure

* Teams predict future delays for a hidden test set.
* Submit predictions to a shared sheet.
* Compute MSE/RMSE on test intervals.
* Optional: Compare lag-feature regression models vs ARIMA models.
* Leaderboard for top-performing models.

---

### 9. Model Comparison & Ensemble Strategies

#### Key Points

* **When to Ensemble:**

	* When individual models capture different aspects of data patterns.

* **Averaging, Stacking:**

	* Simple averaging of predictions.
	* Stacking regressors with meta-model.

* **Random Forests as Ensembles:**

	* Discuss how tree diversity (different splits) leads to robust averaging.

* **Feature Engineering’s Role:**

	* The more diverse the features, the more effective ensembling becomes.

---

### 10. Wrap-up & Reflection 

#### Activities

* Distribute **Prompt Cards**:

	* "What is one engineered feature you'd try next?"
	* "Which model did you trust most and why?"
	* "How will you validate your model on new data?"

* Group sharing & instructor recap of key learning outcomes.

