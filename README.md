# 🍽️ Restaurant Cuisine Classification & Insights

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive data science and machine learning repository focused on restaurant data analysis. This project covers end-to-end workflows including **multiclass cuisine classification**, **spatial/location-based analytics**, **rating regression**, and **content-based recommendation engines**.

---

## 📌 Project Overview

This repository analyzes restaurant profile data (inspired by global restaurant directories like Zomato) to extract business insights and deliver predictive solutions across four dedicated domains:

1. **Cuisine Classification:** Predicts the primary cuisine category using operational features and ratings.
2. **Location-Based Analysis:** Performs geospatial mapping and city-level aggregations of price tiers, ratings, and cuisines.
3. **Restaurant Rating Prediction:** Employs regression pipelines to forecast a restaurant's continuous `Aggregate rating`.
4. **Restaurant Recommendation System:** Suggests top-matching eateries using TF-IDF feature extraction and cosine similarity on user preferences.

---

## 🗂️ Repository Structure

```text
RestaurantCuisineClassificationUsingMachineLearning/
└── machine learning intern/
    └── machine learning intern/
        ├── Dataset .csv
        ├── cuisine classification/
        │   └── cuisine_classification.py
        ├── location based analysis/
        │   └── location_analysis.py
        ├── restaurant rating predict/
        │   └── rating_prediction.py
        └── restaurant recommendation/
            └── recommendation_system.py

```

---

## 🚀 Key Modules & Methodologies

### 1. Cuisine Classification

* **Objective:** Categorize restaurant cuisine offerings based on operational attributes.
* **Techniques:** Missing value imputation, categorical label encoding (`LabelEncoder`), and ensemble modeling via `RandomForestClassifier`.
* **Evaluation:** Weighted Precision, Recall, Accuracy, and standard multi-class classification reports.

### 2. Location-Based Analysis

* **Objective:** Understand geospatial patterns and regional dining trends.
* **Techniques:**
* Coordinate mapping across international latitudes/longitudes using `mpl_toolkits.basemap` and `matplotlib`.
* City-level grouping to determine restaurant density, top-rated cities, and average pricing tiers with `seaborn`.
* Mode analysis to isolate popular cuisines across metro markets.



### 3. Aggregate Rating Prediction

* **Objective:** Estimate customer satisfaction (`Aggregate rating`) prior to review accumulation.
* **Techniques:**
* Clean `ColumnTransformer` preprocessing pipeline.
* Standardized numerical features (`StandardScaler`) with median/mean imputation.
* One-Hot Encoding (`OneHotEncoder`) for categorical parameters.
* Continuous regression modeling (`LinearRegression`) evaluated via Mean Squared Error (MSE) and $R^2$ Score.



### 4. Content-Based Recommendation System

* **Objective:** Match user taste profiles to the most relevant restaurants.
* **Techniques:**
* Metadata unification (Cuisines, Price Tiers, Text Ratings, and Votes).
* TF-IDF vectorization (`TfidfVectorizer`) with English stop-word filtering.
* Pairwise cosine similarity ranking (`linear_kernel`) returning top-$N$ personalized venues.



---

## 📊 Dataset Schema

The input dataset (`Dataset .csv`) contains global restaurant metrics including:

| Field | Description |
| --- | --- |
| `Restaurant ID` / `Name` | Unique identifier and public name of the venue |
| `City` / `Address` / `Locality` | Granular geographic descriptors |
| `Latitude` / `Longitude` | Coordinate pairs for geospatial plotting |
| `Cuisines` | Comma-separated food styles served |
| `Price range` | Scaled price band (e.g., 1–4) |
| `Aggregate rating` / `Rating text` | Numerical review score (0.0–5.0) and categorical descriptor |
| `Votes` | Cumulative count of user reviews submitted |

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/ThiriloseJonesNithish-R/RestaurantCuisineClassificationUsingMachineLearning.git](https://github.com/ThiriloseJonesNithish-R/RestaurantCuisineClassificationUsingMachineLearning.git)
cd RestaurantCuisineClassificationUsingMachineLearning

```


2. **Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn basemap

```


4. **Configure Dataset Path:**
Ensure each script references the correct path to `Dataset .csv` relative to your execution directory:
```python
data_path = "Dataset .csv"  # Place within the module directory or adjust path accordingly

```



---

## 💻 Usage

Run any module directly from your terminal:

* **Train the Cuisine Classifier:**
```bash
python "machine learning intern/machine learning intern/cuisine classification/cuisine_classification.py"

```


* **Generate Geospatial & City Visualizations:**
```bash
python "machine learning intern/machine learning intern/location based analysis/location_analysis.py"

```


* **Run the Rating Prediction Pipeline:**
```bash
python "machine learning intern/machine learning intern/restaurant rating predict/rating_prediction.py"

```


* **Get Personalized Recommendations:**
```bash
python "machine learning intern/machine learning intern/restaurant recommendation/recommendation_system.py"

```



---

## 📈 Dependencies

* [Python 3.8+](https://www.python.org/)
* [pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Basemap](https://matplotlib.org/basemap/)

---

## 📄 License

This repository is distributed under the [MIT License](https://www.google.com/search?q=LICENSE).
